from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import torch

from ..teacher import default_device, load_frozen_teacher
from ..training.artifacts import atomic_write_json, validate_checkpoint
from ..training.config import TrainingRunConfig, load_training_run_config
from ..training.data import build_validation_source
from ..training.engine import (
    GpuRuntimeSampler,
    _enable_fast_cuda_math,
    _evaluate_validation_dual_streaming,
    _evaluate_validation_multi_streaming,
    _evaluate_validation_streaming,
    _gpu_memory_metrics,
    _run_uses_teacher,
    _training_mode,
)
from .training_recipe_eval import _from_pretrained_kwargs, _load_checkpoint_models, resolve_checkpoint_path


def _default_student_device() -> str:
    return default_device()


def _default_teacher_device() -> str:
    try:
        import torch as _torch
    except ImportError:  # pragma: no cover - guarded by project deps remotely
        return "cpu"
    if bool(_torch.cuda.is_available()) and int(_torch.cuda.device_count()) > 1:
        return "cuda:1"
    return _default_student_device()


_DEFAULT_MULTI_BENCHMARK_WINDOWS = 8


def _benchmark_profile_for_windows(*, windows: int) -> str:
    resolved_windows = int(windows)
    if int(resolved_windows) == int(_DEFAULT_MULTI_BENCHMARK_WINDOWS):
        return "default_w8"
    return f"extended_w{resolved_windows}"


def _comparison_metric_keys() -> tuple[str, ...]:
    return (
        "val_ce",
        "teacher_gap",
        "pointer_gold_top1_rate_when_gold_in_candidate",
        "gold_mrr_when_gold_in_candidate",
        "gold_top10_rate_when_gold_in_candidate",
        "selected_copy_matches_gold_rate_on_oracle_positions",
        "copy_false_trigger_rate_on_non_oracle_memory_positions",
        "copy_trigger_rate_on_oracle_positions",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live HF-streaming validation for one or two GFST-HMB checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Evaluate one checkpoint with live streaming validation.")
    single.add_argument("--run-dir", required=True)
    single.add_argument("--checkpoint", default="latest")
    single.add_argument("--output-root", default=None)
    single.add_argument("--device", default=_default_student_device())
    single.add_argument("--teacher-device", default=_default_teacher_device())
    single.add_argument("--seq-len", type=int, default=2048)
    single.add_argument("--windows", type=int, default=32)
    single.add_argument("--eval-kind", choices=("debug", "full"), default="full")
    single.add_argument("--source-manifest", default=None)
    single.add_argument("--keep-existing", action="store_true")

    dual = subparsers.add_parser("dual", help="Evaluate two checkpoints on the same streamed validation batches.")
    dual.add_argument("--primary-run-dir", required=True)
    dual.add_argument("--primary-checkpoint", default="latest")
    dual.add_argument("--primary-label", default="primary")
    dual.add_argument("--secondary-run-dir", required=True)
    dual.add_argument("--secondary-checkpoint", default="latest")
    dual.add_argument("--secondary-label", default="secondary")
    dual.add_argument("--output-root", default=None)
    dual.add_argument("--device", default=_default_student_device())
    dual.add_argument("--teacher-device", default=_default_teacher_device())
    dual.add_argument("--seq-len", type=int, default=2048)
    dual.add_argument("--windows", type=int, default=32)
    dual.add_argument("--eval-kind", choices=("debug", "full"), default="full")
    dual.add_argument("--source-manifest", default=None)
    dual.add_argument("--keep-existing", action="store_true")

    multi = subparsers.add_parser("multi", help="Evaluate multiple checkpoints on the same streamed validation batches.")
    multi.add_argument(
        "--student",
        action="append",
        nargs=3,
        metavar=("LABEL", "RUN_DIR", "CHECKPOINT"),
        required=True,
        help="Repeat for each student as: --student <label> <run_dir> <checkpoint>",
    )
    multi.add_argument("--output-root", default=None)
    multi.add_argument("--device", default=_default_student_device())
    multi.add_argument("--teacher-device", default=_default_student_device())
    multi.add_argument("--seq-len", type=int, default=2048)
    multi.add_argument("--windows", type=int, default=_DEFAULT_MULTI_BENCHMARK_WINDOWS)
    multi.add_argument("--eval-kind", choices=("debug", "full"), default="full")
    multi.add_argument("--source-manifest", default=None)
    multi.add_argument("--gpu-slot-count", default="8")
    multi.add_argument("--host-refill-depth", type=int, default=32)
    multi.add_argument("--teacher-prefetch-slots", type=int, default=16)
    multi.add_argument("--student-stream-count", type=int, default=3)
    multi.add_argument("--cuda-graphs", choices=("auto", "on", "off"), default="auto")
    multi.add_argument("--warmup-windows", type=int, default=2)
    multi.add_argument("--telemetry-interval-ms", type=int, default=200)
    multi.add_argument("--executor", choices=("ring", "legacy"), default="ring")
    multi.add_argument("--keep-existing", action="store_true")
    return parser


def _dtype_for_device(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _teacher_attn_implementation_for_device(device: torch.device) -> str | None:
    if device.type != "cuda":
        return "eager"
    try:
        major, _minor = torch.cuda.get_device_capability(device)
    except Exception:  # pragma: no cover - guarded by remote runtime
        return "eager"
    if int(major) < 8:
        return "eager"
    return None


def _log_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": str(event), **payload}, sort_keys=True), flush=True)


def _start_validation_heartbeat(
    *,
    validation_source: Any,
    label: str,
    interval_seconds: float = 5.0,
    progress_provider: Callable[[], Mapping[str, Any] | None] | None = None,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def _worker() -> None:
        while not stop_event.wait(float(interval_seconds)):
            telemetry = dict(validation_source.telemetry())
            if progress_provider is not None:
                compute_progress = progress_provider()
                if compute_progress:
                    telemetry["compute_progress"] = dict(compute_progress)
                    telemetry.update(dict(compute_progress))
            _log_event(
                "live_stream_eval_heartbeat",
                label=str(label),
                telemetry=telemetry,
            )

    thread = threading.Thread(
        target=_worker,
        name=f"live-stream-eval-heartbeat-{label}",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def _load_run_bundle(run_dir: str | Path, checkpoint_name: str, *, device: torch.device, dtype: torch.dtype) -> tuple[Any, Any, Any]:
    resolved_run_dir = Path(run_dir).resolve()
    checkpoint_path = resolve_checkpoint_path(resolved_run_dir, checkpoint_name).resolve()
    checkpoint = validate_checkpoint(checkpoint_path, require_extended_metadata=False)
    config_path = resolved_run_dir / "train_config.json"
    if config_path.exists():
        run_config = load_training_run_config(config_path)
    else:
        run_config = TrainingRunConfig.from_mapping(checkpoint.payload.get("run_config"))
    tokenizer, student, _ = _load_checkpoint_models(
        run_config=run_config,
        checkpoint=checkpoint,
        device=device,
        dtype=dtype,
        load_teacher=False,
    )
    return run_config, checkpoint, tokenizer, student


def _infer_stream_family(dataset_name: str, dataset_config: str) -> str:
    lowered = f"{dataset_name} {dataset_config}".lower()
    if "wiki" in lowered:
        return "wiki"
    if "code" in lowered or "python" in lowered or "stack" in lowered:
        return "code"
    if "science" in lowered or "pubmed" in lowered or "arxiv" in lowered:
        return "science"
    if "book" in lowered or "pile_books" in lowered:
        return "books"
    return "background"


def _ensure_stream_manifest(
    *,
    run_config: Any,
    output_root: Path,
    source_manifest_override: str | None,
) -> str:
    if source_manifest_override:
        return str(Path(str(source_manifest_override)).resolve())
    configured_path = str(getattr(run_config.data, "source_manifest_path", "") or "")
    if configured_path:
        return configured_path
    dataset_name = str(getattr(run_config.data, "dataset_name", "") or "")
    dataset_config = str(getattr(run_config.data, "dataset_config", "") or "")
    manifest_path = output_root / "live_stream_manifest.json"
    payload = [
        {
            "source_id": "live_eval_source",
            "family": _infer_stream_family(dataset_name, dataset_config),
            "stream_mode": "hf_streaming",
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "train_split": str(getattr(run_config.data, "train_split", "train") or "train"),
            "validation_split": str(getattr(run_config.data, "validation_split", "validation") or "validation"),
            "text_field": str(getattr(run_config.data, "text_field", "text") or "text"),
            "base_weight": 1.0,
            "allow_streaming_only": True,
        }
    ]
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(manifest_path)


def _streaming_eval_inputs(
    *,
    run_config: Any,
    seq_len: int,
    windows: int,
    eval_kind: str,
    output_root: Path,
    source_manifest_override: str | None,
):
    manifest_path = _ensure_stream_manifest(
        run_config=run_config,
        output_root=output_root,
        source_manifest_override=source_manifest_override,
    )
    dataset_config = replace(
        run_config.dataset_stream_config(),
        streaming=True,
        source_manifest_path=str(manifest_path),
        dev_memory_fixed_windows_path="",
        dev_generic_fixed_windows_path="",
    )
    if str(eval_kind) == "debug":
        evaluation_config = SimpleNamespace(
            debug_validation_sequence_lengths=(int(seq_len),),
            debug_validation_num_windows=(int(windows),),
            validation_sequence_lengths=(),
            validation_num_windows=(),
        )
    else:
        evaluation_config = SimpleNamespace(
            debug_validation_sequence_lengths=(),
            debug_validation_num_windows=(),
            validation_sequence_lengths=(int(seq_len),),
            validation_num_windows=(int(windows),),
        )
    return dataset_config, evaluation_config


def _prepare_output_root(path: str | Path | None, *, default_root: Path, keep_existing: bool) -> Path:
    output_root = (
        Path(path).resolve()
        if path is not None
        else default_root / f"live_stream_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    if output_root.exists() and not bool(keep_existing):
        for child in output_root.iterdir():
            if child.is_file() or child.is_symlink():
                if child.suffix == ".log":
                    continue
                child.unlink()
                continue
            shutil.rmtree(child)
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _resolve_gpu_slot_count(
    *,
    value: str,
    device: torch.device,
    seq_len: int,
    student_count: int,
) -> int | str:
    del device, seq_len, student_count
    if str(value) == "auto":
        return "auto"
    if str(value) != "auto":
        resolved = int(value)
        if resolved <= 0:
            raise ValueError("gpu_slot_count must be positive")
        return resolved
    raise ValueError("gpu_slot_count must be positive or 'auto'")


def _load_shared_teacher(*, run_config: Any, device: torch.device, dtype: torch.dtype) -> Any | None:
    if not bool(_run_uses_teacher(run_config)):
        return None
    teacher_kwargs = dict(_from_pretrained_kwargs(run_config, dtype=dtype))
    attn_implementation = _teacher_attn_implementation_for_device(device)
    if attn_implementation is not None:
        teacher_kwargs["attn_implementation"] = str(attn_implementation)
    teacher = load_frozen_teacher(
        str(run_config.model.model_path),
        **teacher_kwargs,
    ).to(device)
    teacher.eval()
    return teacher


def _retrieval_layer_idx(_run_config: Any) -> int:
    return 0


def _augment_result(
    *,
    metrics: dict[str, Any],
    run_config: Any,
    checkpoint: Any,
    seq_len: int,
    windows: int,
    eval_kind: str,
    event: str,
    eval_ms: float,
) -> dict[str, Any]:
    payload = dict(metrics)
    payload.update(
        {
            "event": str(event),
            "checkpoint_path": str(checkpoint.path),
            "checkpoint_step": int(checkpoint.payload.get("step", checkpoint.metadata.get("step", -1))),
            "source_run_id": str(run_config.run_id),
            "sequence_lengths": [int(seq_len)],
            "validation_num_windows": [int(windows)],
            "benchmark_profile": _benchmark_profile_for_windows(windows=int(windows)),
            "eval_kind": str(eval_kind),
            "eval_ms": float(eval_ms),
        }
    )
    return payload


def run_live_stream_eval(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    command = str(args.command)
    if command == "single":
        return _run_single_eval(args)
    if command == "dual":
        return _run_dual_eval(args)
    if command == "multi":
        return _run_multi_eval(args)
    raise ValueError(f"unsupported command: {command}")


def _run_single_eval(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    teacher_device = torch.device(str(args.teacher_device))
    dtype = _dtype_for_device(device)
    _enable_fast_cuda_math(device)
    _enable_fast_cuda_math(teacher_device)

    run_dir = Path(str(args.run_dir)).resolve()
    output_root = _prepare_output_root(
        args.output_root,
        default_root=run_dir,
        keep_existing=bool(args.keep_existing),
    )
    run_config, checkpoint, tokenizer, student = _load_run_bundle(
        run_dir,
        str(args.checkpoint),
        device=device,
        dtype=dtype,
    )
    teacher = _load_shared_teacher(run_config=run_config, device=teacher_device, dtype=dtype)
    dataset_config, evaluation_config = _streaming_eval_inputs(
        run_config=run_config,
        seq_len=int(args.seq_len),
        windows=int(args.windows),
        eval_kind=str(args.eval_kind),
        output_root=output_root,
        source_manifest_override=args.source_manifest,
    )
    validation_source = build_validation_source(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        evaluation_config=evaluation_config,
        text_fallback="",
    )
    _log_event(
        "live_stream_eval_started",
        mode="single",
        benchmark_profile=_benchmark_profile_for_windows(windows=int(args.windows)),
        run_id=str(run_config.run_id),
        checkpoint_path=str(checkpoint.path),
        output_root=str(output_root),
        seq_len=int(args.seq_len),
        windows=int(args.windows),
        eval_kind=str(args.eval_kind),
        device=str(device),
        teacher_device=str(teacher_device),
    )
    heartbeat_stop, heartbeat_thread = _start_validation_heartbeat(
        validation_source=validation_source,
        label="single",
    )
    try:
        eval_start = time.perf_counter()
        metrics = _evaluate_validation_streaming(
            student=student,
            teacher=teacher,
            validation_source=validation_source,
            eval_kind=str(args.eval_kind),
            device=device,
            dtype=dtype,
            training_mode=_training_mode(run_config),
            retrieval_block_len=int(run_config.model.block_len),
            retrieval_layer_idx=int(_retrieval_layer_idx(run_config)),
        )
        payload = _augment_result(
            metrics=metrics,
            run_config=run_config,
            checkpoint=checkpoint,
            seq_len=int(args.seq_len),
            windows=int(args.windows),
            eval_kind=str(args.eval_kind),
            event="live_stream_eval_done",
            eval_ms=float((time.perf_counter() - eval_start) * 1000.0),
        )
        atomic_write_json(output_root / "result.json", payload)
        _log_event(
            "live_stream_eval_finished",
            mode="single",
            benchmark_profile=_benchmark_profile_for_windows(windows=int(args.windows)),
            output_root=str(output_root),
            result_path=str(output_root / "result.json"),
            val_ce=payload.get("val_ce"),
            streaming_validation=payload.get("streaming_validation"),
        )
        return payload
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        validation_source.close()


def _run_dual_eval(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    teacher_device = torch.device(str(args.teacher_device))
    dtype = _dtype_for_device(device)
    _enable_fast_cuda_math(device)
    _enable_fast_cuda_math(teacher_device)

    primary_run_dir = Path(str(args.primary_run_dir)).resolve()
    output_root = _prepare_output_root(
        args.output_root,
        default_root=primary_run_dir,
        keep_existing=bool(args.keep_existing),
    )

    primary_config, primary_checkpoint, tokenizer, primary_student = _load_run_bundle(
        primary_run_dir,
        str(args.primary_checkpoint),
        device=device,
        dtype=dtype,
    )
    secondary_config, secondary_checkpoint, _secondary_tokenizer, secondary_student = _load_run_bundle(
        str(args.secondary_run_dir),
        str(args.secondary_checkpoint),
        device=device,
        dtype=dtype,
    )
    if str(primary_config.model.model_path) != str(secondary_config.model.model_path):
        raise ValueError("dual live eval requires both checkpoints to use the same teacher/model path")
    teacher = _load_shared_teacher(run_config=primary_config, device=teacher_device, dtype=dtype)

    manifest_run_config = primary_config if str(primary_config.data.source_manifest_path) else secondary_config
    dataset_config, evaluation_config = _streaming_eval_inputs(
        run_config=manifest_run_config,
        seq_len=int(args.seq_len),
        windows=int(args.windows),
        eval_kind=str(args.eval_kind),
        output_root=output_root,
        source_manifest_override=args.source_manifest,
    )
    validation_source = build_validation_source(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        evaluation_config=evaluation_config,
        text_fallback="",
    )
    label_a = str(args.primary_label)
    label_b = str(args.secondary_label)
    _log_event(
        "live_stream_eval_started",
        mode="dual",
        benchmark_profile=_benchmark_profile_for_windows(windows=int(args.windows)),
        output_root=str(output_root),
        seq_len=int(args.seq_len),
        windows=int(args.windows),
        eval_kind=str(args.eval_kind),
        device=str(device),
        teacher_device=str(teacher_device),
        primary_checkpoint_path=str(primary_checkpoint.path),
        secondary_checkpoint_path=str(secondary_checkpoint.path),
    )
    heartbeat_stop, heartbeat_thread = _start_validation_heartbeat(
        validation_source=validation_source,
        label="dual",
    )
    try:
        eval_start = time.perf_counter()
        metrics_by_student = _evaluate_validation_dual_streaming(
            students={
                label_a: primary_student,
                label_b: secondary_student,
            },
            teacher=teacher,
            validation_source=validation_source,
            eval_kind=str(args.eval_kind),
            device=device,
            dtype=dtype,
            training_mode=_training_mode(primary_config),
            retrieval_block_len=int(primary_config.model.block_len),
            retrieval_layer_idx=int(_retrieval_layer_idx(primary_config)),
            student_eval_params={
                label_a: {
                    "training_mode": _training_mode(primary_config),
                    "retrieval_block_len": int(primary_config.model.block_len),
                    "retrieval_layer_idx": int(_retrieval_layer_idx(primary_config)),
                },
                label_b: {
                    "training_mode": _training_mode(secondary_config),
                    "retrieval_block_len": int(secondary_config.model.block_len),
                    "retrieval_layer_idx": int(_retrieval_layer_idx(secondary_config)),
                },
            },
        )
        total_eval_ms = float((time.perf_counter() - eval_start) * 1000.0)
        primary_payload = _augment_result(
            metrics=metrics_by_student[label_a],
            run_config=primary_config,
            checkpoint=primary_checkpoint,
            seq_len=int(args.seq_len),
            windows=int(args.windows),
            eval_kind=str(args.eval_kind),
            event="live_stream_eval_done",
            eval_ms=total_eval_ms,
        )
        secondary_payload = _augment_result(
            metrics=metrics_by_student[label_b],
            run_config=secondary_config,
            checkpoint=secondary_checkpoint,
            seq_len=int(args.seq_len),
            windows=int(args.windows),
            eval_kind=str(args.eval_kind),
            event="live_stream_eval_done",
            eval_ms=total_eval_ms,
        )
        atomic_write_json(output_root / f"{label_a}.result.json", primary_payload)
        atomic_write_json(output_root / f"{label_b}.result.json", secondary_payload)
        comparison_keys = _comparison_metric_keys()
        comparison = {
            "event": "dual_live_stream_eval_done",
            "benchmark_profile": _benchmark_profile_for_windows(windows=int(args.windows)),
            "eval_kind": str(args.eval_kind),
            "sequence_lengths": [int(args.seq_len)],
            "validation_num_windows": [int(args.windows)],
            "shared_eval_ms": total_eval_ms,
            "teacher_device": str(teacher_device),
            "student_device": str(device),
            "primary_label": label_a,
            "secondary_label": label_b,
            "primary_checkpoint_path": str(primary_checkpoint.path),
            "secondary_checkpoint_path": str(secondary_checkpoint.path),
            "metrics": {
                key: {
                    label_a: primary_payload.get(key),
                    label_b: secondary_payload.get(key),
                    "delta_secondary_minus_primary": (
                        None
                        if primary_payload.get(key) is None or secondary_payload.get(key) is None
                        else float(secondary_payload[key] - primary_payload[key])
                    ),
                }
                for key in comparison_keys
            },
        }
        atomic_write_json(output_root / "comparison.json", comparison)
        _log_event(
            "live_stream_eval_finished",
            mode="dual",
            benchmark_profile=_benchmark_profile_for_windows(windows=int(args.windows)),
            output_root=str(output_root),
            comparison_path=str(output_root / "comparison.json"),
            primary_label=label_a,
            secondary_label=label_b,
        )
        return {
            "primary": primary_payload,
            "secondary": secondary_payload,
            "comparison": comparison,
        }
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        validation_source.close()


def _run_multi_eval(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.student) < 2:
        raise ValueError("multi live eval requires at least two students")
    device = torch.device(str(args.device))
    teacher_device = torch.device(str(args.teacher_device))
    dtype = _dtype_for_device(device)
    _enable_fast_cuda_math(device)
    _enable_fast_cuda_math(teacher_device)

    student_specs = [(str(label), Path(str(run_dir)).resolve(), str(checkpoint_name)) for label, run_dir, checkpoint_name in args.student]
    output_root = _prepare_output_root(
        args.output_root,
        default_root=student_specs[0][1],
        keep_existing=bool(args.keep_existing),
    )

    loaded_students: dict[str, Any] = {}
    run_configs: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    tokenizer = None
    reference_model_path: str | None = None
    manifest_run_config = None
    student_eval_params: dict[str, dict[str, Any]] = {}

    for label, run_dir, checkpoint_name in student_specs:
        run_config, checkpoint, student_tokenizer, student = _load_run_bundle(
            run_dir,
            checkpoint_name,
            device=device,
            dtype=dtype,
        )
        if tokenizer is None:
            tokenizer = student_tokenizer
        model_path = str(run_config.model.model_path)
        if reference_model_path is None:
            reference_model_path = model_path
        elif model_path != reference_model_path:
            raise ValueError("multi live eval requires all checkpoints to use the same teacher/model path")
        if manifest_run_config is None and str(run_config.data.source_manifest_path):
            manifest_run_config = run_config
        loaded_students[str(label)] = student
        run_configs[str(label)] = run_config
        checkpoints[str(label)] = checkpoint
        student_eval_params[str(label)] = {
            "training_mode": _training_mode(run_config),
            "retrieval_block_len": int(run_config.model.block_len),
            "retrieval_layer_idx": int(_retrieval_layer_idx(run_config)),
        }

    if tokenizer is None:
        raise RuntimeError("failed to load tokenizer for multi live eval")
    if manifest_run_config is None:
        manifest_run_config = next(iter(run_configs.values()))
    teacher = _load_shared_teacher(run_config=manifest_run_config, device=teacher_device, dtype=dtype)
    resolved_gpu_slot_count = _resolve_gpu_slot_count(
        value=str(args.gpu_slot_count),
        device=device,
        seq_len=int(args.seq_len),
        student_count=len(student_specs),
    )
    dataset_config, evaluation_config = _streaming_eval_inputs(
        run_config=manifest_run_config,
        seq_len=int(args.seq_len),
        windows=int(args.windows),
        eval_kind=str(args.eval_kind),
        output_root=output_root,
        source_manifest_override=args.source_manifest,
    )
    validation_source = build_validation_source(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        evaluation_config=evaluation_config,
        text_fallback="",
        host_refill_depth=int(args.host_refill_depth),
        cache_batches=False,
    )
    _log_event(
        "live_stream_eval_started",
        mode="multi",
        benchmark_profile=_benchmark_profile_for_windows(windows=int(args.windows)),
        output_root=str(output_root),
        seq_len=int(args.seq_len),
        windows=int(args.windows),
        eval_kind=str(args.eval_kind),
        device=str(device),
        teacher_device=str(teacher_device),
        executor=str(args.executor),
        gpu_slot_count=(
            int(resolved_gpu_slot_count)
            if not isinstance(resolved_gpu_slot_count, str)
            else str(resolved_gpu_slot_count)
        ),
        host_refill_depth=int(args.host_refill_depth),
        teacher_prefetch_slots=int(args.teacher_prefetch_slots),
        student_stream_count=int(args.student_stream_count),
        cuda_graphs=str(args.cuda_graphs),
        warmup_windows=int(args.warmup_windows),
        student_labels=[str(label) for label, _run_dir, _checkpoint_name in student_specs],
        checkpoint_paths={str(label): str(checkpoints[str(label)].path) for label, _run_dir, _checkpoint_name in student_specs},
    )
    progress_state_lock = threading.Lock()
    progress_state: dict[str, Any] = {
        "executor": str(args.executor),
        "launched_batches": 0,
        "completed_batches": 0,
        "inflight_batches": 0,
        "pending_batches": 0,
        "target_batches": None,
        "completed_fraction": None,
    }

    def _update_progress(payload: Mapping[str, Any]) -> None:
        with progress_state_lock:
            progress_state.clear()
            progress_state.update(dict(payload))

    def _progress_provider() -> Mapping[str, Any]:
        with progress_state_lock:
            return dict(progress_state)

    heartbeat_stop, heartbeat_thread = _start_validation_heartbeat(
        validation_source=validation_source,
        label="multi",
        interval_seconds=max(0.2, float(args.telemetry_interval_ms) / 1000.0),
        progress_provider=_progress_provider,
    )
    runtime_sampler = GpuRuntimeSampler(
        device=device,
        interval_sec=max(0.2, float(args.telemetry_interval_ms) / 1000.0),
    )
    runtime_sampler.start()
    try:
        eval_start = time.perf_counter()
        metrics_by_student = _evaluate_validation_multi_streaming(
            students=loaded_students,
            teacher=teacher,
            validation_source=validation_source,
            eval_kind=str(args.eval_kind),
            device=device,
            dtype=dtype,
            training_mode=_training_mode(manifest_run_config),
            retrieval_block_len=int(manifest_run_config.model.block_len),
            retrieval_layer_idx=int(_retrieval_layer_idx(manifest_run_config)),
            student_eval_params=student_eval_params,
            gpu_slot_count=resolved_gpu_slot_count,
            host_refill_depth=int(args.host_refill_depth),
            teacher_prefetch_slots=int(args.teacher_prefetch_slots),
            student_stream_count=int(args.student_stream_count),
            cuda_graphs=str(args.cuda_graphs),
            warmup_windows=int(args.warmup_windows),
            execution_mode=str(args.executor),
            progress_callback=_update_progress,
        )
        eval_end = time.perf_counter()
        total_eval_ms = float((eval_end - eval_start) * 1000.0)
        gpu_runtime = runtime_sampler.metrics_between(start_time=float(eval_start), end_time=float(eval_end))
        gpu_runtime.update(_gpu_memory_metrics(device))
        results_payload: dict[str, Any] = {}
        for label, _run_dir, _checkpoint_name in student_specs:
            metrics = dict(metrics_by_student[str(label)])
            streaming_validation = dict(metrics.get("streaming_validation", {}))
            streaming_validation.update(
                {
                    "benchmark_profile": _benchmark_profile_for_windows(windows=int(args.windows)),
                    "telemetry_interval_ms": int(args.telemetry_interval_ms),
                    "gpu_runtime": dict(gpu_runtime),
                }
            )
            metrics["streaming_validation"] = streaming_validation
            payload = _augment_result(
                metrics=metrics,
                run_config=run_configs[str(label)],
                checkpoint=checkpoints[str(label)],
                seq_len=int(args.seq_len),
                windows=int(args.windows),
                eval_kind=str(args.eval_kind),
                event="live_stream_eval_done",
                eval_ms=total_eval_ms,
            )
            atomic_write_json(output_root / f"{label}.result.json", payload)
            results_payload[str(label)] = payload
        comparison_keys = _comparison_metric_keys()
        comparison = {
            "event": "multi_live_stream_eval_done",
            "benchmark_profile": _benchmark_profile_for_windows(windows=int(args.windows)),
            "eval_kind": str(args.eval_kind),
            "sequence_lengths": [int(args.seq_len)],
            "validation_num_windows": [int(args.windows)],
            "shared_eval_ms": total_eval_ms,
            "teacher_device": str(teacher_device),
            "student_device": str(device),
            "executor": str(args.executor),
            "gpu_slot_count": (
                int(resolved_gpu_slot_count)
                if not isinstance(resolved_gpu_slot_count, str)
                else str(resolved_gpu_slot_count)
            ),
            "host_refill_depth": int(args.host_refill_depth),
            "teacher_prefetch_slots": int(args.teacher_prefetch_slots),
            "student_stream_count": int(args.student_stream_count),
            "cuda_graphs": str(args.cuda_graphs),
            "warmup_windows": int(args.warmup_windows),
            "telemetry_interval_ms": int(args.telemetry_interval_ms),
            "gpu_runtime": dict(gpu_runtime),
            "labels": [str(label) for label, _run_dir, _checkpoint_name in student_specs],
            "checkpoint_paths": {str(label): str(checkpoints[str(label)].path) for label, _run_dir, _checkpoint_name in student_specs},
            "metrics": {
                key: {
                    str(label): results_payload[str(label)].get(key)
                    for label, _run_dir, _checkpoint_name in student_specs
                }
                for key in comparison_keys
            },
        }
        atomic_write_json(output_root / "comparison.json", comparison)
        _log_event(
            "live_stream_eval_finished",
            mode="multi",
            benchmark_profile=_benchmark_profile_for_windows(windows=int(args.windows)),
            output_root=str(output_root),
            comparison_path=str(output_root / "comparison.json"),
            labels=[str(label) for label, _run_dir, _checkpoint_name in student_specs],
        )
        return {
            "results": results_payload,
            "comparison": comparison,
        }
    finally:
        runtime_sampler.close()
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        validation_source.close()


def main(argv: Sequence[str] | None = None) -> None:
    run_live_stream_eval(argv)


__all__ = [
    "_benchmark_profile_for_windows",
    "_comparison_metric_keys",
    "run_live_stream_eval",
    "main",
]


if __name__ == "__main__":
    main()
