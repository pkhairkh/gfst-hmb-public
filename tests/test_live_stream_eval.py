from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import gfst_hmb_mpuc.diagnostics.live_stream_eval as live_stream_eval


class _DummyStopEvent:
    def set(self) -> None:
        return


class _DummyThread:
    def join(self, timeout: float | None = None) -> None:
        del timeout
        return


class _DummyRuntimeSampler:
    def __init__(self, *, device, interval_sec: float) -> None:
        del device, interval_sec

    def start(self) -> None:
        return

    def close(self) -> None:
        return

    def metrics_between(self, *, start_time: float, end_time: float) -> dict[str, float]:
        del start_time, end_time
        return {
            "gpu_util_avg_pct": 55.0,
            "gpu_util_median_pct": 55.0,
            "gpu_util_p10_pct": 40.0,
        }


class _FakeValidationSource:
    def telemetry(self) -> dict[str, object]:
        return {
            "source": "streaming",
            "failed": False,
            "windows_emitted": {"full": {"2048": 8}},
            "windows_released": {"full": {"2048": 8}},
            "completed_batches": 8,
            "launched_batches": 8,
        }

    def close(self) -> None:
        return


def test_validation_heartbeat_merges_executor_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        live_stream_eval,
        "_log_event",
        lambda event, **payload: logged.append((str(event), dict(payload))),
    )
    source = _FakeValidationSource()
    stop_event, thread = live_stream_eval._start_validation_heartbeat(
        validation_source=source,
        label="multi",
        interval_seconds=0.01,
        progress_provider=lambda: {
            "executor": "legacy",
            "launched_batches": 8,
            "completed_batches": 3,
            "inflight_batches": 5,
            "pending_batches": 5,
        },
    )
    try:
        for _ in range(20):
            if logged:
                break
            live_stream_eval.time.sleep(0.01)
    finally:
        stop_event.set()
        thread.join(timeout=1.0)

    assert logged
    event, payload = logged[-1]
    assert event == "live_stream_eval_heartbeat"
    telemetry = payload["telemetry"]
    assert telemetry["source"] == "streaming"
    assert telemetry["executor"] == "legacy"
    assert telemetry["completed_batches"] == 3
    assert telemetry["compute_progress"]["inflight_batches"] == 5


def test_multi_parser_defaults_to_eight_windows() -> None:
    parser = live_stream_eval._build_parser()
    args = parser.parse_args(
        [
            "multi",
            "--student",
            "step49",
            "/tmp/step49",
            "best",
            "--student",
            "gold",
            "/tmp/gold",
            "best",
        ]
    )

    assert args.windows == 8


def test_remote_benchmark_script_uses_default_w8_profile() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "remote_live_stream_eval_benchmark.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert 'DEFAULT_WINDOWS="${DEFAULT_WINDOWS:-8}"' in script_text
    assert 'run_eval benchmark_legacy' in script_text
    assert 'run_eval benchmark_ring' in script_text
    assert '"benchmark_profile": "default_w8"' in script_text
    assert 'EXTENDED_WINDOWS="${EXTENDED_WINDOWS:-${FULL_WINDOWS:-}}"' in script_text


def test_run_multi_eval_writes_default_w8_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_load_run_bundle(run_dir, checkpoint_name, *, device, dtype):
        del device, dtype
        resolved_run_dir = Path(run_dir)
        run_config = SimpleNamespace(
            model=SimpleNamespace(model_path="Qwen/Qwen3-1.7B", block_len=128),
            data=SimpleNamespace(source_manifest_path="manifest.json"),
            run_id=f"{resolved_run_dir.name}-run",
        )
        checkpoint = SimpleNamespace(
            path=resolved_run_dir / f"{checkpoint_name}.pt",
            payload={"step": 49},
            metadata={},
        )
        return run_config, checkpoint, object(), object()

    def _fake_metrics(*, students, **kwargs):
        del kwargs
        payload = {}
        for idx, label in enumerate(students, start=1):
            payload[str(label)] = {
                "val_ce": 10.0 + float(idx),
                "teacher_gap": 1.0 + (0.1 * float(idx)),
                "pointer_gold_top1_rate_when_gold_in_candidate": 0.2,
                "gold_mrr_when_gold_in_candidate": 0.3,
                "gold_top10_rate_when_gold_in_candidate": 0.4,
                "selected_copy_matches_gold_rate_on_oracle_positions": 0.5,
                "copy_false_trigger_rate_on_non_oracle_memory_positions": 0.6,
                "copy_trigger_rate_on_oracle_positions": 0.7,
                "streaming_validation": {
                    "executor": "ring",
                    "source": "streaming",
                    "failed": False,
                    "windows_emitted": {"full": {"2048": 8}},
                    "windows_released": {"full": {"2048": 8}},
                    "completed_batches": 8,
                    "launched_batches": 8,
                },
            }
        return payload

    monkeypatch.setattr(live_stream_eval, "_enable_fast_cuda_math", lambda device: None)
    monkeypatch.setattr(live_stream_eval, "_load_run_bundle", _fake_load_run_bundle)
    monkeypatch.setattr(live_stream_eval, "_load_shared_teacher", lambda **kwargs: object())
    monkeypatch.setattr(live_stream_eval, "_streaming_eval_inputs", lambda **kwargs: (object(), object()))
    monkeypatch.setattr(live_stream_eval, "build_validation_source", lambda **kwargs: _FakeValidationSource())
    monkeypatch.setattr(
        live_stream_eval,
        "_start_validation_heartbeat",
        lambda **kwargs: (_DummyStopEvent(), _DummyThread()),
    )
    monkeypatch.setattr(live_stream_eval, "_evaluate_validation_multi_streaming", _fake_metrics)
    monkeypatch.setattr(live_stream_eval, "GpuRuntimeSampler", _DummyRuntimeSampler)
    monkeypatch.setattr(live_stream_eval, "_gpu_memory_metrics", lambda device: {"gpu_memory_peak_mb": 1234.0})
    monkeypatch.setattr(live_stream_eval, "_training_mode", lambda run_config: "canonical")
    monkeypatch.setattr(live_stream_eval, "_retrieval_layer_idx", lambda run_config: 0)

    output_root = tmp_path / "live_eval"
    args = SimpleNamespace(
        student=[
            ("step49", str(tmp_path / "step49"), "best"),
            ("step99", str(tmp_path / "step49"), "latest"),
            ("gold", str(tmp_path / "gold"), "best"),
        ],
        output_root=str(output_root),
        device="cpu",
        teacher_device="cpu",
        seq_len=2048,
        windows=8,
        eval_kind="full",
        source_manifest=None,
        gpu_slot_count="auto",
        host_refill_depth=32,
        teacher_prefetch_slots=16,
        student_stream_count=3,
        cuda_graphs="auto",
        warmup_windows=2,
        telemetry_interval_ms=200,
        executor="ring",
        keep_existing=False,
    )

    result = live_stream_eval._run_multi_eval(args)

    assert set(result["results"]) == {"step49", "step99", "gold"}
    assert result["comparison"]["benchmark_profile"] == "default_w8"
    assert result["comparison"]["validation_num_windows"] == [8]
    assert set(result["comparison"]["metrics"]) == set(live_stream_eval._comparison_metric_keys())

    for label in ("step49", "step99", "gold"):
        payload = json.loads((output_root / f"{label}.result.json").read_text(encoding="utf-8"))
        assert payload["benchmark_profile"] == "default_w8"
        assert payload["validation_num_windows"] == [8]
        assert payload["streaming_validation"]["benchmark_profile"] == "default_w8"

    comparison_payload = json.loads((output_root / "comparison.json").read_text(encoding="utf-8"))
    assert comparison_payload["benchmark_profile"] == "default_w8"
    assert comparison_payload["validation_num_windows"] == [8]
    assert set(comparison_payload["metrics"]) == set(live_stream_eval._comparison_metric_keys())
