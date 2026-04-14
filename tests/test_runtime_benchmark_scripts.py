from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from gfst_hmb_mpuc.state import GfstDecodeState


def _load_benchmark_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "runtime_backend_benchmark.py"
    spec = importlib.util.spec_from_file_location("runtime_backend_benchmark_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_benchmark_runs_forward_in_train_mode() -> None:
    bench = _load_benchmark_module()

    class DummyStudent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(7, dtype=torch.float32))
            self.training_flags: list[bool] = []
            self.use_cache_flags: list[bool] = []
            self.last_runtime_profile = {"exact_readout.fake": {"total_ms": 1.0}}

        def forward(self, *, input_ids, use_cache, output_hidden_states=False, past_key_values=None):
            del output_hidden_states, past_key_values
            self.training_flags.append(bool(self.training))
            self.use_cache_flags.append(bool(use_cache))
            batch_size, seq_len = input_ids.shape
            logits = self.weight.view(1, 1, -1).expand(batch_size, seq_len, -1)
            return SimpleNamespace(
                logits=logits,
                generation_logits=None,
                copy_decision_logits=None,
                pointer_scores=None,
                selected_copy_token_ids=None,
            )

    student = DummyStudent()
    input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)
    labels = torch.tensor([[2, 3, -100], [2, 1, -100]], dtype=torch.long)

    result = bench._run_train_benchmark_steps(
        student=student,
        teacher=None,
        input_ids=input_ids,
        labels=labels,
        student_device=torch.device("cpu"),
        teacher_device=torch.device("cpu"),
        warmup_steps=1,
        timed_steps=1,
    )

    assert result["benchmark_kind"] == "train"
    assert student.training_flags
    assert all(student.training_flags)
    assert student.use_cache_flags
    assert all(flag is False for flag in student.use_cache_flags)


def test_inference_benchmark_uses_cache_and_gfst_decode_state() -> None:
    bench = _load_benchmark_module()

    class DummyStudent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5, dtype=torch.float32))
            self.calls: list[dict[str, object]] = []
            self.last_runtime_profile = {"exact_readout.fake": {"total_ms": 1.0}}

        def forward(self, *, input_ids, use_cache, output_hidden_states=False, past_key_values=None):
            del output_hidden_states
            self.calls.append(
                {
                    "use_cache": bool(use_cache),
                    "past_key_values": past_key_values,
                }
            )
            batch_size, seq_len = input_ids.shape
            logits = self.weight.view(1, 1, -1).expand(batch_size, seq_len, -1)
            next_state = past_key_values
            if next_state is None:
                next_state = GfstDecodeState(num_layers=1, batch_size=int(batch_size))
            return SimpleNamespace(
                logits=logits,
                generation_logits=None,
                copy_decision_logits=None,
                pointer_scores=None,
                selected_copy_token_ids=None,
                past_key_values=next_state,
            )

    student = DummyStudent()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    result = bench._run_inference_benchmark_steps(
        student=student,
        input_ids=input_ids,
        student_device=torch.device("cpu"),
        warmup_steps=0,
        timed_steps=1,
        decode_steps=1,
    )

    assert result["benchmark_kind"] == "inference"
    assert result["past_key_values_kind"] == "GfstDecodeState"
    assert len(student.calls) == 2
    assert all(bool(call["use_cache"]) for call in student.calls)
    assert student.calls[0]["past_key_values"] is None
    assert isinstance(student.calls[1]["past_key_values"], GfstDecodeState)


def test_compare_native_reuses_fixed_batch_and_seed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench = _load_benchmark_module()
    base_config = SimpleNamespace(
        model=SimpleNamespace(
            model_path="/tmp/fake-model",
            gfst_feature_backend="native",
            recurrent_kernel_backend="native",
            exact_readout_grouped_reduction_backend="native",
            segmented_reduction_backend="native",
            exact_readout_score_backend="native",
            state_pack_backend="native",
            rms_norm_backend="native",
            use_liger_losses=False,
        ),
        data=SimpleNamespace(micro_batch_size=2),
        artifacts=SimpleNamespace(output_dir=str(tmp_path / "artifacts")),
    )

    seed_calls: list[int] = []
    prepare_calls: list[dict[str, object]] = []
    variant_calls: list[dict[str, object]] = []
    fixed_batch = {
        "input_ids": torch.ones((2, 4), dtype=torch.long),
        "labels": torch.ones((2, 4), dtype=torch.long),
    }

    monkeypatch.setattr(bench, "load_training_run_config", lambda _path: base_config)
    monkeypatch.setattr(
        bench,
        "_prepare_fixed_batch_cpu",
        lambda **kwargs: prepare_calls.append(dict(kwargs)) or fixed_batch,
    )
    monkeypatch.setattr(bench, "_seed_everything", lambda seed: seed_calls.append(int(seed)))
    monkeypatch.setattr(bench, "_ensure_requested_runtime_dependencies", lambda **kwargs: None)

    def _fake_run_variant(**kwargs):
        variant_calls.append(
            {
                "batch_id": id(kwargs["batch_cpu"]),
                "backend": str(getattr(kwargs["run_config"].model, "gfst_feature_backend", "native")),
            }
        )
        payload = {
            "logits": torch.zeros((1, 1, 1), dtype=torch.float32),
            "generation_logits": None,
            "copy_decision_logits": None,
            "pointer_scores": None,
            "selected_copy_token_ids": None,
        }
        return {
            "teacher_forward_ms": 0.0,
            "student_forward_ms": 1.0,
            "exact_readout_ms": 0.0,
            "loss_build_ms": 1.0,
            "backward_ms": 1.0,
            "total_ms": 3.0,
            "tokens_per_sec": 1.0,
            "peak_memory_bytes": 0,
            "loss": 0.0,
            "runtime_profile": None,
            "parity_payload": payload,
            "benchmark_kind": str(kwargs["benchmark_kind"]),
        }

    monkeypatch.setattr(bench, "_run_variant", _fake_run_variant)

    args = bench.build_parser().parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--output",
            str(tmp_path / "bench.json"),
            "--compare-native",
            "--seed",
            "99",
        ]
    )

    result = bench.run_from_args(args)

    assert result["seed"] == 99
    assert len(prepare_calls) == 1
    assert len(variant_calls) == 2
    assert variant_calls[0]["batch_id"] == variant_calls[1]["batch_id"]
    assert seed_calls == [99, 99]
    assert result["parity_deltas"]["loss_abs_delta"] == pytest.approx(0.0, abs=1e-12)


def test_compare_native_records_mathdx_dependency_report_for_sm120_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bench = _load_benchmark_module()
    base_config = SimpleNamespace(
        model=SimpleNamespace(
            model_path="/tmp/fake-model",
            gfst_feature_backend="native",
            recurrent_kernel_backend="native",
            exact_readout_grouped_reduction_backend="native",
            segmented_reduction_backend="native",
            exact_readout_score_backend="native",
            state_pack_backend="native",
            rms_norm_backend="native",
            use_liger_losses=False,
        ),
        data=SimpleNamespace(micro_batch_size=1),
        artifacts=SimpleNamespace(output_dir=str(tmp_path / "artifacts")),
    )

    monkeypatch.setattr(bench, "load_training_run_config", lambda _path: base_config)
    monkeypatch.setattr(
        bench,
        "_prepare_fixed_batch_cpu",
        lambda **kwargs: {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "labels": torch.ones((1, 4), dtype=torch.long),
        },
    )
    monkeypatch.setattr(bench, "_seed_everything", lambda seed: None)
    monkeypatch.setattr(bench, "_ensure_requested_runtime_dependencies", lambda **kwargs: None)
    monkeypatch.setattr(
        bench,
        "mathdx_extension_dependency_report",
        lambda: {"ptxas_opt_level": "O1", "device_lto_enabled": False, "ipo_enabled": False},
    )

    payload = {
        "logits": torch.zeros((1, 1, 1), dtype=torch.float32),
        "generation_logits": None,
        "copy_decision_logits": None,
        "pointer_scores": None,
        "selected_copy_token_ids": None,
    }

    monkeypatch.setattr(
        bench,
        "_run_variant",
        lambda **kwargs: {
            "teacher_forward_ms": 0.0,
            "student_forward_ms": 1.0,
            "exact_readout_ms": 0.0,
            "loss_build_ms": 1.0,
            "backward_ms": 1.0,
            "total_ms": 3.0,
            "tokens_per_sec": 1.0,
            "peak_memory_bytes": 0,
            "loss": 0.0,
            "runtime_profile": None,
            "parity_payload": payload,
            "benchmark_kind": str(kwargs["benchmark_kind"]),
        },
    )

    args = bench.build_parser().parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--output",
            str(tmp_path / "bench.json"),
            "--compare-native",
            "--gfst-feature-backend",
            "sm120_exact",
            "--state-pack-backend",
            "cuda_packed",
        ]
    )

    result = bench.run_from_args(args)

    assert result["mathdx_dependency_report"] == {
        "ptxas_opt_level": "O1",
        "device_lto_enabled": False,
        "ipo_enabled": False,
    }


def test_runtime_profile_deprecations_flag_legacy_fallback_metric() -> None:
    bench = _load_benchmark_module()

    deprecations = bench._runtime_profile_deprecations(
        {
            "gfst.global_features.global_sketch.fallback_row_count": {
                "value_total": 2.0,
                "value_count": 1,
                "value_avg": 2.0,
            }
        }
    )

    assert "gfst.global_features.global_sketch.fallback_row_count" in deprecations
    assert "scalar_summary_fallback_row_count" in deprecations["gfst.global_features.global_sketch.fallback_row_count"]


def test_compare_native_can_emit_direct_microbench_and_deprecation_notes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bench = _load_benchmark_module()
    base_config = SimpleNamespace(
        model=SimpleNamespace(
            model_path="/tmp/fake-model",
            gfst_feature_backend="native",
            recurrent_kernel_backend="native",
            exact_readout_grouped_reduction_backend="native",
            segmented_reduction_backend="native",
            exact_readout_score_backend="native",
            state_pack_backend="native",
            rms_norm_backend="native",
            use_liger_losses=False,
        ),
        data=SimpleNamespace(micro_batch_size=2),
        artifacts=SimpleNamespace(output_dir=str(tmp_path / "artifacts")),
    )

    monkeypatch.setattr(bench, "load_training_run_config", lambda _path: base_config)
    monkeypatch.setattr(
        bench,
        "_prepare_fixed_batch_cpu",
        lambda **kwargs: {
            "input_ids": torch.ones((2, 4), dtype=torch.long),
            "labels": torch.ones((2, 4), dtype=torch.long),
        },
    )
    monkeypatch.setattr(bench, "_seed_everything", lambda seed: None)
    monkeypatch.setattr(bench, "_ensure_requested_runtime_dependencies", lambda **kwargs: None)

    payload = {
        "logits": torch.zeros((1, 1, 1), dtype=torch.float32),
        "generation_logits": None,
        "copy_decision_logits": None,
        "pointer_scores": None,
        "selected_copy_token_ids": None,
    }
    variant_capture_flags: list[bool] = []

    def _fake_run_variant(**kwargs):
        variant_capture_flags.append(bool(kwargs.get("capture_sm120_direct_microbench", False)))
        runtime_profile = None
        sm120_direct_microbench = None
        if str(getattr(kwargs["run_config"].model, "gfst_feature_backend", "native")) == "sm120_exact":
            runtime_profile = {
                "gfst.global_features.global_sketch.fallback_row_count": {
                    "value_total": 2.0,
                    "value_count": 1,
                    "value_avg": 2.0,
                }
            }
            sm120_direct_microbench = {
                "heev": {"input_shape": [2, 4, 4], "avg_ms": 12.5},
                "canonical_summary": {"eigvals_shape": [2, 4], "avg_ms": 18.25},
            }
        result = {
            "teacher_forward_ms": 0.0,
            "student_forward_ms": 1.0,
            "exact_readout_ms": 0.0,
            "loss_build_ms": 1.0,
            "backward_ms": 1.0,
            "total_ms": 3.0,
            "tokens_per_sec": 1.0,
            "peak_memory_bytes": 0,
            "loss": 0.0,
            "runtime_profile": runtime_profile,
            "parity_payload": payload,
            "benchmark_kind": str(kwargs["benchmark_kind"]),
        }
        if sm120_direct_microbench is not None:
            result["sm120_direct_microbench"] = sm120_direct_microbench
        return result

    monkeypatch.setattr(bench, "_run_variant", _fake_run_variant)

    args = bench.build_parser(default_benchmark_kind="inference").parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--output",
            str(tmp_path / "bench.json"),
            "--benchmark-kind",
            "inference",
            "--compare-native",
            "--gfst-feature-backend",
            "sm120_exact",
            "--state-pack-backend",
            "cuda_packed",
            "--include-sm120-direct-microbench",
        ]
    )

    result = bench.run_from_args(args)

    assert variant_capture_flags == [False, True]
    assert result["variant"]["sm120_direct_microbench"]["heev"]["avg_ms"] == pytest.approx(12.5)
    assert "runtime_profile_deprecations" in result
    assert "variant" in result["runtime_profile_deprecations"]
