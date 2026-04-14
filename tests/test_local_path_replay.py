from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gfst_hmb_mpuc.diagnostics.local_path_replay import (
    compare_local_path_bundle,
    local_path_bundle_from_transition_capture,
)


def _make_transition_capture() -> dict[str, object]:
    return {
        "metadata": {
            "layer_idx": 0,
            "batch_idx": 0,
            "token_idx": 191,
            "absolute_position": 191,
            "flag_dims": [1, 2],
            "max_flag_dim": 2,
            "eps": 1e-6,
            "eps_sign": 1e-8,
            "eps_update": 1e-6,
            "gfst_update_rule": "residual_rank_one_v1",
            "forgetting_rate": 0.1,
        },
        "pre_token_inputs": {
            "hidden_token_in": torch.zeros((4,), dtype=torch.float32),
            "reduced_token": torch.tensor([0.75, -1.25, 0.5, 0.25], dtype=torch.float32),
            "query_token": torch.tensor([0.1, -0.2], dtype=torch.float32),
            "absolute_position": 191,
        },
        "local_path": {
            "largest_basis_in": torch.eye(4, 2, dtype=torch.float32),
            "largest_candidate": torch.tensor(
                [
                    [1.0, 0.2],
                    [0.1, 0.95],
                    [0.3, 0.1],
                    [0.2, -0.05],
                ],
                dtype=torch.float32,
            ),
            "largest_basis_out": torch.eye(4, 2, dtype=torch.float32),
            "projected": torch.tensor([0.5, 0.25], dtype=torch.float32),
            "nested_steps": [
                {
                    "nested_index": 0,
                    "dim": 1,
                    "nested_basis_in": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
                    "nested_candidate": torch.tensor([[0.9], [0.25]], dtype=torch.float32),
                    "nested_basis_out": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
                }
            ],
            "local_features": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
        },
    }


def test_local_path_bundle_from_transition_capture_extracts_required_fields() -> None:
    bundle = local_path_bundle_from_transition_capture(_make_transition_capture())

    assert bundle["metadata"]["eps_update"] == pytest.approx(1e-6)
    assert bundle["metadata"]["gfst_update_rule"] == "residual_rank_one_v1"
    assert bundle["metadata"]["forgetting_rate"] == pytest.approx(0.1)
    assert tuple(bundle["largest_basis_in"].shape) == (4, 2)
    assert len(bundle["nested_steps"]) == 1
    assert tuple(bundle["nested_steps"][0]["nested_basis_in"].shape) == (2, 1)


def test_compare_local_path_bundle_is_clean_for_identical_native_replays() -> None:
    bundle = local_path_bundle_from_transition_capture(_make_transition_capture())

    summary = compare_local_path_bundle(
        bundle,
        reference_implementation="native",
        variant_implementation="native",
    )

    assert summary["classification"] == "clean_local_path"
    assert summary["largest_basis_out_max_abs"] == pytest.approx(0.0)
    assert summary["largest_full_rank_projector_delta"] == pytest.approx(0.0)
    assert summary["projected_max_abs"] == pytest.approx(0.0)
    assert summary["local_features_max_abs"] == pytest.approx(0.0)


def test_compare_local_path_bundle_flags_largest_qr_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from gfst_hmb_mpuc.diagnostics import local_path_replay as module

    payloads = {
        "native": {
            "implementation": "native",
            "used_accelerated_kernel": False,
            "largest_candidate": torch.zeros((4, 2), dtype=torch.float32),
            "largest_basis_in": torch.zeros((4, 2), dtype=torch.float32),
            "largest_basis_out": torch.zeros((4, 2), dtype=torch.float32),
            "largest_full_rank_projector": torch.zeros((4, 4), dtype=torch.float32),
            "projected": torch.zeros((2,), dtype=torch.float32),
            "nested_steps": [],
            "local_features": torch.zeros((4,), dtype=torch.float32),
        },
        "candidate": {
            "implementation": "candidate",
            "used_accelerated_kernel": True,
            "largest_candidate": torch.zeros((4, 2), dtype=torch.float32),
            "largest_basis_in": torch.zeros((4, 2), dtype=torch.float32),
            "largest_basis_out": torch.tensor([[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
            "largest_full_rank_projector": torch.zeros((4, 4), dtype=torch.float32),
            "projected": torch.zeros((2,), dtype=torch.float32),
            "nested_steps": [],
            "local_features": torch.zeros((4,), dtype=torch.float32),
        },
    }
    monkeypatch.setattr(module, "replay_local_path_bundle", lambda bundle, *, implementation: payloads[str(implementation)])

    summary = compare_local_path_bundle({"metadata": {"max_flag_dim": 2}})

    assert summary["classification"] == "largest_qr_mismatch"
    assert summary["largest_basis_out_max_abs"] > 0.0


def test_compare_local_path_bundle_flags_nested_qr_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from gfst_hmb_mpuc.diagnostics import local_path_replay as module

    shared = {
        "implementation": "native",
        "used_accelerated_kernel": False,
        "largest_candidate": torch.zeros((4, 2), dtype=torch.float32),
        "largest_basis_in": torch.zeros((4, 2), dtype=torch.float32),
        "largest_basis_out": torch.zeros((4, 2), dtype=torch.float32),
        "largest_full_rank_projector": torch.zeros((4, 4), dtype=torch.float32),
        "projected": torch.zeros((2,), dtype=torch.float32),
        "local_features": torch.zeros((4,), dtype=torch.float32),
    }
    payloads = {
        "native": {
            **shared,
            "nested_steps": [
                {
                    "nested_index": 0,
                    "dim": 1,
                    "nested_candidate": torch.zeros((2, 1), dtype=torch.float32),
                    "nested_basis_in": torch.zeros((2, 1), dtype=torch.float32),
                    "nested_basis_out": torch.zeros((2, 1), dtype=torch.float32),
                    "nested_projector": torch.zeros((2, 2), dtype=torch.float32),
                }
            ],
        },
        "candidate": {
            **shared,
            "implementation": "candidate",
            "used_accelerated_kernel": True,
            "nested_steps": [
                {
                    "nested_index": 0,
                    "dim": 1,
                    "nested_candidate": torch.zeros((2, 1), dtype=torch.float32),
                    "nested_basis_in": torch.zeros((2, 1), dtype=torch.float32),
                    "nested_basis_out": torch.tensor([[0.1], [0.0]], dtype=torch.float32),
                    "nested_projector": torch.zeros((2, 2), dtype=torch.float32),
                }
            ],
        },
    }
    monkeypatch.setattr(module, "replay_local_path_bundle", lambda bundle, *, implementation: payloads[str(implementation)])

    summary = compare_local_path_bundle({"metadata": {"max_flag_dim": 2}})

    assert summary["classification"] == "nested_qr_mismatch"
    assert summary["nested_steps"][0]["nested_basis_out_max_abs"] > 0.0


def test_compare_local_path_bundle_flags_feature_projection_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from gfst_hmb_mpuc.diagnostics import local_path_replay as module

    shared = {
        "largest_candidate": torch.zeros((4, 2), dtype=torch.float32),
        "largest_basis_in": torch.zeros((4, 2), dtype=torch.float32),
        "largest_basis_out": torch.zeros((4, 2), dtype=torch.float32),
        "largest_full_rank_projector": torch.zeros((4, 4), dtype=torch.float32),
        "nested_steps": [],
    }
    payloads = {
        "native": {
            "implementation": "native",
            "used_accelerated_kernel": False,
            **shared,
            "projected": torch.zeros((2,), dtype=torch.float32),
            "local_features": torch.zeros((4,), dtype=torch.float32),
        },
        "candidate": {
            "implementation": "candidate",
            "used_accelerated_kernel": True,
            **shared,
            "projected": torch.tensor([0.1, 0.0], dtype=torch.float32),
            "local_features": torch.zeros((4,), dtype=torch.float32),
        },
    }
    monkeypatch.setattr(module, "replay_local_path_bundle", lambda bundle, *, implementation: payloads[str(implementation)])

    summary = compare_local_path_bundle({"metadata": {"max_flag_dim": 2}})

    assert summary["classification"] == "feature_projection_mismatch"
    assert summary["projected_max_abs"] > 0.0


def test_local_path_replay_script_writes_summary_and_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "gfst_local_path_replay.py"
    spec = importlib.util.spec_from_file_location("test_local_path_replay_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    capture_path = tmp_path / "capture.pt"
    site_id = "layer_0_batch_0_token_191"
    torch.save(
        {
            "native_captures": {site_id: _make_transition_capture()},
            "variant_captures": {site_id: _make_transition_capture()},
        },
        capture_path,
    )
    monkeypatch.setattr(
        module,
        "compare_local_path_bundle",
        lambda bundle, **kwargs: {
            "classification": "clean_local_path",
            "largest_candidate_max_abs": 0.0,
            "largest_basis_out_max_abs": 0.0,
            "largest_full_rank_projector_delta": 0.0,
            "projected_max_abs": 0.0,
            "local_features_max_abs": 0.0,
            "nested_steps": [],
        },
    )
    monkeypatch.setattr(
        module,
        "replay_local_path_bundle",
        lambda bundle, *, implementation: {
            "implementation": str(implementation),
            "used_accelerated_kernel": implementation == "candidate",
            "largest_candidate": torch.zeros((4, 2), dtype=torch.float32),
            "largest_basis_in": torch.zeros((4, 2), dtype=torch.float32),
            "largest_basis_out": torch.zeros((4, 2), dtype=torch.float32),
            "largest_full_rank_projector": torch.zeros((4, 4), dtype=torch.float32),
            "projected": torch.zeros((2,), dtype=torch.float32),
            "nested_steps": [],
            "local_features": torch.zeros((4,), dtype=torch.float32),
        },
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "gfst_local_path_replay.py",
            "--capture-path",
            str(capture_path),
            "--output-root",
            str(tmp_path),
            "--site",
            "0:191",
        ],
    )
    module.main()

    summary_paths = sorted(tmp_path.glob("gfst_local_path_replay_*_summary.json"))
    capture_paths = sorted(tmp_path.glob("gfst_local_path_replay_*_capture.pt"))
    assert summary_paths and capture_paths
    summary = json.loads(summary_paths[-1].read_text(encoding="utf-8"))
    assert summary["site_id"] == site_id
    assert summary["native_input_replay"]["classification"] == "clean_local_path"
