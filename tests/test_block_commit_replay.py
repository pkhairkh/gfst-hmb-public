from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gfst_hmb_mpuc.diagnostics.block_commit_replay import (
    block_commit_bundle_from_capture,
    compare_block_commit_bundle,
)


def _make_capture(*, token_idx: int = 191) -> dict[str, object]:
    projector = torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32))
    reduced = torch.tensor([1.0, -0.5, 0.25, 0.75], dtype=torch.float32)
    weight = torch.zeros((2, 8), dtype=torch.float32)
    weight[0, 0] = 1.0
    weight[1, 4] = 1.0
    return {
        "metadata": {
            "layer_idx": 0,
            "batch_idx": 0,
            "token_idx": int(token_idx),
            "absolute_position": int(token_idx),
            "flag_dims": [1, 2],
            "max_flag_dim": 2,
            "eps": 1e-6,
            "eps_sign": 1e-8,
        },
        "pre_token_inputs": {
            "hidden_token_in": torch.zeros((4,), dtype=torch.float32),
            "reduced_token": reduced,
            "query_token": torch.zeros((2,), dtype=torch.float32),
            "absolute_position": int(token_idx),
        },
        "block_memory": {
            "tracker_before": {
                "largest_basis": torch.eye(4, 2, dtype=torch.float32),
                "projector_sum": torch.zeros((4, 4), dtype=torch.float32),
                "reduced_sum": torch.zeros((4,), dtype=torch.float32),
                "current_block_tokens": 1,
                "current_block_start": int(token_idx - 1),
            },
            "largest_projector": projector,
            "commit_fired": True,
            "commit_inputs": {
                "projector_sum": 2.0 * projector,
                "reduced_sum": 2.0 * reduced,
                "token_count": 2,
                "block_key_proj_weight": weight,
            },
        },
    }


def test_block_commit_bundle_from_capture_uses_commit_inputs() -> None:
    bundle = block_commit_bundle_from_capture(_make_capture())

    assert bundle["token_count"] == 2
    assert tuple(bundle["projector_sum"].shape) == (4, 4)
    assert tuple(bundle["reduced_sum"].shape) == (4,)
    assert tuple(bundle["block_key_proj_weight"].shape) == (2, 8)


def test_compare_block_commit_bundle_is_clean_for_identical_native_replays() -> None:
    bundle = block_commit_bundle_from_capture(_make_capture())

    summary = compare_block_commit_bundle(
        bundle,
        reference_implementation="native",
        variant_implementation="native",
    )

    assert summary["classification"] == "clean_commit"
    assert summary["average_projector_max_abs"] == pytest.approx(0.0)
    assert summary["full_rank_projector_delta"] == pytest.approx(0.0)
    assert summary["top_vector_max_abs"] == pytest.approx(0.0)
    assert summary["block_key_max_abs"] == pytest.approx(0.0)


def test_compare_block_commit_bundle_flags_top_vector_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from gfst_hmb_mpuc.diagnostics import block_commit_replay as module

    replay_payloads = {
        "native": {
            "implementation": "native",
            "used_accelerated_kernel": False,
            "average_projector": torch.eye(4, dtype=torch.float32),
            "summary_basis": torch.eye(4, 2, dtype=torch.float32),
            "prefix_projectors": {1: torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)), 2: torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32))},
            "reduced_mean": torch.zeros((4,), dtype=torch.float32),
            "top_vector": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "block_key": torch.tensor([0.0, 0.0], dtype=torch.float32),
        },
        "candidate": {
            "implementation": "candidate",
            "used_accelerated_kernel": True,
            "average_projector": torch.eye(4, dtype=torch.float32),
            "summary_basis": torch.eye(4, 2, dtype=torch.float32),
            "prefix_projectors": {1: torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)), 2: torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32))},
            "reduced_mean": torch.zeros((4,), dtype=torch.float32),
            "top_vector": torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "block_key": torch.tensor([0.0, 0.0], dtype=torch.float32),
        },
    }
    monkeypatch.setattr(module, "replay_block_commit_bundle", lambda bundle, *, implementation: replay_payloads[str(implementation)])

    summary = compare_block_commit_bundle(
        {
            "projector_sum": torch.eye(4, dtype=torch.float32),
            "reduced_sum": torch.zeros((4,), dtype=torch.float32),
            "token_count": 2,
            "max_flag_dim": 2,
            "flag_dims": (1, 2),
            "eps": 1e-6,
            "eps_sign": 1e-8,
            "block_key_proj_weight": torch.zeros((2, 8), dtype=torch.float32),
            "prefix_dims": (1, 2),
        }
    )

    assert summary["classification"] == "top_vector_mismatch"
    assert summary["full_rank_projector_delta"] == pytest.approx(0.0)
    assert summary["top_vector_max_abs"] > 0.0


def test_block_commit_replay_script_writes_summary_and_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "gfst_block_commit_replay.py"
    spec = importlib.util.spec_from_file_location("test_block_commit_replay_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    capture_path = tmp_path / "capture.pt"
    site_id = "layer_0_batch_0_token_191"
    torch.save(
        {
            "native_captures": {site_id: _make_capture()},
            "variant_captures": {site_id: _make_capture()},
        },
        capture_path,
    )
    monkeypatch.setattr(
        module,
        "compare_block_commit_bundle",
        lambda bundle: {
            "classification": "clean_commit",
            "average_projector_max_abs": 0.0,
            "summary_basis_max_abs": 0.0,
            "prefix_projector_deltas": {"1": 0.0, "2": 0.0},
            "full_rank_projector_delta": 0.0,
            "reduced_mean_max_abs": 0.0,
            "top_vector_max_abs": 0.0,
            "block_key_max_abs": 0.0,
        },
    )
    monkeypatch.setattr(
        module,
        "replay_block_commit_bundle",
        lambda bundle, *, implementation: {
            "implementation": str(implementation),
            "used_accelerated_kernel": implementation == "candidate",
            "average_projector": torch.eye(4, dtype=torch.float32),
            "summary_basis": torch.eye(4, 2, dtype=torch.float32),
            "prefix_projectors": {1: torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)), 2: torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32))},
            "reduced_mean": torch.zeros((4,), dtype=torch.float32),
            "top_vector": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "block_key": torch.zeros((2,), dtype=torch.float32),
        },
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "gfst_block_commit_replay.py",
            "--capture-path",
            str(capture_path),
            "--output-root",
            str(tmp_path),
            "--site",
            "0:191",
        ],
    )
    module.main()

    summary_paths = sorted(tmp_path.glob("gfst_block_commit_replay_*_summary.json"))
    capture_paths = sorted(tmp_path.glob("gfst_block_commit_replay_*_capture.pt"))
    assert summary_paths and capture_paths
    summary = json.loads(summary_paths[-1].read_text(encoding="utf-8"))
    assert summary["site_id"] == site_id
    assert summary["native_input_replay"]["classification"] == "clean_commit"
