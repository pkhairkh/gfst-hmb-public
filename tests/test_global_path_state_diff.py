from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gfst_hmb_mpuc.diagnostics.global_path_state_diff import summarize_global_path_state_diff


def _make_capture() -> dict[str, object]:
    return {
        "metadata": {
            "layer_idx": 0,
            "batch_idx": 0,
            "token_idx": 191,
        },
        "global_path": {
            "summary_count": 3,
            "summary_key_tensor": torch.zeros((3, 2), dtype=torch.float32),
            "summary_block_index_tensor": torch.tensor([0, 1, 2], dtype=torch.long),
            "summary_token_count_tensor": torch.tensor([64, 64, 63], dtype=torch.long),
            "summary_reduced_mean_tensor": torch.zeros((3, 4), dtype=torch.float32),
            "summary_basis_tensor": torch.zeros((3, 4, 2), dtype=torch.float32),
            "summary_projector_tensor": torch.zeros((3, 4, 4), dtype=torch.float32),
            "coarse_indices": torch.tensor([2, 1, 0], dtype=torch.long),
            "coarse_scores": torch.tensor([0.2, 0.1, 0.05], dtype=torch.float32),
            "final_indices": torch.tensor([2, 1], dtype=torch.long),
            "final_scores": torch.tensor([0.2, 0.1], dtype=torch.float32),
            "retrieved_block_ids_tensor": torch.tensor([2, 1], dtype=torch.long),
            "retrieved_prefix_lengths_tensor": torch.tensor([63, 64], dtype=torch.long),
            "current_block_id": 2,
            "current_prefix": 63,
            "returned_block_tensor": torch.tensor([2, 1, -1, -1], dtype=torch.long),
            "returned_score_tensor": torch.tensor([0.2, 0.1, 0.0, 0.0], dtype=torch.float32),
            "returned_prefix_tensor": torch.tensor([63, 64, 0, 0], dtype=torch.long),
            "returned_coarse_block_tensor": torch.tensor([2, 1, 0, -1], dtype=torch.long),
            "returned_coarse_score_tensor": torch.tensor([0.2, 0.1, 0.05, 0.0], dtype=torch.float32),
            "returned_facet_scores": torch.zeros((4, 2), dtype=torch.float32),
            "global_sketch": torch.eye(4, dtype=torch.float32),
            "eig_basis": torch.eye(4, 2, dtype=torch.float32),
            "global_features": torch.zeros((4,), dtype=torch.float32),
        },
    }


def test_global_path_state_diff_is_clean_when_captures_match() -> None:
    capture = _make_capture()
    summary = summarize_global_path_state_diff(capture, capture)

    assert summary["classification"] == "clean_global_path"
    assert summary["first_nonzero_group"] is None
    assert max(summary["summary_cache_deltas"].values()) == pytest.approx(0.0)
    assert max(summary["retrieval_deltas"].values()) == pytest.approx(0.0)
    assert max(summary["output_deltas"].values()) == pytest.approx(0.0)


def test_global_path_state_diff_flags_summary_cache_mismatch() -> None:
    native_capture = _make_capture()
    variant_capture = _make_capture()
    variant_capture["global_path"]["summary_projector_tensor"] = torch.ones((3, 4, 4), dtype=torch.float32)

    summary = summarize_global_path_state_diff(native_capture, variant_capture)

    assert summary["classification"] == "summary_cache_state_mismatch"
    assert summary["first_nonzero_group"] == "summary_cache"
    assert summary["summary_cache_deltas"]["summary_projector_tensor"] > 0.0


def test_global_path_state_diff_flags_retrieval_mismatch() -> None:
    native_capture = _make_capture()
    variant_capture = _make_capture()
    variant_capture["global_path"]["final_scores"] = torch.tensor([0.3, 0.05], dtype=torch.float32)

    summary = summarize_global_path_state_diff(native_capture, variant_capture)

    assert summary["classification"] == "retrieval_state_mismatch"
    assert summary["first_nonzero_group"] == "retrieval_state"
    assert summary["retrieval_deltas"]["final_scores"] > 0.0


def test_global_path_state_diff_flags_output_only_mismatch() -> None:
    native_capture = _make_capture()
    variant_capture = _make_capture()
    variant_capture["global_path"]["eig_basis"] = torch.tensor(
        [[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
    )

    summary = summarize_global_path_state_diff(native_capture, variant_capture)

    assert summary["classification"] == "output_only_mismatch"
    assert summary["first_nonzero_group"] == "output_state"
    assert summary["output_deltas"]["eig_basis"] > 0.0


def test_global_path_state_diff_script_writes_summary_and_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "gfst_global_path_state_diff.py"
    spec = importlib.util.spec_from_file_location("test_global_path_state_diff_script", script_path)
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
        "summarize_global_path_state_diff",
        lambda native_capture, variant_capture: {
            "classification": "clean_global_path",
            "first_nonzero_group": None,
            "summary_cache_deltas": {"summary_key_tensor": 0.0},
            "retrieval_deltas": {"final_scores": 0.0},
            "output_deltas": {"global_features": 0.0},
            "top_differences": [],
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "gfst_global_path_state_diff.py",
            "--capture-path",
            str(capture_path),
            "--output-root",
            str(tmp_path),
            "--site",
            "0:191",
        ],
    )

    module.main()

    summary_files = sorted(tmp_path.glob("gfst_global_path_state_diff_*_summary.json"))
    capture_files = sorted(tmp_path.glob("gfst_global_path_state_diff_*_capture.pt"))
    assert len(summary_files) == 1
    assert len(capture_files) == 1

    payload = json.loads(summary_files[0].read_text(encoding="utf-8"))
    assert payload["classification"] == "clean_global_path"
    assert payload["site_id"] == site_id
