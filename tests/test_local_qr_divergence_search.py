from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gfst_hmb_mpuc.diagnostics.local_qr_divergence_search import (
    binary_search_first_dirty_index,
    summarize_capture_site_delta,
    verify_first_dirty_index,
)


def _capture(offset: float) -> dict[str, object]:
    return {
        "reduced_token": torch.tensor([1.0, 2.0], dtype=torch.float32) + offset,
        "largest_candidate": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32) + offset,
        "largest_basis_in": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "largest_basis_out": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32) + offset,
        "projected": torch.tensor([0.25, -0.5], dtype=torch.float32) + offset,
        "local_features": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32) + offset,
        "nested_steps": [
            {
                "nested_index": 0,
                "nested_candidate": torch.tensor([[0.5], [0.25]], dtype=torch.float32) + offset,
                "nested_basis_in": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
                "nested_basis_out": torch.tensor([[1.0], [0.0]], dtype=torch.float32) + offset,
            }
        ],
    }


def test_summarize_capture_site_delta_reports_inputs_outputs_and_nested() -> None:
    native = _capture(0.0)
    variant = _capture(0.125)

    summary = summarize_capture_site_delta(native, variant)

    assert summary["reduced_token_max_abs"] == pytest.approx(0.125)
    assert summary["largest_candidate_max_abs"] == pytest.approx(0.125)
    assert summary["largest_basis_in_max_abs"] == pytest.approx(0.0)
    assert summary["largest_basis_out_max_abs"] == pytest.approx(0.125)
    assert summary["projected_max_abs"] == pytest.approx(0.125)
    assert summary["local_features_max_abs"] == pytest.approx(0.125)
    assert summary["max_input_max_abs"] == pytest.approx(0.125)
    assert summary["max_output_max_abs"] == pytest.approx(0.125)
    assert summary["max_any_max_abs"] == pytest.approx(0.125)
    assert len(summary["nested_steps"]) == 1
    assert summary["nested_steps"][0]["nested_candidate_max_abs"] == pytest.approx(0.125)
    assert summary["nested_steps"][0]["nested_basis_in_max_abs"] == pytest.approx(0.0)
    assert summary["nested_steps"][0]["nested_basis_out_max_abs"] == pytest.approx(0.125)


def test_binary_search_first_dirty_index_returns_boundary() -> None:
    first_dirty = 9

    def is_dirty(index: int) -> bool:
        return index >= first_dirty

    found = binary_search_first_dirty_index(clean_index=0, dirty_index=16, is_dirty=is_dirty)

    assert found == first_dirty


def test_verify_first_dirty_index_walks_back_inside_window() -> None:
    first_dirty = 9

    def is_dirty(index: int) -> bool:
        return index >= first_dirty

    found = verify_first_dirty_index(
        clean_index=0,
        first_dirty_index=12,
        is_dirty=is_dirty,
        window=4,
    )

    assert found == first_dirty
