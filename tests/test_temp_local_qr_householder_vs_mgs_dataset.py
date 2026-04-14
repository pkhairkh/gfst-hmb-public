from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from gfst_hmb_mpuc.diagnostics.temp_local_qr_householder_vs_mgs_dataset import (
    TEMP_MARKER,
    compare_mgs_against_householder,
    fixture_and_payload_cases,
    householder_reduced_q_with_trace,
    modified_gram_schmidt_q,
)


def _base_payload() -> dict[str, object]:
    return {
        "metadata": {
            "layer_idx": 0,
            "batch_idx": 0,
            "token_idx": 0,
            "absolute_position": 0,
            "flag_dims": [1, 2],
            "max_flag_dim": 2,
            "eps": 1e-6,
            "eps_sign": 1e-8,
            "temporary_marker": TEMP_MARKER,
        },
        "reduced_token": torch.tensor([0.75, -1.25, 0.5, 0.25], dtype=torch.float32),
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
        "nested_steps": [
            {
                "nested_index": 0,
                "dim": 1,
                "nested_basis_in": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
                "nested_candidate": torch.tensor([[0.9], [0.25]], dtype=torch.float32),
            }
        ],
    }


def test_stepwise_householder_trace_returns_reduced_q_and_steps() -> None:
    matrix = torch.tensor(
        [
            [1.0, 0.2],
            [0.1, 0.95],
            [0.3, 0.1],
            [0.2, -0.05],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    traced = householder_reduced_q_with_trace(matrix, eps_sign=1e-8)

    assert tuple(traced["raw_q"].shape) == (4, 2)
    assert tuple(traced["signfixed_q"].shape) == (4, 2)
    assert len(traced["steps"]) == 2
    orth_error = (traced["signfixed_q"].transpose(0, 1) @ traced["signfixed_q"] - torch.eye(2)).abs().max()
    assert float(orth_error.item()) < 1e-5


def test_compare_mgs_against_householder_reports_losses_and_gradients() -> None:
    matrix = torch.tensor(
        [
            [1.0, 0.2],
            [0.1, 0.95],
            [0.3, 0.1],
            [0.2, -0.05],
        ],
        dtype=torch.float32,
    )
    comparison = compare_mgs_against_householder(matrix, prefix_dims=(1, 2), eps_sign=1e-8)

    assert tuple(comparison["mgs_q"].shape) == (4, 2)
    assert tuple(comparison["householder_signfixed_q"].shape) == (4, 2)
    assert "primary_loss" in comparison["metrics"]
    assert "prefix_projector_total_loss" in comparison["metrics"]
    assert comparison["input_grad_summary"]["present"] is True
    assert len(comparison["householder_step_reports"]) == 2
    first_step = comparison["householder_step_reports"][0]
    assert first_step["dominant_operation"] is not None
    assert "v_unit" in first_step["operations"]


def test_fixture_and_payload_cases_include_fixture_and_selected_payloads() -> None:
    payload = _base_payload()
    seam_fixture_bundle = {"native_capture": copy.deepcopy(payload)}
    selected_payloads = [copy.deepcopy(payload)]
    selected_payloads[0]["metadata"]["capture_sequence"] = 7

    cases = fixture_and_payload_cases(
        seam_fixture_bundle=seam_fixture_bundle,
        selected_payloads=selected_payloads,
    )

    case_ids = [str(case["case_id"]) for case in cases]
    assert "fixture_largest" in case_ids
    assert "fixture_nested_0" in case_ids
    assert "payload_7_largest" in case_ids
    assert "payload_7_nested_0" in case_ids


def test_modified_gram_schmidt_returns_orthonormal_columns() -> None:
    matrix = torch.tensor(
        [
            [1.0, 0.0],
            [0.2, 0.9],
            [0.1, 0.3],
            [0.4, -0.1],
        ],
        dtype=torch.float32,
    )
    q = modified_gram_schmidt_q(matrix, eps_sign=1e-8, reorthogonalize=True)
    error = (q.transpose(0, 1) @ q - torch.eye(2)).abs().max()
    assert float(error.item()) < 1e-5
