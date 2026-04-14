from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gfst_hmb_mpuc.diagnostics.temp_local_qr_sign_problem_analysis import (
    DEFAULT_FAMILIES,
    aggregate_case_sign_problem_reports,
    analyze_case_sign_problem,
    generate_random_sign_problem_cases,
    heuristic_sign_problem_findings,
    householder_reduced_q_with_step_features,
)


def _sample_matrix() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.2],
            [0.1, 0.95],
            [0.3, 0.1],
            [0.2, -0.05],
        ],
        dtype=torch.float32,
    )


def test_householder_reduced_q_with_step_features_returns_orthonormal_q() -> None:
    traced = householder_reduced_q_with_step_features(_sample_matrix(), eps_sign=1e-8)

    assert tuple(traced["raw_q"].shape) == (4, 2)
    assert tuple(traced["signfixed_q"].shape) == (4, 2)
    assert len(traced["steps"]) == 2
    assert traced["steps"][0]["step_index"] == 0
    orth_error = (traced["signfixed_q"].transpose(0, 1) @ traced["signfixed_q"] - torch.eye(2)).abs().max()
    assert float(orth_error.item()) < 1e-5


def test_analyze_case_sign_problem_reports_column_level_counts() -> None:
    report = analyze_case_sign_problem(
        _sample_matrix(),
        case_id="toy_case",
        family="toy",
        prefix_dims=(1, 2),
        eps_sign=1e-8,
    )

    assert report["case_id"] == "toy_case"
    assert report["family"] == "toy"
    assert report["counts"]["column_count"] == 2
    assert len(report["column_reports"]) == 2
    assert "raw_q_max_abs" in report["metrics"]
    assert "prefix_projector_max_abs" in report["metrics"]
    assert "raw_alignment" in report["column_reports"][0]
    assert "tail_over_norm" in report["column_reports"][0]["step_features"]


def test_generate_random_sign_problem_cases_is_deterministic() -> None:
    cases_a = generate_random_sign_problem_cases(num_cases=10, seed=123, rows=8, cols=4)
    cases_b = generate_random_sign_problem_cases(num_cases=10, seed=123, rows=8, cols=4)

    assert [case["case_id"] for case in cases_a] == [case["case_id"] for case in cases_b]
    assert [case["family"] for case in cases_a] == [case["family"] for case in cases_b]
    for left, right in zip(cases_a, cases_b, strict=True):
        assert torch.equal(left["matrix"], right["matrix"])
    assert tuple(sorted({case["family"] for case in cases_a})) == tuple(sorted(DEFAULT_FAMILIES))


def test_aggregate_case_sign_problem_reports_produces_family_summary() -> None:
    reports = [
        analyze_case_sign_problem(
            case["matrix"],
            case_id=case["case_id"],
            family=case["family"],
            prefix_dims=(2, 4),
            eps_sign=1e-8,
        )
        for case in generate_random_sign_problem_cases(
            num_cases=10,
            seed=456,
            rows=8,
            cols=4,
        )
    ]
    summary = aggregate_case_sign_problem_reports(
        reports,
        cols=4,
        anchor_small_threshold=1e-5,
        tail_small_threshold=1e-3,
    )
    findings = heuristic_sign_problem_findings(summary)

    assert summary["overall"]["case_count"] == 10
    assert summary["overall"]["column_count"] == 40
    assert "gaussian" in summary["by_family"]
    assert len(summary["overall"]["raw_sign_flip_rate_by_column"]) == 4
    assert len(summary["worst_cases"]) <= 10
    assert len(findings) >= 1
