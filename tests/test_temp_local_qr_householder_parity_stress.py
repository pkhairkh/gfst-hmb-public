from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gfst_hmb_mpuc.diagnostics.temp_local_qr_householder_parity_stress import (
    DEFAULT_REPLAY_THRESHOLD,
    aggregate_candidate_suite_summaries,
    candidate_suite_summary,
    fixture_replay_pair_summary,
    native_suite_baseline,
    rank_candidate_aggregates,
    select_outlier_suite_reports,
)

FIXTURE_PT = Path(__file__).resolve().parent / "fixtures" / "temp_local_qr_first_divergence_v1.pt"


def _suite() -> dict[str, object]:
    q0 = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    q1 = q0.clone()
    q1[0, 0] = 1.0 - 2e-8
    q1[2, 0] = 2e-8
    q2 = q0.clone()
    q2[1, 1] = 1.0 - 3e-8
    q2[2, 1] = 3e-8
    return {
        "suite_id": "fixture_largest",
        "kind": "matrix",
        "candidate": torch.randn((3, 2), dtype=torch.float32),
        "prefix_dims": (1, 2),
        "output_runs": [
            {"stages": {"matrix": {"signfixed_q": q0}}, "local_features": None},
            {"stages": {"matrix": {"signfixed_q": q1}}, "local_features": None},
            {"stages": {"matrix": {"signfixed_q": q2}}, "local_features": None},
        ],
    }


def test_native_suite_baseline_uses_replay_threshold_floor() -> None:
    suite = _suite()
    replay_threshold = 1e-6
    baseline = native_suite_baseline(
        suite,
        output_runs=list(suite["output_runs"]),
        replay_threshold=replay_threshold,
    )

    assert baseline["suite_id"] == "fixture_largest"
    assert baseline["q_ceiling"] >= replay_threshold
    assert baseline["proj_ceiling"] >= replay_threshold
    assert baseline["requires_replay_recalibration"] is False


def test_candidate_suite_summary_marks_suite_clean_when_within_baseline() -> None:
    suite = _suite()
    baseline = {
        "q_ceiling": 1e-4,
        "proj_ceiling": 1e-4,
    }
    summary = candidate_suite_summary(
        suite,
        metrics_runs=[
            {
                "signfixed_q_max_abs": 2e-5,
                "prefix_projector_max_abs": 3e-5,
                "reconstructed_local_feature_max_abs": None,
            },
            {
                "signfixed_q_max_abs": 1e-5,
                "prefix_projector_max_abs": 2e-5,
                "reconstructed_local_feature_max_abs": None,
            },
        ],
        output_runs=list(suite["output_runs"]),
        baseline=baseline,
    )

    assert summary["clean"] is True
    assert summary["max_signfixed_q_max_abs"] == pytest.approx(2e-5)
    assert summary["max_prefix_projector_max_abs"] == pytest.approx(3e-5)


def test_rank_candidate_aggregates_prefers_lower_median_then_p90_then_worst() -> None:
    candidate_records = [
        {
            "operator_name": "b",
            "aggregate": {
                "median_real_prefix_projector_max_abs": 2e-6,
                "p90_real_prefix_projector_max_abs": 4e-6,
                "worst_real_prefix_projector_max_abs": 5e-6,
            },
        },
        {
            "operator_name": "a",
            "aggregate": {
                "median_real_prefix_projector_max_abs": 1e-6,
                "p90_real_prefix_projector_max_abs": 4e-6,
                "worst_real_prefix_projector_max_abs": 6e-6,
            },
        },
    ]
    ranked = rank_candidate_aggregates(candidate_records)
    assert [record["operator_name"] for record in ranked] == ["a", "b"]


def test_select_outlier_suite_reports_orders_by_prefix_then_signfixed() -> None:
    candidate_records = [
        {
            "operator_name": "op0",
            "suite_summaries": [
                {
                    "suite_id": "real_a",
                    "max_prefix_projector_max_abs": 3e-5,
                    "max_signfixed_q_max_abs": 4e-5,
                },
                {
                    "suite_id": "real_b",
                    "max_prefix_projector_max_abs": 2e-5,
                    "max_signfixed_q_max_abs": 9e-5,
                },
            ],
        },
        {
            "operator_name": "op1",
            "suite_summaries": [
                {
                    "suite_id": "real_c",
                    "max_prefix_projector_max_abs": 3e-5,
                    "max_signfixed_q_max_abs": 5e-5,
                }
            ],
        },
    ]
    outliers = select_outlier_suite_reports(candidate_records, top_k=2)
    assert [(item["operator_name"], item["suite_id"]) for item in outliers] == [("op1", "real_c"), ("op0", "real_a")]


def test_aggregate_candidate_suite_summaries_requires_real_suite_set() -> None:
    summaries = [
        {
            "suite_id": "real_a",
            "clean": True,
            "median_prefix_projector_max_abs": 1e-6,
            "p90_prefix_projector_max_abs": 2e-6,
            "max_prefix_projector_max_abs": 3e-6,
        },
        {
            "suite_id": "real_b",
            "clean": False,
            "median_prefix_projector_max_abs": 2e-6,
            "p90_prefix_projector_max_abs": 3e-6,
            "max_prefix_projector_max_abs": 4e-6,
        },
    ]
    aggregate = aggregate_candidate_suite_summaries(summaries, real_suite_ids={"real_a", "real_b"})
    assert aggregate["all_real_suites_clean"] is False
    assert aggregate["clean_real_suite_ids"] == ["real_a"]


def test_fixture_replay_pair_summary_uses_operator_delta_not_reproduction_floor() -> None:
    fixture_bundle = torch.load(FIXTURE_PT, map_location="cpu")
    summary = fixture_replay_pair_summary(
        fixture_bundle,
        operator_name="householder_torch_qr_f32",
        device=torch.device("cpu"),
        replay_q_threshold=1e-7,
        replay_proj_threshold=1e-7,
        replay_local_feature_threshold=1e-8,
    )

    assert summary["passed"] is True
    assert summary["max_operator_signfixed_q_max_abs"] == pytest.approx(0.0, abs=1e-12)
    assert summary["max_operator_prefix_projector_max_abs"] == pytest.approx(0.0, abs=1e-12)
    assert summary["max_operator_local_feature_delta"] == pytest.approx(0.0, abs=1e-12)
    assert summary["max_native_reproduction_error"] > 1e-8
    assert summary["native_floor_exceeds_local_feature_threshold"] is True
