from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .temp_local_qr_sweeps import evaluate_suite_runs, suite_output_deltas


TEMP_MARKER = "TEMP_REMOVE_AFTER_SM120_EXACT_LOCAL_QR_PARITY"
DEFAULT_REPLAY_THRESHOLD = 1e-7
ROUND1_OPERATORS = (
    "householder_cusolver_dn_sgeqrf_sorgqr_current",
    "householder_cusolver_dn_sgeqrf_householder_product",
)
ROUND2_OPERATORS = (
    "householder_cublas_sgeqrf_batched_householder_product",
    "householder_fixed_order_stepwise",
)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor([float(value) for value in values], dtype=torch.float64)
    return float(torch.quantile(tensor, torch.tensor(float(q), dtype=torch.float64)).item())


def native_suite_baseline(
    suite: dict[str, Any],
    *,
    output_runs: list[dict[str, Any]],
    replay_threshold: float = DEFAULT_REPLAY_THRESHOLD,
) -> dict[str, Any]:
    baseline_output = output_runs[0]
    q_deltas: list[float] = []
    projector_deltas: list[float] = []
    local_feature_deltas: list[float] = []
    for current_output in output_runs[1:]:
        deltas = suite_output_deltas(
            suite,
            baseline_output=baseline_output,
            current_output=current_output,
        )
        q_deltas.append(float(deltas["signfixed_q_max_abs"]))
        projector_deltas.append(float(deltas["prefix_projector_max_abs"]))
        local_feature_deltas.append(float(deltas["local_feature_max_abs"]))

    q_p99 = _quantile(q_deltas, 0.99)
    proj_p99 = _quantile(projector_deltas, 0.99)
    local_p99 = _quantile(local_feature_deltas, 0.99)
    suite_id = str(suite["suite_id"])
    return {
        "suite_id": suite_id,
        "q_self_deltas": q_deltas,
        "projector_self_deltas": projector_deltas,
        "local_feature_self_deltas": local_feature_deltas,
        "q_self_max": max(q_deltas, default=0.0),
        "projector_self_max": max(projector_deltas, default=0.0),
        "local_feature_self_max": max(local_feature_deltas, default=0.0),
        "q_ceiling": max(float(q_p99), float(replay_threshold)),
        "proj_ceiling": max(float(proj_p99), float(replay_threshold)),
        "local_feature_ceiling": max(float(local_p99), 1e-8),
        "requires_replay_recalibration": suite_id.startswith("fixture_")
        and (
            max(q_deltas, default=0.0) > float(replay_threshold)
            or max(projector_deltas, default=0.0) > float(replay_threshold)
        ),
    }


def candidate_suite_summary(
    suite: dict[str, Any],
    *,
    metrics_runs: list[dict[str, float | None]],
    output_runs: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_output = output_runs[0]
    determinism_q: list[float] = []
    determinism_projector: list[float] = []
    determinism_local_feature: list[float] = []
    for current_output in output_runs[1:]:
        deltas = suite_output_deltas(
            suite,
            baseline_output=baseline_output,
            current_output=current_output,
        )
        determinism_q.append(float(deltas["signfixed_q_max_abs"]))
        determinism_projector.append(float(deltas["prefix_projector_max_abs"]))
        determinism_local_feature.append(float(deltas["local_feature_max_abs"]))

    q_values = [float(run["signfixed_q_max_abs"] or 0.0) for run in metrics_runs]
    projector_values = [float(run["prefix_projector_max_abs"] or 0.0) for run in metrics_runs]
    local_feature_values = [float(run["reconstructed_local_feature_max_abs"] or 0.0) for run in metrics_runs]
    return {
        "suite_id": str(suite["suite_id"]),
        "median_signfixed_q_max_abs": _quantile(q_values, 0.5),
        "p90_signfixed_q_max_abs": _quantile(q_values, 0.9),
        "max_signfixed_q_max_abs": max(q_values, default=0.0),
        "median_prefix_projector_max_abs": _quantile(projector_values, 0.5),
        "p90_prefix_projector_max_abs": _quantile(projector_values, 0.9),
        "max_prefix_projector_max_abs": max(projector_values, default=0.0),
        "max_local_feature_max_abs": max(local_feature_values, default=0.0),
        "max_determinism_q": max(determinism_q, default=0.0),
        "max_determinism_projector": max(determinism_projector, default=0.0),
        "max_determinism_local_feature": max(determinism_local_feature, default=0.0),
        "clean": (
            max(q_values, default=0.0) <= float(baseline["q_ceiling"])
            and max(projector_values, default=0.0) <= float(baseline["proj_ceiling"])
            and max(determinism_q, default=0.0) <= float(baseline["q_ceiling"])
            and max(determinism_projector, default=0.0) <= float(baseline["proj_ceiling"])
        ),
    }


def aggregate_candidate_suite_summaries(
    suite_summaries: list[dict[str, Any]],
    *,
    real_suite_ids: set[str],
) -> dict[str, Any]:
    real_summaries = [summary for summary in suite_summaries if str(summary["suite_id"]) in real_suite_ids]
    if not real_summaries:
        raise ValueError("expected at least one real-suite summary")
    median_prefix_values = [float(summary["median_prefix_projector_max_abs"]) for summary in real_summaries]
    p90_prefix_values = [float(summary["p90_prefix_projector_max_abs"]) for summary in real_summaries]
    max_prefix_values = [float(summary["max_prefix_projector_max_abs"]) for summary in real_summaries]
    clean_real_suite_ids = [str(summary["suite_id"]) for summary in real_summaries if bool(summary["clean"])]
    return {
        "median_real_prefix_projector_max_abs": _quantile(median_prefix_values, 0.5),
        "p90_real_prefix_projector_max_abs": _quantile(p90_prefix_values, 0.9),
        "worst_real_prefix_projector_max_abs": max(max_prefix_values, default=0.0),
        "all_real_suites_clean": len(clean_real_suite_ids) == len(real_summaries),
        "clean_real_suite_ids": clean_real_suite_ids,
    }


def rank_candidate_aggregates(candidate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidate_records,
        key=lambda record: (
            float(record["aggregate"]["median_real_prefix_projector_max_abs"]),
            float(record["aggregate"]["p90_real_prefix_projector_max_abs"]),
            float(record["aggregate"]["worst_real_prefix_projector_max_abs"]),
        ),
    )


def select_outlier_suite_reports(
    candidate_records: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    outliers: list[dict[str, Any]] = []
    for record in candidate_records:
        operator_name = str(record["operator_name"])
        for suite_summary in record["suite_summaries"]:
            outliers.append(
                {
                    "operator_name": operator_name,
                    "suite_id": str(suite_summary["suite_id"]),
                    "max_prefix_projector_max_abs": float(suite_summary["max_prefix_projector_max_abs"]),
                    "max_signfixed_q_max_abs": float(suite_summary["max_signfixed_q_max_abs"]),
                }
            )
    outliers.sort(
        key=lambda item: (
            float(item["max_prefix_projector_max_abs"]),
            float(item["max_signfixed_q_max_abs"]),
        ),
        reverse=True,
    )
    return outliers[: max(0, int(top_k))]


def fixture_replay_pair_summary(
    fixture_bundle: dict[str, Any],
    *,
    operator_name: str,
    device: torch.device,
    replay_q_threshold: float,
    replay_proj_threshold: float,
    replay_local_feature_threshold: float,
) -> dict[str, Any]:
    capture_summaries: list[dict[str, Any]] = []
    for suite_id, payload_key in (("fixture_native_capture", "native_capture"), ("fixture_variant_capture", "variant_capture")):
        suite = {
            "suite_id": suite_id,
            "kind": "payload_full",
            "payload": fixture_bundle[payload_key],
        }
        native_runs = evaluate_suite_runs(
            suite,
            operator_name="householder_torch_qr_f32",
            device=device,
            repeat_runs=1,
        )
        candidate_runs = evaluate_suite_runs(
            suite,
            operator_name=operator_name,
            device=device,
            repeat_runs=1,
        )
        operator_deltas = suite_output_deltas(
            suite,
            baseline_output=native_runs["output_runs"][0],
            current_output=candidate_runs["output_runs"][0],
        )
        native_metrics = dict(native_runs["metrics_runs"][0])
        candidate_metrics = dict(candidate_runs["metrics_runs"][0])
        native_reproduction_error = float(native_metrics.get("reconstructed_local_feature_max_abs") or 0.0)
        candidate_reproduction_error = float(candidate_metrics.get("reconstructed_local_feature_max_abs") or 0.0)
        reproduction_excess = max(0.0, candidate_reproduction_error - native_reproduction_error)
        capture_summaries.append(
            {
                "suite_id": suite_id,
                "operator_signfixed_q_max_abs": float(operator_deltas["signfixed_q_max_abs"]),
                "operator_prefix_projector_max_abs": float(operator_deltas["prefix_projector_max_abs"]),
                "operator_local_feature_delta": float(operator_deltas["local_feature_max_abs"]),
                "native_reproduction_error": native_reproduction_error,
                "candidate_reproduction_error": candidate_reproduction_error,
                "candidate_reproduction_excess": reproduction_excess,
                "passed": (
                    float(operator_deltas["signfixed_q_max_abs"]) <= float(replay_q_threshold)
                    and float(operator_deltas["prefix_projector_max_abs"]) <= float(replay_proj_threshold)
                    and float(operator_deltas["local_feature_max_abs"]) <= float(replay_local_feature_threshold)
                ),
            }
        )
    return {
        "passed": all(bool(item["passed"]) for item in capture_summaries),
        "thresholds": {
            "operator_signfixed_q_max_abs": float(replay_q_threshold),
            "operator_prefix_projector_max_abs": float(replay_proj_threshold),
            "operator_local_feature_delta": float(replay_local_feature_threshold),
        },
        "captures": capture_summaries,
        "max_operator_signfixed_q_max_abs": max(
            float(item["operator_signfixed_q_max_abs"]) for item in capture_summaries
        ),
        "max_operator_prefix_projector_max_abs": max(
            float(item["operator_prefix_projector_max_abs"]) for item in capture_summaries
        ),
        "max_operator_local_feature_delta": max(
            float(item["operator_local_feature_delta"]) for item in capture_summaries
        ),
        "max_native_reproduction_error": max(
            float(item["native_reproduction_error"]) for item in capture_summaries
        ),
        "max_candidate_reproduction_error": max(
            float(item["candidate_reproduction_error"]) for item in capture_summaries
        ),
        "max_candidate_reproduction_excess": max(
            float(item["candidate_reproduction_excess"]) for item in capture_summaries
        ),
        "native_floor_exceeds_local_feature_threshold": any(
            float(item["native_reproduction_error"]) > float(replay_local_feature_threshold)
            for item in capture_summaries
        ),
    }


def json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
