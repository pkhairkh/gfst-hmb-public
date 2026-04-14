from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from ..gfst import basis_projector, canonicalize_qr_column_signs
from .temp_local_qr_householder_vs_mgs_dataset import modified_gram_schmidt_q


TEMP_MARKER = "TEMP_REMOVE_AFTER_SM120_EXACT_LOCAL_QR_PARITY"
DEFAULT_FAMILIES = (
    "gaussian",
    "column_scaled",
    "tiny_diagonal",
    "near_collinear",
    "low_rank_perturbed",
)


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def _first_significant_entry(column: torch.Tensor, *, eps_sign: float) -> dict[str, float | int | None]:
    mask = column.abs() > float(eps_sign)
    if not bool(mask.any().item()):
        return {
            "row_index": None,
            "value": 0.0,
            "abs_value": 0.0,
        }
    row_index = int(torch.argmax(mask.to(dtype=torch.int64)).item())
    value = float(column[row_index].item())
    return {
        "row_index": row_index,
        "value": value,
        "abs_value": abs(value),
    }


def _prefix_projector_max_abs(
    reference_q: torch.Tensor,
    candidate_q: torch.Tensor,
    *,
    prefix_dims: tuple[int, ...],
) -> float:
    max_abs = 0.0
    max_cols = min(int(reference_q.shape[1]), int(candidate_q.shape[1]))
    for dim in prefix_dims:
        prefix_dim = int(dim)
        if prefix_dim <= 0 or prefix_dim > max_cols:
            continue
        ref_projector = basis_projector(reference_q[:, :prefix_dim])
        cand_projector = basis_projector(candidate_q[:, :prefix_dim])
        max_abs = max(max_abs, float((cand_projector - ref_projector).abs().max().item()))
    return max_abs


def householder_reduced_q_with_step_features(
    matrix: torch.Tensor,
    *,
    eps_sign: float,
    eps_norm: float = 1e-12,
) -> dict[str, Any]:
    if matrix.ndim != 2:
        raise ValueError(f"expected rank-2 matrix, got shape {tuple(matrix.shape)}")
    rows, cols = matrix.shape
    if cols > rows:
        raise ValueError(f"expected rows >= cols for reduced QR, got shape {tuple(matrix.shape)}")

    working = matrix.detach().to(dtype=torch.float32).contiguous().clone()
    q_total = torch.eye(rows, device=working.device, dtype=working.dtype)
    eps = float(eps_norm)
    steps: list[dict[str, float | int]] = []

    for step_idx in range(cols):
        x = working[step_idx:, step_idx]
        x_norm = float(torch.linalg.vector_norm(x).item())
        x0 = float(x[0].item()) if int(x.numel()) > 0 else 0.0
        sign = -1.0 if x0 < 0.0 else 1.0
        alpha = sign * x_norm
        e1 = torch.zeros_like(x)
        if int(e1.numel()) > 0:
            e1[0] = 1.0
        v_raw = x + alpha * e1
        v_norm = float(torch.linalg.vector_norm(v_raw).item())
        if v_norm > eps:
            v_unit = v_raw / max(v_norm, eps)
        else:
            v_unit = torch.zeros_like(v_raw)
            if int(v_unit.numel()) > 0:
                v_unit[0] = 1.0
        panel_before = working[step_idx:, step_idx:]
        panel_after = panel_before - 2.0 * v_unit.unsqueeze(-1) * (v_unit @ panel_before).unsqueeze(0)
        next_working = working.clone()
        next_working[step_idx:, step_idx:] = panel_after
        working = next_working

        q_block_before = q_total[:, step_idx:]
        q_block_after = q_block_before - 2.0 * (q_block_before @ v_unit).unsqueeze(-1) * v_unit.unsqueeze(0)
        next_q_total = q_total.clone()
        next_q_total[:, step_idx:] = q_block_after
        q_total = next_q_total

        norm_sq = max(x_norm * x_norm, 0.0)
        tail_norm = math.sqrt(max(norm_sq - x0 * x0, 0.0))
        steps.append(
            {
                "step_index": int(step_idx),
                "x0": float(x0),
                "x_norm": float(x_norm),
                "alpha": float(alpha),
                "abs_x0_over_norm": float(abs(x0) / max(x_norm, eps)),
                "tail_over_norm": float(tail_norm / max(x_norm, eps)),
                "v_raw0_abs": float(abs(float(v_raw[0].item()))) if int(v_raw.numel()) > 0 else 0.0,
                "v_norm": float(v_norm),
            }
        )

    raw_q = q_total[:, :cols]
    signfixed_q = canonicalize_qr_column_signs(raw_q, eps_sign=float(eps_sign))
    return {
        "raw_q": _cpu_clone(raw_q),
        "signfixed_q": _cpu_clone(signfixed_q),
        "r": _cpu_clone(working[:cols, :]),
        "steps": steps,
    }


def analyze_case_sign_problem(
    matrix: torch.Tensor,
    *,
    case_id: str,
    family: str,
    prefix_dims: tuple[int, ...],
    eps_sign: float,
    reorthogonalize_mgs: bool = False,
    eps_norm: float = 1e-12,
    signfixed_residual_tol: float = 1e-5,
    alignment_tol: float = 0.999,
    anchor_small_threshold: float = 1e-5,
    tail_small_threshold: float = 1e-3,
) -> dict[str, Any]:
    candidate = _cpu_clone(matrix)
    mgs_q = modified_gram_schmidt_q(
        candidate,
        eps_sign=float(eps_sign),
        reorthogonalize=bool(reorthogonalize_mgs),
        eps_norm=float(eps_norm),
    )
    householder = householder_reduced_q_with_step_features(
        candidate,
        eps_sign=float(eps_sign),
        eps_norm=float(eps_norm),
    )
    raw_q = householder["raw_q"]
    signfixed_q = householder["signfixed_q"]

    raw_q_diff = (raw_q - mgs_q).abs()
    signfixed_q_diff = (signfixed_q - mgs_q).abs()
    prefix_projector_max_abs = _prefix_projector_max_abs(mgs_q, signfixed_q, prefix_dims=prefix_dims)

    column_reports: list[dict[str, Any]] = []
    sign_flip_count = 0
    pure_sign_like_count = 0
    residual_after_sign_count = 0
    anchor_small_count = 0
    tail_small_count = 0

    for col_idx in range(int(mgs_q.shape[1])):
        mgs_col = mgs_q[:, col_idx]
        raw_col = raw_q[:, col_idx]
        signfixed_col = signfixed_q[:, col_idx]
        raw_alignment = float(torch.dot(mgs_col, raw_col).item())
        signfixed_alignment = float(torch.dot(mgs_col, signfixed_col).item())
        raw_anchor = _first_significant_entry(raw_col, eps_sign=float(eps_sign))
        mgs_anchor = _first_significant_entry(mgs_col, eps_sign=float(eps_sign))
        step_features = dict(householder["steps"][col_idx])
        sign_flip = bool(raw_alignment < 0.0)
        pure_sign_like = bool(abs(raw_alignment) >= float(alignment_tol) and float(signfixed_q_diff[:, col_idx].max().item()) <= float(signfixed_residual_tol))
        residual_after_sign = bool(float(signfixed_q_diff[:, col_idx].max().item()) > float(signfixed_residual_tol))
        anchor_small = bool(float(raw_anchor["abs_value"]) <= float(anchor_small_threshold))
        tail_small = bool(float(step_features["tail_over_norm"]) <= float(tail_small_threshold))

        if sign_flip:
            sign_flip_count += 1
        if pure_sign_like:
            pure_sign_like_count += 1
        if residual_after_sign:
            residual_after_sign_count += 1
        if anchor_small:
            anchor_small_count += 1
        if tail_small:
            tail_small_count += 1

        column_reports.append(
            {
                "column_index": int(col_idx),
                "raw_alignment": float(raw_alignment),
                "signfixed_alignment": float(signfixed_alignment),
                "raw_sign_flip": bool(sign_flip),
                "pure_sign_like": bool(pure_sign_like),
                "residual_after_sign": bool(residual_after_sign),
                "raw_max_abs_diff": float(raw_q_diff[:, col_idx].max().item()),
                "signfixed_max_abs_diff": float(signfixed_q_diff[:, col_idx].max().item()),
                "mgs_anchor": mgs_anchor,
                "raw_anchor": raw_anchor,
                "anchor_small": bool(anchor_small),
                "tail_small": bool(tail_small),
                "step_features": step_features,
            }
        )

    worst_column = max(
        column_reports,
        key=lambda item: (
            float(item["signfixed_max_abs_diff"]),
            float(item["raw_max_abs_diff"]),
        ),
    )
    return {
        "case_id": str(case_id),
        "family": str(family),
        "matrix_shape": list(candidate.shape),
        "prefix_dims": list(prefix_dims),
        "metrics": {
            "raw_q_max_abs": float(raw_q_diff.max().item()) if int(raw_q_diff.numel()) > 0 else 0.0,
            "signfixed_q_max_abs": float(signfixed_q_diff.max().item()) if int(signfixed_q_diff.numel()) > 0 else 0.0,
            "prefix_projector_max_abs": float(prefix_projector_max_abs),
        },
        "counts": {
            "column_count": int(mgs_q.shape[1]),
            "sign_flip_count": int(sign_flip_count),
            "pure_sign_like_count": int(pure_sign_like_count),
            "residual_after_sign_count": int(residual_after_sign_count),
            "anchor_small_count": int(anchor_small_count),
            "tail_small_count": int(tail_small_count),
        },
        "worst_column": worst_column,
        "step_features": list(householder["steps"]),
        "column_reports": column_reports,
    }


def generate_random_sign_problem_cases(
    *,
    num_cases: int,
    seed: int,
    rows: int = 64,
    cols: int = 32,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
) -> list[dict[str, Any]]:
    if int(num_cases) < 0:
        raise ValueError(f"expected num_cases >= 0, got {num_cases}")
    if int(rows) < int(cols):
        raise ValueError(f"expected rows >= cols, got rows={rows} cols={cols}")
    if not families:
        raise ValueError("expected at least one family")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    cases: list[dict[str, Any]] = []
    rank = max(2, min(int(cols) // 4, 8))

    for case_idx in range(int(num_cases)):
        family = str(families[case_idx % len(families)])
        ordinal = case_idx // len(families)
        if family == "gaussian":
            matrix = torch.randn((rows, cols), generator=generator, dtype=torch.float32)
        elif family == "column_scaled":
            base = torch.randn((rows, cols), generator=generator, dtype=torch.float32)
            scales = torch.logspace(-2.0, 2.0, cols, base=10.0, dtype=torch.float32)
            matrix = base * scales.unsqueeze(0)
        elif family == "tiny_diagonal":
            matrix = torch.randn((rows, cols), generator=generator, dtype=torch.float32)
            for diag_idx in range(cols):
                tiny = 1e-7 * float(torch.randn((), generator=generator, dtype=torch.float32).item())
                matrix[diag_idx, diag_idx] = tiny
        elif family == "near_collinear":
            shared = torch.randn((rows, 1), generator=generator, dtype=torch.float32)
            noise = 1e-4 * torch.randn((rows, cols), generator=generator, dtype=torch.float32)
            matrix = shared.expand(rows, cols).clone() + noise
        elif family == "low_rank_perturbed":
            left = torch.randn((rows, rank), generator=generator, dtype=torch.float32)
            right = torch.randn((rank, cols), generator=generator, dtype=torch.float32)
            perturbation = 1e-4 * torch.randn((rows, cols), generator=generator, dtype=torch.float32)
            matrix = left @ right + perturbation
        else:
            raise ValueError(f"unsupported family {family!r}")
        cases.append(
            {
                "case_id": f"{family}_{ordinal:05d}",
                "family": family,
                "matrix": _cpu_clone(matrix),
            }
        )
    return cases


def _quantile_summary(values: Iterable[float]) -> dict[str, float]:
    values_list = [float(value) for value in values]
    if not values_list:
        return {
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    tensor = torch.tensor(values_list, dtype=torch.float64)
    quantiles = torch.quantile(tensor, torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64))
    return {
        "median": float(quantiles[0].item()),
        "p95": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
        "max": float(tensor.max().item()),
    }


def _init_family_aggregate(cols: int) -> dict[str, Any]:
    return {
        "case_count": 0,
        "column_count": 0,
        "case_signfixed_q_max_abs": [],
        "case_prefix_projector_max_abs": [],
        "case_raw_q_max_abs": [],
        "sign_flip_count": 0,
        "pure_sign_like_count": 0,
        "residual_after_sign_count": 0,
        "anchor_small_count": 0,
        "tail_small_count": 0,
        "anchor_small_and_residual_count": 0,
        "tail_small_and_residual_count": 0,
        "sum_anchor_abs_residual": 0.0,
        "count_anchor_abs_residual": 0,
        "sum_anchor_abs_clean": 0.0,
        "count_anchor_abs_clean": 0,
        "sum_tail_over_norm_residual": 0.0,
        "count_tail_over_norm_residual": 0,
        "sum_tail_over_norm_clean": 0.0,
        "count_tail_over_norm_clean": 0,
        "sign_flip_count_by_column": [0 for _ in range(cols)],
        "residual_count_by_column": [0 for _ in range(cols)],
        "column_total_by_column": [0 for _ in range(cols)],
    }


def aggregate_case_sign_problem_reports(
    case_reports: list[dict[str, Any]],
    *,
    cols: int,
    anchor_small_threshold: float,
    tail_small_threshold: float,
) -> dict[str, Any]:
    overall = _init_family_aggregate(cols)
    by_family: dict[str, dict[str, Any]] = {}
    worst_cases = sorted(
        case_reports,
        key=lambda item: (
            float(item["metrics"]["prefix_projector_max_abs"]),
            float(item["metrics"]["signfixed_q_max_abs"]),
        ),
        reverse=True,
    )[:20]

    for report in case_reports:
        family = str(report["family"])
        family_agg = by_family.setdefault(family, _init_family_aggregate(cols))
        for aggregate in (overall, family_agg):
            aggregate["case_count"] += 1
            aggregate["case_signfixed_q_max_abs"].append(float(report["metrics"]["signfixed_q_max_abs"]))
            aggregate["case_prefix_projector_max_abs"].append(float(report["metrics"]["prefix_projector_max_abs"]))
            aggregate["case_raw_q_max_abs"].append(float(report["metrics"]["raw_q_max_abs"]))
        for column in report["column_reports"]:
            col_idx = int(column["column_index"])
            residual_after_sign = bool(column["residual_after_sign"])
            anchor_small = bool(column["anchor_small"])
            tail_small = bool(column["tail_small"])
            raw_anchor_abs = float(column["raw_anchor"]["abs_value"])
            tail_over_norm = float(column["step_features"]["tail_over_norm"])
            sign_flip = bool(column["raw_sign_flip"])
            pure_sign_like = bool(column["pure_sign_like"])
            for aggregate in (overall, family_agg):
                aggregate["column_count"] += 1
                aggregate["column_total_by_column"][col_idx] += 1
                if sign_flip:
                    aggregate["sign_flip_count"] += 1
                    aggregate["sign_flip_count_by_column"][col_idx] += 1
                if pure_sign_like:
                    aggregate["pure_sign_like_count"] += 1
                if residual_after_sign:
                    aggregate["residual_after_sign_count"] += 1
                    aggregate["residual_count_by_column"][col_idx] += 1
                    aggregate["sum_anchor_abs_residual"] += raw_anchor_abs
                    aggregate["count_anchor_abs_residual"] += 1
                    aggregate["sum_tail_over_norm_residual"] += tail_over_norm
                    aggregate["count_tail_over_norm_residual"] += 1
                else:
                    aggregate["sum_anchor_abs_clean"] += raw_anchor_abs
                    aggregate["count_anchor_abs_clean"] += 1
                    aggregate["sum_tail_over_norm_clean"] += tail_over_norm
                    aggregate["count_tail_over_norm_clean"] += 1
                if anchor_small:
                    aggregate["anchor_small_count"] += 1
                    if residual_after_sign:
                        aggregate["anchor_small_and_residual_count"] += 1
                if tail_small:
                    aggregate["tail_small_count"] += 1
                    if residual_after_sign:
                        aggregate["tail_small_and_residual_count"] += 1

    def _finalize(aggregate: dict[str, Any]) -> dict[str, Any]:
        column_count = max(1, int(aggregate["column_count"]))

        def _rate(numerator: int, denominator: int) -> float:
            if int(denominator) <= 0:
                return 0.0
            return float(numerator) / float(denominator)

        return {
            "case_count": int(aggregate["case_count"]),
            "column_count": int(aggregate["column_count"]),
            "raw_q_max_abs": _quantile_summary(aggregate["case_raw_q_max_abs"]),
            "signfixed_q_max_abs": _quantile_summary(aggregate["case_signfixed_q_max_abs"]),
            "prefix_projector_max_abs": _quantile_summary(aggregate["case_prefix_projector_max_abs"]),
            "raw_sign_flip_rate": _rate(int(aggregate["sign_flip_count"]), int(aggregate["column_count"])),
            "pure_sign_like_column_rate": _rate(int(aggregate["pure_sign_like_count"]), int(aggregate["column_count"])),
            "residual_after_sign_column_rate": _rate(int(aggregate["residual_after_sign_count"]), int(aggregate["column_count"])),
            "anchor_small_column_rate": _rate(int(aggregate["anchor_small_count"]), int(aggregate["column_count"])),
            "tail_small_column_rate": _rate(int(aggregate["tail_small_count"]), int(aggregate["column_count"])),
            "residual_rate_given_anchor_small": _rate(
                int(aggregate["anchor_small_and_residual_count"]),
                int(aggregate["anchor_small_count"]),
            ),
            "residual_rate_given_anchor_not_small": _rate(
                int(aggregate["residual_after_sign_count"]) - int(aggregate["anchor_small_and_residual_count"]),
                int(aggregate["column_count"]) - int(aggregate["anchor_small_count"]),
            ),
            "residual_rate_given_tail_small": _rate(
                int(aggregate["tail_small_and_residual_count"]),
                int(aggregate["tail_small_count"]),
            ),
            "residual_rate_given_tail_not_small": _rate(
                int(aggregate["residual_after_sign_count"]) - int(aggregate["tail_small_and_residual_count"]),
                int(aggregate["column_count"]) - int(aggregate["tail_small_count"]),
            ),
            "mean_anchor_abs_if_residual": float(aggregate["sum_anchor_abs_residual"] / max(1, int(aggregate["count_anchor_abs_residual"]))),
            "mean_anchor_abs_if_clean": float(aggregate["sum_anchor_abs_clean"] / max(1, int(aggregate["count_anchor_abs_clean"]))),
            "mean_tail_over_norm_if_residual": float(
                aggregate["sum_tail_over_norm_residual"] / max(1, int(aggregate["count_tail_over_norm_residual"]))
            ),
            "mean_tail_over_norm_if_clean": float(
                aggregate["sum_tail_over_norm_clean"] / max(1, int(aggregate["count_tail_over_norm_clean"]))
            ),
            "raw_sign_flip_rate_by_column": [
                float(aggregate["sign_flip_count_by_column"][idx]) / max(1, int(aggregate["column_total_by_column"][idx]))
                for idx in range(cols)
            ],
            "residual_after_sign_rate_by_column": [
                float(aggregate["residual_count_by_column"][idx]) / max(1, int(aggregate["column_total_by_column"][idx]))
                for idx in range(cols)
            ],
            "anchor_small_threshold": float(anchor_small_threshold),
            "tail_small_threshold": float(tail_small_threshold),
            "sanity_column_denominator": int(column_count),
        }

    overall_summary = _finalize(overall)
    family_summary = {family: _finalize(aggregate) for family, aggregate in sorted(by_family.items())}
    return {
        "overall": overall_summary,
        "by_family": family_summary,
        "worst_cases": [
            {
                "case_id": str(case["case_id"]),
                "family": str(case["family"]),
                "raw_q_max_abs": float(case["metrics"]["raw_q_max_abs"]),
                "signfixed_q_max_abs": float(case["metrics"]["signfixed_q_max_abs"]),
                "prefix_projector_max_abs": float(case["metrics"]["prefix_projector_max_abs"]),
                "sign_flip_count": int(case["counts"]["sign_flip_count"]),
                "residual_after_sign_count": int(case["counts"]["residual_after_sign_count"]),
                "worst_column": case["worst_column"],
            }
            for case in worst_cases
        ],
    }


def heuristic_sign_problem_findings(summary: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    overall = dict(summary["overall"])
    gaussian = dict(summary["by_family"].get("gaussian", {}))
    near_collinear = dict(summary["by_family"].get("near_collinear", {}))
    low_rank = dict(summary["by_family"].get("low_rank_perturbed", {}))

    if gaussian:
        gaussian_pure = float(gaussian["pure_sign_like_column_rate"])
        gaussian_residual = float(gaussian["residual_after_sign_column_rate"])
        if gaussian_pure >= 0.90 and gaussian_residual <= 0.05:
            findings.append(
                "For well-conditioned Gaussian cases, the raw MGS-vs-Householder disagreement is overwhelmingly a pure column-sign/orientation effect; after sign canonicalization, most columns collapse to near-equality."
            )
    if near_collinear and gaussian:
        if float(near_collinear["residual_after_sign_column_rate"]) > 10.0 * max(1e-12, float(gaussian["residual_after_sign_column_rate"])):
            findings.append(
                "Residual mismatch after sign fixing is concentrated in near-collinear cases, which suggests the remaining problem is conditioning-sensitive and not explained by sign flips alone."
            )
    if low_rank and gaussian:
        if float(low_rank["residual_after_sign_column_rate"]) > 10.0 * max(1e-12, float(gaussian["residual_after_sign_column_rate"])):
            findings.append(
                "Low-rank perturbed cases are far more likely than Gaussian cases to keep a post-canonicalization residual, which points to non-sign numerical drift under ill-conditioning."
            )
    overall_flip_rates = list(overall.get("raw_sign_flip_rate_by_column", []))
    if overall_flip_rates:
        max_rate = max(overall_flip_rates)
        min_rate = min(overall_flip_rates)
        if max_rate - min_rate <= 0.20:
            findings.append(
                "Raw sign flips are broadly distributed across columns rather than isolated to one narrow prefix band, so there is no strong evidence of a single late-column sign-only failure mode."
            )
    if float(overall.get("residual_rate_given_anchor_small", 0.0)) <= float(overall.get("residual_rate_given_anchor_not_small", 0.0)):
        findings.append(
            "Small canonical anchor entries are not the main predictor of post-sign residuals in this sample, which argues against a simple first-nonzero tie-break explanation."
        )
    if float(overall.get("residual_rate_given_tail_small", 0.0)) > float(overall.get("residual_rate_given_tail_not_small", 0.0)):
        findings.append(
            "Cases with very small Householder tail energy are more likely to keep residual mismatch after sign fixing, which is consistent with sensitivity when the column is nearly axis-aligned or nearly rank-degenerate."
        )
    if not findings:
        findings.append("No single sign-only pattern dominated the sample; inspect the family-specific summaries and worst cases directly.")
    return findings


def render_family_markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "# MGS vs Householder Sign-Problem Family Table",
        "",
        "| family | cases | raw_sign_flip_rate | pure_sign_like_rate | residual_after_sign_rate | median_signfixed_q_max_abs | p99_signfixed_q_max_abs | median_prefix_projector_max_abs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, metrics in sorted(summary["by_family"].items()):
        signfixed = dict(metrics["signfixed_q_max_abs"])
        prefix = dict(metrics["prefix_projector_max_abs"])
        lines.append(
            "| {family} | {cases} | {flip:.6e} | {pure:.6e} | {residual:.6e} | {sf_median:.6e} | {sf_p99:.6e} | {pp_median:.6e} |".format(
                family=family,
                cases=int(metrics["case_count"]),
                flip=float(metrics["raw_sign_flip_rate"]),
                pure=float(metrics["pure_sign_like_column_rate"]),
                residual=float(metrics["residual_after_sign_column_rate"]),
                sf_median=float(signfixed["median"]),
                sf_p99=float(signfixed["p99"]),
                pp_median=float(prefix["median"]),
            )
        )
    return "\n".join(lines) + "\n"


def json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
