from __future__ import annotations

from typing import Any

import torch

from ..gfst import (
    alignment_components_from_full_basis_projector_batch,
    basis_projector,
    top_eig_summary_from_eigpairs_batched,
)


def reference_batched_canonical_summary_from_eigpairs(
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    reduced_tokens: torch.Tensor,
    *,
    cutoffs: tuple[int, ...] | list[int],
    target_rank: int,
    eps: float,
) -> dict[str, torch.Tensor]:
    reference = top_eig_summary_from_eigpairs_batched(
        eigvals,
        eigvecs,
        rank=int(target_rank),
        rank_boundaries=tuple(int(dim) for dim in cutoffs),
        output_dtype=torch.float32,
        vector_output_dtype=torch.float32,
    )
    reference["features"] = alignment_components_from_full_basis_projector_batch(
        reduced_tokens.to(device=eigvals.device, dtype=torch.float32),
        reference["full_basis"].to(device=eigvals.device, dtype=torch.float32),
        tuple(int(dim) for dim in cutoffs),
        float(eps),
    )
    return reference


def _max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs - rhs).abs().max().item())


def _prefix_projector_residuals(
    actual_basis: torch.Tensor,
    reference_basis: torch.Tensor,
    *,
    cutoffs: tuple[int, ...] | list[int],
) -> dict[str, float]:
    residuals: dict[str, float] = {}
    for cutoff in tuple(int(dim) for dim in cutoffs):
        actual_projector = basis_projector(actual_basis[:, : int(cutoff)])
        reference_projector = basis_projector(reference_basis[:, : int(cutoff)])
        residuals[str(int(cutoff))] = _max_abs(actual_projector, reference_projector)
    return residuals


def summarize_batched_canonical_summary_diff(
    *,
    actual_basis: torch.Tensor,
    actual_features: torch.Tensor,
    actual_top_vec: torch.Tensor,
    reference_basis: torch.Tensor,
    reference_features: torch.Tensor,
    reference_top_vec: torch.Tensor,
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    reduced_tokens: torch.Tensor,
    global_sketch: torch.Tensor | None,
    cutoffs: tuple[int, ...] | list[int],
    basis_projector_tol: float = 1e-6,
    top_vec_tol: float = 1e-6,
    feature_tol: float = 1e-9,
) -> dict[str, Any]:
    batch_size = int(actual_basis.shape[0])
    row_summaries: list[dict[str, Any]] = []
    first_dirty_row = None
    first_dirty_bucket = "clean"

    for row_idx in range(batch_size):
        projector_residuals = _prefix_projector_residuals(
            actual_basis[row_idx],
            reference_basis[row_idx],
            cutoffs=cutoffs,
        )
        max_projector_residual = max(projector_residuals.values(), default=0.0)
        basis_max_abs = _max_abs(actual_basis[row_idx], reference_basis[row_idx])
        top_vec_max_abs = _max_abs(actual_top_vec[row_idx], reference_top_vec[row_idx])
        features_max_abs = _max_abs(actual_features[row_idx], reference_features[row_idx])
        ortho_residual = _max_abs(
            actual_basis[row_idx].transpose(0, 1) @ actual_basis[row_idx],
            torch.eye(
                int(actual_basis.shape[-1]),
                device=actual_basis.device,
                dtype=actual_basis.dtype,
            ),
        )
        bucket = "clean"
        if max_projector_residual > float(basis_projector_tol):
            bucket = "basis"
        elif top_vec_max_abs > float(top_vec_tol):
            bucket = "top_vec"
        elif features_max_abs > float(feature_tol):
            bucket = "features"
        row_summary = {
            "row_idx": int(row_idx),
            "bucket": bucket,
            "basis_max_abs": float(basis_max_abs),
            "basis_prefix_projector_max_abs": float(max_projector_residual),
            "basis_prefix_projector_residuals": projector_residuals,
            "top_vec_max_abs": float(top_vec_max_abs),
            "features_max_abs": float(features_max_abs),
            "orthonormality_residual": float(ortho_residual),
        }
        row_summaries.append(row_summary)
        if first_dirty_row is None and bucket != "clean":
            first_dirty_row = int(row_idx)
            first_dirty_bucket = str(bucket)

    payload: dict[str, Any] = {
        "batch_size": int(batch_size),
        "cutoffs": [int(dim) for dim in cutoffs],
        "first_dirty_row": first_dirty_row,
        "first_dirty_bucket": first_dirty_bucket,
        "row_summaries": row_summaries,
    }
    if first_dirty_row is not None:
        row_idx = int(first_dirty_row)
        payload["first_dirty_capture"] = {
            "row_idx": row_idx,
            "global_sketch": None if global_sketch is None else global_sketch.detach().to(device="cpu").clone(),
            "eigvals": eigvals.detach().to(device="cpu").clone(),
            "eigvecs": eigvecs.detach().to(device="cpu").clone(),
            "reduced_tokens": reduced_tokens.detach().to(device="cpu").clone(),
            "actual_basis": actual_basis.detach().to(device="cpu").clone(),
            "actual_features": actual_features.detach().to(device="cpu").clone(),
            "actual_top_vec": actual_top_vec.detach().to(device="cpu").clone(),
            "reference_basis": reference_basis.detach().to(device="cpu").clone(),
            "reference_features": reference_features.detach().to(device="cpu").clone(),
            "reference_top_vec": reference_top_vec.detach().to(device="cpu").clone(),
        }
    return payload
