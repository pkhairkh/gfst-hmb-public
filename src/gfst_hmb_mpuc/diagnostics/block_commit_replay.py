from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..gfst import basis_projector, top_eig_summary
from ..runtime_backends import gfst_block_commit, sm120_exact_math_kernels_available

PREFIX_DIAGNOSTIC_DIMS = (4, 8, 16, 32)


def _tensor_max_abs(lhs: torch.Tensor | None, rhs: torch.Tensor | None) -> float:
    if lhs is None and rhs is None:
        return 0.0
    if lhs is None or rhs is None:
        return 1.0
    if tuple(lhs.shape) != tuple(rhs.shape):
        return 1.0
    if int(lhs.numel()) == 0:
        return 0.0
    lhs_f = torch.nan_to_num(lhs.to(dtype=torch.float32), nan=0.0, posinf=1e30, neginf=-1e30)
    rhs_f = torch.nan_to_num(rhs.to(dtype=torch.float32), nan=0.0, posinf=1e30, neginf=-1e30)
    return float((rhs_f - lhs_f).abs().max().item())


def _canonical_prefix_dims(*, max_flag_dim: int) -> tuple[int, ...]:
    dims = sorted({int(dim) for dim in PREFIX_DIAGNOSTIC_DIMS if 0 < int(dim) <= int(max_flag_dim)})
    if int(max_flag_dim) > 0 and int(max_flag_dim) not in dims:
        dims.append(int(max_flag_dim))
    return tuple(dims)


def _prefix_projectors_from_basis(basis: torch.Tensor, *, prefix_dims: tuple[int, ...]) -> dict[int, torch.Tensor]:
    projectors: dict[int, torch.Tensor] = {}
    for dim in prefix_dims:
        if int(dim) <= int(basis.shape[1]):
            projector = basis_projector(basis[:, : int(dim)])
            projectors[int(dim)] = (0.5 * (projector + projector.transpose(0, 1))).to(dtype=basis.dtype)
    return projectors


def _project_block_key(
    weight: torch.Tensor | None,
    *,
    reduced_mean: torch.Tensor,
    top_vector: torch.Tensor,
) -> torch.Tensor | None:
    if weight is None:
        return None
    key_input = torch.cat((reduced_mean, top_vector.to(dtype=reduced_mean.dtype)), dim=0)
    return F.linear(key_input.to(dtype=weight.dtype), weight).to(dtype=torch.float32)


def block_commit_bundle_from_capture(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(capture.get("metadata", {}))
    block_memory = dict(capture.get("block_memory", {}))
    tracker_before = dict(block_memory.get("tracker_before", {}))
    commit_inputs = dict(block_memory.get("commit_inputs") or {})
    pre_token_inputs = dict(capture.get("pre_token_inputs", {}))

    projector_sum = commit_inputs.get("projector_sum")
    if projector_sum is None:
        tracker_projector_sum = tracker_before.get("projector_sum")
        largest_projector = block_memory.get("largest_projector")
        if isinstance(tracker_projector_sum, torch.Tensor) and isinstance(largest_projector, torch.Tensor):
            projector_sum = tracker_projector_sum + largest_projector

    reduced_sum = commit_inputs.get("reduced_sum")
    if reduced_sum is None:
        tracker_reduced_sum = tracker_before.get("reduced_sum")
        reduced_token = pre_token_inputs.get("reduced_token")
        if isinstance(tracker_reduced_sum, torch.Tensor) and isinstance(reduced_token, torch.Tensor):
            reduced_sum = tracker_reduced_sum + reduced_token

    token_count = commit_inputs.get("token_count")
    if token_count is None:
        token_count = int(tracker_before.get("current_block_tokens", 0)) + 1

    return {
        "metadata": metadata,
        "projector_sum": None if projector_sum is None else projector_sum.detach().to(device="cpu", dtype=torch.float32).clone(),
        "reduced_sum": None if reduced_sum is None else reduced_sum.detach().to(device="cpu", dtype=torch.float32).clone(),
        "token_count": int(token_count),
        "max_flag_dim": int(metadata.get("max_flag_dim", 0) or 0),
        "flag_dims": tuple(int(dim) for dim in metadata.get("flag_dims", [])),
        "eps": float(metadata.get("eps", 1e-6)),
        "eps_sign": float(metadata.get("eps_sign", 1e-8)),
        "block_key_proj_weight": None
        if commit_inputs.get("block_key_proj_weight") is None
        else commit_inputs["block_key_proj_weight"].detach().to(device="cpu", dtype=torch.float32).clone(),
        "prefix_dims": _canonical_prefix_dims(max_flag_dim=int(metadata.get("max_flag_dim", 0) or 0)),
    }


def replay_block_commit_bundle(
    bundle: dict[str, Any],
    *,
    implementation: str,
) -> dict[str, Any]:
    projector_sum = bundle["projector_sum"]
    reduced_sum = bundle["reduced_sum"]
    if not isinstance(projector_sum, torch.Tensor) or not isinstance(reduced_sum, torch.Tensor):
        raise ValueError("block commit bundle is missing projector_sum or reduced_sum")

    token_count = int(bundle["token_count"])
    max_flag_dim = int(bundle["max_flag_dim"])
    flag_dims = tuple(int(dim) for dim in bundle["flag_dims"])
    eps_sign = float(bundle["eps_sign"])
    prefix_dims = tuple(int(dim) for dim in bundle["prefix_dims"])
    block_key_proj_weight = bundle.get("block_key_proj_weight")

    if implementation == "native":
        average_projector = projector_sum / float(token_count)
        reduced_mean = reduced_sum / float(token_count)
        summary = top_eig_summary(
            average_projector,
            rank=int(max_flag_dim),
            rank_boundaries=flag_dims,
            eps_sign=float(eps_sign),
        )
        summary_basis = summary["basis"].to(dtype=average_projector.dtype)
        top_vector = summary["top_vector"].to(dtype=reduced_mean.dtype)
        prefix_projectors = _prefix_projectors_from_basis(summary_basis, prefix_dims=prefix_dims)
        block_key = _project_block_key(block_key_proj_weight, reduced_mean=reduced_mean, top_vector=top_vector)
        return {
            "implementation": "native",
            "used_accelerated_kernel": False,
            "average_projector": average_projector.detach().to(device="cpu", dtype=torch.float32).clone(),
            "summary_basis": summary_basis.detach().to(device="cpu", dtype=torch.float32).clone(),
            "prefix_projectors": {
                int(dim): projector.detach().to(device="cpu", dtype=torch.float32).clone()
                for dim, projector in prefix_projectors.items()
            },
            "reduced_mean": reduced_mean.detach().to(device="cpu", dtype=torch.float32).clone(),
            "top_vector": top_vector.detach().to(device="cpu", dtype=torch.float32).clone(),
            "block_key": None if block_key is None else block_key.detach().to(device="cpu", dtype=torch.float32).clone(),
        }

    if implementation != "candidate":
        raise ValueError(f"unsupported block commit replay implementation {implementation!r}")

    if torch.cuda.is_available() and sm120_exact_math_kernels_available():
        device = torch.device("cuda:0")
        accelerated = gfst_block_commit(
            projector_sum=projector_sum.to(device=device, dtype=torch.float32),
            reduced_sum=reduced_sum.to(device=device, dtype=torch.float32),
            token_count=int(token_count),
            max_flag_dim=int(max_flag_dim),
            flag_dims=flag_dims,
        )
        if accelerated is not None:
            summary_basis, reduced_mean, top_vector, average_projector = accelerated
            prefix_projectors = _prefix_projectors_from_basis(summary_basis, prefix_dims=prefix_dims)
            block_key_weight = (
                None
                if block_key_proj_weight is None
                else block_key_proj_weight.to(device=device, dtype=torch.float32)
            )
            block_key = _project_block_key(block_key_weight, reduced_mean=reduced_mean, top_vector=top_vector)
            return {
                "implementation": "candidate",
                "used_accelerated_kernel": True,
                "average_projector": average_projector.detach().to(device="cpu", dtype=torch.float32).clone(),
                "summary_basis": summary_basis.detach().to(device="cpu", dtype=torch.float32).clone(),
                "prefix_projectors": {
                    int(dim): projector.detach().to(device="cpu", dtype=torch.float32).clone()
                    for dim, projector in prefix_projectors.items()
                },
                "reduced_mean": reduced_mean.detach().to(device="cpu", dtype=torch.float32).clone(),
                "top_vector": top_vector.detach().to(device="cpu", dtype=torch.float32).clone(),
                "block_key": None if block_key is None else block_key.detach().to(device="cpu", dtype=torch.float32).clone(),
            }

    fallback = replay_block_commit_bundle(bundle, implementation="native")
    return {
        **fallback,
        "implementation": "candidate",
        "used_accelerated_kernel": False,
    }


def compare_block_commit_bundle(
    bundle: dict[str, Any],
    *,
    reference_implementation: str = "native",
    variant_implementation: str = "candidate",
    clean_tol: float = 1e-8,
    projector_tol: float = 1e-7,
) -> dict[str, Any]:
    reference = replay_block_commit_bundle(bundle, implementation=reference_implementation)
    variant = replay_block_commit_bundle(bundle, implementation=variant_implementation)
    prefix_dims = tuple(int(dim) for dim in bundle["prefix_dims"])
    prefix_projector_deltas = {
        int(dim): _tensor_max_abs(
            reference["prefix_projectors"].get(int(dim)),
            variant["prefix_projectors"].get(int(dim)),
        )
        for dim in prefix_dims
    }
    full_rank = max(prefix_dims, default=int(bundle["max_flag_dim"]))
    full_rank_projector_delta = float(prefix_projector_deltas.get(int(full_rank), 0.0))
    average_projector_max_abs = _tensor_max_abs(reference["average_projector"], variant["average_projector"])
    reduced_mean_max_abs = _tensor_max_abs(reference["reduced_mean"], variant["reduced_mean"])
    top_vector_max_abs = _tensor_max_abs(reference["top_vector"], variant["top_vector"])
    block_key_max_abs = _tensor_max_abs(reference["block_key"], variant["block_key"])

    classification = "other_commit_mismatch"
    if (
        average_projector_max_abs <= float(clean_tol)
        and max(prefix_projector_deltas.values(), default=0.0) <= float(clean_tol)
        and reduced_mean_max_abs <= float(clean_tol)
        and top_vector_max_abs <= float(clean_tol)
        and block_key_max_abs <= float(clean_tol)
    ):
        classification = "clean_commit"
    elif max(prefix_projector_deltas.values(), default=0.0) > float(projector_tol):
        classification = "basis_mismatch"
    elif top_vector_max_abs > float(clean_tol):
        classification = "top_vector_mismatch"
    elif block_key_max_abs > float(clean_tol):
        classification = "block_key_mismatch"

    return {
        "classification": classification,
        "reference_implementation": str(reference_implementation),
        "variant_implementation": str(variant_implementation),
        "reference_used_accelerated_kernel": bool(reference["used_accelerated_kernel"]),
        "variant_used_accelerated_kernel": bool(variant["used_accelerated_kernel"]),
        "average_projector_max_abs": float(average_projector_max_abs),
        "summary_basis_max_abs": _tensor_max_abs(reference["summary_basis"], variant["summary_basis"]),
        "prefix_projector_deltas": {str(dim): float(delta) for dim, delta in prefix_projector_deltas.items()},
        "full_rank_projector_delta": float(full_rank_projector_delta),
        "reduced_mean_max_abs": float(reduced_mean_max_abs),
        "top_vector_max_abs": float(top_vector_max_abs),
        "block_key_max_abs": float(block_key_max_abs),
    }
