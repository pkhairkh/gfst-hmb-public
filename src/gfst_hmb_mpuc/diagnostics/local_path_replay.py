from __future__ import annotations

from typing import Any

import torch

from ..gfst import basis_projector, canonical_qr_batched, update_basis_candidate
from .local_qr_divergence import tensor_max_abs


def _as_cpu_float32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def _symmetrized_projector_delta(reference_basis: torch.Tensor, candidate_basis: torch.Tensor) -> float:
    reference_projector = basis_projector(reference_basis)
    candidate_projector = basis_projector(candidate_basis)
    return tensor_max_abs(
        _as_cpu_float32(0.5 * (reference_projector + reference_projector.transpose(0, 1))),
        _as_cpu_float32(0.5 * (candidate_projector + candidate_projector.transpose(0, 1))),
    )


def local_path_bundle_from_transition_capture(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(capture.get("metadata", {}))
    pre_token_inputs = dict(capture.get("pre_token_inputs", {}))
    local_path = dict(capture.get("local_path", {}))
    nested_steps = sorted(local_path.get("nested_steps", []), key=lambda item: int(item["nested_index"]))
    return {
        "metadata": metadata,
        "reduced_token": _as_cpu_float32(pre_token_inputs["reduced_token"]),
        "largest_basis_in": _as_cpu_float32(local_path["largest_basis_in"]),
        "largest_candidate": _as_cpu_float32(local_path["largest_candidate"]),
        "largest_basis_out_captured": _as_cpu_float32(local_path["largest_basis_out"]),
        "projected_captured": _as_cpu_float32(local_path["projected"]),
        "local_features_captured": _as_cpu_float32(local_path["local_features"]),
        "nested_steps": [
            {
                "nested_index": int(step["nested_index"]),
                "dim": int(step["dim"]),
                "nested_basis_in": _as_cpu_float32(step["nested_basis_in"]),
                "nested_candidate": _as_cpu_float32(step["nested_candidate"]),
                "nested_basis_out_captured": _as_cpu_float32(step["nested_basis_out"]),
            }
            for step in nested_steps
        ],
    }


def _native_qr_operator(candidate: torch.Tensor, eps_sign: float) -> tuple[torch.Tensor, bool]:
    result = canonical_qr_batched(candidate.unsqueeze(0), eps_sign=float(eps_sign)).squeeze(0)
    return _as_cpu_float32(result), False


def _local_alignment_features_from_tracker(
    reduced_token: torch.Tensor,
    largest_basis: torch.Tensor,
    nested_bases: list[torch.Tensor],
    *,
    eps: float,
) -> torch.Tensor:
    denominator = torch.clamp(reduced_token.square().sum(), min=float(eps))
    projected = largest_basis.transpose(0, 1) @ reduced_token
    cumulative: list[torch.Tensor] = []
    for nested_basis in nested_bases:
        nested_projection = nested_basis.transpose(0, 1) @ projected
        cumulative.append(nested_projection.square().sum() / denominator)
    cumulative.append(projected.square().sum() / denominator)
    cumulative_tensor = torch.stack(cumulative, dim=0).to(dtype=torch.float32)
    previous = torch.cat((torch.zeros_like(cumulative_tensor[:1]), cumulative_tensor[:-1]), dim=0)
    increments = cumulative_tensor - previous
    return torch.stack((cumulative_tensor, increments), dim=-1).reshape(-1)


def replay_local_path_bundle(
    bundle: dict[str, Any],
    *,
    implementation: str,
) -> dict[str, Any]:
    metadata = dict(bundle["metadata"])
    reduced_token = _as_cpu_float32(bundle["reduced_token"])
    largest_basis_in = _as_cpu_float32(bundle["largest_basis_in"])
    eps_sign = float(metadata["eps_sign"])
    if implementation not in {"native", "candidate"}:
        raise ValueError(f"unsupported local-path implementation {implementation!r}")
    used_accelerated = False

    largest_candidate = _as_cpu_float32(
        update_basis_candidate(
            largest_basis_in,
            reduced_token,
            float(metadata["forgetting_rate"]),
            update_rule=str(metadata["gfst_update_rule"]),
            eps_update=float(metadata["eps_update"]),
        )
    )
    largest_basis_out = _native_qr_operator(largest_candidate, eps_sign)[0]

    projected = _as_cpu_float32(largest_basis_out.transpose(0, 1) @ reduced_token)
    nested_bases: list[torch.Tensor] = []
    nested_steps: list[dict[str, Any]] = []
    for bundle_step in bundle["nested_steps"]:
        nested_candidate = _as_cpu_float32(bundle_step["nested_candidate"])
        nested_basis_in = _as_cpu_float32(bundle_step["nested_basis_in"])
        nested_basis_full, accelerated = _native_qr_operator(nested_candidate, eps_sign)
        used_accelerated = used_accelerated or bool(accelerated)
        dim = int(bundle_step["dim"])
        nested_basis_out = _as_cpu_float32(nested_basis_full[:, :dim])
        nested_bases.append(nested_basis_out)
        nested_steps.append(
            {
                "nested_index": int(bundle_step["nested_index"]),
                "dim": dim,
                "nested_candidate": nested_candidate,
                "nested_basis_in": nested_basis_in,
                "nested_basis_out": nested_basis_out,
                "nested_projector": _as_cpu_float32(basis_projector(nested_basis_out)),
            }
        )
    local_features = _as_cpu_float32(
        _local_alignment_features_from_tracker(
            reduced_token,
            largest_basis_out,
            nested_bases,
            eps=float(metadata["eps"]),
        )
    )
    return {
        "implementation": str(implementation),
        "used_accelerated_kernel": bool(used_accelerated),
        "largest_candidate": largest_candidate,
        "largest_basis_in": largest_basis_in,
        "largest_basis_out": largest_basis_out,
        "largest_full_rank_projector": _as_cpu_float32(
            basis_projector(largest_basis_out[:, : int(metadata["max_flag_dim"])])
        ),
        "projected": projected,
        "nested_steps": nested_steps,
        "local_features": local_features,
    }


def compare_local_path_bundle(
    bundle: dict[str, Any],
    *,
    reference_implementation: str = "native",
    variant_implementation: str = "candidate",
    clean_tol: float = 1e-8,
    projector_tol: float = 1e-7,
) -> dict[str, Any]:
    reference = replay_local_path_bundle(bundle, implementation=reference_implementation)
    variant = replay_local_path_bundle(bundle, implementation=variant_implementation)

    largest_candidate_max_abs = tensor_max_abs(reference["largest_candidate"], variant["largest_candidate"])
    largest_basis_out_max_abs = tensor_max_abs(reference["largest_basis_out"], variant["largest_basis_out"])
    largest_full_rank_projector_delta = _symmetrized_projector_delta(
        reference["largest_basis_out"][:, : int(bundle["metadata"]["max_flag_dim"])],
        variant["largest_basis_out"][:, : int(bundle["metadata"]["max_flag_dim"])],
    )
    projected_max_abs = tensor_max_abs(reference["projected"], variant["projected"])
    local_features_max_abs = tensor_max_abs(reference["local_features"], variant["local_features"])

    nested_summaries: list[dict[str, Any]] = []
    nested_dirty = False
    for reference_step, variant_step in zip(reference["nested_steps"], variant["nested_steps"]):
        nested_candidate_max_abs = tensor_max_abs(reference_step["nested_candidate"], variant_step["nested_candidate"])
        nested_basis_out_max_abs = tensor_max_abs(reference_step["nested_basis_out"], variant_step["nested_basis_out"])
        nested_projector_delta = _symmetrized_projector_delta(
            reference_step["nested_basis_out"],
            variant_step["nested_basis_out"],
        )
        nested_summaries.append(
            {
                "nested_index": int(reference_step["nested_index"]),
                "dim": int(reference_step["dim"]),
                "nested_candidate_max_abs": float(nested_candidate_max_abs),
                "nested_basis_out_max_abs": float(nested_basis_out_max_abs),
                "nested_projector_delta": float(nested_projector_delta),
            }
        )
        nested_dirty = nested_dirty or (
            float(nested_basis_out_max_abs) > float(clean_tol)
            or float(nested_projector_delta) > float(projector_tol)
        )

    classification = "other_local_path_mismatch"
    if (
        float(largest_candidate_max_abs) <= float(clean_tol)
        and float(largest_basis_out_max_abs) <= float(clean_tol)
        and float(largest_full_rank_projector_delta) <= float(projector_tol)
        and not nested_dirty
        and float(projected_max_abs) <= float(clean_tol)
        and float(local_features_max_abs) <= float(clean_tol)
    ):
        classification = "clean_local_path"
    elif float(largest_basis_out_max_abs) > float(clean_tol) or float(largest_full_rank_projector_delta) > float(projector_tol):
        classification = "largest_qr_mismatch"
    elif nested_dirty:
        classification = "nested_qr_mismatch"
    elif float(projected_max_abs) > float(clean_tol) or float(local_features_max_abs) > float(clean_tol):
        classification = "feature_projection_mismatch"

    return {
        "classification": classification,
        "reference_implementation": str(reference_implementation),
        "variant_implementation": str(variant_implementation),
        "reference_used_accelerated_kernel": bool(reference["used_accelerated_kernel"]),
        "variant_used_accelerated_kernel": bool(variant["used_accelerated_kernel"]),
        "largest_candidate_max_abs": float(largest_candidate_max_abs),
        "largest_basis_out_max_abs": float(largest_basis_out_max_abs),
        "largest_full_rank_projector_delta": float(largest_full_rank_projector_delta),
        "projected_max_abs": float(projected_max_abs),
        "local_features_max_abs": float(local_features_max_abs),
        "nested_steps": nested_summaries,
    }
