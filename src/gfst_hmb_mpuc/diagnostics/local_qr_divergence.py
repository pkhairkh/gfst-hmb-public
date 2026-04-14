from __future__ import annotations

from typing import Any, Callable

import torch

from ..gfst import basis_projector, canonicalize_qr_column_signs


TensorQrOperator = Callable[[torch.Tensor, float], torch.Tensor]


def _as_cpu_float32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def tensor_max_abs(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"tensor shape mismatch: reference={tuple(reference.shape)} candidate={tuple(candidate.shape)}"
        )
    if int(reference.numel()) == 0:
        return 0.0
    return float((candidate - reference).abs().max().item())


def find_earliest_local_feature_divergence(
    native_traces: list[torch.Tensor | None],
    variant_traces: list[torch.Tensor | None],
) -> dict[str, Any] | None:
    layer_count = min(len(native_traces), len(variant_traces))
    for layer_idx in range(layer_count):
        native_trace = native_traces[layer_idx]
        variant_trace = variant_traces[layer_idx]
        if native_trace is None or variant_trace is None:
            continue
        if tuple(native_trace.shape) != tuple(variant_trace.shape):
            raise ValueError(
                f"feature trace shape mismatch at layer {layer_idx}: "
                f"{tuple(native_trace.shape)} vs {tuple(variant_trace.shape)}"
            )
        if native_trace.ndim != 3:
            raise ValueError(f"expected rank-3 feature trace, got shape {tuple(native_trace.shape)}")
        local_dim = int(native_trace.shape[-1]) // 2
        local_delta = (variant_trace[:, :, :local_dim] - native_trace[:, :, :local_dim]).abs()
        nonzero = torch.nonzero(local_delta > 0, as_tuple=False)
        if int(nonzero.numel()) == 0:
            continue
        first = nonzero[0]
        batch_idx = int(first[0].item())
        token_idx = int(first[1].item())
        feature_idx = int(first[2].item())
        return {
            "layer_idx": int(layer_idx),
            "batch_idx": batch_idx,
            "token_idx": token_idx,
            "feature_idx": feature_idx,
            "delta": float(local_delta[batch_idx, token_idx, feature_idx].item()),
            "layer_local_max_abs": float(local_delta.max().item()),
            "local_dim": local_dim,
        }
    return None


def compute_prefix_projector_deltas(
    reference_basis: torch.Tensor,
    candidate_basis: torch.Tensor,
    prefix_dims: tuple[int, ...] | list[int],
) -> dict[str, float]:
    reference = _as_cpu_float32(reference_basis)
    candidate = _as_cpu_float32(candidate_basis)
    max_prefix = min(int(reference.shape[1]), int(candidate.shape[1]))
    deltas: dict[str, float] = {}
    for dim in prefix_dims:
        prefix_dim = int(dim)
        if prefix_dim <= 0 or prefix_dim > max_prefix:
            continue
        ref_projector = basis_projector(reference[:, :prefix_dim])
        cand_projector = basis_projector(candidate[:, :prefix_dim])
        deltas[str(prefix_dim)] = tensor_max_abs(ref_projector, cand_projector)
    return deltas


def native_compact_qr(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
    compact_factors, tau = torch.geqrf(candidate)
    explicit_q = torch.linalg.householder_product(compact_factors, tau)
    return {
        "compact_factors": _as_cpu_float32(compact_factors),
        "tau": _as_cpu_float32(tau),
        "explicit_q": _as_cpu_float32(explicit_q),
    }


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


def reconstruct_local_qr_capture_from_explicit_q(
    payload: dict[str, Any],
    *,
    largest_q: torch.Tensor,
    nested_qs: list[torch.Tensor],
    eps_sign: float | None = None,
    apply_sign_fix: bool = True,
) -> dict[str, Any]:
    metadata = dict(payload["metadata"])
    resolved_eps_sign = float(metadata["eps_sign"] if eps_sign is None else eps_sign)
    reduced_token = _as_cpu_float32(payload["reduced_token"])
    largest_basis_out = _as_cpu_float32(largest_q)
    if apply_sign_fix:
        largest_basis_out = _as_cpu_float32(
            canonicalize_qr_column_signs(largest_basis_out.to(dtype=torch.float64), eps_sign=resolved_eps_sign)
        )
    projected = _as_cpu_float32(largest_basis_out.transpose(0, 1) @ reduced_token)

    nested_results: list[dict[str, Any]] = []
    local_bases: list[torch.Tensor] = []
    nested_payloads = sorted(payload.get("nested_steps", []), key=lambda item: int(item["nested_index"]))
    for index, nested_step in enumerate(nested_payloads):
        nested_basis_full = _as_cpu_float32(nested_qs[index])
        if apply_sign_fix:
            nested_basis_full = _as_cpu_float32(
                canonicalize_qr_column_signs(nested_basis_full.to(dtype=torch.float64), eps_sign=resolved_eps_sign)
            )
        dim = int(nested_step["dim"])
        nested_basis_out = nested_basis_full[:, :dim]
        composed_basis = _as_cpu_float32(largest_basis_out @ nested_basis_out)
        nested_results.append(
            {
                "nested_index": int(nested_step["nested_index"]),
                "dim": dim,
                "nested_basis_out": nested_basis_out,
                "composed_basis": composed_basis,
            }
        )
        local_bases.append(composed_basis)

    max_flag_dim = int(metadata["max_flag_dim"])
    local_bases.append(_as_cpu_float32(largest_basis_out[:, :max_flag_dim]))
    local_features = _local_alignment_features_from_tracker(
        reduced_token,
        largest_basis_out,
        [step["nested_basis_out"] for step in nested_results],
        eps=float(metadata["eps"]),
    )
    return {
        "largest_basis_out": largest_basis_out,
        "projected": projected,
        "nested_steps": nested_results,
        "local_bases": local_bases,
        "local_features": _as_cpu_float32(local_features),
    }


def reconstruct_local_qr_capture(
    payload: dict[str, Any],
    *,
    qr_operator: TensorQrOperator,
    eps_sign: float | None = None,
) -> dict[str, Any]:
    metadata = dict(payload["metadata"])
    resolved_eps_sign = float(metadata["eps_sign"] if eps_sign is None else eps_sign)
    largest_basis_out = _as_cpu_float32(qr_operator(_as_cpu_float32(payload["largest_candidate"]), resolved_eps_sign))
    nested_qs = [
        _as_cpu_float32(qr_operator(_as_cpu_float32(nested_step["nested_candidate"]), resolved_eps_sign))
        for nested_step in sorted(payload.get("nested_steps", []), key=lambda item: int(item["nested_index"]))
    ]
    return reconstruct_local_qr_capture_from_explicit_q(
        payload,
        largest_q=largest_basis_out,
        nested_qs=nested_qs,
        eps_sign=resolved_eps_sign,
        apply_sign_fix=False,
    )


def analyze_local_qr_reflector_mismatch(
    payload: dict[str, Any],
    *,
    native_compact_qr_operator: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    kernel_compact_qr_operator: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    tau_atol: float = 1e-7,
    explicit_q_atol: float = 1e-7,
    prefix_projector_atol: float = 1e-7,
    local_feature_atol: float = 1e-8,
) -> dict[str, Any]:
    metadata = dict(payload["metadata"])
    eps_sign = float(metadata["eps_sign"])
    flag_dims = tuple(int(dim) for dim in metadata["flag_dims"])
    max_flag_dim = int(metadata["max_flag_dim"])
    prefix_dims = tuple(dim for dim in flag_dims if dim <= max_flag_dim)

    stage_payloads: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
        ("largest", {"candidate": payload["largest_candidate"], "dim": max_flag_dim}, prefix_dims)
    ]
    for nested_step in sorted(payload.get("nested_steps", []), key=lambda item: int(item["nested_index"])):
        dim = int(nested_step["dim"])
        stage_payloads.append(
            (
                f"nested_{int(nested_step['nested_index'])}",
                {"candidate": nested_step["nested_candidate"], "dim": dim},
                tuple(prefix for prefix in flag_dims if prefix <= dim),
            )
        )

    stage_summaries: list[dict[str, Any]] = []
    largest_native_raw: torch.Tensor | None = None
    largest_kernel_raw: torch.Tensor | None = None
    nested_native_raw: list[torch.Tensor] = []
    nested_kernel_raw: list[torch.Tensor] = []
    max_tau_delta = 0.0
    max_explicit_q_delta = 0.0
    max_signfixed_q_delta = 0.0
    max_prefix_delta = 0.0

    for stage_name, stage_payload, stage_prefix_dims in stage_payloads:
        candidate = _as_cpu_float32(stage_payload["candidate"])
        native = native_compact_qr_operator(candidate)
        kernel = kernel_compact_qr_operator(candidate)
        native_q_raw = _as_cpu_float32(native["explicit_q"])
        kernel_q_raw = _as_cpu_float32(kernel["explicit_q"])
        native_q_signfixed = _as_cpu_float32(
            canonicalize_qr_column_signs(native_q_raw.to(dtype=torch.float64), eps_sign=eps_sign)
        )
        kernel_q_signfixed = _as_cpu_float32(
            canonicalize_qr_column_signs(kernel_q_raw.to(dtype=torch.float64), eps_sign=eps_sign)
        )
        compact_delta = tensor_max_abs(native["compact_factors"], kernel["compact_factors"])
        tau_delta = tensor_max_abs(native["tau"], kernel["tau"])
        explicit_q_delta = tensor_max_abs(native_q_raw, kernel_q_raw)
        signfixed_q_delta = tensor_max_abs(native_q_signfixed, kernel_q_signfixed)
        prefix_projector_deltas = compute_prefix_projector_deltas(native_q_signfixed, kernel_q_signfixed, stage_prefix_dims)
        stage_prefix_max = max(prefix_projector_deltas.values(), default=0.0)
        max_tau_delta = max(max_tau_delta, tau_delta)
        max_explicit_q_delta = max(max_explicit_q_delta, explicit_q_delta)
        max_signfixed_q_delta = max(max_signfixed_q_delta, signfixed_q_delta)
        max_prefix_delta = max(max_prefix_delta, stage_prefix_max)
        stage_summaries.append(
            {
                "stage": stage_name,
                "compact_factors_max_abs": compact_delta,
                "tau_max_abs": tau_delta,
                "explicit_q_max_abs": explicit_q_delta,
                "signfixed_q_max_abs": signfixed_q_delta,
                "prefix_projector_max_abs": prefix_projector_deltas,
            }
        )
        if stage_name == "largest":
            largest_native_raw = native_q_raw
            largest_kernel_raw = kernel_q_raw
        else:
            nested_native_raw.append(native_q_raw)
            nested_kernel_raw.append(kernel_q_raw)

    if largest_native_raw is None or largest_kernel_raw is None:
        raise ValueError("payload did not contain a largest QR stage")

    native_signfixed = reconstruct_local_qr_capture_from_explicit_q(
        payload,
        largest_q=largest_native_raw,
        nested_qs=nested_native_raw,
        eps_sign=eps_sign,
        apply_sign_fix=True,
    )
    kernel_signfixed = reconstruct_local_qr_capture_from_explicit_q(
        payload,
        largest_q=largest_kernel_raw,
        nested_qs=nested_kernel_raw,
        eps_sign=eps_sign,
        apply_sign_fix=True,
    )
    rollout_local_features = _as_cpu_float32(payload["local_features"])
    native_local_feature_reproduction = tensor_max_abs(rollout_local_features, native_signfixed["local_features"])
    kernel_local_feature_reproduction = tensor_max_abs(rollout_local_features, kernel_signfixed["local_features"])
    reconstructed_local_feature_max_abs = tensor_max_abs(native_signfixed["local_features"], kernel_signfixed["local_features"])

    if max_tau_delta > float(tau_atol) or any(
        float(stage["compact_factors_max_abs"]) > float(explicit_q_atol) for stage in stage_summaries
    ):
        classification = "reflector_generation_mismatch"
    elif max_explicit_q_delta > float(explicit_q_atol):
        classification = "q_expansion_mismatch"
    elif max(max_prefix_delta, reconstructed_local_feature_max_abs, max_signfixed_q_delta) > max(
        float(prefix_projector_atol),
        float(local_feature_atol),
    ):
        classification = "post_signfix_prefix_mismatch"
    else:
        classification = "unreproduced"

    return {
        "metadata": metadata,
        "classification": classification,
        "tau_max_abs": max_tau_delta,
        "explicit_q_max_abs": max_explicit_q_delta,
        "signfixed_q_max_abs": max_signfixed_q_delta,
        "prefix_projector_max_abs": max_prefix_delta,
        "reconstructed_local_feature_max_abs": reconstructed_local_feature_max_abs,
        "native_local_feature_reproduction_error": native_local_feature_reproduction,
        "kernel_local_feature_reproduction_error": kernel_local_feature_reproduction,
        "stages": stage_summaries,
    }


def _stage_summary(
    *,
    stage_name: str,
    reference_basis_in: torch.Tensor,
    candidate_basis_in: torch.Tensor,
    reference_candidate: torch.Tensor,
    candidate_candidate: torch.Tensor,
    reference_basis_native: torch.Tensor,
    reference_basis_kernel: torch.Tensor,
    candidate_basis_native: torch.Tensor,
    candidate_basis_kernel: torch.Tensor,
    prefix_dims: tuple[int, ...] | list[int],
) -> dict[str, Any]:
    return {
        "stage": str(stage_name),
        "basis_in_max_abs": tensor_max_abs(reference_basis_in, candidate_basis_in),
        "candidate_max_abs": tensor_max_abs(reference_candidate, candidate_candidate),
        "operator_basis_max_abs_on_reference_candidate": tensor_max_abs(reference_basis_native, reference_basis_kernel),
        "operator_basis_max_abs_on_variant_candidate": tensor_max_abs(candidate_basis_native, candidate_basis_kernel),
        "candidate_basis_max_abs_under_native_operator": tensor_max_abs(reference_basis_native, candidate_basis_native),
        "candidate_basis_max_abs_under_kernel_operator": tensor_max_abs(reference_basis_kernel, candidate_basis_kernel),
        "operator_prefix_projector_max_abs_on_reference_candidate": compute_prefix_projector_deltas(
            reference_basis_native,
            reference_basis_kernel,
            prefix_dims,
        ),
        "operator_prefix_projector_max_abs_on_variant_candidate": compute_prefix_projector_deltas(
            candidate_basis_native,
            candidate_basis_kernel,
            prefix_dims,
        ),
        "candidate_prefix_projector_max_abs_under_native_operator": compute_prefix_projector_deltas(
            reference_basis_native,
            candidate_basis_native,
            prefix_dims,
        ),
        "candidate_prefix_projector_max_abs_under_kernel_operator": compute_prefix_projector_deltas(
            reference_basis_kernel,
            candidate_basis_kernel,
            prefix_dims,
        ),
    }


def analyze_local_qr_divergence(
    native_payload: dict[str, Any],
    variant_payload: dict[str, Any],
    *,
    native_qr_operator: TensorQrOperator,
    kernel_qr_operator: TensorQrOperator,
) -> dict[str, Any]:
    native_native = reconstruct_local_qr_capture(native_payload, qr_operator=native_qr_operator)
    native_kernel = reconstruct_local_qr_capture(native_payload, qr_operator=kernel_qr_operator)
    variant_native = reconstruct_local_qr_capture(variant_payload, qr_operator=native_qr_operator)
    variant_kernel = reconstruct_local_qr_capture(variant_payload, qr_operator=kernel_qr_operator)

    native_rollout_features = _as_cpu_float32(native_payload["local_features"])
    variant_rollout_features = _as_cpu_float32(variant_payload["local_features"])
    rollout_local_feature_max_abs = tensor_max_abs(native_rollout_features, variant_rollout_features)
    native_reproduction_error = tensor_max_abs(native_native["local_features"], native_rollout_features)
    variant_reproduction_error = tensor_max_abs(variant_kernel["local_features"], variant_rollout_features)

    operator_effect = max(
        tensor_max_abs(native_native["local_features"], native_kernel["local_features"]),
        tensor_max_abs(variant_native["local_features"], variant_kernel["local_features"]),
    )
    candidate_effect = max(
        tensor_max_abs(native_native["local_features"], variant_native["local_features"]),
        tensor_max_abs(native_kernel["local_features"], variant_kernel["local_features"]),
    )
    reference_scale = max(rollout_local_feature_max_abs, 1e-8)
    reproduction_tolerance = max(1e-7, 0.1 * reference_scale)
    if max(native_reproduction_error, variant_reproduction_error) > reproduction_tolerance:
        classification = "unreproduced"
    elif operator_effect >= (0.5 * reference_scale) and candidate_effect < (0.25 * reference_scale):
        classification = "qr_operator_mismatch"
    elif candidate_effect >= (0.5 * reference_scale) and operator_effect < (0.25 * reference_scale):
        classification = "candidate_input_mismatch"
    elif operator_effect >= (0.25 * reference_scale) and candidate_effect >= (0.25 * reference_scale):
        classification = "mixed"
    else:
        classification = "unreproduced"

    metadata = dict(native_payload["metadata"])
    flag_dims = tuple(int(dim) for dim in metadata["flag_dims"])
    largest_prefix_dims = tuple(dim for dim in flag_dims if dim <= int(metadata["max_flag_dim"]))

    stage_summaries: list[dict[str, Any]] = [
        _stage_summary(
            stage_name="largest",
            reference_basis_in=_as_cpu_float32(native_payload["largest_basis_in"]),
            candidate_basis_in=_as_cpu_float32(variant_payload["largest_basis_in"]),
            reference_candidate=_as_cpu_float32(native_payload["largest_candidate"]),
            candidate_candidate=_as_cpu_float32(variant_payload["largest_candidate"]),
            reference_basis_native=native_native["largest_basis_out"],
            reference_basis_kernel=native_kernel["largest_basis_out"],
            candidate_basis_native=variant_native["largest_basis_out"],
            candidate_basis_kernel=variant_kernel["largest_basis_out"],
            prefix_dims=largest_prefix_dims,
        )
    ]

    nested_summaries: list[dict[str, Any]] = []
    native_nested_payloads = sorted(native_payload.get("nested_steps", []), key=lambda item: int(item["nested_index"]))
    variant_nested_payloads = sorted(variant_payload.get("nested_steps", []), key=lambda item: int(item["nested_index"]))
    for index, (native_nested, variant_nested) in enumerate(zip(native_nested_payloads, variant_nested_payloads)):
        native_nested_native = native_native["nested_steps"][index]
        native_nested_kernel = native_kernel["nested_steps"][index]
        variant_nested_native = variant_native["nested_steps"][index]
        variant_nested_kernel = variant_kernel["nested_steps"][index]
        dim = int(native_nested["dim"])
        nested_prefix_dims = tuple(prefix for prefix in flag_dims if prefix <= dim)
        nested_summary = _stage_summary(
            stage_name=f"nested_{index}",
            reference_basis_in=_as_cpu_float32(native_nested["nested_basis_in"]),
            candidate_basis_in=_as_cpu_float32(variant_nested["nested_basis_in"]),
            reference_candidate=_as_cpu_float32(native_nested["nested_candidate"]),
            candidate_candidate=_as_cpu_float32(variant_nested["nested_candidate"]),
            reference_basis_native=native_nested_native["nested_basis_out"],
            reference_basis_kernel=native_nested_kernel["nested_basis_out"],
            candidate_basis_native=variant_nested_native["nested_basis_out"],
            candidate_basis_kernel=variant_nested_kernel["nested_basis_out"],
            prefix_dims=nested_prefix_dims,
        )
        nested_summary["composed_basis_max_abs_under_native_operator"] = tensor_max_abs(
            native_nested_native["composed_basis"],
            variant_nested_native["composed_basis"],
        )
        nested_summary["composed_basis_max_abs_under_kernel_operator"] = tensor_max_abs(
            native_nested_kernel["composed_basis"],
            variant_nested_kernel["composed_basis"],
        )
        nested_summary["composed_basis_operator_max_abs_on_reference_candidate"] = tensor_max_abs(
            native_nested_native["composed_basis"],
            native_nested_kernel["composed_basis"],
        )
        nested_summary["composed_basis_operator_max_abs_on_variant_candidate"] = tensor_max_abs(
            variant_nested_native["composed_basis"],
            variant_nested_kernel["composed_basis"],
        )
        nested_summaries.append(nested_summary)

    composed_basis_deltas: dict[str, Any] = {}
    for basis_index, dim in enumerate(flag_dims):
        composed_basis_deltas[str(int(dim))] = {
            "candidate_effect_under_native_operator": tensor_max_abs(
                native_native["local_bases"][basis_index],
                variant_native["local_bases"][basis_index],
            ),
            "candidate_effect_under_kernel_operator": tensor_max_abs(
                native_kernel["local_bases"][basis_index],
                variant_kernel["local_bases"][basis_index],
            ),
            "operator_effect_on_reference_candidate": tensor_max_abs(
                native_native["local_bases"][basis_index],
                native_kernel["local_bases"][basis_index],
            ),
            "operator_effect_on_variant_candidate": tensor_max_abs(
                variant_native["local_bases"][basis_index],
                variant_kernel["local_bases"][basis_index],
            ),
            "prefix_projector_candidate_effect_under_native_operator": compute_prefix_projector_deltas(
                native_native["local_bases"][basis_index],
                variant_native["local_bases"][basis_index],
                (int(dim),),
            ),
            "prefix_projector_candidate_effect_under_kernel_operator": compute_prefix_projector_deltas(
                native_kernel["local_bases"][basis_index],
                variant_kernel["local_bases"][basis_index],
                (int(dim),),
            ),
        }

    return {
        "metadata": metadata,
        "classification": classification,
        "rollout_local_feature_max_abs": rollout_local_feature_max_abs,
        "native_reproduction_error": native_reproduction_error,
        "variant_reproduction_error": variant_reproduction_error,
        "operator_effect_local_feature_max_abs": operator_effect,
        "candidate_effect_local_feature_max_abs": candidate_effect,
        "largest": stage_summaries[0],
        "nested": nested_summaries,
        "composed_bases": composed_basis_deltas,
        "local_feature_cross_eval": {
            "native_qr_on_native_candidate": _as_cpu_float32(native_native["local_features"]),
            "kernel_qr_on_native_candidate": _as_cpu_float32(native_kernel["local_features"]),
            "native_qr_on_variant_candidate": _as_cpu_float32(variant_native["local_features"]),
            "kernel_qr_on_variant_candidate": _as_cpu_float32(variant_kernel["local_features"]),
        },
    }
