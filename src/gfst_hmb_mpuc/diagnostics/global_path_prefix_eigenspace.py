from __future__ import annotations

from typing import Any

import torch

from ..gfst import (
    _DEFAULT_EIG_CLUSTER_ATOL,
    _DEFAULT_EIG_CLUSTER_RTOL,
    _descending_top_eig_cluster_spans,
    _symmetric_eigpairs_with_fallback,
    alignment_components,
    basis_projector,
    top_eig_summary,
)
from ..runtime_backends.sm120 import (
    _requires_native_summary_fallback,
    gfst_global_sketch,
    gfst_heev,
    gfst_reference_global_sketch,
    gfst_summary_from_sketch,
    sm120_exact_math_kernels_available,
)

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


def _canonical_prefix_dims(
    *,
    requested: tuple[int, ...] | list[int] | None,
    max_flag_dim: int,
) -> tuple[int, ...]:
    base = PREFIX_DIAGNOSTIC_DIMS if requested is None else tuple(int(dim) for dim in requested)
    dims = sorted({int(dim) for dim in base if int(dim) > 0 and int(dim) <= int(max_flag_dim)})
    if int(max_flag_dim) > 0 and int(max_flag_dim) not in dims:
        dims.append(int(max_flag_dim))
    return tuple(dims)


def _cluster_diagnosis_from_eigvals(
    eigvals: torch.Tensor,
    *,
    prefix_dims: tuple[int, ...],
    max_rank: int,
    atol: float = _DEFAULT_EIG_CLUSTER_ATOL,
    rtol: float = _DEFAULT_EIG_CLUSTER_RTOL,
) -> dict[int, dict[str, Any]]:
    _descending_vals, spans = _descending_top_eig_cluster_spans(
        eigvals,
        rank=int(max_rank),
        atol=float(atol),
        rtol=float(rtol),
    )
    diagnostics: dict[int, dict[str, Any]] = {}
    consumed = 0
    boundary_index = 0
    sorted_prefixes = sorted({int(dim) for dim in prefix_dims if int(dim) > 0 and int(dim) <= int(max_rank)})
    for start, end in spans:
        cluster_size = int(end - start + 1)
        while boundary_index < len(sorted_prefixes) and int(sorted_prefixes[boundary_index]) <= consumed + cluster_size:
            boundary = int(sorted_prefixes[boundary_index])
            diagnostics[int(boundary)] = {
                "cluster_start_rank": int(consumed + 1),
                "cluster_end_rank": int(consumed + cluster_size),
                "cluster_size": int(cluster_size),
                "splits_cluster": bool(boundary < consumed + cluster_size),
            }
            boundary_index += 1
        consumed += cluster_size
        if boundary_index >= len(sorted_prefixes):
            break
    return diagnostics


def _sorted_descending_eigvals_and_gaps(
    eigvals: torch.Tensor,
    *,
    max_rank: int,
) -> tuple[list[float], list[float]]:
    selected_rank = max(0, min(int(max_rank), int(eigvals.numel())))
    if selected_rank == 0:
        return [], []
    descending = torch.flip(eigvals[-selected_rank:].to(dtype=torch.float64), dims=(0,))
    values = [float(value.item()) for value in descending]
    gaps = [
        float((descending[index] - descending[index + 1]).item())
        for index in range(max(0, int(descending.numel()) - 1))
    ]
    return values, gaps


def _summary_count_from_capture(global_path: dict[str, Any]) -> int:
    explicit = global_path.get("summary_count")
    if explicit is not None:
        return int(explicit)
    tensor = global_path.get("summary_block_index_tensor")
    if isinstance(tensor, torch.Tensor):
        return int(tensor.shape[0])
    return 0


def global_path_bundle_from_capture(
    capture: dict[str, Any],
    *,
    prefix_dims: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    metadata = dict(capture.get("metadata", {}))
    pre_token_inputs = dict(capture.get("pre_token_inputs", {}))
    global_path = dict(capture.get("global_path", {}))
    max_flag_dim = int(metadata.get("max_flag_dim", 0) or 0)
    if max_flag_dim <= 0:
        summary_basis_tensor = global_path.get("summary_basis_tensor")
        if isinstance(summary_basis_tensor, torch.Tensor) and summary_basis_tensor.ndim == 3:
            max_flag_dim = int(summary_basis_tensor.shape[-1])
    dims = _canonical_prefix_dims(requested=prefix_dims, max_flag_dim=int(max_flag_dim))
    return {
        "metadata": metadata,
        "reduced_token": pre_token_inputs["reduced_token"].detach().to(device="cpu", dtype=torch.float32).clone(),
        "query_token": pre_token_inputs["query_token"].detach().to(device="cpu", dtype=torch.float32).clone(),
        "summary_projectors": global_path["summary_projector_tensor"].detach().to(device="cpu", dtype=torch.float32).clone(),
        "summary_basis_tensor": global_path["summary_basis_tensor"].detach().to(device="cpu", dtype=torch.float32).clone(),
        "summary_block_indices": global_path["summary_block_index_tensor"].detach().to(device="cpu", dtype=torch.long).clone(),
        "summary_token_counts": global_path["summary_token_count_tensor"].detach().to(device="cpu", dtype=torch.long).clone(),
        "final_indices": global_path["final_indices"].detach().to(device="cpu", dtype=torch.long).clone(),
        "final_scores": global_path["final_scores"].detach().to(device="cpu", dtype=torch.float32).clone(),
        "prefix_dims": dims,
        "flag_dims": tuple(int(dim) for dim in metadata.get("flag_dims", [])),
        "max_flag_dim": int(max_flag_dim),
        "retrieval_temperature": float(metadata.get("retrieval_temperature", 1.0)),
        "eps": float(metadata.get("eps", 1e-6)),
        "summary_count": int(_summary_count_from_capture(global_path)),
    }


def replay_global_path_bundle(
    bundle: dict[str, Any],
    *,
    implementation: str,
) -> dict[str, Any]:
    summary_projectors = bundle["summary_projectors"]
    final_indices = bundle["final_indices"]
    final_scores = bundle["final_scores"]
    reduced_token = bundle["reduced_token"]
    max_flag_dim = int(bundle["max_flag_dim"])
    prefix_dims = tuple(int(dim) for dim in bundle["prefix_dims"])
    flag_dims = tuple(int(dim) for dim in bundle["flag_dims"])
    retrieval_temperature = float(bundle["retrieval_temperature"])
    eps = float(bundle["eps"])

    if implementation == "native":
        global_sketch = gfst_reference_global_sketch(
            summary_projectors,
            final_indices,
            final_scores,
            retrieval_temperature=float(retrieval_temperature),
        )
        summary = top_eig_summary(
            global_sketch,
            rank=int(max_flag_dim),
            rank_boundaries=prefix_dims,
        )
        eig_basis = summary["basis"]
        eigvals = summary["eigvals"]
        prefix_projectors = {
            int(dim): projector
            for dim, projector in dict(summary["prefix_projectors"]).items()
        }
        global_features = alignment_components(
            reduced_token,
            [
                prefix_projectors.get(int(dim), basis_projector(eig_basis[:, : int(dim)]))
                for dim in flag_dims
            ],
            float(eps),
        )
        return {
            "implementation": "native",
            "global_sketch": global_sketch.detach().to(device="cpu", dtype=torch.float32).clone(),
            "eig_basis": eig_basis.detach().to(device="cpu", dtype=torch.float32).clone(),
            "eigvals": eigvals.detach().to(device="cpu", dtype=torch.float64).clone(),
            "prefix_projectors": {
                int(dim): projector.detach().to(device="cpu", dtype=torch.float32).clone()
                for dim, projector in prefix_projectors.items()
            },
            "global_features": global_features.detach().to(device="cpu", dtype=torch.float32).clone(),
        }

    if implementation != "candidate":
        raise ValueError(f"unsupported global replay implementation {implementation!r}")

    if not torch.cuda.is_available():
        raise RuntimeError("candidate global-path replay requires CUDA")

    device = torch.device("cuda:0")
    summary_projectors_cuda = summary_projectors.to(device=device, dtype=torch.float32)
    final_indices_cuda = final_indices.to(device=device, dtype=torch.long)
    final_scores_cuda = final_scores.to(device=device, dtype=torch.float32)
    reduced_token_cuda = reduced_token.to(device=device, dtype=torch.float32)

    accelerated = None
    if sm120_exact_math_kernels_available():
        accelerated = gfst_global_sketch(
            summary_projectors=summary_projectors_cuda,
            final_indices=final_indices_cuda,
            final_scores=final_scores_cuda,
            reduced_token=reduced_token_cuda,
            retrieval_temperature=float(retrieval_temperature),
            flag_dims=flag_dims,
            max_flag_dim=int(max_flag_dim),
            eps=float(eps),
        )
    if accelerated is not None:
        global_features, global_sketch, eig_basis = accelerated
    else:
        global_sketch = gfst_reference_global_sketch(
            summary_projectors_cuda,
            final_indices_cuda,
            final_scores_cuda,
            retrieval_temperature=float(retrieval_temperature),
        )
        summary = gfst_summary_from_sketch(
            global_sketch,
            max_flag_dim=int(max_flag_dim),
            rank_boundaries=prefix_dims,
            basis_dtype=global_sketch.dtype,
            vector_dtype=global_sketch.dtype,
        )
        eig_basis = summary["basis"]
        prefix_projectors = {
            int(dim): projector
            for dim, projector in dict(summary["prefix_projectors"]).items()
        }
        global_features = alignment_components(
            reduced_token_cuda.to(dtype=torch.float32),
            [
                prefix_projectors.get(int(dim), basis_projector(eig_basis[:, : int(dim)]))
                for dim in flag_dims
            ],
            float(eps),
        )

    heev_outputs = gfst_heev(global_sketch)
    if heev_outputs is None:
        summary = gfst_summary_from_sketch(
            global_sketch,
            max_flag_dim=int(max_flag_dim),
            rank_boundaries=prefix_dims,
            basis_dtype=global_sketch.dtype,
            vector_dtype=global_sketch.dtype,
        )
        eigvals = summary["eigvals"]
        prefix_projectors = {
            int(dim): projector
            for dim, projector in dict(summary["prefix_projectors"]).items()
        }
    else:
        eigvals, eigvecs = heev_outputs
        if _requires_native_summary_fallback(
            eigvals,
            rank_boundaries=tuple(sorted({int(max_flag_dim), *prefix_dims})),
        ):
            summary = gfst_summary_from_sketch(
                global_sketch,
                max_flag_dim=int(max_flag_dim),
                rank_boundaries=prefix_dims,
                basis_dtype=global_sketch.dtype,
                vector_dtype=global_sketch.dtype,
            )
        else:
            summary = gfst_summary_from_sketch(
                global_sketch,
                max_flag_dim=int(max_flag_dim),
                rank_boundaries=prefix_dims,
                eigvals=eigvals,
                eigvecs=eigvecs,
                basis_dtype=global_sketch.dtype,
                vector_dtype=global_sketch.dtype,
            )
        eigvals = summary["eigvals"]
        prefix_projectors = {
            int(dim): projector
            for dim, projector in dict(summary["prefix_projectors"]).items()
        }

    return {
        "implementation": "candidate",
        "global_sketch": global_sketch.detach().to(device="cpu", dtype=torch.float32).clone(),
        "eig_basis": eig_basis.detach().to(device="cpu", dtype=torch.float32).clone(),
        "eigvals": eigvals.detach().to(device="cpu", dtype=torch.float64).clone(),
        "prefix_projectors": {
            int(dim): projector.detach().to(device="cpu", dtype=torch.float32).clone()
            for dim, projector in prefix_projectors.items()
        },
        "global_features": global_features.detach().to(device="cpu", dtype=torch.float32).clone(),
    }


def compare_global_path_bundle(
    bundle: dict[str, Any],
    *,
    reference_implementation: str = "native",
    variant_implementation: str = "candidate",
    prefix_dims: tuple[int, ...] | list[int] | None = None,
    projector_tol: float = 1e-5,
    prefix_delta_tol: float = 1e-6,
) -> dict[str, Any]:
    dims = _canonical_prefix_dims(requested=prefix_dims or bundle["prefix_dims"], max_flag_dim=int(bundle["max_flag_dim"]))
    reference = replay_global_path_bundle(bundle, implementation=reference_implementation)
    variant = replay_global_path_bundle(bundle, implementation=variant_implementation)
    prefix_projector_deltas = {
        int(dim): _tensor_max_abs(
            reference["prefix_projectors"].get(int(dim)),
            variant["prefix_projectors"].get(int(dim)),
        )
        for dim in dims
    }
    full_rank = int(max(dims))
    full_projector_delta = float(prefix_projector_deltas[int(full_rank)])
    unstable_cutoffs = [
        int(dim)
        for dim in dims
        if int(dim) != int(full_rank) and float(prefix_projector_deltas[int(dim)]) > max(float(prefix_delta_tol), full_projector_delta * 10.0)
    ]
    reference_clusters = _cluster_diagnosis_from_eigvals(
        reference["eigvals"],
        prefix_dims=dims,
        max_rank=int(full_rank),
    )
    variant_clusters = _cluster_diagnosis_from_eigvals(
        variant["eigvals"],
        prefix_dims=dims,
        max_rank=int(full_rank),
    )
    classification = "other_global_path_mismatch"
    if (
        _tensor_max_abs(reference["global_sketch"], variant["global_sketch"]) <= float(prefix_delta_tol)
        and _tensor_max_abs(reference["global_features"], variant["global_features"]) <= float(prefix_delta_tol)
        and max(prefix_projector_deltas.values(), default=0.0) <= float(prefix_delta_tol)
    ):
        classification = "clean_boundary"
    elif unstable_cutoffs and full_projector_delta <= float(projector_tol):
        first_cutoff = int(unstable_cutoffs[0])
        if bool(reference_clusters.get(first_cutoff, {}).get("splits_cluster")) or bool(
            variant_clusters.get(first_cutoff, {}).get("splits_cluster")
        ):
            classification = "prefix_cutoff_inside_degenerate_cluster"

    reference_top_vals, reference_gaps = _sorted_descending_eigvals_and_gaps(reference["eigvals"], max_rank=int(full_rank))
    variant_top_vals, variant_gaps = _sorted_descending_eigvals_and_gaps(variant["eigvals"], max_rank=int(full_rank))
    return {
        "classification": classification,
        "reference_implementation": str(reference_implementation),
        "variant_implementation": str(variant_implementation),
        "global_sketch_max_abs": _tensor_max_abs(reference["global_sketch"], variant["global_sketch"]),
        "eig_basis_max_abs": _tensor_max_abs(reference["eig_basis"], variant["eig_basis"]),
        "global_features_max_abs": _tensor_max_abs(reference["global_features"], variant["global_features"]),
        "prefix_projector_deltas": {str(dim): float(delta) for dim, delta in prefix_projector_deltas.items()},
        "full_rank_projector_delta": float(full_projector_delta),
        "unstable_cutoffs": [int(dim) for dim in unstable_cutoffs],
        "reference_cutoff_clusters": {str(dim): value for dim, value in reference_clusters.items()},
        "variant_cutoff_clusters": {str(dim): value for dim, value in variant_clusters.items()},
        "reference_top_eigenvalues_desc": reference_top_vals,
        "reference_adjacent_gaps": reference_gaps,
        "variant_top_eigenvalues_desc": variant_top_vals,
        "variant_adjacent_gaps": variant_gaps,
    }


def summarize_boundary_spectrum_site(
    native_capture: dict[str, Any],
    variant_capture: dict[str, Any],
    *,
    prefix_dims: tuple[int, ...] | list[int] | None = None,
    retrieval_tol: float = 1e-6,
    clean_tol: float = 1e-6,
    projector_tol: float = 1e-5,
) -> dict[str, Any]:
    native_bundle = global_path_bundle_from_capture(native_capture, prefix_dims=prefix_dims)
    variant_bundle = global_path_bundle_from_capture(variant_capture, prefix_dims=prefix_dims)
    dims = _canonical_prefix_dims(requested=native_bundle["prefix_dims"], max_flag_dim=int(native_bundle["max_flag_dim"]))

    native_summary = top_eig_summary(
        native_capture["global_path"]["global_sketch"],
        rank=int(native_bundle["max_flag_dim"]),
        rank_boundaries=dims,
    )
    variant_summary = top_eig_summary(
        variant_capture["global_path"]["global_sketch"],
        rank=int(variant_bundle["max_flag_dim"]),
        rank_boundaries=dims,
    )
    prefix_projector_deltas = {
        int(dim): _tensor_max_abs(
            dict(native_summary["prefix_projectors"]).get(int(dim)),
            dict(variant_summary["prefix_projectors"]).get(int(dim)),
        )
        for dim in dims
    }
    full_rank = int(max(dims))
    full_projector_delta = float(prefix_projector_deltas[int(full_rank)])
    native_clusters = _cluster_diagnosis_from_eigvals(
        native_summary["eigvals"],
        prefix_dims=dims,
        max_rank=int(full_rank),
    )
    variant_clusters = _cluster_diagnosis_from_eigvals(
        variant_summary["eigvals"],
        prefix_dims=dims,
        max_rank=int(full_rank),
    )

    coarse_indices_delta = _tensor_max_abs(
        native_capture["global_path"].get("coarse_indices"),
        variant_capture["global_path"].get("coarse_indices"),
    )
    final_indices_delta = _tensor_max_abs(
        native_capture["global_path"].get("final_indices"),
        variant_capture["global_path"].get("final_indices"),
    )
    coarse_scores_delta = _tensor_max_abs(
        native_capture["global_path"].get("coarse_scores"),
        variant_capture["global_path"].get("coarse_scores"),
    )
    final_scores_delta = _tensor_max_abs(
        native_capture["global_path"].get("final_scores"),
        variant_capture["global_path"].get("final_scores"),
    )
    retrieval_mismatch = any(
        float(delta) > float(retrieval_tol)
        for delta in (coarse_indices_delta, final_indices_delta, coarse_scores_delta, final_scores_delta)
    )
    unstable_cutoffs = [
        int(dim)
        for dim in dims
        if int(dim) != int(full_rank) and float(prefix_projector_deltas[int(dim)]) > max(float(clean_tol), full_projector_delta * 10.0)
    ]
    classification = "other_global_path_mismatch"
    if (
        not retrieval_mismatch
        and _tensor_max_abs(native_capture["global_path"].get("global_sketch"), variant_capture["global_path"].get("global_sketch"))
        <= float(clean_tol)
        and _tensor_max_abs(native_capture["global_path"].get("global_features"), variant_capture["global_path"].get("global_features"))
        <= float(clean_tol)
        and _tensor_max_abs(
            native_capture.get("post_feature", {}).get("hidden_token_out"),
            variant_capture.get("post_feature", {}).get("hidden_token_out"),
        )
        <= float(clean_tol)
        and max(prefix_projector_deltas.values(), default=0.0) <= float(clean_tol)
    ):
        classification = "clean_boundary"
    elif retrieval_mismatch:
        classification = "retrieval_mismatch"
    elif unstable_cutoffs and full_projector_delta <= float(projector_tol):
        first_cutoff = int(unstable_cutoffs[0])
        if bool(native_clusters.get(first_cutoff, {}).get("splits_cluster")) or bool(
            variant_clusters.get(first_cutoff, {}).get("splits_cluster")
        ):
            classification = "degenerate_prefix_boundary"

    native_top_vals, native_gaps = _sorted_descending_eigvals_and_gaps(native_summary["eigvals"], max_rank=int(full_rank))
    variant_top_vals, variant_gaps = _sorted_descending_eigvals_and_gaps(variant_summary["eigvals"], max_rank=int(full_rank))
    return {
        "site_id": str(
            f"layer_{int(native_capture['metadata']['layer_idx'])}_batch_{int(native_capture['metadata']['batch_idx'])}"
            f"_token_{int(native_capture['metadata']['token_idx'])}"
        ),
        "classification": classification,
        "summary_counts": {
            "native": int(native_bundle["summary_count"]),
            "variant": int(variant_bundle["summary_count"]),
        },
        "retrieval": {
            "coarse_indices_delta": float(coarse_indices_delta),
            "coarse_scores_delta": float(coarse_scores_delta),
            "final_indices_delta": float(final_indices_delta),
            "final_scores_delta": float(final_scores_delta),
        },
        "global_sketch_max_abs": _tensor_max_abs(
            native_capture["global_path"].get("global_sketch"),
            variant_capture["global_path"].get("global_sketch"),
        ),
        "eig_basis_max_abs": _tensor_max_abs(
            native_capture["global_path"].get("eig_basis"),
            variant_capture["global_path"].get("eig_basis"),
        ),
        "global_features_max_abs": _tensor_max_abs(
            native_capture["global_path"].get("global_features"),
            variant_capture["global_path"].get("global_features"),
        ),
        "hidden_token_out_max_abs": _tensor_max_abs(
            native_capture.get("post_feature", {}).get("hidden_token_out"),
            variant_capture.get("post_feature", {}).get("hidden_token_out"),
        ),
        "prefix_projector_deltas": {str(dim): float(delta) for dim, delta in prefix_projector_deltas.items()},
        "full_rank_projector_delta": float(full_projector_delta),
        "unstable_cutoffs": [int(dim) for dim in unstable_cutoffs],
        "native_cutoff_clusters": {str(dim): value for dim, value in native_clusters.items()},
        "variant_cutoff_clusters": {str(dim): value for dim, value in variant_clusters.items()},
        "native_top_eigenvalues_desc": native_top_vals,
        "native_adjacent_gaps": native_gaps,
        "variant_top_eigenvalues_desc": variant_top_vals,
        "variant_adjacent_gaps": variant_gaps,
    }


def summarize_boundary_spectrum_probe(
    *,
    native_captures: dict[str, dict[str, Any]],
    variant_captures: dict[str, dict[str, Any]],
    prefix_dims: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    site_ids = sorted(set(native_captures.keys()) | set(variant_captures.keys()))
    site_summaries: list[dict[str, Any]] = []
    first_dirty_site = None
    for site_id in site_ids:
        native_capture = native_captures.get(site_id)
        variant_capture = variant_captures.get(site_id)
        if native_capture is None or variant_capture is None:
            summary = {
                "site_id": site_id,
                "classification": "other_global_path_mismatch",
                "summary_counts": {"native": 0, "variant": 0},
                "retrieval": {
                    "coarse_indices_delta": 1.0,
                    "coarse_scores_delta": 1.0,
                    "final_indices_delta": 1.0,
                    "final_scores_delta": 1.0,
                },
                "global_sketch_max_abs": 1.0,
                "eig_basis_max_abs": 1.0,
                "global_features_max_abs": 1.0,
                "hidden_token_out_max_abs": 1.0,
                "prefix_projector_deltas": {},
                "full_rank_projector_delta": 1.0,
                "unstable_cutoffs": [],
                "native_cutoff_clusters": {},
                "variant_cutoff_clusters": {},
                "native_top_eigenvalues_desc": [],
                "native_adjacent_gaps": [],
                "variant_top_eigenvalues_desc": [],
                "variant_adjacent_gaps": [],
            }
        else:
            summary = summarize_boundary_spectrum_site(
                native_capture,
                variant_capture,
                prefix_dims=prefix_dims,
            )
        if first_dirty_site is None and str(summary["classification"]) != "clean_boundary":
            first_dirty_site = str(summary["site_id"])
        site_summaries.append(summary)
    return {
        "site_summaries": site_summaries,
        "first_dirty_site": first_dirty_site,
    }
