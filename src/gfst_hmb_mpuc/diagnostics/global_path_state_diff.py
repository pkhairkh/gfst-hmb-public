from __future__ import annotations

from typing import Any

import torch

from .block_boundary_transition_probe import transition_value_delta


SUMMARY_CACHE_FIELDS = (
    "summary_count",
    "summary_cache_rebuilt",
    "summary_cache_rebuild_reasons",
    "summary_key_tensor",
    "summary_block_index_tensor",
    "summary_token_count_tensor",
    "summary_reduced_mean_tensor",
    "summary_basis_tensor",
    "summary_projector_tensor",
)

RETRIEVAL_FIELDS = (
    "coarse_indices",
    "coarse_scores",
    "final_indices",
    "final_scores",
    "selected_weights",
    "retrieved_block_ids_tensor",
    "retrieved_prefix_lengths_tensor",
    "current_block_id",
    "current_prefix",
    "returned_block_tensor",
    "returned_score_tensor",
    "returned_prefix_tensor",
    "returned_coarse_block_tensor",
    "returned_coarse_score_tensor",
    "returned_facet_scores",
)

OUTPUT_FIELDS = (
    "global_sketch",
    "eig_basis",
    "global_features",
)

FEATURE_MATH_FIELDS = (
    "feature_actual_path",
    "feature_reduced_token",
    "feature_projector_debug",
    "feature_basis_native_debug",
    "feature_basis_float32_debug",
    "feature_actual_minus_projector_ref_max_abs",
    "feature_actual_minus_basis_native_ref_max_abs",
    "feature_actual_minus_basis_float32_ref_max_abs",
)


def _field_deltas(
    native_capture: dict[str, Any],
    variant_capture: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, float]:
    return {
        str(field): float(transition_value_delta(native_capture.get(field), variant_capture.get(field)))
        for field in fields
    }


def _flatten_leaf_deltas(
    native_value: Any,
    variant_value: Any,
    *,
    prefix: str,
) -> list[tuple[float, str]]:
    if isinstance(native_value, dict) and isinstance(variant_value, dict):
        rows: list[tuple[float, str]] = []
        for key in sorted(set(native_value.keys()) | set(variant_value.keys())):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(
                _flatten_leaf_deltas(
                    native_value.get(key),
                    variant_value.get(key),
                    prefix=child_prefix,
                )
            )
        return rows
    if isinstance(native_value, (list, tuple)) and isinstance(variant_value, (list, tuple)):
        rows = []
        for index, (lhs, rhs) in enumerate(zip(native_value, variant_value)):
            child_prefix = f"{prefix}[{index}]"
            rows.extend(_flatten_leaf_deltas(lhs, rhs, prefix=child_prefix))
        if len(native_value) != len(variant_value):
            rows.append((1.0, f"{prefix}.length"))
        return rows
    return [(float(transition_value_delta(native_value, variant_value)), prefix)]


def summarize_global_path_state_diff(
    native_capture: dict[str, Any],
    variant_capture: dict[str, Any],
) -> dict[str, Any]:
    native_global = dict(native_capture.get("global_path", {}))
    variant_global = dict(variant_capture.get("global_path", {}))

    summary_cache_deltas = _field_deltas(native_global, variant_global, SUMMARY_CACHE_FIELDS)
    retrieval_deltas = _field_deltas(native_global, variant_global, RETRIEVAL_FIELDS)
    feature_math_deltas = _field_deltas(native_global, variant_global, FEATURE_MATH_FIELDS)
    output_deltas = _field_deltas(native_global, variant_global, OUTPUT_FIELDS)

    all_leaf_rows = _flatten_leaf_deltas(native_global, variant_global, prefix="global_path")
    nonzero_leaf_rows = [
        {"path": str(path), "max_abs": float(delta)}
        for delta, path in sorted(all_leaf_rows, key=lambda item: (-float(item[0]), str(item[1])))
        if float(delta) > 0.0
    ]

    summary_cache_max = max(summary_cache_deltas.values(), default=0.0)
    retrieval_max = max(retrieval_deltas.values(), default=0.0)
    feature_math_max = max(feature_math_deltas.values(), default=0.0)
    output_max = max(output_deltas.values(), default=0.0)

    classification = "clean_global_path"
    first_nonzero_group = None
    if summary_cache_max > 0.0:
        classification = "summary_cache_state_mismatch"
        first_nonzero_group = "summary_cache"
    elif retrieval_max > 0.0:
        classification = "retrieval_state_mismatch"
        first_nonzero_group = "retrieval_state"
    elif feature_math_max > 0.0:
        classification = "feature_math_mismatch"
        first_nonzero_group = "feature_math"
    elif output_max > 0.0:
        classification = "output_only_mismatch"
        first_nonzero_group = "output_state"

    return {
        "classification": classification,
        "first_nonzero_group": first_nonzero_group,
        "summary_cache_deltas": summary_cache_deltas,
        "retrieval_deltas": retrieval_deltas,
        "feature_math_deltas": feature_math_deltas,
        "output_deltas": output_deltas,
        "top_differences": nonzero_leaf_rows[:20],
    }
