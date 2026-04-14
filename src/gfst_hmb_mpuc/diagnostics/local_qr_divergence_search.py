from __future__ import annotations

from typing import Any, Callable

import torch

from .local_qr_divergence import tensor_max_abs


def summarize_capture_site_delta(
    native_capture: dict[str, Any],
    variant_capture: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "reduced_token_max_abs": tensor_max_abs(native_capture["reduced_token"], variant_capture["reduced_token"]),
        "largest_candidate_max_abs": tensor_max_abs(
            native_capture["largest_candidate"],
            variant_capture["largest_candidate"],
        ),
        "largest_basis_in_max_abs": tensor_max_abs(
            native_capture["largest_basis_in"],
            variant_capture["largest_basis_in"],
        ),
        "largest_basis_out_max_abs": tensor_max_abs(
            native_capture["largest_basis_out"],
            variant_capture["largest_basis_out"],
        ),
        "projected_max_abs": tensor_max_abs(native_capture["projected"], variant_capture["projected"]),
        "local_features_max_abs": tensor_max_abs(
            native_capture["local_features"],
            variant_capture["local_features"],
        ),
        "nested_steps": [],
    }
    max_input_abs = max(
        summary["reduced_token_max_abs"],
        summary["largest_candidate_max_abs"],
        summary["largest_basis_in_max_abs"],
    )
    max_output_abs = max(
        summary["largest_basis_out_max_abs"],
        summary["projected_max_abs"],
        summary["local_features_max_abs"],
    )
    nested_native = list(native_capture.get("nested_steps", []))
    nested_variant = list(variant_capture.get("nested_steps", []))
    for native_step, variant_step in zip(nested_native, nested_variant):
        nested_summary = {
            "nested_index": int(native_step["nested_index"]),
            "nested_candidate_max_abs": tensor_max_abs(
                native_step["nested_candidate"],
                variant_step["nested_candidate"],
            ),
            "nested_basis_in_max_abs": tensor_max_abs(
                native_step["nested_basis_in"],
                variant_step["nested_basis_in"],
            ),
            "nested_basis_out_max_abs": tensor_max_abs(
                native_step["nested_basis_out"],
                variant_step["nested_basis_out"],
            ),
        }
        summary["nested_steps"].append(nested_summary)
        max_input_abs = max(
            max_input_abs,
            nested_summary["nested_candidate_max_abs"],
            nested_summary["nested_basis_in_max_abs"],
        )
        max_output_abs = max(max_output_abs, nested_summary["nested_basis_out_max_abs"])
    summary["max_input_max_abs"] = float(max_input_abs)
    summary["max_output_max_abs"] = float(max_output_abs)
    summary["max_any_max_abs"] = float(max(max_input_abs, max_output_abs))
    return summary


def binary_search_first_dirty_index(
    *,
    clean_index: int,
    dirty_index: int,
    is_dirty: Callable[[int], bool],
) -> int:
    if int(clean_index) >= int(dirty_index):
        raise ValueError(f"expected clean_index < dirty_index, got {clean_index} >= {dirty_index}")
    if bool(is_dirty(int(clean_index))):
        raise ValueError(f"clean_index={clean_index} is already dirty")
    if not bool(is_dirty(int(dirty_index))):
        raise ValueError(f"dirty_index={dirty_index} is not dirty")
    lo = int(clean_index)
    hi = int(dirty_index)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if bool(is_dirty(mid)):
            hi = mid
        else:
            lo = mid
    return hi


def verify_first_dirty_index(
    *,
    clean_index: int,
    first_dirty_index: int,
    is_dirty: Callable[[int], bool],
    window: int,
) -> int:
    start = max(int(clean_index) + 1, int(first_dirty_index) - max(int(window), 0))
    for index in range(start, int(first_dirty_index) + 1):
        if bool(is_dirty(index)):
            return index
    return int(first_dirty_index)
