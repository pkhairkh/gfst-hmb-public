from __future__ import annotations

from typing import Any

import torch

from .block_boundary_transition_probe import summarize_transition_site, transition_value_delta
from .global_path_state_diff import summarize_global_path_state_diff


def _token_value(run: dict[str, Any], key: str, *, token_idx: int, batch_idx: int = 0) -> Any:
    value = run.get(str(key))
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu")
        if tensor.ndim >= 2:
            if int(tensor.shape[0]) <= int(batch_idx) or int(tensor.shape[1]) <= int(token_idx):
                return None
            return tensor[int(batch_idx), int(token_idx)].clone()
        return tensor.clone()
    return value


def _field_deltas(
    native_run: dict[str, Any],
    variant_run: dict[str, Any],
    *,
    token_idx: int,
    fields: tuple[str, ...],
) -> dict[str, float]:
    return {
        str(field): float(
            transition_value_delta(
                _token_value(native_run, str(field), token_idx=int(token_idx)),
                _token_value(variant_run, str(field), token_idx=int(token_idx)),
            )
        )
        for field in fields
    }


def _tensor_to_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        if int(value.numel()) == 0:
            return []
        return [int(item) for item in value.to(device="cpu", dtype=torch.long).reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _feature_math_summary(capture: dict[str, Any]) -> dict[str, Any]:
    global_path = dict(capture.get("global_path", {}))
    return {
        "feature_actual_path": global_path.get("feature_actual_path"),
        "feature_actual_minus_projector_ref_max_abs": global_path.get("feature_actual_minus_projector_ref_max_abs"),
        "feature_actual_minus_basis_native_ref_max_abs": global_path.get(
            "feature_actual_minus_basis_native_ref_max_abs"
        ),
        "feature_actual_minus_basis_float32_ref_max_abs": global_path.get(
            "feature_actual_minus_basis_float32_ref_max_abs"
        ),
    }


def summarize_commit_invariants(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(capture.get("metadata", {}))
    block_memory = dict(capture.get("block_memory", {}))
    tracker_before = dict(block_memory.get("tracker_before", {}))
    tracker_after = dict(block_memory.get("tracker_after_reset") or {})
    appended_summary = dict(block_memory.get("appended_summary") or {})
    absolute_position = int(metadata.get("absolute_position", -1))
    block_len = max(1, int(metadata.get("block_len", 1)))
    expected_block_index = (
        int(block_memory["expected_closed_block_index"])
        if block_memory.get("expected_closed_block_index") is not None
        else int(tracker_before.get("current_block_start", 0)) // block_len
    )
    summary_count_before = int(block_memory.get("summary_count_before", 0))
    summary_count_after = int(block_memory.get("summary_count_after", summary_count_before))
    appended_block_index = None if not appended_summary else int(appended_summary.get("block_index", -1))
    appended_end_pos = None if not appended_summary else int(appended_summary.get("end_pos", -1))
    tracker_reset_ok = (
        int(tracker_after.get("current_block_tokens", -1)) == 0
        and int(tracker_after.get("current_block_start", -1)) == int(absolute_position + 1)
    )
    return {
        "commit_fired": bool(block_memory.get("commit_fired", False)),
        "summary_count_before": int(summary_count_before),
        "summary_count_after": int(summary_count_after),
        "summary_count_increment": int(summary_count_after - summary_count_before),
        "expected_block_index": int(expected_block_index),
        "appended_summary_present": bool(appended_summary),
        "appended_block_index": appended_block_index,
        "appended_end_pos": appended_end_pos,
        "appended_block_index_matches_expected": (
            appended_block_index is not None and int(appended_block_index) == int(expected_block_index)
        ),
        "tracker_reset_ok": bool(tracker_reset_ok),
        "summary_cache_invalidated": bool(block_memory.get("summary_cache_invalidated", False)),
        "summary_block_indices_before": [
            int(item) for item in block_memory.get("summary_block_indices_before", [])
        ],
        "summary_block_indices_after": [
            int(item) for item in block_memory.get("summary_block_indices_after", [])
        ],
    }


def summarize_consume_invariants(
    *,
    commit_capture: dict[str, Any],
    consume_capture: dict[str, Any],
) -> dict[str, Any]:
    commit_block = dict(commit_capture.get("block_memory", {}))
    appended_summary = dict(commit_block.get("appended_summary") or {})
    committed_block_index = None if not appended_summary else int(appended_summary.get("block_index", -1))
    committed_token_count = None if not appended_summary else int(appended_summary.get("token_count", -1))
    global_path = dict(consume_capture.get("global_path", {}))
    summary_indices = _tensor_to_int_list(global_path.get("summary_block_index_tensor"))
    summary_token_counts = _tensor_to_int_list(global_path.get("summary_token_count_tensor"))
    matching_positions = (
        [index for index, value in enumerate(summary_indices) if int(value) == int(committed_block_index)]
        if committed_block_index is not None
        else []
    )
    matching_token_counts = [
        int(summary_token_counts[index])
        for index in matching_positions
        if index < len(summary_token_counts)
    ]
    return {
        "committed_block_index": committed_block_index,
        "committed_token_count": committed_token_count,
        "summary_cache_rebuilt": bool(global_path.get("summary_cache_rebuilt", False)),
        "summary_cache_rebuild_reasons": list(global_path.get("summary_cache_rebuild_reasons", [])),
        "summary_count_seen_by_consumer": int(global_path.get("summary_count", 0)),
        "summary_block_indices_seen_by_consumer": summary_indices,
        "summary_token_counts_seen_by_consumer": summary_token_counts,
        "committed_block_seen_count": int(len(matching_positions)),
        "committed_block_seen_exactly_once": bool(len(matching_positions) == 1),
        "committed_block_token_counts_seen": matching_token_counts,
        "current_block_id": global_path.get("current_block_id"),
        "current_prefix": global_path.get("current_prefix"),
    }


READOUT_INPUT_FIELDS = (
    "retrieval_queries",
    "retrieved_block_indices",
    "retrieved_block_scores",
    "retrieved_prefix_lengths",
    "coarse_retrieved_block_indices",
    "coarse_retrieved_block_scores",
    "coarse_retrieved_facet_scores",
)

READOUT_OUTPUT_FIELDS = (
    "pointer_scores",
    "router_feature_tensor",
    "selected_copy_token_ids",
    "pointer_candidate_token_ids",
    "pointer_candidate_blocks",
    "pointer_candidate_positions",
    "pointer_candidate_mask",
    "copy_decision_logits",
    "generation_logits",
    "logits",
)


def summarize_block_handoff_probe(
    *,
    native_commit_capture: dict[str, Any],
    variant_commit_capture: dict[str, Any],
    native_consume_capture: dict[str, Any],
    variant_consume_capture: dict[str, Any],
    native_run: dict[str, Any],
    variant_run: dict[str, Any],
    commit_token_idx: int,
    consume_token_idx: int,
) -> dict[str, Any]:
    commit_site_summary = summarize_transition_site(native_commit_capture, variant_commit_capture)
    consume_site_summary = summarize_transition_site(native_consume_capture, variant_consume_capture)
    consume_global_path = summarize_global_path_state_diff(native_consume_capture, variant_consume_capture)
    post_feature_hidden_delta = float(
        transition_value_delta(
            dict(native_consume_capture.get("post_feature", {})).get("hidden_token_out"),
            dict(variant_consume_capture.get("post_feature", {})).get("hidden_token_out"),
        )
    )
    readout_input_deltas = _field_deltas(
        native_run,
        variant_run,
        token_idx=int(consume_token_idx),
        fields=READOUT_INPUT_FIELDS,
    )
    readout_output_deltas = _field_deltas(
        native_run,
        variant_run,
        token_idx=int(consume_token_idx),
        fields=READOUT_OUTPUT_FIELDS,
    )
    readout_input_max = max(readout_input_deltas.values(), default=0.0)
    readout_output_max = max(readout_output_deltas.values(), default=0.0)

    first_dirty_bucket = "clean"
    if not bool(commit_site_summary.get("clean", False)):
        first_dirty_bucket = "commit_path"
    elif str(consume_site_summary.get("first_differing_stage") or "") == "global_path":
        group = consume_global_path.get("first_nonzero_group")
        if group == "summary_cache":
            first_dirty_bucket = "summary_cache"
        elif group == "retrieval_state":
            first_dirty_bucket = "retrieval_state"
        elif group == "feature_math":
            first_dirty_bucket = "global_feature_math"
        elif group == "output_state":
            first_dirty_bucket = "global_fusion_output"
        else:
            first_dirty_bucket = "global_path"
    elif float(post_feature_hidden_delta) > 0.0:
        first_dirty_bucket = "layer0_post_feature"
    elif float(readout_input_max) > 0.0:
        first_dirty_bucket = "exact_readout_inputs"
    elif float(readout_output_max) > 0.0:
        first_dirty_bucket = "exact_readout_outputs"

    return {
        "zero_based_commit_token_idx": int(commit_token_idx),
        "zero_based_consume_token_idx": int(consume_token_idx),
        "human_commit_token_number": int(commit_token_idx) + 1,
        "human_consume_token_number": int(consume_token_idx) + 1,
        "first_dirty_bucket": first_dirty_bucket,
        "commit_site_summary": commit_site_summary,
        "consume_site_summary": consume_site_summary,
        "native_commit_invariants": summarize_commit_invariants(native_commit_capture),
        "variant_commit_invariants": summarize_commit_invariants(variant_commit_capture),
        "native_consume_invariants": summarize_consume_invariants(
            commit_capture=native_commit_capture,
            consume_capture=native_consume_capture,
        ),
        "variant_consume_invariants": summarize_consume_invariants(
            commit_capture=variant_commit_capture,
            consume_capture=variant_consume_capture,
        ),
        "consume_global_path": consume_global_path,
        "native_consume_feature_math": _feature_math_summary(native_consume_capture),
        "variant_consume_feature_math": _feature_math_summary(variant_consume_capture),
        "consume_post_feature_hidden_delta": float(post_feature_hidden_delta),
        "readout_input_deltas": readout_input_deltas,
        "readout_output_deltas": readout_output_deltas,
    }
