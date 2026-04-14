from __future__ import annotations

from typing import Any

import torch


TRANSITION_STAGE_ORDER = (
    "pre_token_inputs",
    "local_path",
    "global_path",
    "block_memory",
    "post_feature",
)

_DEBUG_ONLY_TRANSITION_KEYS = frozenset({"accelerated_global_features_raw"})


def transition_site_id(*, layer_idx: int, token_idx: int, batch_idx: int = 0) -> str:
    return f"layer_{int(layer_idx)}_batch_{int(batch_idx)}_token_{int(token_idx)}"


def _tensor_delta(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if tuple(lhs.shape) != tuple(rhs.shape):
        return 1.0
    if lhs.dtype == torch.bool or rhs.dtype == torch.bool:
        return 0.0 if bool(torch.equal(lhs, rhs)) else 1.0
    if int(lhs.numel()) == 0:
        return 0.0
    lhs_f = torch.nan_to_num(lhs.to(dtype=torch.float32), nan=0.0, posinf=1e30, neginf=-1e30)
    rhs_f = torch.nan_to_num(rhs.to(dtype=torch.float32), nan=0.0, posinf=1e30, neginf=-1e30)
    return float((rhs_f - lhs_f).abs().max().item())


def transition_value_delta(lhs: Any, rhs: Any) -> float:
    if lhs is None and rhs is None:
        return 0.0
    if lhs is None or rhs is None:
        return 1.0
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        return _tensor_delta(lhs, rhs)
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        keys = (set(lhs.keys()) | set(rhs.keys())) - set(_DEBUG_ONLY_TRANSITION_KEYS)
        return max((transition_value_delta(lhs.get(key), rhs.get(key)) for key in keys), default=0.0)
    if isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)):
        pairwise = [transition_value_delta(l_item, r_item) for l_item, r_item in zip(lhs, rhs)]
        if len(lhs) != len(rhs):
            pairwise.append(1.0)
        return max(pairwise, default=0.0)
    if isinstance(lhs, (bool, int, float, str)) and isinstance(rhs, (bool, int, float, str)):
        return 0.0 if lhs == rhs else float(abs(float(rhs) - float(lhs))) if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)) else 1.0
    return 0.0 if lhs == rhs else 1.0


def summarize_transition_site(
    native_capture: dict[str, Any],
    variant_capture: dict[str, Any],
) -> dict[str, Any]:
    stage_max_abs = {
        stage: float(transition_value_delta(native_capture.get(stage), variant_capture.get(stage)))
        for stage in TRANSITION_STAGE_ORDER
    }
    first_differing_stage = next((stage for stage in TRANSITION_STAGE_ORDER if stage_max_abs[stage] > 0.0), None)
    classification_map = {
        "pre_token_inputs": "pretoken_input_mismatch",
        "local_path": "local_path_mismatch",
        "global_path": "global_path_mismatch",
        "block_memory": "block_commit_mismatch",
        "post_feature": "post_feature_delta_mismatch",
    }
    metadata = dict(native_capture.get("metadata", {}))
    metadata.update({key: value for key, value in dict(variant_capture.get("metadata", {})).items() if key not in metadata})
    return {
        "site_id": transition_site_id(
            layer_idx=int(metadata.get("layer_idx", -1)),
            batch_idx=int(metadata.get("batch_idx", 0)),
            token_idx=int(metadata.get("token_idx", -1)),
        ),
        "metadata": metadata,
        "stage_max_abs": stage_max_abs,
        "first_differing_stage": first_differing_stage,
        "classification": None if first_differing_stage is None else classification_map[first_differing_stage],
        "clean": bool(first_differing_stage is None),
    }


def summarize_transition_probe(
    *,
    native_captures: dict[str, dict[str, Any]],
    variant_captures: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    site_ids = sorted(set(native_captures.keys()) | set(variant_captures.keys()))
    site_summaries = []
    by_id: dict[str, dict[str, Any]] = {}
    for site_id in site_ids:
        native_capture = native_captures.get(site_id)
        variant_capture = variant_captures.get(site_id)
        if native_capture is None or variant_capture is None:
            summary = {
                "site_id": site_id,
                "metadata": {},
                "stage_max_abs": {stage: 1.0 for stage in TRANSITION_STAGE_ORDER},
                "first_differing_stage": "pre_token_inputs",
                "classification": "pretoken_input_mismatch",
                "clean": False,
            }
        else:
            summary = summarize_transition_site(native_capture, variant_capture)
        site_summaries.append(summary)
        by_id[site_id] = summary

    layer0_191 = by_id.get(transition_site_id(layer_idx=0, batch_idx=0, token_idx=191))
    layer0_192 = by_id.get(transition_site_id(layer_idx=0, batch_idx=0, token_idx=192))
    layer1_192 = by_id.get(transition_site_id(layer_idx=1, batch_idx=0, token_idx=192))

    if layer0_192 is not None and float(layer0_192["stage_max_abs"]["pre_token_inputs"]) > 0.0:
        top_level = "upstream_of_layer_mismatch"
    elif layer0_191 is not None and not bool(layer0_191["clean"]):
        top_level = str(layer0_191["classification"])
    elif layer0_192 is not None and not bool(layer0_192["clean"]):
        top_level = str(layer0_192["classification"])
    elif layer1_192 is not None and not bool(layer1_192["clean"]):
        top_level = "upstream_of_layer_mismatch"
    else:
        top_level = "upstream_of_layer_mismatch"

    cross_layer_hidden_bridge = {
        "layer0_token192_hidden_token_out_delta": float(
            transition_value_delta(
                native_captures.get(transition_site_id(layer_idx=0, batch_idx=0, token_idx=192), {})
                .get("post_feature", {})
                .get("hidden_token_out"),
                variant_captures.get(transition_site_id(layer_idx=0, batch_idx=0, token_idx=192), {})
                .get("post_feature", {})
                .get("hidden_token_out"),
            )
        ),
        "layer1_token192_hidden_token_in_delta": float(
            transition_value_delta(
                native_captures.get(transition_site_id(layer_idx=1, batch_idx=0, token_idx=192), {})
                .get("pre_token_inputs", {})
                .get("hidden_token_in"),
                variant_captures.get(transition_site_id(layer_idx=1, batch_idx=0, token_idx=192), {})
                .get("pre_token_inputs", {})
                .get("hidden_token_in"),
            )
        ),
    }
    return {
        "classification": top_level,
        "site_summaries": site_summaries,
        "cross_layer_hidden_bridge": cross_layer_hidden_bridge,
    }
