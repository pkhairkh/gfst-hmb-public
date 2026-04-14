from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
from torch.nn.utils.rnn import pad_sequence

from .config import FaissGpuRetrievalConfig, GfstHmbConfig, MpucGfstHmbConfig
from .gfst import (
    alignment_components,
    alignment_components_from_bases,
    alignment_components_debug,
    alignment_components_from_full_basis_debug,
    alignment_components_from_full_basis_batch,
    apply_rope,
    basis_projector,
    canonical_qr_batched,
    init_basis,
    top_eig_summary,
    update_basis,
    update_basis_candidate,
)
from .perf import RuntimeProfileRecorder, collapse_logsumexp_scores_and_occurrence, get_liger_rms_norm_cls, runtime_range
from .retrieval import FaissGpuRetriever
from .runtime_backends import exact_block_score as sm120_exact_block_score
from .runtime_backends import gfst_block_commit as sm120_gfst_block_commit
from .runtime_backends import gfst_canonical_summary_from_eigpairs as sm120_gfst_canonical_summary_from_eigpairs
from .runtime_backends import gfst_heev as sm120_gfst_heev
from .runtime_backends import gfst_global_sketch as sm120_gfst_global_sketch
from .runtime_backends import gfst_local_qr as sm120_gfst_local_qr
from .runtime_backends import gfst_summary_from_sketch as sm120_gfst_summary_from_sketch
from .runtime_backends import gfst_summary_from_sketch_batched as sm120_gfst_summary_from_sketch_batched
from .runtime_backends import mpuc_recurrent_scan as sm120_mpuc_recurrent_scan
from .runtime_backends import sm120_exact_math_kernels_available
from .runtime_backends import sm120 as _sm120_runtime
from .state import (
    _CENTROID_COUNT_QUANT_SCALE,
    _CENTROID_VALUE_QUANT_SCALE,
    _CENTROID_WEIGHT_QUANT_SCALE,
    _CoreLayerCacheState,
    ExactBlockRecord,
    GfstBlockSummary,
    GfstDecodeState,
)

try:
    from transformers import AutoModelForCausalLM
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError:
    AutoModelForCausalLM = None

    @dataclass
    class CausalLMOutputWithPast:
        loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None
        past_key_values: Any = None
        hidden_states: Any = None
        attentions: Any = None

@dataclass
class _PerTokenResult:
    delta: torch.Tensor
    features: torch.Tensor
    query: torch.Tensor
    retrieved_block_ids: torch.Tensor
    retrieved_scores: torch.Tensor
    retrieved_prefix_lengths: torch.Tensor
    coarse_retrieved_block_ids: torch.Tensor
    coarse_retrieved_scores: torch.Tensor
    coarse_retrieved_facet_scores: torch.Tensor
    centroid_block_id: torch.Tensor
    centroid_assignment_ids: torch.Tensor
    centroid_assignment_weights: torch.Tensor
    centroid_assignment_logits: torch.Tensor
    centroid_token_key: torch.Tensor
    centroid_token_value: torch.Tensor
    centroid_valid_mask: torch.Tensor
    gate_mean: float
    gate_std: float
    delta_rms: float


@dataclass
class _ReadoutTrace:
    queries: torch.Tensor
    retrieved_block_ids: torch.Tensor
    retrieved_scores: torch.Tensor
    retrieved_prefix_lengths: torch.Tensor
    coarse_retrieved_block_ids: torch.Tensor
    coarse_retrieved_scores: torch.Tensor
    coarse_retrieved_facet_scores: torch.Tensor


@dataclass
class _CentroidTrace:
    block_ids: torch.Tensor
    assignment_ids: torch.Tensor
    assignment_weights: torch.Tensor
    assignment_logits: torch.Tensor
    token_keys: torch.Tensor
    token_values: torch.Tensor
    valid_mask: torch.Tensor


@dataclass
class _PrecomputedCentroidWrite:
    block_ids: torch.Tensor
    assignment_ids: torch.Tensor
    assignment_weights: torch.Tensor
    assignment_logits: torch.Tensor
    token_keys: torch.Tensor
    token_values: torch.Tensor
    valid_mask: torch.Tensor


@dataclass
class _GlobalFeatureResponse:
    global_features: torch.Tensor
    retrieved_block_ids: torch.Tensor
    retrieved_scores: torch.Tensor
    retrieved_prefix_lengths: torch.Tensor
    coarse_retrieved_block_ids: torch.Tensor
    coarse_retrieved_scores: torch.Tensor
    coarse_retrieved_facet_scores: torch.Tensor


@dataclass
class _PreparedGlobalRequest:
    state: GfstDecodeState
    batch_idx: int
    reduced_token: torch.Tensor
    selected_projectors: torch.Tensor
    final_scores: torch.Tensor
    retrieved_block_ids_tensor: torch.Tensor
    retrieved_prefix_lengths_tensor: torch.Tensor
    empty_blocks: torch.Tensor
    empty_scores: torch.Tensor
    empty_prefixes: torch.Tensor
    coarse_block_tensor: torch.Tensor
    coarse_score_tensor: torch.Tensor
    facet_scores: torch.Tensor
    transition_global: dict[str, Any] | None


@dataclass(frozen=True)
class CentroidRecoveryControl:
    enabled: bool = False
    mode: str = "off"
    temperature: float = 1.0
    noise_scale: float = 0.0
    load_bias_scale: float = 0.0
    load_bias_clip: float = 0.0


@dataclass
class PaperFaithfulCausalLMOutputWithPast:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    past_key_values: Any = None
    hidden_states: Any = None
    attentions: Any = None
    generation_logits: torch.Tensor | None = None
    copy_decision_logits: torch.Tensor | None = None
    pointer_scores: torch.Tensor | None = None
    retrieval_queries: torch.Tensor | None = None
    retrieved_block_indices: torch.Tensor | None = None
    retrieved_block_scores: torch.Tensor | None = None
    retrieved_prefix_lengths: torch.Tensor | None = None
    coarse_retrieved_block_indices: torch.Tensor | None = None
    coarse_retrieved_block_scores: torch.Tensor | None = None
    coarse_retrieved_facet_scores: torch.Tensor | None = None
    selected_copy_token_ids: torch.Tensor | None = None
    pointer_candidate_token_ids: torch.Tensor | None = None
    pointer_candidate_blocks: torch.Tensor | None = None
    pointer_candidate_positions: torch.Tensor | None = None
    pointer_candidate_mask: torch.Tensor | None = None
    router_feature_tensor: torch.Tensor | None = None
    centroid_block_ids: torch.Tensor | None = None
    centroid_assignment_ids: torch.Tensor | None = None
    centroid_assignment_weights: torch.Tensor | None = None
    centroid_assignment_logits: torch.Tensor | None = None
    centroid_token_keys: torch.Tensor | None = None
    centroid_token_values: torch.Tensor | None = None
    centroid_valid_mask: torch.Tensor | None = None
    centroid_query_heads: torch.Tensor | None = None
    centroid_selected_block_slots: torch.Tensor | None = None
    centroid_selected_cell_ids: torch.Tensor | None = None
    centroid_selected_alphas: torch.Tensor | None = None
    centroid_selection_entropy: torch.Tensor | None = None
    centroid_selection_margin: torch.Tensor | None = None
    centroid_confidence: torch.Tensor | None = None
    centroid_selected_cells_per_head: torch.Tensor | None = None
    boundary_logits: torch.Tensor | None = None


def _tensor_rms(tensor: torch.Tensor) -> float:
    return float(torch.sqrt(tensor.detach().float().square().mean()).item())


def _detach_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous().clone()


def _detach_cpu_float32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def _clone_cpu_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _detach_cpu_tensor(value)
    if isinstance(value, dict):
        return {str(key): _clone_cpu_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu_value(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return copy.deepcopy(value)


def _summary_to_debug_dict(summary: GfstBlockSummary) -> dict[str, Any]:
    return {
        "basis": _detach_cpu_float32(summary.basis),
        "key": _detach_cpu_float32(summary.key),
        "block_index": int(summary.block_index),
        "start_pos": int(summary.start_pos),
        "end_pos": int(summary.end_pos),
        "token_count": int(summary.token_count),
        "reduced_mean": None if summary.reduced_mean is None else _detach_cpu_float32(summary.reduced_mean),
    }


def _tensor_max_abs_delta(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if tuple(lhs.shape) != tuple(rhs.shape):
        return 1.0
    if int(lhs.numel()) == 0:
        return 0.0
    lhs_f = lhs.detach().to(dtype=torch.float32)
    rhs_f = rhs.detach().to(dtype=torch.float32)
    return float((rhs_f - lhs_f).abs().max().item())


def _capture_global_feature_math_debug(
    *,
    transition_global: dict[str, Any] | None,
    reduced_token: torch.Tensor,
    full_basis: torch.Tensor,
    rank_boundaries: tuple[int, ...] | list[int],
    eps: float,
    actual_global_features: torch.Tensor,
    actual_path: str,
    prefix_projectors: dict[int, torch.Tensor] | None = None,
) -> None:
    if transition_global is None:
        return
    ordered_boundaries = tuple(int(boundary) for boundary in rank_boundaries if int(boundary) > 0)
    if prefix_projectors is None:
        prefix_projector_list = [basis_projector(full_basis[:, : int(boundary)]) for boundary in ordered_boundaries]
    else:
        prefix_projector_list = [
            prefix_projectors.get(int(boundary), basis_projector(full_basis[:, : int(boundary)]))
            for boundary in ordered_boundaries
        ]
    projector_debug = alignment_components_debug(
        reduced_token,
        prefix_projector_list,
        float(eps),
        boundaries=ordered_boundaries,
    )
    basis_native_debug = alignment_components_from_full_basis_debug(
        reduced_token.to(device=full_basis.device, dtype=full_basis.dtype),
        full_basis,
        ordered_boundaries,
        float(eps),
    )
    basis_float32_debug = alignment_components_from_full_basis_debug(
        reduced_token.to(device=full_basis.device, dtype=torch.float32),
        full_basis.to(device=full_basis.device, dtype=torch.float32),
        ordered_boundaries,
        float(eps),
    )
    transition_global["feature_actual_path"] = str(actual_path)
    transition_global["feature_reduced_token"] = _detach_cpu_tensor(reduced_token)
    transition_global["feature_projector_debug"] = _clone_cpu_value(projector_debug)
    transition_global["feature_basis_native_debug"] = _clone_cpu_value(basis_native_debug)
    transition_global["feature_basis_float32_debug"] = _clone_cpu_value(basis_float32_debug)
    transition_global["feature_actual_minus_projector_ref_max_abs"] = _tensor_max_abs_delta(
        actual_global_features,
        projector_debug["output"],
    )
    transition_global["feature_actual_minus_basis_native_ref_max_abs"] = _tensor_max_abs_delta(
        actual_global_features,
        basis_native_debug["output"],
    )
    transition_global["feature_actual_minus_basis_float32_ref_max_abs"] = _tensor_max_abs_delta(
        actual_global_features,
        basis_float32_debug["output"],
    )


_TEMP_LOCAL_QR_POOL_MARKER = "TEMP_REMOVE_AFTER_SM120_EXACT_LOCAL_QR_PARITY"
_TEMP_LOCAL_QR_POOL_LIMIT = 0
_TEMP_LOCAL_QR_POOL_RECORDS: list[dict[str, Any]] = []
_TEMP_LOCAL_QR_POOL_SEQUENCE = 0


def _temp_local_qr_pool_reset(limit: int = 0) -> None:
    global _TEMP_LOCAL_QR_POOL_LIMIT, _TEMP_LOCAL_QR_POOL_RECORDS, _TEMP_LOCAL_QR_POOL_SEQUENCE
    _TEMP_LOCAL_QR_POOL_LIMIT = max(0, int(limit))
    _TEMP_LOCAL_QR_POOL_RECORDS = []
    _TEMP_LOCAL_QR_POOL_SEQUENCE = 0


def _temp_local_qr_pool_has_capacity() -> bool:
    return int(_TEMP_LOCAL_QR_POOL_LIMIT) > 0 and len(_TEMP_LOCAL_QR_POOL_RECORDS) < int(_TEMP_LOCAL_QR_POOL_LIMIT)


def _temp_local_qr_pool_records() -> list[dict[str, Any]]:
    return copy.deepcopy(_TEMP_LOCAL_QR_POOL_RECORDS)


def _temp_local_qr_pool_append(payload: dict[str, Any]) -> None:
    global _TEMP_LOCAL_QR_POOL_SEQUENCE
    if not _temp_local_qr_pool_has_capacity():
        return
    payload_copy = copy.deepcopy(payload)
    metadata = dict(payload_copy.get("metadata", {}))
    metadata["temporary_marker"] = _TEMP_LOCAL_QR_POOL_MARKER
    metadata["capture_sequence"] = int(_TEMP_LOCAL_QR_POOL_SEQUENCE)
    payload_copy["metadata"] = metadata
    _TEMP_LOCAL_QR_POOL_RECORDS.append(payload_copy)
    _TEMP_LOCAL_QR_POOL_SEQUENCE += 1


def _topk_sparsemax_projection(logits: torch.Tensor, *, k: int, temperature: float) -> torch.Tensor:
    if int(logits.numel()) == 0:
        return torch.zeros_like(logits)
    scaled = logits.float() / max(float(temperature), 1e-6)
    k_eff = max(1, min(int(k), int(scaled.shape[-1])))
    top_values, top_indices = torch.topk(scaled, k=k_eff, dim=-1)
    sorted_values, _ = torch.sort(top_values, dim=-1, descending=True)
    cumsum = sorted_values.cumsum(dim=-1)
    support_range = torch.arange(1, k_eff + 1, device=scaled.device, dtype=scaled.dtype)
    support_range = support_range.view((1,) * (sorted_values.dim() - 1) + (k_eff,))
    support = 1.0 + support_range * sorted_values > cumsum
    support_size = support.to(dtype=torch.long).sum(dim=-1, keepdim=True).clamp_min(1)
    tau = (cumsum.gather(dim=-1, index=support_size - 1) - 1.0) / support_size.to(dtype=scaled.dtype)
    projected_top = torch.clamp(top_values - tau, min=0.0)
    projected = torch.zeros_like(scaled)
    projected.scatter_(dim=-1, index=top_indices, src=projected_top)
    projected_sum = projected.sum(dim=-1, keepdim=True)
    return torch.where(projected_sum > 0.0, projected / projected_sum.clamp_min(1e-6), projected).to(dtype=logits.dtype)


def _masked_pointer_log_probs(pointer_scores: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    mask = candidate_mask.to(device=pointer_scores.device, dtype=torch.bool)
    has_candidates = mask.any(dim=-1, keepdim=True)
    masked_scores = pointer_scores.float().masked_fill(~mask, float("-inf"))
    safe_scores = torch.where(has_candidates, masked_scores, torch.zeros_like(masked_scores))
    log_probs = F.log_softmax(safe_scores, dim=-1)
    return torch.where(mask, log_probs, torch.full_like(log_probs, float("-inf")))


class ExactRecallReadout(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        key_dim: int,
        router_hidden_dim: int = 256,
        centroid_heads: int = 4,
        centroid_key_dim: int = 32,
        centroid_value_dim: int = 16,
        cells_per_block: int = 16,
        centroid_top_cells_per_head: int = 8,
        centroid_selection_policy: str = "bh_fdr",
        centroid_fdr_q: float = 0.10,
        centroid_score_normalization: str = "robust_zscore",
        centroid_background_subtract: bool = True,
        centroid_count_penalty: float = 0.5,
        centroid_self_field_weight: float = 0.15,
        centroid_min_selected_cells_per_head: int = 4,
        centroid_selection_temperature: float = 1.0,
        centroid_gain_entropy_low: float = 0.35,
        centroid_gain_entropy_high: float = 0.75,
        centroid_gain_margin_low: float = 0.25,
        centroid_gain_margin_high: float = 1.0,
        pointer_value_source: str = "token_embedding",
        centroid_recall_value_source: str = "token_embedding",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.within_block_proj = nn.Linear(hidden_size, key_dim, bias=False, device=device, dtype=dtype)
        self.centroid_query_proj = nn.Linear(
            key_dim,
            int(centroid_heads) * int(centroid_key_dim),
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.centroid_token_proj = nn.Linear(
            hidden_size,
            int(centroid_heads) * int(centroid_value_dim),
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.eval_pointer_mode = "token_plus_centroid_recall"
        self.centroid_heads = int(centroid_heads)
        self.centroid_key_dim = int(centroid_key_dim)
        self.centroid_value_dim = int(centroid_value_dim)
        self.cells_per_block = int(cells_per_block)
        self.centroid_top_cells_per_head = int(centroid_top_cells_per_head)
        self.centroid_selection_policy = str(centroid_selection_policy)
        self.centroid_fdr_q = float(centroid_fdr_q)
        self.centroid_score_normalization = str(centroid_score_normalization)
        self.centroid_background_subtract = bool(centroid_background_subtract)
        self.centroid_count_penalty = float(centroid_count_penalty)
        self.centroid_self_field_weight = float(centroid_self_field_weight)
        self.centroid_min_selected_cells_per_head = int(centroid_min_selected_cells_per_head)
        self.centroid_selection_temperature = float(centroid_selection_temperature)
        self.centroid_gain_entropy_low = float(centroid_gain_entropy_low)
        self.centroid_gain_entropy_high = float(centroid_gain_entropy_high)
        self.centroid_gain_margin_low = float(centroid_gain_margin_low)
        self.centroid_gain_margin_high = float(centroid_gain_margin_high)
        self.router_hidden_dim = int(router_hidden_dim)
        self.pointer_value_source = str(pointer_value_source)
        self.centroid_recall_value_source = str(centroid_recall_value_source)
        self.grouped_reduction_backend = "native"
        self.segmented_reduction_backend = "native"
        self.exact_readout_score_backend = "native"
        self.runtime_profiler_mode = "off"
        self._runtime_profile_recorder: RuntimeProfileRecorder | None = None
        self.centroid_block_score_lambda = 0.10
        # Exact readout otherwise embeds full token prefixes in a single slab.
        # Chunking keeps replay/eval memory bounded without changing semantics.
        self.token_score_chunk_size = 256
        self.centroid_recall_chunk_size = 256
        self.centroid_gamma_m = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))
        self.centroid_gamma_r = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))
        self.router_query_norm = nn.LayerNorm(key_dim, elementwise_affine=True, device=device, dtype=torch.float32)
        self.router_in_proj = nn.Linear(key_dim + 8, self.router_hidden_dim, bias=True, device=device, dtype=torch.float32)
        self.router_out_proj = nn.Linear(self.router_hidden_dim, 1, bias=True, device=device, dtype=torch.float32)
        nn.init.zeros_(self.router_out_proj.weight)
        nn.init.zeros_(self.router_out_proj.bias)

    def set_eval_pointer_mode(self, mode: str) -> None:
        mode_str = str(mode)
        if mode_str != "token_plus_centroid_recall":
            raise ValueError(f"unsupported eval pointer mode: {mode_str!r}")
        self.eval_pointer_mode = mode_str

    def set_grouped_reduction_backend(self, backend: str) -> None:
        backend_str = str(backend)
        if backend_str not in {"native", "torch_scatter", "cuda_segmented"}:
            raise ValueError("grouped reduction backend must be one of {'native', 'torch_scatter', 'cuda_segmented'}")
        self.grouped_reduction_backend = backend_str
        self.segmented_reduction_backend = backend_str

    def set_segmented_reduction_backend(self, backend: str) -> None:
        self.set_grouped_reduction_backend(backend)

    def set_exact_readout_score_backend(self, backend: str) -> None:
        backend_str = str(backend)
        if backend_str not in {"native", "sm120_cutlass"}:
            raise ValueError("exact readout score backend must be one of {'native', 'sm120_cutlass'}")
        self.exact_readout_score_backend = backend_str

    def set_runtime_profiler_mode(self, mode: str) -> None:
        mode_str = str(mode)
        if mode_str not in {"off", "nvtx", "benchmark"}:
            raise ValueError("runtime profiler mode must be one of {'off', 'nvtx', 'benchmark'}")
        self.runtime_profiler_mode = mode_str

    def set_runtime_profile_recorder(self, recorder: RuntimeProfileRecorder | None) -> None:
        self._runtime_profile_recorder = recorder

    def _generation_copy_gap(self, generation_logits: torch.Tensor, copy_token_id: int) -> torch.Tensor:
        copy_logit = generation_logits[..., int(copy_token_id)].float()
        if int(generation_logits.shape[-1]) <= 1:
            best_noncopy = torch.zeros_like(copy_logit)
        else:
            noncopy_mask = torch.ones(
                int(generation_logits.shape[-1]),
                device=generation_logits.device,
                dtype=torch.bool,
            )
            noncopy_mask[int(copy_token_id)] = False
            best_noncopy = generation_logits[..., noncopy_mask].float().max(dim=-1).values
        return copy_logit - best_noncopy

    def _router_hidden(self, router_features: torch.Tensor) -> torch.Tensor:
        router_input = router_features.to(dtype=torch.float32)
        weight = self.router_in_proj.weight.to(dtype=torch.float32)
        bias = None if self.router_in_proj.bias is None else self.router_in_proj.bias.to(dtype=torch.float32)
        return F.silu(F.linear(router_input, weight, bias))

    def _router_logits(self, router_features: torch.Tensor) -> torch.Tensor:
        hidden = self._router_hidden(router_features)
        weight = self.router_out_proj.weight.to(dtype=torch.float32)
        bias = None if self.router_out_proj.bias is None else self.router_out_proj.bias.to(dtype=torch.float32)
        return F.linear(hidden.to(dtype=torch.float32), weight, bias).squeeze(-1)

    def _router_feature_tensor(
        self,
        *,
        query: torch.Tensor,
        token_scores: torch.Tensor,
        token_mask: torch.Tensor,
        retrieved_block_ids: torch.Tensor,
        retrieved_scores: torch.Tensor,
        centroid_confidence: torch.Tensor,
        generation_copy_gap: torch.Tensor,
    ) -> torch.Tensor:
        query_input = query.to(dtype=torch.float32)
        query_feature = F.layer_norm(
            query_input,
            normalized_shape=(int(query_input.shape[-1]),),
            weight=self.router_query_norm.weight.to(dtype=torch.float32),
            bias=self.router_query_norm.bias.to(dtype=torch.float32),
            eps=float(self.router_query_norm.eps),
        )
        valid_token_scores = token_scores[token_mask]
        if int(valid_token_scores.numel()) > 0:
            pointer_top1 = valid_token_scores.max()
            if int(valid_token_scores.numel()) > 1:
                top2 = torch.topk(valid_token_scores, k=2, dim=-1).values
                pointer_margin = top2[0] - top2[1]
                pointer_probs = torch.softmax(valid_token_scores, dim=-1)
                pointer_entropy = -(
                    pointer_probs * pointer_probs.clamp_min(1e-8).log()
                ).sum() / math.log(float(valid_token_scores.numel()))
            else:
                pointer_margin = torch.zeros((), device=query.device, dtype=torch.float32)
                pointer_entropy = torch.zeros((), device=query.device, dtype=torch.float32)
            candidate_count = torch.log1p(
                torch.tensor(float(valid_token_scores.numel()), device=query.device, dtype=torch.float32)
            )
        else:
            zero = torch.zeros((), device=query.device, dtype=torch.float32)
            pointer_top1 = zero
            pointer_margin = zero
            pointer_entropy = zero
            candidate_count = zero
        valid_retrieved = retrieved_block_ids.ge(0)
        if bool(valid_retrieved.any().item()):
            step_retrieved_scores = retrieved_scores[valid_retrieved].to(dtype=torch.float32)
            retrieval_max = step_retrieved_scores.max()
            retrieval_mean = step_retrieved_scores.mean()
        else:
            retrieval_max = torch.zeros((), device=query.device, dtype=torch.float32)
            retrieval_mean = torch.zeros((), device=query.device, dtype=torch.float32)
        scalar_features = torch.stack(
            (
                pointer_top1.to(dtype=torch.float32),
                pointer_margin.to(dtype=torch.float32),
                pointer_entropy.to(dtype=torch.float32),
                candidate_count.to(dtype=torch.float32),
                retrieval_max.to(dtype=torch.float32),
                retrieval_mean.to(dtype=torch.float32),
                centroid_confidence.to(dtype=torch.float32),
                generation_copy_gap.to(dtype=torch.float32),
            ),
            dim=0,
        )
        return torch.cat((query_feature, scalar_features), dim=0)

    def _record_hidden_state_tensor(
        self,
        *,
        record: ExactBlockRecord,
        prefix_length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        hidden_states = record.hidden_states
        if hidden_states is None or int(prefix_length) <= 0:
            return None
        if torch.is_tensor(hidden_states):
            hidden_tensor = hidden_states
        elif isinstance(hidden_states, (list, tuple)) and hidden_states:
            hidden_tensor = torch.stack(
                [torch.as_tensor(hidden, device=device) for hidden in hidden_states[: int(prefix_length)]],
                dim=0,
            )
        else:
            return None
        if hidden_tensor.ndim != 2:
            return None
        return hidden_tensor[: int(prefix_length)].to(device=device)

    def _score_block_tokens(
        self,
        *,
        query: torch.Tensor,
        embedding_layer: nn.Embedding,
        record: ExactBlockRecord,
        prefix_length: int,
        block_score: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with runtime_range(
            "exact_readout.score_block_tokens",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=query.device,
        ):
            token_tensor = record.token_id_tensor
            if token_tensor is None:
                token_ids = list(record.token_ids[: int(prefix_length)])
                if not token_ids:
                    empty_scores = query.new_empty((0,), dtype=torch.float32)
                    empty_long = torch.empty((0,), device=query.device, dtype=torch.long)
                    return empty_scores, empty_long, empty_long, empty_long
                token_tensor = torch.tensor(token_ids, device=query.device, dtype=torch.long)
            else:
                token_tensor = token_tensor[: int(prefix_length)].to(device=query.device, dtype=torch.long)
            if int(token_tensor.numel()) == 0:
                empty_scores = query.new_empty((0,), dtype=torch.float32)
                empty_long = torch.empty((0,), device=query.device, dtype=torch.long)
                return empty_scores, empty_long, empty_long, empty_long
            proj_dtype = self.within_block_proj.weight.dtype
            query_vector = query.to(dtype=proj_dtype)
            chunk_size = max(1, int(self.token_score_chunk_size))
            hidden_state_tensor = None
            if str(self.pointer_value_source) == "exact_hidden_state":
                hidden_state_tensor = self._record_hidden_state_tensor(
                    record=record,
                    prefix_length=int(token_tensor.shape[0]),
                    device=query.device,
                )
            token_score_chunks: list[torch.Tensor] = []
            for chunk_start in range(0, int(token_tensor.shape[0]), chunk_size):
                chunk_ids = token_tensor[chunk_start : chunk_start + chunk_size]
                if hidden_state_tensor is not None:
                    token_features = hidden_state_tensor[chunk_start : chunk_start + chunk_size].to(dtype=proj_dtype)
                else:
                    token_features = embedding_layer(chunk_ids).to(dtype=proj_dtype)
                accelerated_scores = None
                if str(self.exact_readout_score_backend) == "sm120_cutlass":
                    accelerated_scores = sm120_exact_block_score(query_vector, token_features, self.within_block_proj.weight)
                if accelerated_scores is not None:
                    token_score_chunks.append(accelerated_scores.float())
                else:
                    token_score_chunks.append((self.within_block_proj(token_features) @ query_vector).float())
            token_scores = torch.cat(token_score_chunks, dim=0)

            scores = token_scores
            block_ids = torch.full_like(token_tensor, fill_value=int(record.block_id))
            positions = torch.arange(int(token_tensor.shape[0]), device=query.device, dtype=torch.long)
            return scores, token_tensor, block_ids, positions

    def _build_centroid_candidates(
        self,
        *,
        query: torch.Tensor,
        embedding_layer: nn.Embedding,
        block_entries: list[tuple[ExactBlockRecord, int, float]],
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        dict[str, torch.Tensor],
    ] | None:
        with runtime_range(
            "exact_readout.build_centroid_candidates",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=query.device,
        ):
            if not block_entries:
                return None

            flat_keys: list[torch.Tensor] = []
            flat_values: list[torch.Tensor] = []
            flat_block_indices: list[torch.Tensor] = []
            flat_cell_indices: list[torch.Tensor] = []
            flat_block_biases: list[torch.Tensor] = []
            flat_counts: list[torch.Tensor] = []
            flat_self_fields: list[torch.Tensor] = []
            usable_entries: list[tuple[ExactBlockRecord, int, float]] = []
            for block_index, (record, prefix_length, block_score) in enumerate(block_entries):
                cell_keys = record.cell_keys
                cell_values_q = record.cell_values_q
                cell_counts = record.cell_counts
                token_cell_ids = record.token_cell_id_tensor
                token_cell_weights_q = record.token_cell_weight_q_tensor
                if cell_keys is None or cell_values_q is None or cell_counts is None or token_cell_ids is None:
                    usable_entries.append((record, prefix_length, block_score))
                    continue
                device = query.device
                counts = cell_counts.to(device=device, dtype=torch.long)
                valid_cells = counts.gt(0)
                if not bool(valid_cells.any().item()):
                    usable_entries.append((record, prefix_length, block_score))
                    continue
                valid_cell_ids = torch.nonzero(valid_cells, as_tuple=False).reshape(-1).to(dtype=torch.long)
                flat_keys.append(cell_keys.to(device=device, dtype=torch.float32).index_select(0, valid_cell_ids))
                flat_values.append(
                    cell_values_q.to(device=device, dtype=torch.float32).index_select(0, valid_cell_ids)
                    / float(_CENTROID_VALUE_QUANT_SCALE)
                )
                flat_counts.append(
                    cell_counts.to(device=device, dtype=torch.float32).index_select(0, valid_cell_ids)
                    / float(_CENTROID_COUNT_QUANT_SCALE)
                )
                local_counts = cell_counts.to(device=device, dtype=torch.float32).index_select(0, valid_cell_ids)
                local_self_field = torch.log1p(local_counts.clamp_min(0.0))
                local_self_field = local_self_field - local_self_field.mean()
                flat_self_fields.append(local_self_field)
                flat_block_indices.append(
                    torch.full((int(valid_cell_ids.numel()),), int(block_index), device=device, dtype=torch.long)
                )
                flat_cell_indices.append(valid_cell_ids)
                flat_block_biases.append(
                    torch.full((int(valid_cell_ids.numel()),), float(block_score), device=device, dtype=torch.float32)
                )
                usable_entries.append((record, prefix_length, block_score))

            if not flat_keys:
                return None

            device = query.device
            all_keys = torch.cat(flat_keys, dim=0)
            all_values = torch.cat(flat_values, dim=0)
            all_block_indices = torch.cat(flat_block_indices, dim=0)
            all_cell_indices = torch.cat(flat_cell_indices, dim=0)
            all_block_biases = torch.cat(flat_block_biases, dim=0)
            all_counts = torch.cat(flat_counts, dim=0)
            all_self_fields = torch.cat(flat_self_fields, dim=0)

            query_dtype = self.centroid_query_proj.weight.dtype
            query_heads = self.centroid_query_proj(query.to(dtype=query_dtype)).reshape(
                int(self.centroid_heads),
                int(self.centroid_key_dim),
            ).float()
            raw_scores = (query_heads @ all_keys.transpose(0, 1)) / math.sqrt(float(self.centroid_key_dim))
            raw_scores = raw_scores + float(self.centroid_block_score_lambda) * all_block_biases.unsqueeze(0)

            if bool(self.centroid_background_subtract):
                centered = raw_scores - raw_scores.median(dim=-1, keepdim=True).values
            else:
                centered = raw_scores
            mad = centered.abs().median(dim=-1, keepdim=True).values
            sigma = torch.clamp(1.4826 * mad, min=1e-4)
            standardized_scores = centered / sigma
            standardized_scores = standardized_scores - float(self.centroid_count_penalty) * torch.log1p(
                all_counts.clamp_min(0.0)
            ).unsqueeze(0)
            standardized_scores = standardized_scores + float(self.centroid_self_field_weight) * all_self_fields.unsqueeze(0)

            selection_width = int(self.centroid_top_cells_per_head)
            topk = min(selection_width, int(all_keys.shape[0]))
            if selection_width <= 0 or topk <= 0:
                return None

            activation_mass = torch.zeros(
                (len(usable_entries), int(self.cells_per_block)),
                device=device,
                dtype=torch.float32,
            )
            selected_values = torch.zeros(
                (int(self.centroid_heads), int(selection_width), int(self.centroid_value_dim)),
                device=device,
                dtype=torch.float32,
            )
            log_alpha = torch.full(
                (int(self.centroid_heads), int(selection_width)),
                float("-inf"),
                device=device,
                dtype=torch.float32,
            )
            selected_block_slots = torch.full(
                (int(self.centroid_heads), int(selection_width)),
                -1,
                device=device,
                dtype=torch.long,
            )
            selected_cell_ids = torch.full(
                (int(self.centroid_heads), int(selection_width)),
                -1,
                device=device,
                dtype=torch.long,
            )
            selected_alphas = torch.zeros(
                (int(self.centroid_heads), int(selection_width)),
                device=device,
                dtype=torch.float32,
            )
            selection_entropy = torch.zeros((int(self.centroid_heads),), device=device, dtype=torch.float32)
            selection_margin = torch.zeros((int(self.centroid_heads),), device=device, dtype=torch.float32)
            selection_significance = torch.zeros((int(self.centroid_heads),), device=device, dtype=torch.float32)
            selected_counts = torch.zeros((int(self.centroid_heads),), device=device, dtype=torch.float32)

            for head_index in range(int(self.centroid_heads)):
                head_scores = standardized_scores[head_index]
                ranked_scores, ranked_indices = torch.topk(head_scores, k=topk, dim=-1)
                head_pvalues = 0.5 * torch.erfc(head_scores / math.sqrt(2.0))
                sorted_pvalues, sorted_indices = torch.sort(head_pvalues, dim=-1)
                rank_positions = torch.arange(
                    1,
                    int(sorted_pvalues.shape[0]) + 1,
                    device=device,
                    dtype=torch.float32,
                )
                thresholds = float(self.centroid_fdr_q) * (rank_positions / float(max(1, int(sorted_pvalues.shape[0]))))
                significant = sorted_pvalues.le(thresholds)
                if bool(significant.any().item()):
                    keep_upto = int(torch.nonzero(significant, as_tuple=False).reshape(-1)[-1].item()) + 1
                    kept_indices = sorted_indices[:keep_upto]
                    kept_scores = head_scores.index_select(0, kept_indices)
                    kept_pvalues = head_pvalues.index_select(0, kept_indices)
                    reorder = torch.argsort(kept_scores, descending=True)
                    kept_indices = kept_indices.index_select(0, reorder)
                    kept_scores = kept_scores.index_select(0, reorder)
                    kept_pvalues = kept_pvalues.index_select(0, reorder)
                else:
                    kept_scores = ranked_scores.new_empty((0,))
                    kept_indices = ranked_indices.new_empty((0,), dtype=torch.long)
                    kept_pvalues = ranked_scores.new_empty((0,))
                if int(kept_indices.numel()) < int(self.centroid_min_selected_cells_per_head):
                    keep_count = min(int(self.centroid_min_selected_cells_per_head), int(ranked_indices.numel()))
                    kept_scores = ranked_scores[:keep_count]
                    kept_indices = ranked_indices[:keep_count]
                    kept_pvalues = head_pvalues.index_select(0, kept_indices)
                if int(kept_indices.numel()) == 0:
                    continue
                keep_count = int(min(int(selection_width), int(kept_indices.numel())))
                kept_scores = kept_scores[:keep_count]
                kept_indices = kept_indices[:keep_count]
                kept_pvalues = kept_pvalues[:keep_count]
                alpha = torch.softmax(kept_scores / float(self.centroid_selection_temperature), dim=-1)
                alpha = alpha[:keep_count]
                head_block_slots = all_block_indices.index_select(0, kept_indices)
                head_cell_ids = all_cell_indices.index_select(0, kept_indices)
                selected_values[head_index, :keep_count] = all_values.index_select(0, kept_indices)
                selected_block_slots[head_index, :keep_count] = head_block_slots
                selected_cell_ids[head_index, :keep_count] = head_cell_ids
                selected_alphas[head_index, :keep_count] = alpha
                log_alpha[head_index, :keep_count] = torch.log(alpha.clamp_min(1e-8))
                selected_counts[head_index] = float(keep_count)
                if keep_count > 1:
                    entropy = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum()
                    selection_entropy[head_index] = entropy / math.log(float(keep_count))
                    selection_margin[head_index] = kept_scores[0] - kept_scores[1]
                else:
                    selection_entropy[head_index] = 0.0
                    selection_margin[head_index] = 0.0
                selection_significance[head_index] = (-torch.log10(kept_pvalues.clamp_min(1e-12))).mean()
                activation_mass.index_put_(
                    (
                        head_block_slots,
                        head_cell_ids,
                    ),
                    alpha,
                    accumulate=True,
                )

            if not bool(selected_counts.gt(0).any().item()):
                return None

            mean_selection_entropy = (
                selection_entropy[selected_counts.gt(0)].mean()
                if bool(selected_counts.gt(0).any().item())
                else torch.zeros((), device=device, dtype=torch.float32)
            )
            mean_selection_margin = (
                selection_margin[selected_counts.gt(0)].mean()
                if bool(selected_counts.gt(0).any().item())
                else torch.zeros((), device=device, dtype=torch.float32)
            )
            mean_selection_significance = (
                selection_significance[selected_counts.gt(0)].mean()
                if bool(selected_counts.gt(0).any().item())
                else torch.zeros((), device=device, dtype=torch.float32)
            )
            entropy_conf = torch.clamp(
                (float(self.centroid_gain_entropy_high) - mean_selection_entropy)
                / max(1e-6, float(self.centroid_gain_entropy_high - self.centroid_gain_entropy_low)),
                min=0.0,
                max=1.0,
            )
            significance_conf = torch.clamp(
                mean_selection_significance / 3.0,
                min=0.0,
                max=1.0,
            )
            centroid_confidence = entropy_conf * significance_conf

            gamma_m = self.centroid_gamma_m.to(device=device, dtype=torch.float32)
            gamma_r = self.centroid_gamma_r.to(device=device, dtype=torch.float32)
            candidate_scores: list[torch.Tensor] = []
            candidate_token_ids: list[torch.Tensor] = []
            candidate_blocks: list[torch.Tensor] = []
            candidate_positions: list[torch.Tensor] = []
            token_proj_dtype = self.centroid_token_proj.weight.dtype
            recall_scale = math.sqrt(float(self.centroid_value_dim))
            recall_chunk_size = max(1, int(self.centroid_recall_chunk_size))

            for block_index, (record, prefix_length, block_score) in enumerate(usable_entries):
                if record.token_cell_id_tensor is None:
                    continue
                token_cell_ids = record.token_cell_id_tensor[: int(prefix_length)].to(device=device, dtype=torch.long)
                if int(token_cell_ids.numel()) == 0:
                    continue
                hidden_state_tensor = None
                if str(self.centroid_recall_value_source) == "exact_hidden_state":
                    hidden_state_tensor = self._record_hidden_state_tensor(
                        record=record,
                        prefix_length=int(prefix_length),
                        device=device,
                    )
                block_mass = activation_mass[block_index]
                safe_cell_ids = token_cell_ids.clamp(0, max(0, int(self.cells_per_block) - 1))
                gathered_mass = block_mass.index_select(0, safe_cell_ids.reshape(-1)).reshape_as(safe_cell_ids)
                if record.token_cell_weight_q_tensor is None:
                    per_token_mass = gathered_mass[..., 0] if gathered_mass.ndim > 1 else gathered_mass
                else:
                    token_cell_weights_q = record.token_cell_weight_q_tensor[: int(prefix_length)].to(
                        device=device,
                        dtype=torch.float32,
                    )
                    token_cell_weights = token_cell_weights_q / float(_CENTROID_WEIGHT_QUANT_SCALE)
                    per_token_mass = (gathered_mass.to(dtype=torch.float32) * token_cell_weights).sum(dim=-1)
                base_scores, token_ids, block_ids, positions = self._score_block_tokens(
                    query=query,
                    embedding_layer=embedding_layer,
                    record=record,
                    prefix_length=prefix_length,
                    block_score=block_score,
                )
                if int(base_scores.numel()) == 0:
                    continue
                recall_score_chunks: list[torch.Tensor] = []
                for chunk_start in range(0, int(token_ids.shape[0]), recall_chunk_size):
                    token_id_chunk = token_ids[chunk_start : chunk_start + recall_chunk_size]
                    if hidden_state_tensor is not None:
                        token_features = hidden_state_tensor[chunk_start : chunk_start + recall_chunk_size].to(
                            dtype=token_proj_dtype
                        )
                    else:
                        token_features = embedding_layer(token_id_chunk.to(dtype=torch.long)).to(dtype=token_proj_dtype)
                    centroid_features = self.centroid_token_proj(token_features).reshape(
                        int(token_id_chunk.shape[0]),
                        int(self.centroid_heads),
                        int(self.centroid_value_dim),
                    ).float()
                    seed_scores = torch.einsum("thd,hkd->thk", centroid_features, selected_values) / float(recall_scale)
                    recall_score_chunks.append(torch.logsumexp(seed_scores + log_alpha.unsqueeze(0), dim=(1, 2)))
                recall_scores = torch.cat(recall_score_chunks, dim=0)
                final_scores = base_scores + (gamma_m * centroid_confidence) * per_token_mass + (
                    gamma_r * centroid_confidence
                ) * recall_scores
                candidate_scores.append(final_scores)
                candidate_token_ids.append(token_ids)
                candidate_blocks.append(block_ids)
                candidate_positions.append(positions)

            if not candidate_scores:
                return None
            return (
                candidate_scores,
                candidate_token_ids,
                candidate_blocks,
                candidate_positions,
                {
                    "selected_block_slots": selected_block_slots,
                    "selected_cell_ids": selected_cell_ids,
                    "selected_alphas": selected_alphas,
                    "selection_entropy": selection_entropy,
                    "selection_margin": selection_margin,
                    "confidence": centroid_confidence,
                    "selected_cells_per_head": selected_counts,
                },
            )

    def forward(
        self,
        *,
        generation_logits: torch.Tensor,
        readout_trace: _ReadoutTrace | None,
        state: GfstDecodeState,
        embedding_layer: nn.Embedding,
        copy_token_id: int,
        block_len: int,
        cache_capacity: int,
    ) -> dict[str, torch.Tensor]:
        device = generation_logits.device
        batch_size, seq_len, _ = generation_logits.shape
        with runtime_range(
            "exact_readout.forward",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=device,
        ):
            generation_log_probs = F.log_softmax(generation_logits.float(), dim=-1)
        if readout_trace is None:
            selected_copy_token_ids = torch.full((batch_size, seq_len), -1, device=device, dtype=torch.long)
            empty_scores = generation_logits.new_full((batch_size, seq_len, 1), float("-inf"))
            empty_long = torch.full((batch_size, seq_len, 1), -1, device=device, dtype=torch.long)
            empty_mask = torch.zeros((batch_size, seq_len, 1), device=device, dtype=torch.bool)
            empty_facet_scores = generation_logits.new_full((batch_size, seq_len, 1, 1), float("-inf"))
            empty_selected = torch.full(
                (batch_size, seq_len, int(self.centroid_heads), int(self.centroid_top_cells_per_head)),
                -1,
                device=device,
                dtype=torch.long,
            )
            empty_alpha = torch.zeros(
                (batch_size, seq_len, int(self.centroid_heads), int(self.centroid_top_cells_per_head)),
                device=device,
                dtype=torch.float32,
            )
            empty_head_stats = torch.zeros(
                (batch_size, seq_len, int(self.centroid_heads)),
                device=device,
                dtype=torch.float32,
            )
            empty_conf = torch.zeros((batch_size, seq_len), device=device, dtype=torch.float32)
            return {
                "logits": generation_log_probs,
                "generation_logits": generation_logits,
                "copy_decision_logits": generation_logits.new_zeros((batch_size, seq_len, 2), dtype=torch.float32),
                "pointer_scores": empty_scores,
                "retrieved_block_indices": empty_long,
                "retrieved_block_scores": empty_scores,
                "retrieved_prefix_lengths": empty_long,
                "coarse_retrieved_block_indices": empty_long,
                "coarse_retrieved_block_scores": empty_scores,
                "coarse_retrieved_facet_scores": empty_facet_scores,
                "selected_copy_token_ids": selected_copy_token_ids,
                "pointer_candidate_token_ids": empty_long,
                "pointer_candidate_blocks": empty_long,
                "pointer_candidate_positions": empty_long,
                "pointer_candidate_mask": empty_mask,
                "centroid_selected_block_slots": empty_selected,
                "centroid_selected_cell_ids": empty_selected,
                "centroid_selected_alphas": empty_alpha,
                "centroid_selection_entropy": empty_head_stats,
                "centroid_selection_margin": empty_head_stats,
                "centroid_confidence": empty_conf,
                "centroid_selected_cells_per_head": empty_head_stats,
            }

        flat_pointer_scores: list[torch.Tensor] = []
        flat_candidate_token_ids: list[torch.Tensor] = []
        flat_candidate_blocks: list[torch.Tensor] = []
        flat_candidate_positions: list[torch.Tensor] = []
        flat_candidate_masks: list[torch.Tensor] = []
        selected_copy_token_ids: list[torch.Tensor] = []
        centroid_selected_block_slots: list[torch.Tensor] = []
        centroid_selected_cell_ids: list[torch.Tensor] = []
        centroid_selected_alphas: list[torch.Tensor] = []
        centroid_selection_entropy: list[torch.Tensor] = []
        centroid_selection_margin: list[torch.Tensor] = []
        centroid_confidence: list[torch.Tensor] = []
        centroid_selected_cells_per_head: list[torch.Tensor] = []
        router_features: list[torch.Tensor] = []

        for batch_idx in range(batch_size):
            for token_idx in range(seq_len):
                query = readout_trace.queries[batch_idx, token_idx]
                candidate_scores: list[torch.Tensor] = []
                candidate_token_ids: list[torch.Tensor] = []
                candidate_blocks: list[torch.Tensor] = []
                candidate_positions: list[torch.Tensor] = []
                centroid_block_entries: list[tuple[ExactBlockRecord, int, float]] = []
                step_selected_block_slots = torch.full(
                    (int(self.centroid_heads), int(self.centroid_top_cells_per_head)),
                    -1,
                    device=device,
                    dtype=torch.long,
                )
                step_selected_cell_ids = torch.full_like(step_selected_block_slots, -1)
                step_selected_alphas = torch.zeros(
                    (int(self.centroid_heads), int(self.centroid_top_cells_per_head)),
                    device=device,
                    dtype=torch.float32,
                )
                step_selection_entropy = torch.zeros((int(self.centroid_heads),), device=device, dtype=torch.float32)
                step_selection_margin = torch.zeros((int(self.centroid_heads),), device=device, dtype=torch.float32)
                step_confidence = torch.zeros((), device=device, dtype=torch.float32)
                step_selected_cells = torch.zeros((int(self.centroid_heads),), device=device, dtype=torch.float32)
                for block_id_tensor, prefix_tensor, block_score_tensor in zip(
                    readout_trace.retrieved_block_ids[batch_idx, token_idx],
                    readout_trace.retrieved_prefix_lengths[batch_idx, token_idx],
                    readout_trace.retrieved_scores[batch_idx, token_idx],
                ):
                    block_id = int(block_id_tensor.item())
                    prefix_length = int(prefix_tensor.item())
                    block_score = float(block_score_tensor.item())
                    if block_id < 0 or prefix_length <= 0:
                        continue
                    record = state.load_exact_block(
                        batch_idx=batch_idx,
                        block_id=block_id,
                        block_len=int(block_len),
                        cache_capacity=int(cache_capacity),
                    )
                    if record is None:
                        continue
                    centroid_block_entries.append((record, prefix_length, block_score))

                centroid_candidates = self._build_centroid_candidates(
                    query=query,
                    embedding_layer=embedding_layer,
                    block_entries=centroid_block_entries,
                )
                if centroid_candidates is not None:
                    (
                        candidate_scores,
                        candidate_token_ids,
                        candidate_blocks,
                        candidate_positions,
                        centroid_diag,
                    ) = centroid_candidates
                    step_selected_block_slots = centroid_diag["selected_block_slots"]
                    step_selected_cell_ids = centroid_diag["selected_cell_ids"]
                    step_selected_alphas = centroid_diag["selected_alphas"]
                    step_selection_entropy = centroid_diag["selection_entropy"]
                    step_selection_margin = centroid_diag["selection_margin"]
                    step_confidence = centroid_diag["confidence"]
                    step_selected_cells = centroid_diag["selected_cells_per_head"]
                else:
                    candidate_scores = []
                    candidate_token_ids = []
                    candidate_blocks = []
                    candidate_positions = []
                    for record, prefix_length, block_score in centroid_block_entries:
                        scores, token_ids, block_ids, positions = self._score_block_tokens(
                            query=query,
                            embedding_layer=embedding_layer,
                            record=record,
                            prefix_length=prefix_length,
                            block_score=block_score,
                        )
                        if int(scores.numel()) == 0:
                            continue
                        candidate_scores.append(scores)
                        candidate_token_ids.append(token_ids)
                        candidate_blocks.append(block_ids)
                        candidate_positions.append(positions)

                if candidate_scores:
                    flat_scores = torch.cat(candidate_scores, dim=0).float()
                    flat_token_ids = torch.cat(candidate_token_ids, dim=0)
                    flat_blocks = torch.cat(candidate_blocks, dim=0)
                    flat_positions = torch.cat(candidate_positions, dim=0)
                    unique_token_ids, grouped_scores, grouped_blocks, grouped_positions = collapse_logsumexp_scores_and_occurrence(
                        flat_scores,
                        flat_token_ids,
                        flat_blocks,
                        flat_positions,
                        backend=self.segmented_reduction_backend,
                    )
                    order = torch.argsort(grouped_scores, descending=True)
                    flat_scores = grouped_scores.index_select(0, order)
                    flat_token_ids = unique_token_ids.index_select(0, order)
                    flat_blocks = grouped_blocks.index_select(0, order)
                    flat_positions = grouped_positions.index_select(0, order)
                    flat_mask = torch.ones_like(flat_token_ids, dtype=torch.bool)
                    selected_copy_token_ids.append(flat_token_ids[0])
                else:
                    flat_scores = generation_logits.new_full((1,), float("-inf"))
                    flat_token_ids = torch.full((1,), -1, device=device, dtype=torch.long)
                    flat_blocks = torch.full((1,), -1, device=device, dtype=torch.long)
                    flat_positions = torch.full((1,), -1, device=device, dtype=torch.long)
                    flat_mask = torch.zeros((1,), device=device, dtype=torch.bool)
                    selected_copy_token_ids.append(torch.tensor(-1, device=device, dtype=torch.long))

                flat_pointer_scores.append(flat_scores)
                flat_candidate_token_ids.append(flat_token_ids)
                flat_candidate_blocks.append(flat_blocks)
                flat_candidate_positions.append(flat_positions)
                flat_candidate_masks.append(flat_mask)
                router_features.append(
                    self._router_feature_tensor(
                        query=query,
                        token_scores=flat_scores,
                        token_mask=flat_mask,
                        retrieved_block_ids=readout_trace.retrieved_block_ids[batch_idx, token_idx],
                        retrieved_scores=readout_trace.retrieved_scores[batch_idx, token_idx],
                        centroid_confidence=step_confidence,
                        generation_copy_gap=self._generation_copy_gap(
                            generation_logits[batch_idx, token_idx],
                            int(copy_token_id),
                        ),
                    )
                )
                centroid_selected_block_slots.append(step_selected_block_slots)
                centroid_selected_cell_ids.append(step_selected_cell_ids)
                centroid_selected_alphas.append(step_selected_alphas)
                centroid_selection_entropy.append(step_selection_entropy)
                centroid_selection_margin.append(step_selection_margin)
                centroid_confidence.append(step_confidence)
                centroid_selected_cells_per_head.append(step_selected_cells)

        pointer_scores = pad_sequence(flat_pointer_scores, batch_first=True, padding_value=float("-inf")).reshape(
            batch_size,
            seq_len,
            -1,
        )
        candidate_token_ids = pad_sequence(flat_candidate_token_ids, batch_first=True, padding_value=-1).reshape(
            batch_size,
            seq_len,
            -1,
        )
        candidate_blocks = pad_sequence(flat_candidate_blocks, batch_first=True, padding_value=-1).reshape(
            batch_size,
            seq_len,
            -1,
        )
        candidate_positions = pad_sequence(flat_candidate_positions, batch_first=True, padding_value=-1).reshape(
            batch_size,
            seq_len,
            -1,
        )
        candidate_mask = pad_sequence(flat_candidate_masks, batch_first=True, padding_value=False).reshape(
            batch_size,
            seq_len,
            -1,
        )
        router_feature_tensor = torch.stack(router_features, dim=0).reshape(batch_size, seq_len, -1)
        selected_copy_tensor = torch.stack(selected_copy_token_ids, dim=0).reshape(batch_size, seq_len)
        centroid_selected_block_slots_tensor = torch.stack(centroid_selected_block_slots, dim=0).reshape(
            batch_size,
            seq_len,
            int(self.centroid_heads),
            int(self.centroid_top_cells_per_head),
        )
        centroid_selected_cell_ids_tensor = torch.stack(centroid_selected_cell_ids, dim=0).reshape(
            batch_size,
            seq_len,
            int(self.centroid_heads),
            int(self.centroid_top_cells_per_head),
        )
        centroid_selected_alphas_tensor = torch.stack(centroid_selected_alphas, dim=0).reshape(
            batch_size,
            seq_len,
            int(self.centroid_heads),
            int(self.centroid_top_cells_per_head),
        )
        centroid_selection_entropy_tensor = torch.stack(centroid_selection_entropy, dim=0).reshape(
            batch_size,
            seq_len,
            int(self.centroid_heads),
        )
        centroid_selection_margin_tensor = torch.stack(centroid_selection_margin, dim=0).reshape(
            batch_size,
            seq_len,
            int(self.centroid_heads),
        )
        centroid_confidence_tensor = torch.stack(centroid_confidence, dim=0).reshape(batch_size, seq_len)
        centroid_selected_cells_per_head_tensor = torch.stack(centroid_selected_cells_per_head, dim=0).reshape(
            batch_size,
            seq_len,
            int(self.centroid_heads),
        )
        router_logits = self._router_logits(router_feature_tensor).float()
        has_candidates = candidate_mask.any(dim=-1)
        router_logits = router_logits.masked_fill(~has_candidates, 0.0)
        router_prob = torch.sigmoid(router_logits) * has_candidates.to(dtype=torch.float32)
        copy_decision_logits = torch.stack((torch.zeros_like(router_logits), router_logits), dim=-1)
        pointer_log_probs = _masked_pointer_log_probs(pointer_scores.float(), candidate_mask)
        final_logits = generation_log_probs + torch.log((1.0 - router_prob).clamp_min(1e-8)).unsqueeze(-1)
        log_gate = torch.where(
            has_candidates,
            torch.log(router_prob.clamp_min(1e-8)),
            torch.full_like(router_prob, float("-inf")),
        )
        safe_candidate_ids = candidate_token_ids.clamp_min(0)
        copy_token_log_probs = (pointer_log_probs + log_gate.unsqueeze(-1)).masked_fill(
            ~candidate_mask,
            float("-inf"),
        )
        copy_logits = torch.full_like(final_logits, float("-inf")).scatter_reduce(
            dim=-1,
            index=safe_candidate_ids,
            src=copy_token_log_probs,
            reduce="amax",
            include_self=True,
        )
        final_logits = torch.logaddexp(final_logits, copy_logits)

        return {
            "logits": final_logits,
            "generation_logits": generation_logits,
            "copy_decision_logits": copy_decision_logits,
            "pointer_scores": pointer_scores,
            "router_feature_tensor": router_feature_tensor,
            "retrieved_block_indices": readout_trace.retrieved_block_ids,
            "retrieved_block_scores": readout_trace.retrieved_scores,
            "retrieved_prefix_lengths": readout_trace.retrieved_prefix_lengths,
            "coarse_retrieved_block_indices": readout_trace.coarse_retrieved_block_ids,
            "coarse_retrieved_block_scores": readout_trace.coarse_retrieved_scores,
            "coarse_retrieved_facet_scores": readout_trace.coarse_retrieved_facet_scores,
            "selected_copy_token_ids": selected_copy_tensor,
            "pointer_candidate_token_ids": candidate_token_ids,
            "pointer_candidate_blocks": candidate_blocks,
            "pointer_candidate_positions": candidate_positions,
            "pointer_candidate_mask": candidate_mask,
            "centroid_selected_block_slots": centroid_selected_block_slots_tensor,
            "centroid_selected_cell_ids": centroid_selected_cell_ids_tensor,
            "centroid_selected_alphas": centroid_selected_alphas_tensor,
            "centroid_selection_entropy": centroid_selection_entropy_tensor,
            "centroid_selection_margin": centroid_selection_margin_tensor,
            "centroid_confidence": centroid_confidence_tensor,
            "centroid_selected_cells_per_head": centroid_selected_cells_per_head_tensor,
        }


class GfstHmbSublayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        layer_idx: int,
        gfst_config: GfstHmbConfig,
        rope_theta: float = 10000.0,
        retrieval_config: FaissGpuRetrievalConfig | None = None,
        retriever: FaissGpuRetriever | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.layer_idx = int(layer_idx)
        self.gfst_config = gfst_config
        self.rope_theta = float(rope_theta)
        self.retrieval_config = retrieval_config
        self.retriever = retriever

        self.reduce_proj = nn.Linear(self.hidden_size, gfst_config.reduced_dim, bias=False)
        self.block_key_proj = nn.Linear(gfst_config.reduced_dim * 2, gfst_config.key_dim, bias=False)
        self.query_proj = nn.Linear(gfst_config.reduced_dim, gfst_config.key_dim, bias=False)
        self.cell_assign_proj = nn.Linear(self.hidden_size, gfst_config.cells_per_block, bias=False)
        self.cell_key_proj = nn.Linear(self.hidden_size, gfst_config.centroid_key_dim, bias=False)
        self.cell_value_proj = nn.Linear(self.hidden_size, gfst_config.centroid_value_dim, bias=False)
        self.centroid_phase_proj = nn.Linear(max(1, int(gfst_config.centroid_phase_bands)) * 2, self.hidden_size, bias=False)
        self.feature_gate = nn.Linear(gfst_config.feature_dim, self.hidden_size)
        self.feature_value = nn.Linear(gfst_config.feature_dim, self.hidden_size)
        self.training_path_mode = "fast"
        self.disable_residual = False
        self.gfst_feature_backend = "native"
        self.eval_router_mode = "baseline"
        self.eval_router_shortlist_depth: int | None = None
        self.eval_router_facet_count = 4
        self.eval_router_facet_aggregation = "max"
        self.force_store_exact_hidden_states = False
        self.force_centroid_activation = False
        self.centroid_recovery_control: CentroidRecoveryControl | None = None
        self.debug_capture_features = False
        self.runtime_profiler_mode = "off"
        self._runtime_profile_recorder: RuntimeProfileRecorder | None = None
        self.register_buffer(
            "centroid_usage_ema",
            torch.full(
                (int(gfst_config.cells_per_block),),
                1.0 / float(max(1, int(gfst_config.cells_per_block))),
                dtype=torch.float32,
            ),
        )
        self._last_feature_trace: torch.Tensor | None = None
        self._last_readout_trace: _ReadoutTrace | None = None
        self._last_centroid_trace: _CentroidTrace | None = None
        self._last_forward_stats: dict[str, float] | None = None
        self._local_qr_capture_target: tuple[int, int] | None = None
        self._pending_local_qr_capture: dict[str, Any] | None = None
        self._last_local_qr_capture: dict[str, Any] | None = None
        self._transition_probe_targets: set[tuple[int, int]] = set()
        self._last_transition_probe_captures: dict[tuple[int, int], dict[str, Any]] = {}
        self._active_transition_probe_capture: dict[str, Any] | None = None
        self._active_transition_probe_key: tuple[int, int] | None = None
        self._current_transition_token_context: dict[str, Any] | None = None

    def set_debug_capture(self, enabled: bool) -> None:
        self.debug_capture_features = bool(enabled)
        if not self.debug_capture_features:
            self._last_feature_trace = None
            self._pending_local_qr_capture = None
            self._last_local_qr_capture = None
            self._transition_probe_targets = set()
            self._last_transition_probe_captures = {}
            self._active_transition_probe_capture = None
            self._active_transition_probe_key = None
            self._current_transition_token_context = None

    @property
    def last_feature_trace(self) -> torch.Tensor | None:
        return self._last_feature_trace

    @property
    def last_readout_trace(self) -> _ReadoutTrace | None:
        return self._last_readout_trace

    @property
    def last_centroid_trace(self) -> _CentroidTrace | None:
        return self._last_centroid_trace

    @property
    def last_forward_stats(self) -> dict[str, float] | None:
        return None if self._last_forward_stats is None else dict(self._last_forward_stats)

    def set_disable_residual(self, enabled: bool) -> None:
        self.disable_residual = bool(enabled)

    def set_gfst_feature_backend(self, backend: str) -> None:
        backend_str = str(backend)
        if backend_str not in {"native", "sm120_exact"}:
            raise ValueError("gfst feature backend must be one of {'native', 'sm120_exact'}")
        self.gfst_feature_backend = backend_str

    def set_eval_router_mode(
        self,
        mode: str,
        *,
        shortlist_depth: int | None = None,
        facet_count: int = 4,
        facet_aggregation: str = "max",
    ) -> None:
        mode_str = str(mode)
        allowed = {"baseline", "multifacet_rerank", "multifacet_rerank_plus_score_bias"}
        if mode_str not in allowed:
            raise ValueError(f"unsupported eval router mode: {mode_str!r}")
        facet_mode = str(facet_aggregation)
        if facet_mode not in {"max", "logsumexp"}:
            raise ValueError(f"unsupported eval router facet aggregation: {facet_mode!r}")
        self.eval_router_mode = mode_str
        self.eval_router_shortlist_depth = None if shortlist_depth is None else max(1, int(shortlist_depth))
        self.eval_router_facet_count = max(1, int(facet_count))
        self.eval_router_facet_aggregation = facet_mode

    def set_force_store_exact_hidden_states(self, enabled: bool) -> None:
        self.force_store_exact_hidden_states = bool(enabled)

    def set_force_centroid_activation(self, enabled: bool) -> None:
        self.force_centroid_activation = bool(enabled)

    def set_centroid_recovery_control(self, control: CentroidRecoveryControl | None) -> None:
        self.centroid_recovery_control = control

    def set_runtime_profiler_mode(self, mode: str) -> None:
        mode_str = str(mode)
        if mode_str not in {"off", "nvtx", "benchmark"}:
            raise ValueError("runtime profiler mode must be one of {'off', 'nvtx', 'benchmark'}")
        self.runtime_profiler_mode = mode_str

    def set_runtime_profile_recorder(self, recorder: RuntimeProfileRecorder | None) -> None:
        self._runtime_profile_recorder = recorder

    def _set_local_qr_capture_target(self, *, batch_idx: int, token_idx: int) -> None:
        self._local_qr_capture_target = (int(batch_idx), int(token_idx))
        self._pending_local_qr_capture = None
        self._last_local_qr_capture = None

    def _clear_local_qr_capture_target(self) -> None:
        self._local_qr_capture_target = None
        self._pending_local_qr_capture = None
        self._last_local_qr_capture = None

    def _get_local_qr_capture(self) -> dict[str, Any] | None:
        return self._last_local_qr_capture

    def _set_transition_probe_targets(self, *, targets: list[tuple[int, int]]) -> None:
        self._transition_probe_targets = {(int(batch_idx), int(token_idx)) for batch_idx, token_idx in targets}
        self._last_transition_probe_captures = {}
        self._active_transition_probe_capture = None
        self._active_transition_probe_key = None
        self._current_transition_token_context = None

    def _clear_transition_probe_targets(self) -> None:
        self._transition_probe_targets = set()
        self._last_transition_probe_captures = {}
        self._active_transition_probe_capture = None
        self._active_transition_probe_key = None
        self._current_transition_token_context = None

    def _get_transition_probe_captures(self) -> dict[tuple[int, int], dict[str, Any]]:
        return {key: copy.deepcopy(value) for key, value in self._last_transition_probe_captures.items()}

    def _should_capture_transition_probe(self, *, batch_idx: int | None, token_idx: int | None) -> bool:
        if batch_idx is None or token_idx is None:
            return False
        key = (int(batch_idx), int(token_idx))
        return key in self._transition_probe_targets and key not in self._last_transition_probe_captures

    def _stash_transition_token_context(
        self,
        *,
        batch_idx: int | None,
        token_idx: int | None,
        absolute_position: int,
        hidden_token: torch.Tensor,
        reduced_token: torch.Tensor,
        query_token: torch.Tensor,
    ) -> None:
        if batch_idx is None or token_idx is None:
            self._current_transition_token_context = None
            return
        self._current_transition_token_context = {
            "batch_idx": int(batch_idx),
            "token_idx": int(token_idx),
            "absolute_position": int(absolute_position),
            "hidden_token": hidden_token,
            "reduced_token": reduced_token,
            "query_token": query_token,
        }

    def _begin_transition_probe_capture(
        self,
        *,
        batch_idx: int,
        token_idx: int,
        absolute_position: int,
        hidden_token: torch.Tensor,
        reduced_token: torch.Tensor,
        query_token: torch.Tensor,
    ) -> dict[str, Any]:
        capture = {
            "metadata": {
                "layer_idx": int(self.layer_idx),
                "batch_idx": int(batch_idx),
                "token_idx": int(token_idx),
                "absolute_position": int(absolute_position),
                "flag_dims": [int(dim) for dim in self.gfst_config.flag_dims],
                "max_flag_dim": int(self.gfst_config.max_flag_dim),
                "eps": float(self.gfst_config.eps),
                "eps_sign": float(self.gfst_config.eps_sign),
                "eps_update": float(self.gfst_config.eps_update),
                "gfst_update_rule": str(self.gfst_config.gfst_update_rule),
                "forgetting_rate": float(self.gfst_config.forgetting_rates[-1]),
                "retrieval_temperature": float(self.gfst_config.retrieval_temperature),
                "block_len": int(self.gfst_config.block_len),
            },
            "pre_token_inputs": {
                "hidden_token_in": _detach_cpu_float32(hidden_token),
                "reduced_token": _detach_cpu_float32(reduced_token),
                "query_token": _detach_cpu_float32(query_token),
                "absolute_position": int(absolute_position),
            },
            "local_path": None,
            "global_path": None,
            "block_memory": None,
            "post_feature": None,
        }
        key = (int(batch_idx), int(token_idx))
        self._active_transition_probe_capture = capture
        self._active_transition_probe_key = key
        return capture

    def _finish_transition_probe_capture(self) -> None:
        if self._active_transition_probe_capture is None or self._active_transition_probe_key is None:
            return
        self._last_transition_probe_captures[self._active_transition_probe_key] = copy.deepcopy(
            self._active_transition_probe_capture
        )
        self._active_transition_probe_capture = None
        self._active_transition_probe_key = None

    def _should_record_local_qr_pool(self) -> bool:
        return str(self.gfst_feature_backend) == "native" and _temp_local_qr_pool_has_capacity()

    def _should_capture_local_qr(self, *, batch_idx: int | None, token_idx: int | None) -> bool:
        if batch_idx is None or token_idx is None:
            return False
        if self._local_qr_capture_target is None:
            return False
        if self._last_local_qr_capture is not None or self._pending_local_qr_capture is not None:
            return False
        return self._local_qr_capture_target == (int(batch_idx), int(token_idx))

    def _finalize_local_qr_capture(self, local_features: torch.Tensor) -> None:
        if self._pending_local_qr_capture is None:
            return
        capture_payload = self._pending_local_qr_capture
        capture_payload["local_features"] = _detach_cpu_float32(local_features)
        record_targeted = bool(capture_payload.pop("_record_targeted", False))
        record_pool = bool(capture_payload.pop("_record_pool", False))
        if record_targeted:
            self._last_local_qr_capture = copy.deepcopy(capture_payload)
        if record_pool:
            _temp_local_qr_pool_append(capture_payload)
        self._pending_local_qr_capture = None

    def _use_sm120_exact_math_kernels(self) -> bool:
        return str(self.gfst_feature_backend) == "sm120_exact" and sm120_exact_math_kernels_available()

    def _centroid_recovery_bias(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ema = getattr(self, "centroid_usage_ema", None)
        if ema is None:
            return torch.zeros((int(self.gfst_config.cells_per_block),), device=device, dtype=dtype)
        usage = ema.to(device=device, dtype=dtype)
        total = usage.sum()
        if float(total.item()) <= 0.0:
            return torch.zeros_like(usage)
        usage = usage / total.clamp_min(1e-6)
        uniform = torch.full_like(usage, 1.0 / float(max(1, int(usage.numel()))))
        control = self.centroid_recovery_control
        if control is None:
            return torch.zeros_like(usage)
        bias = float(control.load_bias_scale) * (uniform - usage)
        clip = float(control.load_bias_clip)
        if clip > 0.0:
            bias = torch.clamp(bias, min=-clip, max=clip)
        return bias

    def _apply_centroid_recovery_control(self, assign_logits: torch.Tensor) -> torch.Tensor:
        control = self.centroid_recovery_control
        if control is None or not bool(control.enabled) or str(control.mode) == "off" or not self.training:
            return assign_logits
        adjusted = assign_logits.float() / max(float(control.temperature), 1e-6)
        adjusted = adjusted + self._centroid_recovery_bias(device=adjusted.device, dtype=adjusted.dtype)
        noise_scale = float(control.noise_scale)
        if noise_scale > 0.0:
            uniforms = torch.rand_like(adjusted).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniforms))
            adjusted = adjusted + noise_scale * gumbel
        return adjusted

    def _record_feature_trace(self, feature_batches: list[torch.Tensor] | None) -> None:
        if feature_batches is None:
            self._last_feature_trace = None
            return
        self._last_feature_trace = torch.stack(feature_batches, dim=0)

    def _record_readout_trace(
        self,
        *,
        query_batches: list[torch.Tensor],
        retrieved_block_batches: list[torch.Tensor],
        retrieved_score_batches: list[torch.Tensor],
        retrieved_prefix_batches: list[torch.Tensor],
        coarse_retrieved_block_batches: list[torch.Tensor],
        coarse_retrieved_score_batches: list[torch.Tensor],
        coarse_retrieved_facet_batches: list[torch.Tensor],
    ) -> None:
        self._last_readout_trace = _ReadoutTrace(
            queries=torch.stack(query_batches, dim=0),
            retrieved_block_ids=torch.stack(retrieved_block_batches, dim=0),
            retrieved_scores=torch.stack(retrieved_score_batches, dim=0),
            retrieved_prefix_lengths=torch.stack(retrieved_prefix_batches, dim=0),
            coarse_retrieved_block_ids=torch.stack(coarse_retrieved_block_batches, dim=0),
            coarse_retrieved_scores=torch.stack(coarse_retrieved_score_batches, dim=0),
            coarse_retrieved_facet_scores=torch.stack(coarse_retrieved_facet_batches, dim=0),
        )

    def _record_centroid_trace(
        self,
        *,
        block_id_batches: list[torch.Tensor],
        assignment_id_batches: list[torch.Tensor],
        assignment_weight_batches: list[torch.Tensor],
        assignment_logit_batches: list[torch.Tensor],
        token_key_batches: list[torch.Tensor],
        token_value_batches: list[torch.Tensor],
        valid_mask_batches: list[torch.Tensor],
    ) -> None:
        self._last_centroid_trace = _CentroidTrace(
            block_ids=torch.stack(block_id_batches, dim=0),
            assignment_ids=torch.stack(assignment_id_batches, dim=0),
            assignment_weights=torch.stack(assignment_weight_batches, dim=0),
            assignment_logits=torch.stack(assignment_logit_batches, dim=0),
            token_keys=torch.stack(token_key_batches, dim=0),
            token_values=torch.stack(token_value_batches, dim=0),
            valid_mask=torch.stack(valid_mask_batches, dim=0),
        )

    def _centroid_phase_features(self, *, local_offset: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        period = max(1.0, float(self.gfst_config.centroid_phase_period))
        bands = torch.arange(int(self.gfst_config.centroid_phase_bands), device=device, dtype=torch.float32)
        angles = (2.0 * math.pi * float(local_offset) / period) * torch.pow(2.0, bands)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=0).to(dtype=dtype)

    def _apply_centroid_phase_rotation(self, vector: torch.Tensor, *, local_offset: int) -> torch.Tensor:
        if not bool(self.gfst_config.centroid_phase_binding_enabled):
            return vector
        flat = vector.reshape(-1)
        pair_dim = int(flat.shape[0]) // 2
        if pair_dim <= 0:
            return vector
        even_dim = pair_dim * 2
        base = flat[:even_dim].reshape(pair_dim, 2)
        pair_positions = torch.arange(pair_dim, device=flat.device, dtype=torch.float32)
        period = max(1.0, float(self.gfst_config.centroid_phase_period))
        angles = (2.0 * math.pi * float(local_offset) / period) * torch.pow(
            2.0,
            pair_positions / max(1.0, float(pair_dim)),
        )
        cos = torch.cos(angles).to(dtype=base.dtype)
        sin = torch.sin(angles).to(dtype=base.dtype)
        rotated = torch.stack(
            (
                base[:, 0] * cos - base[:, 1] * sin,
                base[:, 0] * sin + base[:, 1] * cos,
            ),
            dim=-1,
        ).reshape(even_dim)
        if even_dim == int(flat.shape[0]):
            return rotated.reshape_as(vector)
        return torch.cat((rotated, flat[even_dim:]), dim=0).reshape_as(vector)

    def _centroid_phase_features_batch(
        self,
        *,
        local_offsets: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        period = max(1.0, float(self.gfst_config.centroid_phase_period))
        offsets = local_offsets.to(device=device, dtype=torch.float32).reshape(-1, 1)
        bands = torch.arange(int(self.gfst_config.centroid_phase_bands), device=device, dtype=torch.float32).reshape(1, -1)
        angles = (2.0 * math.pi * offsets / period) * torch.pow(2.0, bands)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1).to(dtype=dtype)

    def _apply_centroid_phase_rotation_batch(
        self,
        vectors: torch.Tensor,
        *,
        local_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.gfst_config.centroid_phase_binding_enabled):
            return vectors
        flat = vectors.reshape(-1, int(vectors.shape[-1]))
        pair_dim = int(flat.shape[-1]) // 2
        if pair_dim <= 0:
            return vectors
        even_dim = pair_dim * 2
        base = flat[:, :even_dim].reshape(int(flat.shape[0]), pair_dim, 2)
        pair_positions = torch.arange(pair_dim, device=flat.device, dtype=torch.float32)
        period = max(1.0, float(self.gfst_config.centroid_phase_period))
        offsets = local_offsets.to(device=flat.device, dtype=torch.float32).reshape(-1, 1)
        angles = (2.0 * math.pi * offsets / period) * torch.pow(2.0, pair_positions / max(1.0, float(pair_dim)))
        cos = torch.cos(angles).to(dtype=base.dtype)
        sin = torch.sin(angles).to(dtype=base.dtype)
        rotated = torch.stack(
            (
                base[..., 0] * cos - base[..., 1] * sin,
                base[..., 0] * sin + base[..., 1] * cos,
            ),
            dim=-1,
        ).reshape(int(flat.shape[0]), even_dim)
        if even_dim == int(flat.shape[-1]):
            return rotated.reshape_as(vectors)
        return torch.cat((rotated, flat[:, even_dim:]), dim=-1).reshape_as(vectors)

    def _centroid_write_input(self, *, hidden_token: torch.Tensor, absolute_position: int) -> torch.Tensor:
        centroid_input = hidden_token.to(dtype=self.cell_assign_proj.weight.dtype)
        if not bool(self.gfst_config.centroid_phase_binding_enabled):
            return centroid_input
        local_offset = int(absolute_position) % int(self.gfst_config.block_len)
        phase_features = self._centroid_phase_features(
            local_offset=local_offset,
            device=hidden_token.device,
            dtype=self.centroid_phase_proj.weight.dtype,
        )
        phase_delta = self.centroid_phase_proj(phase_features).to(dtype=centroid_input.dtype)
        return centroid_input + phase_delta

    def _project_centroid_assignments(self, assign_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assignment_top_k = max(
            1,
            min(int(self.gfst_config.centroid_assignment_top_k), int(self.gfst_config.cells_per_block)),
        )
        _, top_ids = torch.topk(assign_logits, k=assignment_top_k, dim=-1)
        full_weights = _topk_sparsemax_projection(
            assign_logits,
            k=assignment_top_k,
            temperature=float(self.gfst_config.centroid_assignment_temperature),
        )
        top_weights = full_weights.gather(dim=-1, index=top_ids).to(dtype=torch.float32)
        if float(top_weights.sum().item()) <= 0.0:
            top_weights = torch.zeros_like(top_weights, dtype=torch.float32)
            top_weights[..., 0] = 1.0
        return top_ids.to(dtype=torch.long), top_weights

    def _precompute_centroid_write_payloads(
        self,
        *,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
    ) -> _PrecomputedCentroidWrite | None:
        centroid_activation_enabled = bool(self.gfst_config.centroid_activation_enabled) or bool(self.force_centroid_activation)
        if int(self.layer_idx) != 0 or not centroid_activation_enabled:
            return None
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch_size * seq_len, hidden_size)
        flat_positions = positions.reshape(batch_size * seq_len).to(device=hidden_states.device, dtype=torch.long)
        centroid_input = flat_hidden.to(dtype=self.cell_assign_proj.weight.dtype)
        local_offsets = torch.remainder(flat_positions, int(self.gfst_config.block_len))
        if bool(self.gfst_config.centroid_phase_binding_enabled):
            phase_features = self._centroid_phase_features_batch(
                local_offsets=local_offsets,
                device=hidden_states.device,
                dtype=self.centroid_phase_proj.weight.dtype,
            )
            phase_delta = self.centroid_phase_proj(phase_features).to(dtype=centroid_input.dtype)
            centroid_input = centroid_input + phase_delta
        assign_logits = self.cell_assign_proj(centroid_input).float()
        projected_assignment_logits = self._apply_centroid_recovery_control(assign_logits)
        assignment_ids, assignment_weights = self._project_centroid_assignments(projected_assignment_logits)
        token_keys = self.cell_key_proj(centroid_input).float()
        token_values = self.cell_value_proj(centroid_input).float()
        if bool(self.gfst_config.centroid_phase_binding_enabled):
            token_keys = self._apply_centroid_phase_rotation_batch(token_keys, local_offsets=local_offsets).float()
            token_values = self._apply_centroid_phase_rotation_batch(token_values, local_offsets=local_offsets).float()
        return _PrecomputedCentroidWrite(
            block_ids=(flat_positions // int(self.gfst_config.block_len)).reshape(batch_size, seq_len),
            assignment_ids=assignment_ids.reshape(batch_size, seq_len, -1),
            assignment_weights=assignment_weights.reshape(batch_size, seq_len, -1),
            assignment_logits=assign_logits.reshape(batch_size, seq_len, -1),
            token_keys=token_keys.reshape(batch_size, seq_len, -1),
            token_values=token_values.reshape(batch_size, seq_len, -1),
            valid_mask=torch.ones((batch_size, seq_len), device=hidden_states.device, dtype=torch.bool),
        )

    def _prefill_forward_fast(
        self,
        *,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        past_key_values: GfstDecodeState,
    ) -> tuple[torch.Tensor, None]:
        with runtime_range(
            "gfst.tokenwise_forward",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=hidden_states.device,
        ):
            batch_size, seq_len, _ = hidden_states.shape
            reduced_input = hidden_states.to(dtype=self.reduce_proj.weight.dtype)
            reduced_tokens = apply_rope(self.reduce_proj(reduced_input), positions, theta=self.rope_theta)
            query_tokens = self.query_proj(reduced_tokens.to(dtype=self.query_proj.weight.dtype)).float()
            precomputed_centroid = self._precompute_centroid_write_payloads(
                hidden_states=hidden_states,
                positions=positions,
            )
            positions_host = positions.detach().to(device="cpu", dtype=torch.long)
            layer_states = [past_key_values.layer_state(self.layer_idx, batch_idx) for batch_idx in range(batch_size)]

            padded_size = int(self.gfst_config.retrieval_depth) + 1
            coarse_padded_size = max(
                int(self.gfst_config.retrieval_depth),
                int(self.eval_router_shortlist_depth or int(self.gfst_config.retrieval_depth)),
            )
            facet_count = max(1, int(self.eval_router_facet_count))
            assignment_top_k = max(
                1,
                min(int(self.gfst_config.centroid_assignment_top_k), int(self.gfst_config.cells_per_block)),
            )

            outputs = torch.empty_like(hidden_states)
            feature_trace = (
                torch.empty(
                    (batch_size, seq_len, int(self.gfst_config.feature_dim)),
                    device="cpu",
                    dtype=torch.float32,
                )
                if self.debug_capture_features
                else None
            )
            retrieved_block_ids = torch.empty((batch_size, seq_len, padded_size), device=hidden_states.device, dtype=torch.long)
            retrieved_scores = torch.empty((batch_size, seq_len, padded_size), device=hidden_states.device, dtype=torch.float32)
            retrieved_prefix_lengths = torch.empty(
                (batch_size, seq_len, padded_size),
                device=hidden_states.device,
                dtype=torch.long,
            )
            coarse_retrieved_block_ids = torch.empty(
                (batch_size, seq_len, coarse_padded_size),
                device=hidden_states.device,
                dtype=torch.long,
            )
            coarse_retrieved_scores = torch.empty(
                (batch_size, seq_len, coarse_padded_size),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            coarse_retrieved_facet_scores = torch.empty(
                (batch_size, seq_len, coarse_padded_size, facet_count),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            centroid_block_ids = torch.full((batch_size, seq_len), -1, device=hidden_states.device, dtype=torch.long)
            centroid_assignment_ids = torch.full(
                (batch_size, seq_len, assignment_top_k),
                -1,
                device=hidden_states.device,
                dtype=torch.long,
            )
            centroid_assignment_weights = torch.zeros(
                (batch_size, seq_len, assignment_top_k),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            centroid_assignment_logits = torch.zeros(
                (batch_size, seq_len, int(self.gfst_config.cells_per_block)),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            centroid_token_keys = torch.zeros(
                (batch_size, seq_len, int(self.gfst_config.centroid_key_dim)),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            centroid_token_values = torch.zeros(
                (batch_size, seq_len, int(self.gfst_config.centroid_value_dim)),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            centroid_valid_mask = torch.zeros((batch_size, seq_len), device=hidden_states.device, dtype=torch.bool)
            if int(self.layer_idx) == 0:
                centroid_block_ids.copy_(torch.div(positions, int(self.gfst_config.block_len), rounding_mode="floor"))
                if precomputed_centroid is not None:
                    centroid_block_ids.copy_(precomputed_centroid.block_ids.to(device=hidden_states.device, dtype=torch.long))
                    centroid_assignment_ids.copy_(
                        precomputed_centroid.assignment_ids.to(device=hidden_states.device, dtype=torch.long)
                    )
                    centroid_assignment_weights.copy_(
                        precomputed_centroid.assignment_weights.to(device=hidden_states.device, dtype=torch.float32)
                    )
                    centroid_assignment_logits.copy_(
                        precomputed_centroid.assignment_logits.to(device=hidden_states.device, dtype=torch.float32)
                    )
                    centroid_token_keys.copy_(precomputed_centroid.token_keys.to(device=hidden_states.device, dtype=torch.float32))
                    centroid_token_values.copy_(
                        precomputed_centroid.token_values.to(device=hidden_states.device, dtype=torch.float32)
                    )
                    centroid_valid_mask.copy_(precomputed_centroid.valid_mask.to(device=hidden_states.device, dtype=torch.bool))

            gate_mean_sum = 0.0
            gate_std_sum = 0.0
            delta_rms_sum = 0.0
            stat_count = 0

            for token_idx in range(seq_len):
                batchable_rows: list[int] = []
                batchable_local_features: dict[int, torch.Tensor] = {}
                prepared_rows: list[int] = []
                prepared_requests: list[_PreparedGlobalRequest] = []
                early_responses: dict[int, _GlobalFeatureResponse] = {}
                batch_exact_enabled = self._use_sm120_exact_math_kernels()
                for batch_idx in range(batch_size):
                    hidden_token = hidden_states[batch_idx, token_idx]
                    reduced_token = reduced_tokens[batch_idx, token_idx]
                    query_token = query_tokens[batch_idx, token_idx]
                    absolute_position = int(positions_host[batch_idx, token_idx])
                    should_probe = self._should_capture_transition_probe(batch_idx=batch_idx, token_idx=token_idx)
                    if should_probe or not batch_exact_enabled:
                        token_result = self._token_step(
                            state=past_key_values,
                            batch_idx=batch_idx,
                            hidden_token=hidden_token,
                            reduced_token=reduced_token,
                            query_token=query_token,
                            layer_state=layer_states[batch_idx],
                            absolute_position=absolute_position,
                            precomputed_centroid=precomputed_centroid,
                            token_idx=token_idx,
                        )
                        outputs[batch_idx, token_idx] = token_result.delta
                        retrieved_block_ids[batch_idx, token_idx] = token_result.retrieved_block_ids
                        retrieved_scores[batch_idx, token_idx] = token_result.retrieved_scores
                        retrieved_prefix_lengths[batch_idx, token_idx] = token_result.retrieved_prefix_lengths
                        coarse_retrieved_block_ids[batch_idx, token_idx] = token_result.coarse_retrieved_block_ids
                        coarse_retrieved_scores[batch_idx, token_idx] = token_result.coarse_retrieved_scores
                        coarse_retrieved_facet_scores[batch_idx, token_idx] = token_result.coarse_retrieved_facet_scores
                        if feature_trace is not None:
                            feature_trace[batch_idx, token_idx].copy_(
                                token_result.features.detach().to(device="cpu", dtype=torch.float32)
                            )
                        gate_mean_sum += float(token_result.gate_mean)
                        gate_std_sum += float(token_result.gate_std)
                        delta_rms_sum += float(token_result.delta_rms)
                        stat_count += 1
                        continue
                    batchable_rows.append(batch_idx)
                    if int(self.layer_idx) == 0:
                        token_id = past_key_values.consume_prepared_token(batch_idx, absolute_position)
                        cell_ids = None
                        cell_weights = None
                        cell_key = None
                        cell_value = None
                        if precomputed_centroid is not None:
                            cell_ids = centroid_assignment_ids[batch_idx, token_idx]
                            cell_weights = centroid_assignment_weights[batch_idx, token_idx]
                            cell_key = centroid_token_keys[batch_idx, token_idx]
                            cell_value = centroid_token_values[batch_idx, token_idx]
                        past_key_values.append_exact_token(
                            batch_idx=batch_idx,
                            absolute_position=absolute_position,
                            token_id=token_id,
                            block_len=self.gfst_config.block_len,
                            hidden_state=hidden_token
                            if bool(self.gfst_config.store_exact_hidden_states) or bool(self.force_store_exact_hidden_states)
                            else None,
                            exact_hidden_store_mode=str(getattr(self.gfst_config, "exact_hidden_store_mode", "tuple") or "tuple"),
                            cells_per_block=int(self.gfst_config.cells_per_block),
                            centroid_key_dim=int(self.gfst_config.centroid_key_dim),
                            centroid_value_dim=int(self.gfst_config.centroid_value_dim),
                            cell_ids=None if cell_ids is None else cell_ids.detach(),
                            cell_weights=None if cell_weights is None else cell_weights.detach(),
                            cell_key=cell_key,
                            cell_value=cell_value,
                        )
                    with runtime_range(
                        "gfst.token_features",
                        mode=self.runtime_profiler_mode,
                        recorder=self._runtime_profile_recorder,
                        device=reduced_token.device,
                    ):
                        with runtime_range(
                            "gfst.update_local_projectors",
                            mode=self.runtime_profiler_mode,
                            recorder=self._runtime_profile_recorder,
                            device=reduced_token.device,
                        ):
                            _ = self._update_local_bases(
                                layer_states[batch_idx].local_tracker,
                                reduced_token.float(),
                                batch_idx=batch_idx,
                                token_idx=token_idx,
                                absolute_position=absolute_position,
                            )
                            local_features = self._local_alignment_features(
                                layer_states[batch_idx].local_tracker,
                                reduced_token.float(),
                            )
                            self._finalize_local_qr_capture(local_features)
                        prepared_or_response = self._prepare_global_request(
                            state=past_key_values,
                            layer_state=layer_states[batch_idx],
                            summaries=layer_states[batch_idx].block_summaries,
                            reduced_token=reduced_token.float(),
                            query_token=query_token.float(),
                            batch_idx=batch_idx,
                            absolute_position=absolute_position,
                        )
                    batchable_local_features[batch_idx] = local_features
                    if isinstance(prepared_or_response, _GlobalFeatureResponse):
                        early_responses[batch_idx] = prepared_or_response
                    else:
                        prepared_rows.append(batch_idx)
                        prepared_requests.append(prepared_or_response)

                resolved_batch_responses = self._resolve_global_requests_batched(prepared_requests)
                for response_batch_idx, response in zip(prepared_rows, resolved_batch_responses):
                    early_responses[response_batch_idx] = response

                for batch_idx in batchable_rows:
                    hidden_token = hidden_states[batch_idx, token_idx]
                    reduced_token = reduced_tokens[batch_idx, token_idx]
                    absolute_position = int(positions_host[batch_idx, token_idx])
                    global_response = early_responses[batch_idx]
                    features = torch.cat((batchable_local_features[batch_idx], global_response.global_features), dim=0)
                    self._update_block_memory(
                        past_key_values,
                        layer_states[batch_idx],
                        reduced_token.float(),
                        absolute_position,
                        batch_idx=batch_idx,
                    )
                    delta, gate_mean, gate_std, delta_rms = self._delta_from_features(
                        features=features,
                        hidden_dtype=hidden_token.dtype,
                    )
                    outputs[batch_idx, token_idx] = delta
                    retrieved_block_ids[batch_idx, token_idx] = global_response.retrieved_block_ids
                    retrieved_scores[batch_idx, token_idx] = global_response.retrieved_scores
                    retrieved_prefix_lengths[batch_idx, token_idx] = global_response.retrieved_prefix_lengths
                    coarse_retrieved_block_ids[batch_idx, token_idx] = global_response.coarse_retrieved_block_ids
                    coarse_retrieved_scores[batch_idx, token_idx] = global_response.coarse_retrieved_scores
                    coarse_retrieved_facet_scores[batch_idx, token_idx] = global_response.coarse_retrieved_facet_scores
                    if feature_trace is not None:
                        feature_trace[batch_idx, token_idx].copy_(features.detach().to(device="cpu", dtype=torch.float32))
                    gate_mean_sum += float(gate_mean)
                    gate_std_sum += float(gate_std)
                    delta_rms_sum += float(delta_rms)
                    stat_count += 1

            self._last_feature_trace = feature_trace
            self._last_readout_trace = _ReadoutTrace(
                queries=query_tokens,
                retrieved_block_ids=retrieved_block_ids,
                retrieved_scores=retrieved_scores,
                retrieved_prefix_lengths=retrieved_prefix_lengths,
                coarse_retrieved_block_ids=coarse_retrieved_block_ids,
                coarse_retrieved_scores=coarse_retrieved_scores,
                coarse_retrieved_facet_scores=coarse_retrieved_facet_scores,
            )
            self._last_centroid_trace = _CentroidTrace(
                block_ids=centroid_block_ids,
                assignment_ids=centroid_assignment_ids,
                assignment_weights=centroid_assignment_weights,
                assignment_logits=centroid_assignment_logits,
                token_keys=centroid_token_keys,
                token_values=centroid_token_values,
                valid_mask=centroid_valid_mask,
            )
            self._last_forward_stats = {
                "gfst_branch_rms": float(delta_rms_sum / max(1, stat_count)),
                "gate_activation_mean": float(gate_mean_sum / max(1, stat_count)),
                "gate_activation_std": float(gate_std_sum / max(1, stat_count)),
            }
            return outputs, None

    def _training_forward_fast(
        self,
        *,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        past_key_values: GfstDecodeState,
    ) -> tuple[torch.Tensor, None]:
        with runtime_range(
            "gfst.tokenwise_forward",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=hidden_states.device,
        ):
            batch_size, seq_len, _ = hidden_states.shape
            positions_host = positions.detach().to(device="cpu", dtype=torch.long)
            reduced_input = hidden_states.to(dtype=self.reduce_proj.weight.dtype)
            reduced_tokens = apply_rope(self.reduce_proj(reduced_input), positions, theta=self.rope_theta)
            query_tokens = self.query_proj(reduced_tokens.to(dtype=self.query_proj.weight.dtype)).float()
            precomputed_centroid = self._precompute_centroid_write_payloads(
                hidden_states=hidden_states,
                positions=positions,
            )
            padded_size = int(self.gfst_config.retrieval_depth) + 1
            coarse_padded_size = max(
                int(self.gfst_config.retrieval_depth),
                int(self.eval_router_shortlist_depth or int(self.gfst_config.retrieval_depth)),
            )
            facet_count = max(1, int(self.eval_router_facet_count))
            assignment_top_k = max(
                1,
                min(int(self.gfst_config.centroid_assignment_top_k), int(self.gfst_config.cells_per_block)),
            )
            outputs = torch.empty_like(hidden_states)
            feature_trace = (
                torch.empty(
                    (batch_size, seq_len, int(self.gfst_config.feature_dim)),
                    device="cpu",
                    dtype=torch.float32,
                )
                if self.debug_capture_features
                else None
            )
            retrieved_block_ids = torch.empty((batch_size, seq_len, padded_size), device=hidden_states.device, dtype=torch.long)
            retrieved_scores = torch.empty((batch_size, seq_len, padded_size), device=hidden_states.device, dtype=torch.float32)
            retrieved_prefix_lengths = torch.empty(
                (batch_size, seq_len, padded_size),
                device=hidden_states.device,
                dtype=torch.long,
            )
            coarse_retrieved_block_ids = torch.empty(
                (batch_size, seq_len, coarse_padded_size),
                device=hidden_states.device,
                dtype=torch.long,
            )
            coarse_retrieved_scores = torch.empty(
                (batch_size, seq_len, coarse_padded_size),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            coarse_retrieved_facet_scores = torch.empty(
                (batch_size, seq_len, coarse_padded_size, facet_count),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            centroid_block_ids = torch.full((batch_size, seq_len), -1, device=hidden_states.device, dtype=torch.long)
            centroid_assignment_ids = torch.full(
                (batch_size, seq_len, assignment_top_k),
                -1,
                device=hidden_states.device,
                dtype=torch.long,
            )
            centroid_assignment_weights = torch.zeros(
                (batch_size, seq_len, assignment_top_k),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            centroid_assignment_logits = torch.zeros(
                (batch_size, seq_len, int(self.gfst_config.cells_per_block)),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            centroid_token_keys = torch.zeros(
                (batch_size, seq_len, int(self.gfst_config.centroid_key_dim)),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            centroid_token_values = torch.zeros(
                (batch_size, seq_len, int(self.gfst_config.centroid_value_dim)),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            centroid_valid_mask = torch.zeros((batch_size, seq_len), device=hidden_states.device, dtype=torch.bool)
            if int(self.layer_idx) == 0:
                centroid_block_ids.copy_(torch.div(positions, int(self.gfst_config.block_len), rounding_mode="floor"))
                if precomputed_centroid is not None:
                    centroid_block_ids.copy_(precomputed_centroid.block_ids.to(device=hidden_states.device, dtype=torch.long))
                    centroid_assignment_ids.copy_(
                        precomputed_centroid.assignment_ids.to(device=hidden_states.device, dtype=torch.long)
                    )
                    centroid_assignment_weights.copy_(
                        precomputed_centroid.assignment_weights.to(device=hidden_states.device, dtype=torch.float32)
                    )
                    centroid_assignment_logits.copy_(
                        precomputed_centroid.assignment_logits.to(device=hidden_states.device, dtype=torch.float32)
                    )
                    centroid_token_keys.copy_(precomputed_centroid.token_keys.to(device=hidden_states.device, dtype=torch.float32))
                    centroid_token_values.copy_(
                        precomputed_centroid.token_values.to(device=hidden_states.device, dtype=torch.float32)
                    )
                    centroid_valid_mask.copy_(precomputed_centroid.valid_mask.to(device=hidden_states.device, dtype=torch.bool))

            gate_mean_sum = 0.0
            gate_std_sum = 0.0
            delta_rms_sum = 0.0
            stat_count = 0
            layer_states = [past_key_values.layer_state(self.layer_idx, batch_idx) for batch_idx in range(batch_size)]
            for token_idx in range(seq_len):
                for batch_idx in range(batch_size):
                    token_result = self._token_step(
                        state=past_key_values,
                        batch_idx=batch_idx,
                        hidden_token=hidden_states[batch_idx, token_idx],
                        reduced_token=reduced_tokens[batch_idx, token_idx],
                        query_token=query_tokens[batch_idx, token_idx],
                        layer_state=layer_states[batch_idx],
                        absolute_position=int(positions_host[batch_idx, token_idx]),
                        precomputed_centroid=precomputed_centroid,
                        token_idx=token_idx,
                    )
                    outputs[batch_idx, token_idx] = token_result.delta
                    retrieved_block_ids[batch_idx, token_idx] = token_result.retrieved_block_ids
                    retrieved_scores[batch_idx, token_idx] = token_result.retrieved_scores
                    retrieved_prefix_lengths[batch_idx, token_idx] = token_result.retrieved_prefix_lengths
                    coarse_retrieved_block_ids[batch_idx, token_idx] = token_result.coarse_retrieved_block_ids
                    coarse_retrieved_scores[batch_idx, token_idx] = token_result.coarse_retrieved_scores
                    coarse_retrieved_facet_scores[batch_idx, token_idx] = token_result.coarse_retrieved_facet_scores
                    if feature_trace is not None:
                        feature_trace[batch_idx, token_idx].copy_(
                            token_result.features.detach().to(device="cpu", dtype=torch.float32)
                        )
                    gate_mean_sum += float(token_result.gate_mean)
                    gate_std_sum += float(token_result.gate_std)
                    delta_rms_sum += float(token_result.delta_rms)
                    stat_count += 1

            self._last_feature_trace = feature_trace
            self._last_readout_trace = _ReadoutTrace(
                queries=query_tokens,
                retrieved_block_ids=retrieved_block_ids,
                retrieved_scores=retrieved_scores,
                retrieved_prefix_lengths=retrieved_prefix_lengths,
                coarse_retrieved_block_ids=coarse_retrieved_block_ids,
                coarse_retrieved_scores=coarse_retrieved_scores,
                coarse_retrieved_facet_scores=coarse_retrieved_facet_scores,
            )
            self._last_centroid_trace = _CentroidTrace(
                block_ids=centroid_block_ids,
                assignment_ids=centroid_assignment_ids,
                assignment_weights=centroid_assignment_weights,
                assignment_logits=centroid_assignment_logits,
                token_keys=centroid_token_keys,
                token_values=centroid_token_values,
                valid_mask=centroid_valid_mask,
            )
            self._last_forward_stats = {
                "gfst_branch_rms": float(delta_rms_sum / max(1, stat_count)),
                "gate_activation_mean": float(gate_mean_sum / max(1, stat_count)),
                "gate_activation_std": float(gate_std_sum / max(1, stat_count)),
            }
            return outputs, None

    def _initialise_local_tracker(self, tracker_state, *, device: torch.device) -> None:
        math_dtype = torch.float32
        if tracker_state.largest_basis is None:
            tracker_state.largest_basis = init_basis(
                self.gfst_config.reduced_dim,
                self.gfst_config.max_flag_dim,
                device=device,
                dtype=math_dtype,
            )
        if len(tracker_state.nested_bases) != max(0, len(self.gfst_config.flag_dims) - 1):
            tracker_state.nested_bases = [
                init_basis(self.gfst_config.max_flag_dim, dim, device=device, dtype=math_dtype)
                for dim in self.gfst_config.flag_dims[:-1]
            ]

    def _update_local_bases(
        self,
        tracker_state,
        reduced_token: torch.Tensor,
        *,
        batch_idx: int | None = None,
        token_idx: int | None = None,
        absolute_position: int | None = None,
    ) -> list[torch.Tensor]:
        cfg = self.gfst_config
        self._initialise_local_tracker(tracker_state, device=reduced_token.device)
        if (
            self._active_transition_probe_capture is None
            and self._should_capture_transition_probe(batch_idx=batch_idx, token_idx=token_idx)
            and self._current_transition_token_context is not None
        ):
            context = self._current_transition_token_context
            self._begin_transition_probe_capture(
                batch_idx=int(context["batch_idx"]),
                token_idx=int(context["token_idx"]),
                absolute_position=int(context["absolute_position"]),
                hidden_token=context["hidden_token"],
                reduced_token=context["reduced_token"],
                query_token=context["query_token"],
            )
        transition_capture = self._active_transition_probe_capture
        transition_local: dict[str, Any] | None = None
        if transition_capture is not None:
            transition_local = {
                "largest_basis_in": _detach_cpu_float32(tracker_state.largest_basis),
                "nested_steps": [],
            }
        capture_payload: dict[str, Any] | None = None
        capture_targeted = self._should_capture_local_qr(batch_idx=batch_idx, token_idx=token_idx)
        capture_pool = self._should_record_local_qr_pool()
        if capture_targeted or capture_pool:
            capture_payload = {
                "_record_targeted": bool(capture_targeted),
                "_record_pool": bool(capture_pool),
                "metadata": {
                    "layer_idx": int(self.layer_idx),
                    "batch_idx": None if batch_idx is None else int(batch_idx),
                    "token_idx": None if token_idx is None else int(token_idx),
                    "absolute_position": None if absolute_position is None else int(absolute_position),
                    "flag_dims": [int(dim) for dim in cfg.flag_dims],
                    "max_flag_dim": int(cfg.max_flag_dim),
                    "eps": float(cfg.eps),
                    "eps_sign": float(cfg.eps_sign),
                    "eps_update": float(cfg.eps_update),
                    "gfst_update_rule": str(cfg.gfst_update_rule),
                    "forgetting_rate": float(cfg.forgetting_rates[-1]),
                },
                "reduced_token": _detach_cpu_float32(reduced_token),
                "largest_basis_in": _detach_cpu_float32(tracker_state.largest_basis),
                "nested_steps": [],
            }
        largest_candidate = update_basis_candidate(
            tracker_state.largest_basis,
            reduced_token,
            cfg.forgetting_rates[-1],
            update_rule=str(cfg.gfst_update_rule),
            eps_update=float(cfg.eps_update),
        )
        if capture_payload is not None:
            capture_payload["largest_candidate"] = _detach_cpu_float32(largest_candidate)
        if transition_local is not None:
            transition_local["largest_candidate"] = _detach_cpu_float32(largest_candidate)
        tracker_state.largest_basis = canonical_qr_batched(
            largest_candidate.unsqueeze(0),
            eps_sign=float(cfg.eps_sign),
        ).squeeze(0)
        if capture_payload is not None:
            capture_payload["largest_basis_out"] = _detach_cpu_float32(tracker_state.largest_basis)
        if transition_local is not None:
            transition_local["largest_basis_out"] = _detach_cpu_float32(tracker_state.largest_basis)
        projected = tracker_state.largest_basis.transpose(0, 1) @ reduced_token
        if capture_payload is not None:
            capture_payload["projected"] = _detach_cpu_float32(projected)
        if transition_local is not None:
            transition_local["projected"] = _detach_cpu_float32(projected)
        local_bases = []
        for idx, dim in enumerate(cfg.flag_dims[:-1]):
            nested_capture: dict[str, Any] | None = None
            transition_nested_capture: dict[str, Any] | None = None
            if capture_payload is not None:
                nested_capture = {
                    "nested_index": int(idx),
                    "dim": int(dim),
                    "nested_basis_in": _detach_cpu_float32(tracker_state.nested_bases[idx]),
                }
            if transition_local is not None:
                transition_nested_capture = {
                    "nested_index": int(idx),
                    "dim": int(dim),
                    "nested_basis_in": _detach_cpu_float32(tracker_state.nested_bases[idx]),
                }
            nested_candidate = update_basis_candidate(
                tracker_state.nested_bases[idx],
                projected,
                cfg.forgetting_rates[idx],
                update_rule=str(cfg.gfst_update_rule),
                eps_update=float(cfg.eps_update),
            )
            if nested_capture is not None:
                nested_capture["nested_candidate"] = _detach_cpu_float32(nested_candidate)
            if transition_nested_capture is not None:
                transition_nested_capture["nested_candidate"] = _detach_cpu_float32(nested_candidate)
            nested_basis = canonical_qr_batched(
                nested_candidate.unsqueeze(0),
                eps_sign=float(cfg.eps_sign),
            ).squeeze(0)
            tracker_state.nested_bases[idx] = nested_basis[:, :dim]
            if nested_capture is not None:
                nested_capture["nested_basis_out"] = _detach_cpu_float32(tracker_state.nested_bases[idx])
                capture_payload["nested_steps"].append(nested_capture)
            if transition_nested_capture is not None:
                transition_nested_capture["nested_basis_out"] = _detach_cpu_float32(tracker_state.nested_bases[idx])
                transition_local["nested_steps"].append(transition_nested_capture)
            local_bases.append(tracker_state.largest_basis @ tracker_state.nested_bases[idx])
        local_bases.append(tracker_state.largest_basis[:, : cfg.max_flag_dim])
        if capture_payload is not None:
            self._pending_local_qr_capture = capture_payload
        if transition_capture is not None and transition_local is not None:
            transition_capture["local_path"] = transition_local
        return local_bases

    def _local_alignment_features(self, tracker_state, reduced_token: torch.Tensor) -> torch.Tensor:
        cfg = self.gfst_config
        projected = tracker_state.largest_basis.transpose(0, 1) @ reduced_token
        denominator = torch.clamp(reduced_token.square().sum(), min=float(cfg.eps))
        cumulative: list[torch.Tensor] = []
        for nested_basis in tracker_state.nested_bases:
            nested_projection = nested_basis.transpose(0, 1) @ projected
            cumulative.append(nested_projection.square().sum() / denominator)
        cumulative.append(projected.square().sum() / denominator)
        cumulative_tensor = torch.stack(cumulative, dim=0).to(dtype=reduced_token.dtype)
        previous = torch.cat((torch.zeros_like(cumulative_tensor[:1]), cumulative_tensor[:-1]), dim=0)
        increments = cumulative_tensor - previous
        return torch.stack((cumulative_tensor, increments), dim=-1).reshape(-1)

    def _initialise_block_tracker(self, block_tracker, *, device: torch.device, start_pos: int) -> None:
        math_dtype = torch.float32
        if block_tracker.largest_basis is None:
            block_tracker.largest_basis = init_basis(
                self.gfst_config.reduced_dim,
                self.gfst_config.max_flag_dim,
                device=device,
                dtype=math_dtype,
            )
        if block_tracker.projector_sum is None:
            block_tracker.projector_sum = torch.zeros(
                self.gfst_config.reduced_dim,
                self.gfst_config.reduced_dim,
                device=device,
                dtype=math_dtype,
            )
        if block_tracker.reduced_sum is None:
            block_tracker.reduced_sum = torch.zeros(
                self.gfst_config.reduced_dim,
                device=device,
                dtype=math_dtype,
            )
        if block_tracker.current_block_tokens == 0:
            block_tracker.current_block_start = int(start_pos)

    def _reset_block_tracker(self, block_tracker, *, device: torch.device, start_pos: int) -> None:
        block_tracker.largest_basis = init_basis(
            self.gfst_config.reduced_dim,
            self.gfst_config.max_flag_dim,
            device=device,
            dtype=torch.float32,
        )
        block_tracker.projector_sum = torch.zeros(
            self.gfst_config.reduced_dim,
            self.gfst_config.reduced_dim,
            device=device,
            dtype=torch.float32,
        )
        block_tracker.factor_buffer = None
        block_tracker.reduced_sum = torch.zeros(
            self.gfst_config.reduced_dim,
            device=device,
            dtype=torch.float32,
        )
        block_tracker.current_block_start = int(start_pos)
        block_tracker.current_block_tokens = 0

    def _update_local_projectors(self, tracker_state, reduced_token: torch.Tensor) -> list[torch.Tensor]:
        cfg = self.gfst_config
        local_bases = self._update_local_bases(tracker_state, reduced_token)
        return [basis_projector(basis[:, :dim]) for basis, dim in zip(local_bases, cfg.flag_dims)]

    def _exact_oracle(self, keys_tensor: torch.Tensor, query_token: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices, scores = FaissGpuRetriever.exact_topk(keys_tensor, query_token, top_k)
        return indices, scores

    def _summary_tensor_cache(
        self,
        *,
        layer_state,
        device: torch.device,
        debug_capture: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        with runtime_range(
            "gfst.summary_tensor_cache",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=device,
        ):
            summaries = layer_state.block_summaries
            if not summaries:
                if debug_capture is not None:
                    debug_capture["summary_cache_rebuilt"] = False
                    debug_capture["summary_cache_rebuild_reasons"] = []
                return None
            facet_count = max(1, int(self.eval_router_facet_count))
            cache_device = str(device)
            rebuild_reasons: list[str] = []
            if layer_state.summary_key_tensor is None:
                rebuild_reasons.append("summary_key_tensor_missing")
            if layer_state.summary_block_index_tensor is None:
                rebuild_reasons.append("summary_block_index_tensor_missing")
            if layer_state.summary_token_count_tensor is None:
                rebuild_reasons.append("summary_token_count_tensor_missing")
            if layer_state.summary_reduced_mean_tensor is None:
                rebuild_reasons.append("summary_reduced_mean_tensor_missing")
            if layer_state.summary_facet_basis_tensor is None:
                rebuild_reasons.append("summary_facet_basis_tensor_missing")
            if layer_state.summary_basis_tensor is None:
                rebuild_reasons.append("summary_basis_tensor_missing")
            if (
                str(self.gfst_feature_backend) in {"native", "sm120_exact"}
                and layer_state.summary_projector_tensor is None
            ):
                rebuild_reasons.append("summary_projector_tensor_missing")
            if str(layer_state.summary_cache_device) != cache_device:
                rebuild_reasons.append("summary_cache_device_mismatch")
            if int(layer_state.summary_cache_facet_count) != facet_count:
                rebuild_reasons.append("summary_cache_facet_count_mismatch")
            if layer_state.summary_key_tensor is not None and int(layer_state.summary_key_tensor.shape[0]) != len(summaries):
                rebuild_reasons.append("summary_count_mismatch")
            if rebuild_reasons:
                reduced_dim = int(self.gfst_config.reduced_dim)
                keys: list[torch.Tensor] = []
                block_indices: list[int] = []
                token_counts: list[int] = []
                reduced_means: list[torch.Tensor] = []
                facet_bases: list[torch.Tensor] = []
                bases: list[torch.Tensor] = []
                projectors: list[torch.Tensor] | None = (
                    [] if str(self.gfst_feature_backend) in {"native", "sm120_exact"} else None
                )
                for summary in summaries:
                    keys.append(summary.key.to(device=device, dtype=torch.float32))
                    block_indices.append(int(summary.block_index))
                    token_counts.append(int(summary.token_count))
                    mean = summary.reduced_mean
                    if mean is None:
                        mean = torch.zeros(reduced_dim, device=device, dtype=torch.float32)
                    else:
                        mean = mean.to(device=device, dtype=torch.float32)
                    reduced_means.append(mean)
                    basis = summary.basis.to(device=device, dtype=torch.float32)
                    max_flag_dim = int(self.gfst_config.max_flag_dim)
                    if int(basis.shape[1]) < max_flag_dim:
                        padded_full_basis = torch.zeros((reduced_dim, max_flag_dim), device=device, dtype=torch.float32)
                        padded_full_basis[:, : int(basis.shape[1])] = basis
                        basis = padded_full_basis
                    else:
                        basis = basis[:, :max_flag_dim]
                    bases.append(basis)
                    if projectors is not None:
                        projectors.append(basis_projector(basis))
                    facet_basis = basis[:, :facet_count]
                    if int(facet_basis.shape[1]) < facet_count:
                        padded_facet_basis = torch.zeros((reduced_dim, facet_count), device=device, dtype=torch.float32)
                        padded_facet_basis[:, : int(facet_basis.shape[1])] = facet_basis
                        facet_basis = padded_facet_basis
                    facet_bases.append(facet_basis.transpose(0, 1))
                layer_state.summary_key_tensor = torch.stack(keys, dim=0)
                layer_state.summary_block_index_tensor = torch.tensor(block_indices, device=device, dtype=torch.long)
                layer_state.summary_token_count_tensor = torch.tensor(token_counts, device=device, dtype=torch.long)
                layer_state.summary_reduced_mean_tensor = torch.stack(reduced_means, dim=0)
                layer_state.summary_facet_basis_tensor = torch.stack(facet_bases, dim=0)
                layer_state.summary_basis_tensor = torch.stack(bases, dim=0)
                layer_state.summary_projector_tensor = None if projectors is None else torch.stack(projectors, dim=0)
                layer_state.summary_cache_device = cache_device
                layer_state.summary_cache_facet_count = facet_count
            if debug_capture is not None:
                debug_capture["summary_cache_rebuilt"] = bool(rebuild_reasons)
                debug_capture["summary_cache_rebuild_reasons"] = list(rebuild_reasons)
            return (
                layer_state.summary_key_tensor,
                layer_state.summary_block_index_tensor,
                layer_state.summary_token_count_tensor,
                layer_state.summary_reduced_mean_tensor,
                layer_state.summary_facet_basis_tensor,
                layer_state.summary_basis_tensor,
                layer_state.summary_projector_tensor,
            )

    def _facet_scores(
        self,
        *,
        layer_state,
        indices: torch.Tensor,
        query_token: torch.Tensor,
    ) -> torch.Tensor:
        with runtime_range(
            "gfst.facet_scores",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=query_token.device,
        ):
            if int(indices.numel()) == 0:
                return torch.empty(
                    (0, int(self.eval_router_facet_count)),
                    device=query_token.device,
                    dtype=torch.float32,
                )
            facet_count = max(1, int(self.eval_router_facet_count))
            summary_cache = self._summary_tensor_cache(layer_state=layer_state, device=query_token.device)
            if summary_cache is None:
                return torch.empty(
                    (0, facet_count),
                    device=query_token.device,
                    dtype=torch.float32,
                )
            _, _, _, reduced_mean_tensor, facet_basis_tensor, _, _ = summary_cache
            gather_indices = indices.to(device=query_token.device, dtype=torch.long)
            mean_tensor = reduced_mean_tensor.index_select(0, gather_indices)
            basis_tensor = facet_basis_tensor.index_select(0, gather_indices)
            mean_tensor = mean_tensor.unsqueeze(1).expand(-1, facet_count, -1)
            facet_inputs = torch.cat((mean_tensor, basis_tensor), dim=-1)
            projected = self.block_key_proj(
                facet_inputs.reshape(-1, int(facet_inputs.shape[-1])).to(dtype=self.block_key_proj.weight.dtype)
            ).reshape(int(indices.shape[0]), facet_count, int(self.gfst_config.key_dim))
            query = query_token.to(dtype=projected.dtype)
            return torch.einsum("bfk,k->bf", projected, query).float()

    def _global_feature_empty_tensors(
        self,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.gfst_config
        padded_size = int(cfg.retrieval_depth) + 1
        coarse_padded_size = max(
            int(cfg.retrieval_depth),
            int(self.eval_router_shortlist_depth or int(cfg.retrieval_depth)),
        )
        facet_count = max(1, int(self.eval_router_facet_count))
        empty_blocks = torch.full((padded_size,), -1, device=device, dtype=torch.long)
        empty_scores = torch.zeros((padded_size,), device=device, dtype=torch.float32)
        empty_prefixes = torch.zeros((padded_size,), device=device, dtype=torch.long)
        empty_coarse_blocks = torch.full((coarse_padded_size,), -1, device=device, dtype=torch.long)
        empty_coarse_scores = torch.full((coarse_padded_size,), float("-inf"), device=device, dtype=torch.float32)
        empty_facet_scores = torch.full(
            (coarse_padded_size, facet_count),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        return (
            empty_blocks,
            empty_scores,
            empty_prefixes,
            empty_coarse_blocks,
            empty_coarse_scores,
            empty_facet_scores,
        )

    def _empty_global_feature_response(
        self,
        *,
        state: GfstDecodeState,
        batch_idx: int,
        device: torch.device,
        transition_global: dict[str, Any] | None,
        empty_blocks: torch.Tensor,
        empty_scores: torch.Tensor,
        empty_prefixes: torch.Tensor,
        empty_coarse_blocks: torch.Tensor,
        empty_coarse_scores: torch.Tensor,
        empty_facet_scores: torch.Tensor,
    ) -> _GlobalFeatureResponse:
        global_features = torch.zeros(self.gfst_config.feature_dim // 2, device=device, dtype=torch.float32)
        block_tensor = empty_blocks.clone()
        score_tensor = empty_scores.clone()
        prefix_tensor = empty_prefixes.clone()
        current_block_id = state.current_exact_block_id(batch_idx, self.gfst_config.block_len)
        current_prefix = state.current_exact_block_length(batch_idx)
        if current_block_id is not None and current_prefix > 0:
            block_tensor[0] = int(current_block_id)
            prefix_tensor[0] = int(current_prefix)
        if transition_global is not None:
            transition_global["current_block_id"] = None if current_block_id is None else int(current_block_id)
            transition_global["current_prefix"] = int(current_prefix)
            transition_global["global_features"] = _detach_cpu_float32(global_features)
            transition_global["returned_block_tensor"] = _detach_cpu_tensor(block_tensor)
            transition_global["returned_score_tensor"] = _detach_cpu_tensor(score_tensor)
            transition_global["returned_prefix_tensor"] = _detach_cpu_tensor(prefix_tensor)
            transition_global["returned_coarse_block_tensor"] = _detach_cpu_tensor(empty_coarse_blocks)
            transition_global["returned_coarse_score_tensor"] = _detach_cpu_tensor(empty_coarse_scores)
            transition_global["returned_facet_scores"] = _detach_cpu_tensor(empty_facet_scores)
        return _GlobalFeatureResponse(
            global_features=global_features,
            retrieved_block_ids=block_tensor,
            retrieved_scores=score_tensor,
            retrieved_prefix_lengths=prefix_tensor,
            coarse_retrieved_block_ids=empty_coarse_blocks,
            coarse_retrieved_scores=empty_coarse_scores,
            coarse_retrieved_facet_scores=empty_facet_scores,
        )

    def _prepare_global_request(
        self,
        *,
        state: GfstDecodeState,
        layer_state,
        summaries: list[GfstBlockSummary],
        reduced_token: torch.Tensor,
        query_token: torch.Tensor,
        batch_idx: int,
        absolute_position: int | None = None,
    ) -> _PreparedGlobalRequest | _GlobalFeatureResponse:
        del absolute_position
        cfg = self.gfst_config
        transition_capture = self._active_transition_probe_capture
        transition_global: dict[str, Any] | None = None
        if transition_capture is not None:
            transition_global = {
                "summary_key_tensor": None,
                "summary_count": 0,
                "summary_cache_rebuilt": False,
                "summary_cache_rebuild_reasons": [],
                "summary_block_index_tensor": None,
                "summary_token_count_tensor": None,
                "summary_reduced_mean_tensor": None,
                "summary_basis_tensor": None,
                "summary_projector_tensor": None,
                "coarse_indices": None,
                "coarse_scores": None,
                "final_indices": None,
                "final_scores": None,
                "selected_weights": None,
                "retrieved_block_ids_tensor": None,
                "retrieved_prefix_lengths_tensor": None,
                "current_block_id": None,
                "current_prefix": None,
                "global_sketch": None,
                "eig_basis": None,
                "accelerated_global_features_raw": None,
                "global_features": None,
                "feature_actual_path": None,
                "feature_reduced_token": None,
                "feature_projector_debug": None,
                "feature_basis_native_debug": None,
                "feature_basis_float32_debug": None,
                "feature_actual_minus_projector_ref_max_abs": None,
                "feature_actual_minus_basis_native_ref_max_abs": None,
                "feature_actual_minus_basis_float32_ref_max_abs": None,
                "returned_block_tensor": None,
                "returned_score_tensor": None,
                "returned_prefix_tensor": None,
                "returned_coarse_block_tensor": None,
                "returned_coarse_score_tensor": None,
                "returned_facet_scores": None,
            }
            transition_capture["global_path"] = transition_global
        (
            empty_blocks,
            empty_scores,
            empty_prefixes,
            empty_coarse_blocks,
            empty_coarse_scores,
            empty_facet_scores,
        ) = self._global_feature_empty_tensors(device=reduced_token.device)
        if not summaries:
            return self._empty_global_feature_response(
                state=state,
                batch_idx=batch_idx,
                device=reduced_token.device,
                transition_global=transition_global,
                empty_blocks=empty_blocks,
                empty_scores=empty_scores,
                empty_prefixes=empty_prefixes,
                empty_coarse_blocks=empty_coarse_blocks,
                empty_coarse_scores=empty_coarse_scores,
                empty_facet_scores=empty_facet_scores,
            )

        top_k = min(cfg.retrieval_depth, len(summaries))
        coarse_top_k = min(int(self.eval_router_shortlist_depth or top_k), len(summaries))
        summary_cache = self._summary_tensor_cache(
            layer_state=layer_state,
            device=reduced_token.device,
            debug_capture=transition_global,
        )
        if summary_cache is None:
            return self._empty_global_feature_response(
                state=state,
                batch_idx=batch_idx,
                device=reduced_token.device,
                transition_global=transition_global,
                empty_blocks=empty_blocks,
                empty_scores=empty_scores,
                empty_prefixes=empty_prefixes,
                empty_coarse_blocks=empty_coarse_blocks,
                empty_coarse_scores=empty_coarse_scores,
                empty_facet_scores=empty_facet_scores,
            )
        (
            keys_tensor,
            summary_block_indices,
            summary_token_counts,
            summary_reduced_means,
            _summary_facet_bases,
            summary_basis_tensor,
            summary_projectors,
        ) = summary_cache
        if transition_global is not None:
            transition_global["summary_key_tensor"] = _detach_cpu_float32(keys_tensor)
            transition_global["summary_count"] = int(keys_tensor.shape[0])
            transition_global["summary_block_index_tensor"] = _detach_cpu_tensor(summary_block_indices)
            transition_global["summary_token_count_tensor"] = _detach_cpu_tensor(summary_token_counts)
            transition_global["summary_reduced_mean_tensor"] = _detach_cpu_float32(summary_reduced_means)
            transition_global["summary_basis_tensor"] = _detach_cpu_float32(summary_basis_tensor)
            transition_global["summary_projector_tensor"] = (
                None if summary_projectors is None else _detach_cpu_float32(summary_projectors)
            )
        with runtime_range(
            "gfst.global_features.retrieval_search",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=reduced_token.device,
        ):
            if self.retriever is None:
                coarse_indices, coarse_scores = self._exact_oracle(keys_tensor, query_token, coarse_top_k)
            else:
                coarse_indices, coarse_scores = self.retriever.search(
                    state=state,
                    layer_idx=self.layer_idx,
                    batch_idx=batch_idx,
                    keys_tensor=keys_tensor,
                    query_tensor=query_token,
                    top_k=coarse_top_k,
                )
        if transition_global is not None:
            transition_global["coarse_indices"] = _detach_cpu_tensor(coarse_indices)
            transition_global["coarse_scores"] = _detach_cpu_float32(coarse_scores)
        if int(coarse_indices.numel()) == 0:
            return self._empty_global_feature_response(
                state=state,
                batch_idx=batch_idx,
                device=reduced_token.device,
                transition_global=transition_global,
                empty_blocks=empty_blocks,
                empty_scores=empty_scores,
                empty_prefixes=empty_prefixes,
                empty_coarse_blocks=empty_coarse_blocks,
                empty_coarse_scores=empty_coarse_scores,
                empty_facet_scores=empty_facet_scores,
            )

        coarse_block_tensor = empty_coarse_blocks.clone()
        coarse_score_tensor = empty_coarse_scores.clone()
        coarse_count = min(int(coarse_block_tensor.shape[0]), int(coarse_indices.numel()))
        if coarse_count > 0:
            coarse_block_tensor[:coarse_count] = summary_block_indices.index_select(
                0,
                coarse_indices[:coarse_count].to(device=reduced_token.device, dtype=torch.long),
            )
            coarse_score_tensor[:coarse_count] = coarse_scores[:coarse_count].to(
                device=reduced_token.device,
                dtype=torch.float32,
            )

        facet_scores = empty_facet_scores.clone()
        rerank_scores = coarse_scores
        final_indices = coarse_indices[:top_k]
        final_scores = coarse_scores[:top_k]
        router_mode = str(self.eval_router_mode)
        if router_mode in {"multifacet_rerank", "multifacet_rerank_plus_score_bias"}:
            computed_facet_scores = self._facet_scores(
                layer_state=layer_state,
                indices=coarse_indices,
                query_token=query_token,
            )
            facet_count_used = min(int(facet_scores.shape[1]), int(computed_facet_scores.shape[1]))
            facet_scores[: int(computed_facet_scores.shape[0]), :facet_count_used] = computed_facet_scores[
                :, :facet_count_used
            ]
            if str(self.eval_router_facet_aggregation) == "logsumexp":
                rerank_scores = torch.logsumexp(computed_facet_scores, dim=-1)
            else:
                rerank_scores = computed_facet_scores.max(dim=-1).values
            rerank_order = torch.argsort(rerank_scores, descending=True)
            final_indices = coarse_indices[rerank_order[:top_k]]
            final_scores = rerank_scores[rerank_order[:top_k]]
        if transition_global is not None:
            transition_global["final_indices"] = _detach_cpu_tensor(final_indices)
            transition_global["final_scores"] = _detach_cpu_float32(final_scores)
        if summary_projectors is None:
            raise RuntimeError("GFST global feature path requires summary projectors")
        gathered_indices = final_indices.to(device=reduced_token.device, dtype=torch.long)
        return _PreparedGlobalRequest(
            state=state,
            batch_idx=int(batch_idx),
            reduced_token=reduced_token,
            selected_projectors=summary_projectors.index_select(0, gathered_indices),
            final_scores=final_scores.to(device=reduced_token.device, dtype=torch.float32),
            retrieved_block_ids_tensor=summary_block_indices.index_select(0, gathered_indices),
            retrieved_prefix_lengths_tensor=summary_token_counts.index_select(0, gathered_indices),
            empty_blocks=empty_blocks,
            empty_scores=empty_scores,
            empty_prefixes=empty_prefixes,
            coarse_block_tensor=coarse_block_tensor,
            coarse_score_tensor=coarse_score_tensor,
            facet_scores=facet_scores,
            transition_global=transition_global,
        )

    def _finalize_prepared_global_response(
        self,
        *,
        request: _PreparedGlobalRequest,
        global_features: torch.Tensor,
        global_sketch: torch.Tensor,
        eig_basis: torch.Tensor,
        accelerated_global_features_raw: torch.Tensor | None = None,
    ) -> _GlobalFeatureResponse:
        if request.transition_global is not None:
            request.transition_global["global_sketch"] = _detach_cpu_float32(global_sketch)
            request.transition_global["eig_basis"] = _detach_cpu_float32(eig_basis)
            if accelerated_global_features_raw is not None:
                request.transition_global["accelerated_global_features_raw"] = _detach_cpu_float32(
                    accelerated_global_features_raw
                )
            request.transition_global["global_features"] = _detach_cpu_float32(global_features)
            request.transition_global["retrieved_block_ids_tensor"] = _detach_cpu_tensor(
                request.retrieved_block_ids_tensor
            )
            request.transition_global["retrieved_prefix_lengths_tensor"] = _detach_cpu_tensor(
                request.retrieved_prefix_lengths_tensor
            )

        retrieved_block_ids = [int(value) for value in request.retrieved_block_ids_tensor.tolist()]
        retrieved_prefix_lengths = [int(value) for value in request.retrieved_prefix_lengths_tensor.tolist()]
        current_block_id = request.state.current_exact_block_id(request.batch_idx, self.gfst_config.block_len)
        current_prefix = request.state.current_exact_block_length(request.batch_idx)
        if (
            current_block_id is not None
            and current_prefix > 0
            and int(current_block_id) not in retrieved_block_ids
        ):
            retrieved_block_ids.append(int(current_block_id))
            retrieved_prefix_lengths.append(int(current_prefix))
            request.final_scores = torch.cat(
                (
                    request.final_scores,
                    torch.zeros((1,), device=request.final_scores.device, dtype=request.final_scores.dtype),
                ),
                dim=0,
            )

        block_tensor = request.empty_blocks.clone()
        score_tensor = request.empty_scores.clone()
        prefix_tensor = request.empty_prefixes.clone()
        count = min(int(block_tensor.shape[0]), len(retrieved_block_ids))
        if count > 0:
            block_tensor[:count] = torch.tensor(retrieved_block_ids[:count], device=block_tensor.device, dtype=torch.long)
            score_tensor[:count] = request.final_scores[:count].to(device=score_tensor.device, dtype=torch.float32)
            prefix_tensor[:count] = torch.tensor(
                retrieved_prefix_lengths[:count],
                device=prefix_tensor.device,
                dtype=torch.long,
            )
        if request.transition_global is not None:
            request.transition_global["current_block_id"] = None if current_block_id is None else int(current_block_id)
            request.transition_global["current_prefix"] = int(current_prefix)
            request.transition_global["returned_block_tensor"] = _detach_cpu_tensor(block_tensor)
            request.transition_global["returned_score_tensor"] = _detach_cpu_tensor(score_tensor)
            request.transition_global["returned_prefix_tensor"] = _detach_cpu_tensor(prefix_tensor)
            request.transition_global["returned_coarse_block_tensor"] = _detach_cpu_tensor(request.coarse_block_tensor)
            request.transition_global["returned_coarse_score_tensor"] = _detach_cpu_tensor(request.coarse_score_tensor)
            request.transition_global["returned_facet_scores"] = _detach_cpu_tensor(request.facet_scores)
        return _GlobalFeatureResponse(
            global_features=global_features,
            retrieved_block_ids=block_tensor,
            retrieved_scores=score_tensor,
            retrieved_prefix_lengths=prefix_tensor,
            coarse_retrieved_block_ids=request.coarse_block_tensor,
            coarse_retrieved_scores=request.coarse_score_tensor,
            coarse_retrieved_facet_scores=request.facet_scores,
        )

    def _resolve_single_global_request(
        self,
        request: _PreparedGlobalRequest,
    ) -> _GlobalFeatureResponse:
        cfg = self.gfst_config
        with runtime_range(
            "gfst.global_features.global_sketch",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=request.reduced_token.device,
        ):
            selected_weights = torch.softmax(request.final_scores * float(cfg.retrieval_temperature), dim=0).to(
                device=request.selected_projectors.device,
                dtype=request.selected_projectors.dtype,
            )
            if request.transition_global is not None:
                request.transition_global["selected_weights"] = _detach_cpu_float32(selected_weights)
            accelerated_global = None
            if self._use_sm120_exact_math_kernels():
                projector_bank = request.selected_projectors
                scalar_indices = torch.arange(
                    int(projector_bank.shape[0]),
                    device=projector_bank.device,
                    dtype=torch.long,
                )
                accelerated_global = sm120_gfst_global_sketch(
                    summary_projectors=projector_bank,
                    final_indices=scalar_indices,
                    final_scores=request.final_scores.to(device=projector_bank.device, dtype=torch.float32),
                    reduced_token=request.reduced_token,
                    retrieval_temperature=float(cfg.retrieval_temperature),
                    flag_dims=tuple(int(dim) for dim in cfg.flag_dims),
                    max_flag_dim=int(cfg.max_flag_dim),
                    eps=float(cfg.eps),
                )
            if accelerated_global is not None:
                accelerated_features, global_sketch, eig_basis = accelerated_global
                global_features = accelerated_features.to(device=request.reduced_token.device, dtype=torch.float32)
                accelerated_raw = global_features
                if request.transition_global is not None:
                    feature_summary = top_eig_summary(
                        global_sketch,
                        rank=int(cfg.max_flag_dim),
                        rank_boundaries=tuple(int(dim) for dim in cfg.flag_dims),
                    )
                    _capture_global_feature_math_debug(
                        transition_global=request.transition_global,
                        reduced_token=request.reduced_token.to(device=global_sketch.device),
                        full_basis=feature_summary["full_basis"],
                        rank_boundaries=tuple(int(dim) for dim in cfg.flag_dims),
                        eps=float(cfg.eps),
                        actual_global_features=global_features.to(device=global_sketch.device),
                        actual_path="canonical_projector",
                        prefix_projectors={
                            int(dim): projector for dim, projector in dict(feature_summary["prefix_projectors"]).items()
                        },
                    )
            else:
                with runtime_range(
                    "gfst.global_features.global_sketch.accum",
                    mode=self.runtime_profiler_mode,
                    recorder=self._runtime_profile_recorder,
                    device=request.reduced_token.device,
                ):
                    global_sketch = torch.einsum("k,kij->ij", selected_weights, request.selected_projectors)
                with runtime_range(
                    "gfst.global_features.global_sketch.summary",
                    mode=self.runtime_profiler_mode,
                    recorder=self._runtime_profile_recorder,
                    device=request.reduced_token.device,
                ):
                    eig_summary = top_eig_summary(
                        global_sketch,
                        rank=int(cfg.max_flag_dim),
                        rank_boundaries=tuple(int(dim) for dim in cfg.flag_dims),
                    )
                    eig_basis = eig_summary["basis"]
                    prefix_projectors = {
                        int(dim): projector for dim, projector in dict(eig_summary["prefix_projectors"]).items()
                    }
                with runtime_range(
                    "gfst.global_features.global_sketch.feature",
                    mode=self.runtime_profiler_mode,
                    recorder=self._runtime_profile_recorder,
                    device=request.reduced_token.device,
                ):
                    global_features = alignment_components(
                        request.reduced_token,
                        [
                            prefix_projectors.get(int(dim), basis_projector(eig_basis[:, : int(dim)]))
                            for dim in cfg.flag_dims
                        ],
                        cfg.eps,
                    )
                    if request.transition_global is not None:
                        _capture_global_feature_math_debug(
                            transition_global=request.transition_global,
                            reduced_token=request.reduced_token.to(device=global_sketch.device),
                            full_basis=eig_summary["full_basis"],
                            rank_boundaries=tuple(int(dim) for dim in cfg.flag_dims),
                            eps=float(cfg.eps),
                            actual_global_features=global_features.to(device=global_sketch.device),
                            actual_path="canonical_projector",
                            prefix_projectors=prefix_projectors,
                        )
                accelerated_raw = None
        return self._finalize_prepared_global_response(
            request=request,
            global_features=global_features,
            global_sketch=global_sketch,
            eig_basis=eig_basis,
            accelerated_global_features_raw=accelerated_raw,
        )

    def _resolve_global_requests_batched(
        self,
        requests: list[_PreparedGlobalRequest],
    ) -> list[_GlobalFeatureResponse]:
        if not requests:
            return []
        cfg = self.gfst_config
        responses: list[_GlobalFeatureResponse | None] = [None] * len(requests)
        groups: dict[tuple[int, int, int, str, int | None], list[tuple[int, _PreparedGlobalRequest]]] = {}
        for idx, request in enumerate(requests):
            selected_shape = request.selected_projectors.shape
            key = (
                int(selected_shape[0]),
                int(selected_shape[1]),
                int(selected_shape[2]),
                str(request.selected_projectors.device.type),
                request.selected_projectors.device.index,
            )
            groups.setdefault(key, []).append((idx, request))

        for grouped_requests in groups.values():
            scalar_grouped_requests: list[tuple[int, _PreparedGlobalRequest]] = []
            batched_grouped_requests: list[tuple[int, _PreparedGlobalRequest]] = []
            for idx, request in grouped_requests:
                if request.transition_global is not None:
                    scalar_grouped_requests.append((idx, request))
                else:
                    batched_grouped_requests.append((idx, request))
            for idx, request in scalar_grouped_requests:
                responses[idx] = self._resolve_single_global_request(request)
            if not batched_grouped_requests:
                continue
            if len(batched_grouped_requests) == 1:
                idx, request = batched_grouped_requests[0]
                responses[idx] = self._resolve_single_global_request(request)
                continue
            with runtime_range(
                "gfst.global_features.global_sketch",
                mode=self.runtime_profiler_mode,
                recorder=self._runtime_profile_recorder,
                device=batched_grouped_requests[0][1].reduced_token.device,
            ):
                with runtime_range(
                    "gfst.global_features.global_sketch.accum",
                    mode=self.runtime_profiler_mode,
                    recorder=self._runtime_profile_recorder,
                    device=batched_grouped_requests[0][1].reduced_token.device,
                ):
                    selected_projectors = torch.stack(
                        [request.selected_projectors for _, request in batched_grouped_requests],
                        dim=0,
                    )
                    final_scores = torch.stack(
                        [request.final_scores for _, request in batched_grouped_requests],
                        dim=0,
                    )
                    reduced_tokens = torch.stack(
                        [request.reduced_token for _, request in batched_grouped_requests],
                        dim=0,
                    ).to(device=selected_projectors.device, dtype=torch.float32)
                    weights = torch.softmax(final_scores * float(cfg.retrieval_temperature), dim=-1).to(
                        device=selected_projectors.device,
                        dtype=selected_projectors.dtype,
                    )
                    global_sketches = torch.einsum("bk,bkij->bij", weights, selected_projectors)
                with runtime_range(
                    "gfst.global_features.global_sketch.heev",
                    mode=self.runtime_profiler_mode,
                    recorder=self._runtime_profile_recorder,
                    device=batched_grouped_requests[0][1].reduced_token.device,
                ):
                    heev_outputs = sm120_gfst_heev(global_sketches)
                if heev_outputs is None:
                    for idx, request in batched_grouped_requests:
                        responses[idx] = self._resolve_single_global_request(request)
                    continue
                eigvals_batch, eigvecs_batch = heev_outputs
                target_rank = max(int(cfg.max_flag_dim), max((int(dim) for dim in cfg.flag_dims), default=0), 1)
                full_basis_rank = min(int(global_sketches.shape[-1]), int(target_rank))
                basis_rank = min(int(global_sketches.shape[-1]), int(cfg.max_flag_dim))
                full_basis_batch = torch.zeros(
                    (int(global_sketches.shape[0]), int(global_sketches.shape[-1]), int(full_basis_rank)),
                    device=global_sketches.device,
                    dtype=global_sketches.dtype,
                )
                basis_batch = torch.zeros(
                    (int(global_sketches.shape[0]), int(global_sketches.shape[-1]), int(basis_rank)),
                    device=global_sketches.device,
                    dtype=global_sketches.dtype,
                )
                with runtime_range(
                    "gfst.global_features.global_sketch.summary",
                    mode=self.runtime_profiler_mode,
                    recorder=self._runtime_profile_recorder,
                    device=batched_grouped_requests[0][1].reduced_token.device,
                ):
                    degenerate_boundary_mask: list[bool] = []
                    with runtime_range(
                        "gfst.global_features.global_sketch.summary.classify",
                        mode=self.runtime_profiler_mode,
                        recorder=self._runtime_profile_recorder,
                        device=batched_grouped_requests[0][1].reduced_token.device,
                    ):
                        for row_idx in range(len(batched_grouped_requests)):
                            degenerate_boundary_mask.append(
                                bool(
                                    _sm120_runtime._requires_native_summary_fallback(
                                        eigvals_batch[row_idx],
                                        rank_boundaries=tuple(
                                            sorted({int(cfg.max_flag_dim), *(int(dim) for dim in cfg.flag_dims)})
                                        ),
                                    )
                                )
                            )
                    degenerate_boundary_count = int(sum(1 for value in degenerate_boundary_mask if value))
                    stable_row_count = int(len(degenerate_boundary_mask) - degenerate_boundary_count)
                    canonical_summary = None
                    if (not self.training) and _sm120_runtime.experimental_canonical_summary_enabled():
                        with runtime_range(
                            "gfst.global_features.global_sketch.summary.experimental_op",
                            mode=self.runtime_profiler_mode,
                            recorder=self._runtime_profile_recorder,
                            device=batched_grouped_requests[0][1].reduced_token.device,
                        ):
                            canonical_summary = sm120_gfst_canonical_summary_from_eigpairs(
                                eigvals_batch,
                                eigvecs_batch,
                                reduced_tokens,
                                cutoffs=tuple(int(dim) for dim in cfg.flag_dims),
                                target_rank=int(cfg.max_flag_dim),
                                eps=float(cfg.eps),
                            )
                    canonical_summary_op_row_count = int(degenerate_boundary_count if canonical_summary is not None else 0)
                    scalar_summary_fallback_row_count = int(degenerate_boundary_count if canonical_summary is None else 0)
                    if self._runtime_profile_recorder is not None:
                        self._runtime_profile_recorder.add_value(
                            "gfst.global_features.global_sketch.degenerate_boundary_row_count",
                            float(degenerate_boundary_count),
                        )
                        self._runtime_profile_recorder.add_value(
                            "gfst.global_features.global_sketch.canonical_summary_op_row_count",
                            float(canonical_summary_op_row_count),
                        )
                        self._runtime_profile_recorder.add_value(
                            "gfst.global_features.global_sketch.scalar_summary_fallback_row_count",
                            float(scalar_summary_fallback_row_count),
                        )
                        self._runtime_profile_recorder.add_value(
                            "gfst.global_features.global_sketch.stable_row_count",
                            float(stable_row_count),
                        )
                        self._runtime_profile_recorder.add_value(
                            "gfst.global_features.global_sketch.fallback_row_count",
                            float(degenerate_boundary_count),
                        )
                    if canonical_summary is not None:
                        with runtime_range(
                            "gfst.global_features.global_sketch.summary.scatter_finalize",
                            mode=self.runtime_profiler_mode,
                            recorder=self._runtime_profile_recorder,
                            device=batched_grouped_requests[0][1].reduced_token.device,
                        ):
                            basis_values, global_features_values, _top_vec = canonical_summary
                            basis_batch = basis_values.to(device=global_sketches.device, dtype=global_sketches.dtype)
                            full_basis_batch = basis_batch
                            global_features_batch = global_features_values.to(
                                device=global_sketches.device,
                                dtype=torch.float32,
                            )
                    else:
                        stable_indices = [
                            row_idx for row_idx, degenerate in enumerate(degenerate_boundary_mask) if not degenerate
                        ]
                        fallback_indices = [
                            row_idx for row_idx, degenerate in enumerate(degenerate_boundary_mask) if degenerate
                        ]
                        stable_summary = None
                        if stable_indices:
                            stable_index_tensor = torch.tensor(
                                stable_indices,
                                device=global_sketches.device,
                                dtype=torch.long,
                            )
                            stable_summary = sm120_gfst_summary_from_sketch_batched(
                                global_sketches.index_select(0, stable_index_tensor),
                                max_flag_dim=int(cfg.max_flag_dim),
                                rank_boundaries=tuple(int(dim) for dim in cfg.flag_dims),
                                eigvals=eigvals_batch.index_select(0, stable_index_tensor),
                                eigvecs=eigvecs_batch.index_select(0, stable_index_tensor),
                                basis_dtype=global_sketches.dtype,
                                vector_dtype=global_sketches.dtype,
                            )
                        fallback_summary = None
                        if fallback_indices:
                            fallback_index_tensor = torch.tensor(
                                fallback_indices,
                                device=global_sketches.device,
                                dtype=torch.long,
                            )
                            with runtime_range(
                                "gfst.global_features.global_sketch.summary.scalar_fallback_rows",
                                mode=self.runtime_profiler_mode,
                                recorder=self._runtime_profile_recorder,
                                device=batched_grouped_requests[0][1].reduced_token.device,
                            ):
                                fallback_summary = sm120_gfst_summary_from_sketch_batched(
                                    global_sketches.index_select(0, fallback_index_tensor),
                                    max_flag_dim=int(cfg.max_flag_dim),
                                    rank_boundaries=tuple(int(dim) for dim in cfg.flag_dims),
                                    basis_dtype=global_sketches.dtype,
                                    vector_dtype=global_sketches.dtype,
                                )
                        with runtime_range(
                            "gfst.global_features.global_sketch.summary.scatter_finalize",
                            mode=self.runtime_profiler_mode,
                            recorder=self._runtime_profile_recorder,
                            device=batched_grouped_requests[0][1].reduced_token.device,
                        ):
                            if stable_indices and stable_summary is not None:
                                full_basis_batch.index_copy_(0, stable_index_tensor, stable_summary["full_basis"])
                                basis_batch.index_copy_(0, stable_index_tensor, stable_summary["basis"])
                            if fallback_indices and fallback_summary is not None:
                                full_basis_batch.index_copy_(0, fallback_index_tensor, fallback_summary["full_basis"])
                                basis_batch.index_copy_(0, fallback_index_tensor, fallback_summary["basis"])
                with runtime_range(
                    "gfst.global_features.global_sketch.feature",
                    mode=self.runtime_profiler_mode,
                    recorder=self._runtime_profile_recorder,
                    device=batched_grouped_requests[0][1].reduced_token.device,
                ):
                    if canonical_summary is None:
                        global_features_batch = alignment_components_from_full_basis_batch(
                            reduced_tokens,
                            full_basis_batch.to(device=selected_projectors.device, dtype=torch.float32),
                            tuple(int(dim) for dim in cfg.flag_dims),
                            float(cfg.eps),
                        )
                for row_idx, (response_idx, request) in enumerate(batched_grouped_requests):
                    responses[response_idx] = self._finalize_prepared_global_response(
                        request=request,
                        global_features=global_features_batch[row_idx].to(
                            device=request.reduced_token.device,
                            dtype=torch.float32,
                        ),
                        global_sketch=global_sketches[row_idx],
                        eig_basis=basis_batch[row_idx],
                    )
        if any(response is None for response in responses):
            raise RuntimeError("batched global request resolution left unresolved rows")
        return [response for response in responses if response is not None]

    def _global_features(
        self,
        *,
        state: GfstDecodeState,
        layer_state,
        summaries: list[GfstBlockSummary],
        reduced_token: torch.Tensor,
        query_token: torch.Tensor,
        batch_idx: int,
        absolute_position: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with runtime_range(
            "gfst.global_features",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=reduced_token.device,
        ):
            prepared_or_response = self._prepare_global_request(
                state=state,
                layer_state=layer_state,
                summaries=summaries,
                reduced_token=reduced_token,
                query_token=query_token,
                batch_idx=batch_idx,
                absolute_position=absolute_position,
            )
            if isinstance(prepared_or_response, _GlobalFeatureResponse):
                response = prepared_or_response
            else:
                response = self._resolve_single_global_request(prepared_or_response)
            return (
                response.global_features,
                response.retrieved_block_ids,
                response.retrieved_scores,
                response.retrieved_prefix_lengths,
                response.coarse_retrieved_block_ids,
                response.coarse_retrieved_scores,
                response.coarse_retrieved_facet_scores,
            )

    def _update_block_memory(
        self,
        state: GfstDecodeState,
        layer_state,
        reduced_token: torch.Tensor,
        absolute_position: int,
        batch_idx: int | None = None,
    ) -> None:
        with runtime_range(
            "gfst.update_block_memory",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=reduced_token.device,
        ):
            cfg = self.gfst_config
            block_tracker = layer_state.block_tracker
            self._initialise_block_tracker(block_tracker, device=reduced_token.device, start_pos=absolute_position)
            transition_capture = self._active_transition_probe_capture
            transition_block: dict[str, Any] | None = None
            if transition_capture is not None:
                transition_block = {
                    "tracker_before": {
                        "largest_basis": _detach_cpu_float32(block_tracker.largest_basis),
                        "projector_sum": _detach_cpu_float32(block_tracker.projector_sum),
                        "reduced_sum": _detach_cpu_float32(block_tracker.reduced_sum),
                        "current_block_tokens": int(block_tracker.current_block_tokens),
                        "current_block_start": int(block_tracker.current_block_start),
                    },
                    "summary_count_before": int(len(layer_state.block_summaries)),
                    "summary_block_indices_before": [int(summary.block_index) for summary in layer_state.block_summaries],
                    "summary_cache_present_before_invalidation": {
                        "summary_key_tensor": bool(layer_state.summary_key_tensor is not None),
                        "summary_block_index_tensor": bool(layer_state.summary_block_index_tensor is not None),
                        "summary_token_count_tensor": bool(layer_state.summary_token_count_tensor is not None),
                        "summary_reduced_mean_tensor": bool(layer_state.summary_reduced_mean_tensor is not None),
                        "summary_facet_basis_tensor": bool(layer_state.summary_facet_basis_tensor is not None),
                        "summary_basis_tensor": bool(layer_state.summary_basis_tensor is not None),
                        "summary_projector_tensor": bool(layer_state.summary_projector_tensor is not None),
                    },
                    "block_candidate": None,
                    "updated_largest_basis": None,
                    "largest_projector": None,
                    "commit_fired": False,
                    "expected_closed_block_index": None,
                    "average_projector": None,
                    "summary_basis": None,
                    "reduced_mean": None,
                    "top_vector": None,
                    "block_key": None,
                    "commit_inputs": None,
                    "appended_summary": None,
                    "summary_count_after": int(len(layer_state.block_summaries)),
                    "summary_block_indices_after": [int(summary.block_index) for summary in layer_state.block_summaries],
                    "summary_cache_invalidated": False,
                    "summary_cache_present_after_invalidation": None,
                    "tracker_after_reset": None,
                }
                transition_capture["block_memory"] = transition_block

            block_candidate = update_basis_candidate(
                block_tracker.largest_basis,
                reduced_token,
                cfg.forgetting_rates[-1],
                update_rule=str(cfg.gfst_update_rule),
                eps_update=float(cfg.eps_update),
            )
            if transition_block is not None:
                transition_block["block_candidate"] = _detach_cpu_float32(block_candidate)
            # Keep block-summary accumulation on the same canonical QR path as the live local feature stack.
            block_tracker.largest_basis = canonical_qr_batched(
                block_candidate.unsqueeze(0),
                eps_sign=float(cfg.eps_sign),
            ).squeeze(0)
            largest_projector = basis_projector(block_tracker.largest_basis[:, : cfg.max_flag_dim])
            if transition_block is not None:
                transition_block["updated_largest_basis"] = _detach_cpu_float32(block_tracker.largest_basis)
                transition_block["largest_projector"] = _detach_cpu_float32(largest_projector)
            block_tracker.projector_sum = block_tracker.projector_sum + largest_projector
            block_tracker.reduced_sum = block_tracker.reduced_sum + reduced_token
            block_tracker.current_block_tokens += 1

            if block_tracker.current_block_tokens < cfg.block_len:
                return

            accelerated_commit = None
            if self._use_sm120_exact_math_kernels():
                accelerated_commit = sm120_gfst_block_commit(
                    projector_sum=block_tracker.projector_sum,
                    reduced_sum=block_tracker.reduced_sum,
                    token_count=int(block_tracker.current_block_tokens),
                    max_flag_dim=int(cfg.max_flag_dim),
                    flag_dims=tuple(int(dim) for dim in cfg.flag_dims),
                )
            if accelerated_commit is not None:
                summary_basis, reduced_mean, top_vector, average_projector = accelerated_commit
            else:
                average_projector = block_tracker.projector_sum / float(block_tracker.current_block_tokens)
                summary = top_eig_summary(
                    average_projector,
                    rank=int(cfg.max_flag_dim),
                    rank_boundaries=tuple(int(dim) for dim in cfg.flag_dims),
                )
                summary_basis = summary["basis"]
                reduced_mean = block_tracker.reduced_sum / float(block_tracker.current_block_tokens)
                top_vector = summary["top_vector"]
            key_input = torch.cat((reduced_mean, top_vector.to(dtype=reduced_mean.dtype)), dim=0)
            block_key = self.block_key_proj(key_input.to(dtype=self.block_key_proj.weight.dtype)).float()
            if transition_block is not None:
                transition_block["commit_fired"] = True
                transition_block["average_projector"] = _detach_cpu_float32(average_projector)
                transition_block["summary_basis"] = _detach_cpu_float32(summary_basis)
                transition_block["reduced_mean"] = _detach_cpu_float32(reduced_mean)
                transition_block["top_vector"] = _detach_cpu_float32(top_vector)
                transition_block["block_key"] = _detach_cpu_float32(block_key)
                transition_block["commit_inputs"] = {
                    "projector_sum": _detach_cpu_float32(block_tracker.projector_sum),
                    "reduced_sum": _detach_cpu_float32(block_tracker.reduced_sum),
                    "token_count": int(block_tracker.current_block_tokens),
                    "block_key_proj_weight": _detach_cpu_float32(self.block_key_proj.weight),
                }
            block_index = int(block_tracker.current_block_start) // int(cfg.block_len)
            if transition_block is not None:
                transition_block["expected_closed_block_index"] = int(block_index)
            layer_state.block_summaries.append(
                GfstBlockSummary(
                    basis=summary_basis.float(),
                    key=block_key,
                    block_index=block_index,
                    start_pos=int(block_tracker.current_block_start),
                    end_pos=int(absolute_position + 1),
                    token_count=int(block_tracker.current_block_tokens),
                    reduced_mean=reduced_mean.float(),
                )
            )
            layer_state.block_summaries.sort(key=lambda summary: int(summary.block_index))
            if transition_block is not None:
                transition_block["appended_summary"] = _summary_to_debug_dict(layer_state.block_summaries[-1])
                transition_block["summary_count_after"] = int(len(layer_state.block_summaries))
                transition_block["summary_block_indices_after"] = [
                    int(summary.block_index) for summary in layer_state.block_summaries
                ]
            layer_state.summary_key_tensor = None
            layer_state.summary_block_index_tensor = None
            layer_state.summary_token_count_tensor = None
            layer_state.summary_reduced_mean_tensor = None
            layer_state.summary_facet_basis_tensor = None
            layer_state.summary_basis_tensor = None
            layer_state.summary_projector_tensor = None
            layer_state.summary_cache_device = None
            layer_state.summary_cache_facet_count = 0
            if transition_block is not None:
                transition_block["summary_cache_invalidated"] = True
                transition_block["summary_cache_present_after_invalidation"] = {
                    "summary_key_tensor": bool(layer_state.summary_key_tensor is not None),
                    "summary_block_index_tensor": bool(layer_state.summary_block_index_tensor is not None),
                    "summary_token_count_tensor": bool(layer_state.summary_token_count_tensor is not None),
                    "summary_reduced_mean_tensor": bool(layer_state.summary_reduced_mean_tensor is not None),
                    "summary_facet_basis_tensor": bool(layer_state.summary_facet_basis_tensor is not None),
                    "summary_basis_tensor": bool(layer_state.summary_basis_tensor is not None),
                    "summary_projector_tensor": bool(layer_state.summary_projector_tensor is not None),
                }
            self._reset_block_tracker(block_tracker, device=reduced_token.device, start_pos=absolute_position + 1)
            if transition_block is not None:
                transition_block["tracker_after_reset"] = {
                    "largest_basis": _detach_cpu_float32(block_tracker.largest_basis),
                    "projector_sum": _detach_cpu_float32(block_tracker.projector_sum),
                    "reduced_sum": _detach_cpu_float32(block_tracker.reduced_sum),
                    "current_block_tokens": int(block_tracker.current_block_tokens),
                    "current_block_start": int(block_tracker.current_block_start),
                }
            state.mark_mutated()

    def _token_features(
        self,
        *,
        state: GfstDecodeState,
        batch_idx: int,
        token_idx: int | None,
        reduced_token: torch.Tensor,
        query_token: torch.Tensor,
        layer_state,
        absolute_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with runtime_range(
            "gfst.token_features",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=reduced_token.device,
        ):
            cfg = self.gfst_config
            reduced_token = reduced_token.float()
            query_token = query_token.float()

            with runtime_range(
                "gfst.update_local_projectors",
                mode=self.runtime_profiler_mode,
                recorder=self._runtime_profile_recorder,
                device=reduced_token.device,
            ):
                local_bases = self._update_local_bases(
                    layer_state.local_tracker,
                    reduced_token,
                    batch_idx=batch_idx,
                    token_idx=token_idx,
                    absolute_position=absolute_position,
                )
                _ = local_bases
                local_features = self._local_alignment_features(layer_state.local_tracker, reduced_token)
                if self._active_transition_probe_capture is not None:
                    local_capture = self._active_transition_probe_capture.get("local_path")
                    if isinstance(local_capture, dict):
                        local_capture["local_features"] = _detach_cpu_float32(local_features)
                self._finalize_local_qr_capture(local_features)
            (
                global_features,
                retrieved_block_ids,
                retrieved_scores,
                retrieved_prefix_lengths,
                coarse_retrieved_block_ids,
                coarse_retrieved_scores,
                coarse_retrieved_facet_scores,
            ) = self._global_features(
                state=state,
                layer_state=layer_state,
                summaries=layer_state.block_summaries,
                reduced_token=reduced_token,
                query_token=query_token,
                batch_idx=batch_idx,
                absolute_position=absolute_position,
            )
            features = torch.cat((local_features, global_features), dim=0)
            self._update_block_memory(
                state,
                layer_state,
                reduced_token,
                absolute_position,
                batch_idx=batch_idx,
            )
            return (
                features,
                retrieved_block_ids,
                retrieved_scores,
                retrieved_prefix_lengths,
                coarse_retrieved_block_ids,
                coarse_retrieved_scores,
                coarse_retrieved_facet_scores,
            )

    def _delta_from_features(
        self,
        *,
        features: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, float, float, float]:
        with runtime_range(
            "gfst.delta_from_features",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=features.device,
        ):
            feature_input = features.to(dtype=self.feature_gate.weight.dtype)
            gate = torch.sigmoid(self.feature_gate(feature_input))
            delta = gate * self.feature_value(feature_input)
            if self.disable_residual:
                delta = torch.zeros_like(delta)
            return (
                delta.to(dtype=hidden_dtype),
                float(gate.detach().float().mean().item()),
                float(gate.detach().float().std(unbiased=False).item()),
                _tensor_rms(delta),
            )

    def _delta_from_features_batch(
        self,
        *,
        features: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with runtime_range(
            "gfst.delta_from_features",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=features.device,
        ):
            feature_input = features.to(dtype=self.feature_gate.weight.dtype)
            gate = torch.sigmoid(self.feature_gate(feature_input))
            delta = gate * self.feature_value(feature_input)
            if self.disable_residual:
                delta = torch.zeros_like(delta)
            gate_float = gate.detach().float()
            delta_float = delta.detach().float()
            return (
                delta.to(dtype=hidden_dtype),
                gate_float.mean(dim=-1),
                gate_float.std(dim=-1, unbiased=False),
                torch.sqrt(delta_float.square().mean(dim=-1)),
            )

    def _token_step(
        self,
        *,
        state: GfstDecodeState,
        batch_idx: int,
        hidden_token: torch.Tensor,
        reduced_token: torch.Tensor,
        query_token: torch.Tensor,
        layer_state,
        absolute_position: int,
        precomputed_centroid: _PrecomputedCentroidWrite | None = None,
        token_idx: int | None = None,
        structured: bool = True,
    ) -> _PerTokenResult | tuple[Any, ...]:
        transition_capture: dict[str, Any] | None = None
        self._stash_transition_token_context(
            batch_idx=batch_idx,
            token_idx=token_idx,
            absolute_position=absolute_position,
            hidden_token=hidden_token,
            reduced_token=reduced_token,
            query_token=query_token,
        )
        if self._should_capture_transition_probe(batch_idx=batch_idx, token_idx=token_idx):
            transition_capture = self._begin_transition_probe_capture(
                batch_idx=int(batch_idx),
                token_idx=int(token_idx),
                absolute_position=int(absolute_position),
                hidden_token=hidden_token,
                reduced_token=reduced_token,
                query_token=query_token,
            )
        if int(self.layer_idx) == 0:
            token_id = state.consume_prepared_token(batch_idx, absolute_position)
            assignment_top_k = max(1, min(int(self.gfst_config.centroid_assignment_top_k), int(self.gfst_config.cells_per_block)))
            cell_ids = None
            cell_weights = None
            cell_key = None
            cell_value = None
            centroid_block_id = torch.tensor(
                int(absolute_position // int(self.gfst_config.block_len)),
                device=hidden_token.device,
                dtype=torch.long,
            )
            centroid_assignment_ids = torch.full((assignment_top_k,), -1, device=hidden_token.device, dtype=torch.long)
            centroid_assignment_weights = torch.zeros((assignment_top_k,), device=hidden_token.device, dtype=torch.float32)
            centroid_assignment_logits = torch.zeros(
                (int(self.gfst_config.cells_per_block),), device=hidden_token.device, dtype=torch.float32
            )
            centroid_token_key = torch.zeros(
                (int(self.gfst_config.centroid_key_dim),), device=hidden_token.device, dtype=torch.float32
            )
            centroid_token_value = torch.zeros(
                (int(self.gfst_config.centroid_value_dim),), device=hidden_token.device, dtype=torch.float32
            )
            centroid_valid_mask = torch.tensor(False, device=hidden_token.device, dtype=torch.bool)
            if precomputed_centroid is not None and token_idx is not None:
                centroid_block_id = precomputed_centroid.block_ids[batch_idx, token_idx].to(
                    device=hidden_token.device,
                    dtype=torch.long,
                )
                centroid_assignment_ids = precomputed_centroid.assignment_ids[batch_idx, token_idx].to(
                    device=hidden_token.device,
                    dtype=torch.long,
                )
                centroid_assignment_weights = precomputed_centroid.assignment_weights[batch_idx, token_idx].to(
                    device=hidden_token.device,
                    dtype=torch.float32,
                )
                centroid_assignment_logits = precomputed_centroid.assignment_logits[batch_idx, token_idx].to(
                    device=hidden_token.device,
                    dtype=torch.float32,
                )
                centroid_token_key = precomputed_centroid.token_keys[batch_idx, token_idx].to(
                    device=hidden_token.device,
                    dtype=torch.float32,
                )
                centroid_token_value = precomputed_centroid.token_values[batch_idx, token_idx].to(
                    device=hidden_token.device,
                    dtype=torch.float32,
                )
                centroid_valid_mask = precomputed_centroid.valid_mask[batch_idx, token_idx].to(
                    device=hidden_token.device,
                    dtype=torch.bool,
                )
                if bool(centroid_valid_mask.item()):
                    cell_ids = centroid_assignment_ids
                    cell_weights = centroid_assignment_weights
                    cell_key = centroid_token_key
                    cell_value = centroid_token_value
            else:
                centroid_activation_enabled = bool(self.gfst_config.centroid_activation_enabled) or bool(self.force_centroid_activation)
                if centroid_activation_enabled:
                    centroid_input = self._centroid_write_input(hidden_token=hidden_token, absolute_position=absolute_position)
                    assign_logits = self.cell_assign_proj(centroid_input).float()
                    centroid_assignment_logits = assign_logits
                    projected_assignment_logits = self._apply_centroid_recovery_control(assign_logits)
                    cell_ids, centroid_assignment_weights = self._project_centroid_assignments(projected_assignment_logits)
                    cell_weights = centroid_assignment_weights
                    centroid_assignment_ids = cell_ids
                    cell_key = self.cell_key_proj(centroid_input).float()
                    cell_value = self.cell_value_proj(centroid_input).float()
                    local_offset = int(absolute_position) % int(self.gfst_config.block_len)
                    cell_key = self._apply_centroid_phase_rotation(cell_key, local_offset=local_offset).float()
                    cell_value = self._apply_centroid_phase_rotation(cell_value, local_offset=local_offset).float()
                    centroid_token_key = cell_key
                    centroid_token_value = cell_value
                    centroid_valid_mask = torch.tensor(True, device=hidden_token.device, dtype=torch.bool)
            state.append_exact_token(
                batch_idx=batch_idx,
                absolute_position=absolute_position,
                token_id=token_id,
                block_len=self.gfst_config.block_len,
                hidden_state=hidden_token
                if bool(self.gfst_config.store_exact_hidden_states) or bool(self.force_store_exact_hidden_states)
                else None,
                exact_hidden_store_mode=str(getattr(self.gfst_config, "exact_hidden_store_mode", "tuple") or "tuple"),
                cells_per_block=int(self.gfst_config.cells_per_block),
                centroid_key_dim=int(self.gfst_config.centroid_key_dim),
                centroid_value_dim=int(self.gfst_config.centroid_value_dim),
                cell_ids=None if cell_ids is None else cell_ids.detach(),
                cell_weights=None if cell_weights is None else cell_weights.detach(),
                cell_key=cell_key,
                cell_value=cell_value,
            )
        else:
            assignment_top_k = max(1, min(int(self.gfst_config.centroid_assignment_top_k), int(self.gfst_config.cells_per_block)))
            centroid_block_id = torch.tensor(-1, device=hidden_token.device, dtype=torch.long)
            centroid_assignment_ids = torch.full((assignment_top_k,), -1, device=hidden_token.device, dtype=torch.long)
            centroid_assignment_weights = torch.zeros((assignment_top_k,), device=hidden_token.device, dtype=torch.float32)
            centroid_assignment_logits = torch.zeros(
                (int(self.gfst_config.cells_per_block),),
                device=hidden_token.device,
                dtype=torch.float32,
            )
            centroid_token_key = torch.zeros(
                (int(self.gfst_config.centroid_key_dim),),
                device=hidden_token.device,
                dtype=torch.float32,
            )
            centroid_token_value = torch.zeros(
                (int(self.gfst_config.centroid_value_dim),),
                device=hidden_token.device,
                dtype=torch.float32,
            )
            centroid_valid_mask = torch.tensor(False, device=hidden_token.device, dtype=torch.bool)
        (
            features,
            retrieved_block_ids,
            retrieved_scores,
            retrieved_prefix_lengths,
            coarse_retrieved_block_ids,
            coarse_retrieved_scores,
            coarse_retrieved_facet_scores,
        ) = self._token_features(
            state=state,
            batch_idx=batch_idx,
            token_idx=token_idx,
            reduced_token=reduced_token,
            query_token=query_token,
            layer_state=layer_state,
            absolute_position=absolute_position,
        )
        delta, gate_mean, gate_std, delta_rms = self._delta_from_features(
            features=features,
            hidden_dtype=hidden_token.dtype,
        )
        if transition_capture is not None:
            transition_capture["post_feature"] = {
                "features": _detach_cpu_float32(features),
                "gate_mean": float(gate_mean),
                "gate_std": float(gate_std),
                "delta_rms": float(delta_rms),
                "delta": _detach_cpu_float32(delta),
                "hidden_token_out": _detach_cpu_float32(hidden_token + delta.to(dtype=hidden_token.dtype)),
            }
            self._finish_transition_probe_capture()
        self._current_transition_token_context = None
        result_values = (
            delta,
            features,
            query_token,
            retrieved_block_ids,
            retrieved_scores,
            retrieved_prefix_lengths,
            coarse_retrieved_block_ids,
            coarse_retrieved_scores,
            coarse_retrieved_facet_scores,
            centroid_block_id,
            centroid_assignment_ids,
            centroid_assignment_weights,
            centroid_assignment_logits,
            centroid_token_key,
            centroid_token_value,
            centroid_valid_mask,
            gate_mean,
            gate_std,
            delta_rms,
        )
        if not structured:
            return result_values
        return _PerTokenResult(*result_values)

    def _tokenwise_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        past_key_values: GfstDecodeState,
    ) -> tuple[torch.Tensor, None]:
        with runtime_range(
            "gfst.tokenwise_forward",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=hidden_states.device,
        ):
            batch_size, seq_len, _ = hidden_states.shape
            positions_host = positions.detach().to(device="cpu", dtype=torch.long)
            reduced_input = hidden_states.to(dtype=self.reduce_proj.weight.dtype)
            reduced_tokens = apply_rope(self.reduce_proj(reduced_input), positions, theta=self.rope_theta)
            query_tokens = self.query_proj(reduced_tokens.to(dtype=self.query_proj.weight.dtype)).float()
            precomputed_centroid = self._precompute_centroid_write_payloads(
                hidden_states=hidden_states,
                positions=positions,
            )
            outputs = []
            feature_batches: list[torch.Tensor] | None = [] if self.debug_capture_features else None
            query_batches: list[torch.Tensor] = []
            retrieved_block_batches: list[torch.Tensor] = []
            retrieved_score_batches: list[torch.Tensor] = []
            retrieved_prefix_batches: list[torch.Tensor] = []
            coarse_retrieved_block_batches: list[torch.Tensor] = []
            coarse_retrieved_score_batches: list[torch.Tensor] = []
            coarse_retrieved_facet_batches: list[torch.Tensor] = []
            centroid_block_id_batches: list[torch.Tensor] = []
            centroid_assignment_id_batches: list[torch.Tensor] = []
            centroid_assignment_weight_batches: list[torch.Tensor] = []
            centroid_assignment_logit_batches: list[torch.Tensor] = []
            centroid_token_key_batches: list[torch.Tensor] = []
            centroid_token_value_batches: list[torch.Tensor] = []
            centroid_valid_mask_batches: list[torch.Tensor] = []
            gate_means: list[float] = []
            gate_stds: list[float] = []
            delta_rms_values: list[float] = []
            for batch_idx in range(batch_size):
                sample_outputs = []
                sample_features: list[torch.Tensor] | None = [] if feature_batches is not None else None
                sample_queries: list[torch.Tensor] = []
                sample_retrieved_blocks: list[torch.Tensor] = []
                sample_retrieved_scores: list[torch.Tensor] = []
                sample_retrieved_prefixes: list[torch.Tensor] = []
                sample_coarse_retrieved_blocks: list[torch.Tensor] = []
                sample_coarse_retrieved_scores: list[torch.Tensor] = []
                sample_coarse_retrieved_facets: list[torch.Tensor] = []
                sample_centroid_block_ids: list[torch.Tensor] = []
                sample_centroid_assignment_ids: list[torch.Tensor] = []
                sample_centroid_assignment_weights: list[torch.Tensor] = []
                sample_centroid_assignment_logits: list[torch.Tensor] = []
                sample_centroid_token_keys: list[torch.Tensor] = []
                sample_centroid_token_values: list[torch.Tensor] = []
                sample_centroid_valid_masks: list[torch.Tensor] = []
                layer_state = past_key_values.layer_state(self.layer_idx, batch_idx)
                for token_idx in range(seq_len):
                    token_result = self._token_step(
                        state=past_key_values,
                        batch_idx=batch_idx,
                        hidden_token=hidden_states[batch_idx, token_idx],
                        reduced_token=reduced_tokens[batch_idx, token_idx],
                        query_token=query_tokens[batch_idx, token_idx],
                        layer_state=layer_state,
                        absolute_position=int(positions_host[batch_idx, token_idx]),
                        precomputed_centroid=precomputed_centroid,
                        token_idx=token_idx,
                    )
                    sample_outputs.append(token_result.delta)
                    gate_means.append(float(token_result.gate_mean))
                    gate_stds.append(float(token_result.gate_std))
                    delta_rms_values.append(float(token_result.delta_rms))
                    if sample_features is not None:
                        sample_features.append(token_result.features.detach().to(device="cpu", dtype=torch.float32))
                    sample_queries.append(token_result.query)
                    sample_retrieved_blocks.append(token_result.retrieved_block_ids)
                    sample_retrieved_scores.append(token_result.retrieved_scores)
                    sample_retrieved_prefixes.append(token_result.retrieved_prefix_lengths)
                    sample_coarse_retrieved_blocks.append(token_result.coarse_retrieved_block_ids)
                    sample_coarse_retrieved_scores.append(token_result.coarse_retrieved_scores)
                    sample_coarse_retrieved_facets.append(token_result.coarse_retrieved_facet_scores)
                    sample_centroid_block_ids.append(token_result.centroid_block_id)
                    sample_centroid_assignment_ids.append(token_result.centroid_assignment_ids)
                    sample_centroid_assignment_weights.append(token_result.centroid_assignment_weights)
                    sample_centroid_assignment_logits.append(token_result.centroid_assignment_logits)
                    sample_centroid_token_keys.append(token_result.centroid_token_key)
                    sample_centroid_token_values.append(token_result.centroid_token_value)
                    sample_centroid_valid_masks.append(token_result.centroid_valid_mask)
                outputs.append(torch.stack(sample_outputs, dim=0))
                if sample_features is not None:
                    feature_batches.append(torch.stack(sample_features, dim=0))
                query_batches.append(torch.stack(sample_queries, dim=0))
                retrieved_block_batches.append(torch.stack(sample_retrieved_blocks, dim=0))
                retrieved_score_batches.append(torch.stack(sample_retrieved_scores, dim=0))
                retrieved_prefix_batches.append(torch.stack(sample_retrieved_prefixes, dim=0))
                coarse_retrieved_block_batches.append(torch.stack(sample_coarse_retrieved_blocks, dim=0))
                coarse_retrieved_score_batches.append(torch.stack(sample_coarse_retrieved_scores, dim=0))
                coarse_retrieved_facet_batches.append(torch.stack(sample_coarse_retrieved_facets, dim=0))
                centroid_block_id_batches.append(torch.stack(sample_centroid_block_ids, dim=0))
                centroid_assignment_id_batches.append(torch.stack(sample_centroid_assignment_ids, dim=0))
                centroid_assignment_weight_batches.append(torch.stack(sample_centroid_assignment_weights, dim=0))
                centroid_assignment_logit_batches.append(torch.stack(sample_centroid_assignment_logits, dim=0))
                centroid_token_key_batches.append(torch.stack(sample_centroid_token_keys, dim=0))
                centroid_token_value_batches.append(torch.stack(sample_centroid_token_values, dim=0))
                centroid_valid_mask_batches.append(torch.stack(sample_centroid_valid_masks, dim=0))
            self._record_feature_trace(feature_batches)
            self._record_readout_trace(
                query_batches=query_batches,
                retrieved_block_batches=retrieved_block_batches,
                retrieved_score_batches=retrieved_score_batches,
                retrieved_prefix_batches=retrieved_prefix_batches,
                coarse_retrieved_block_batches=coarse_retrieved_block_batches,
                coarse_retrieved_score_batches=coarse_retrieved_score_batches,
                coarse_retrieved_facet_batches=coarse_retrieved_facet_batches,
            )
            self._record_centroid_trace(
                block_id_batches=centroid_block_id_batches,
                assignment_id_batches=centroid_assignment_id_batches,
                assignment_weight_batches=centroid_assignment_weight_batches,
                assignment_logit_batches=centroid_assignment_logit_batches,
                token_key_batches=centroid_token_key_batches,
                token_value_batches=centroid_token_value_batches,
                valid_mask_batches=centroid_valid_mask_batches,
            )
            self._last_forward_stats = {
                "gfst_branch_rms": float(sum(delta_rms_values) / max(1, len(delta_rms_values))),
                "gate_activation_mean": float(sum(gate_means) / max(1, len(gate_means))),
                "gate_activation_std": float(sum(gate_stds) / max(1, len(gate_stds))),
            }
            return torch.stack(outputs, dim=0), None

    def _resolve_positions(
        self,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        position_ids: torch.LongTensor | None,
        past_key_values: GfstDecodeState,
    ) -> torch.LongTensor:
        if position_ids is not None:
            if position_ids.ndim == 1:
                return position_ids.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=torch.long)
            return position_ids.to(device=device, dtype=torch.long)
        return torch.stack(
            [
                torch.arange(
                    past_key_values.samples[batch_idx].seen_tokens,
                    past_key_values.samples[batch_idx].seen_tokens + seq_len,
                    device=device,
                    dtype=torch.long,
                )
                for batch_idx in range(batch_size)
            ],
            dim=0,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: GfstDecodeState | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, cache_position, position_embeddings, kwargs
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len == 0:
            return hidden_states, None
        if past_key_values is None:
            past_key_values = GfstDecodeState(num_layers=self.num_layers, batch_size=batch_size)
        if not isinstance(past_key_values, GfstDecodeState):
            raise TypeError(f"expected GfstDecodeState, got {type(past_key_values)!r}")

        past_key_values.ensure_num_layers(self.num_layers)
        past_key_values.ensure_batch_size(batch_size)
        positions = self._resolve_positions(
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        if self.training and self.training_path_mode == "fast" and not bool(use_cache) and seq_len > 1:
            return self._training_forward_fast(
                hidden_states=hidden_states,
                positions=positions,
                past_key_values=past_key_values,
            )
        if (not self.training) and self.training_path_mode == "fast" and seq_len > 1:
            return self._prefill_forward_fast(
                hidden_states=hidden_states,
                positions=positions,
                past_key_values=past_key_values,
            )

        return self._tokenwise_forward(
            hidden_states=hidden_states,
            positions=positions,
            past_key_values=past_key_values,
        )


GfstHmbAttention = GfstHmbSublayer


class _SimpleRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, *, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(tensor.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (tensor * inv_rms.to(dtype=tensor.dtype)) * self.weight


def _make_rms_norm(
    hidden_size: int,
    *,
    eps: float = 1e-6,
    backend: str = "native",
    device=None,
    dtype=None,
) -> nn.Module:
    backend_str = str(backend)
    if backend_str not in {"native", "liger"}:
        raise ValueError("rms norm backend must be one of {'native', 'liger'}")
    if backend_str == "liger":
        liger_cls = get_liger_rms_norm_cls()
        if liger_cls is not None:
            norm = liger_cls(int(hidden_size), eps=float(eps))
            if device is not None or dtype is not None:
                norm = norm.to(device=device, dtype=dtype)
            return norm
    return _SimpleRMSNorm(int(hidden_size), eps=float(eps), device=device, dtype=dtype)


def _resolve_rms_norm_eps(norm: nn.Module) -> float:
    if hasattr(norm, "eps"):
        return float(getattr(norm, "eps"))
    if hasattr(norm, "variance_epsilon"):
        return float(getattr(norm, "variance_epsilon"))
    return 1e-6


def _swap_rms_norm_backend(norm: nn.Module, backend: str) -> nn.Module:
    if isinstance(norm, nn.LayerNorm):
        return norm
    weight = getattr(norm, "weight", None)
    if weight is None:
        return norm
    replacement = _make_rms_norm(
        int(weight.shape[0]),
        eps=_resolve_rms_norm_eps(norm),
        backend=backend,
        device=weight.device,
        dtype=weight.dtype,
    )
    replacement.train(norm.training)
    replacement_weight = getattr(replacement, "weight", None)
    if replacement_weight is not None:
        with torch.no_grad():
            replacement_weight.copy_(weight.detach().to(device=replacement_weight.device, dtype=replacement_weight.dtype))
        replacement_weight.requires_grad_(bool(weight.requires_grad))
    return replacement


class _LowRankBridge(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, *, device=None, dtype=None) -> None:
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_dim, bias=False, device=device, dtype=dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(hidden)))


class _CausalDepthwiseBranch(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_dim: int,
        *,
        kernel_size: int,
        dilation: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.out_dim = int(out_dim)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.buffer_len = int((self.kernel_size - 1) * self.dilation)
        self.conv = nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=self.hidden_size,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.proj = nn.Linear(self.hidden_size, self.out_dim, bias=False, device=device, dtype=dtype)

    def allocate_buffer(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.buffer_len <= 0:
            return None
        return torch.zeros(batch_size, self.hidden_size, self.buffer_len, device=device, dtype=dtype)

    def forward_sequence(self, hidden: torch.Tensor) -> torch.Tensor:
        signal = hidden.transpose(1, 2)
        if self.buffer_len > 0:
            signal = F.pad(signal, (self.buffer_len, 0))
        convolved = self.conv(signal).transpose(1, 2)
        return self.proj(convolved)

    def forward_step(self, hidden_t: torch.Tensor, buffer: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        signal_t = hidden_t.unsqueeze(-1)
        if self.buffer_len > 0:
            if buffer is None:
                buffer = self.allocate_buffer(int(hidden_t.shape[0]), device=hidden_t.device, dtype=hidden_t.dtype)
            signal = torch.cat((buffer, signal_t), dim=-1)
            next_buffer = signal[:, :, -self.buffer_len :]
        else:
            signal = signal_t
            next_buffer = None
        convolved = self.conv(signal).squeeze(-1)
        return self.proj(convolved), next_buffer

    def buffer_from_sequence(self, hidden: torch.Tensor) -> torch.Tensor | None:
        if self.buffer_len <= 0:
            return None
        signal = hidden.transpose(1, 2)
        seq_len = int(signal.shape[-1])
        if seq_len >= self.buffer_len:
            return signal[:, :, -self.buffer_len :].contiguous()
        padded = torch.zeros(
            int(signal.shape[0]),
            int(signal.shape[1]),
            self.buffer_len,
            device=signal.device,
            dtype=signal.dtype,
        )
        padded[:, :, -seq_len:] = signal
        return padded


class MpucGfstHmbBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        layer_idx: int,
        gfst_config: GfstHmbConfig,
        retrieval_config: FaissGpuRetrievalConfig,
        rope_theta: float,
        retriever: FaissGpuRetriever,
        mpuc_cfg: dict[str, Any],
        norm_cls: Callable[..., nn.Module],
        residual_in_fp32: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = int(layer_idx)
        self.residual_in_fp32 = bool(residual_in_fp32)
        self.norm = norm_cls(hidden_size, **factory_kwargs)
        self.hidden_size = int(hidden_size)
        self.fast_dim = int(hidden_size // 2)
        self.mid_dim = int(hidden_size // 4)
        self.slow_dim = int(hidden_size - self.fast_dim - self.mid_dim)
        self.cross_rank = int(dict(mpuc_cfg).get("cross_rank", max(64, hidden_size // 16)))
        self.semantic_rank = int(dict(mpuc_cfg).get("semantic_rank", max(64, hidden_size // 16)))
        branch_specs = tuple(dict(mpuc_cfg).get("branch_specs", ((3, 1), (5, 2), (9, 4))))
        if len(branch_specs) != 3:
            raise ValueError("mpuc_cfg.branch_specs must contain exactly three (kernel, dilation) pairs")
        (fast_spec, mid_spec, slow_spec) = branch_specs
        self.fast_branch = _CausalDepthwiseBranch(hidden_size, self.fast_dim, kernel_size=int(fast_spec[0]), dilation=int(fast_spec[1]), **factory_kwargs)
        self.mid_branch = _CausalDepthwiseBranch(hidden_size, self.mid_dim, kernel_size=int(mid_spec[0]), dilation=int(mid_spec[1]), **factory_kwargs)
        self.slow_branch = _CausalDepthwiseBranch(hidden_size, self.slow_dim, kernel_size=int(slow_spec[0]), dilation=int(slow_spec[1]), **factory_kwargs)
        self.fast_from_mid = _LowRankBridge(self.mid_dim, self.fast_dim, self.cross_rank, **factory_kwargs)
        self.fast_from_slow = _LowRankBridge(self.slow_dim, self.fast_dim, self.cross_rank, **factory_kwargs)
        self.mid_from_fast = _LowRankBridge(self.fast_dim, self.mid_dim, self.cross_rank, **factory_kwargs)
        self.mid_from_slow = _LowRankBridge(self.slow_dim, self.mid_dim, self.cross_rank, **factory_kwargs)
        self.slow_from_fast = _LowRankBridge(self.fast_dim, self.slow_dim, self.cross_rank, **factory_kwargs)
        self.slow_from_mid = _LowRankBridge(self.mid_dim, self.slow_dim, self.cross_rank, **factory_kwargs)
        self.fast_gate_proj = nn.Linear(self.fast_dim, self.fast_dim, **factory_kwargs)
        self.fast_cand_proj = nn.Linear(self.fast_dim, self.fast_dim, **factory_kwargs)
        self.mid_gate_proj = nn.Linear(self.mid_dim, self.mid_dim, **factory_kwargs)
        self.mid_cand_proj = nn.Linear(self.mid_dim, self.mid_dim, **factory_kwargs)
        self.slow_gate_proj = nn.Linear(self.slow_dim, self.slow_dim, **factory_kwargs)
        self.slow_cand_proj = nn.Linear(self.slow_dim, self.slow_dim, **factory_kwargs)
        boundary_hidden = max(8, hidden_size // 16)
        self.boundary_mlp = nn.Sequential(
            nn.Linear(4, boundary_hidden, **factory_kwargs),
            nn.SiLU(),
            nn.Linear(boundary_hidden, 1, **factory_kwargs),
        )
        self.semantic_in = nn.Linear(hidden_size, self.semantic_rank, bias=False, **factory_kwargs)
        self.semantic_cond = nn.Linear(self.mid_dim + self.slow_dim, self.semantic_rank, bias=False, **factory_kwargs)
        self.semantic_out = nn.Linear(self.semantic_rank, hidden_size, bias=False, **factory_kwargs)
        self.fusion_proj = nn.Linear(hidden_size, hidden_size, bias=False, **factory_kwargs)
        self.gfst = GfstHmbSublayer(
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_idx=layer_idx,
            gfst_config=gfst_config,
            rope_theta=rope_theta,
            retrieval_config=retrieval_config,
            retriever=retriever,
        )
        self.recurrent_kernel_backend = "native"
        self.recurrent_compile_mode = "max-autotune-no-cudagraphs"
        self.runtime_profiler_mode = "off"
        self._runtime_profile_recorder: RuntimeProfileRecorder | None = None
        self._compiled_forward_sequence_impl: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] | None = None
        self.last_forward_stats: dict[str, float] | None = None
        self.last_boundary_logits: torch.Tensor | None = None

    def _normalise(self, residual: torch.Tensor) -> torch.Tensor:
        weight = getattr(self.norm, "weight", None)
        norm_dtype = residual.dtype if weight is None else weight.dtype
        return self.norm(residual.to(dtype=norm_dtype))

    def _zero_state(self, batch_size: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, dim, device=device, dtype=dtype)

    def _update_band(
        self,
        prev: torch.Tensor,
        drive: torch.Tensor,
        *,
        gate_proj: nn.Linear,
        cand_proj: nn.Linear,
        gate_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(gate_proj(drive))
        if gate_scale is not None:
            gate = gate * gate_scale
        cand = torch.tanh(cand_proj(drive))
        return ((1.0 - gate) * prev) + (gate * cand), gate

    def _semantic_residual(self, token_state: torch.Tensor, mid_state: torch.Tensor, slow_state: torch.Tensor) -> torch.Tensor:
        base = self.semantic_in(token_state)
        cond = torch.sigmoid(self.semantic_cond(torch.cat((mid_state, slow_state), dim=-1)))
        return self.semantic_out(base * cond)

    def _token_step(
        self,
        *,
        token_state: torch.Tensor,
        fast_drive: torch.Tensor,
        mid_drive: torch.Tensor,
        slow_drive: torch.Tensor,
        fast_prev: torch.Tensor,
        mid_prev: torch.Tensor,
        slow_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast_state, fast_gate = self._update_band(
            fast_prev,
            fast_drive,
            gate_proj=self.fast_gate_proj,
            cand_proj=self.fast_cand_proj,
        )
        mid_drive_total = mid_drive + self.mid_from_fast(fast_state) + self.mid_from_slow(slow_prev)
        mid_state, mid_gate = self._update_band(
            mid_prev,
            mid_drive_total,
            gate_proj=self.mid_gate_proj,
            cand_proj=self.mid_cand_proj,
        )
        slow_drive_total = slow_drive + self.slow_from_fast(fast_state) + self.slow_from_mid(mid_state)
        boundary_features = torch.cat(
            (
                fast_gate.mean(dim=-1, keepdim=True),
                mid_gate.mean(dim=-1, keepdim=True),
                torch.sqrt((mid_drive_total - mid_prev).float().pow(2).mean(dim=-1, keepdim=True) + 1e-6),
                torch.sqrt((slow_drive_total - slow_prev).float().pow(2).mean(dim=-1, keepdim=True) + 1e-6),
            ),
            dim=-1,
        )
        boundary_features = boundary_features.to(dtype=self.boundary_mlp[0].weight.dtype)
        boundary_logit = self.boundary_mlp(boundary_features).squeeze(-1)
        slow_state, _slow_gate = self._update_band(
            slow_prev,
            slow_drive_total,
            gate_proj=self.slow_gate_proj,
            cand_proj=self.slow_cand_proj,
            gate_scale=torch.sigmoid(boundary_logit).unsqueeze(-1),
        )
        fused = self.fusion_proj(torch.cat((fast_state, mid_state, slow_state), dim=-1)) + self._semantic_residual(
            token_state,
            mid_state,
            slow_state,
        )
        return fused, fast_state, mid_state, slow_state, boundary_logit, fast_gate

    def _forward_sequence(
        self,
        normalised: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, _CoreLayerCacheState]:
        with runtime_range(
            "backbone.recurrent_sequence",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=normalised.device,
        ):
            if str(self.recurrent_kernel_backend) == "sm120_cutlass":
                accelerated = sm120_mpuc_recurrent_scan(
                    normalised,
                    self.fast_branch.forward_sequence(normalised),
                    self.mid_branch.forward_sequence(normalised),
                    self.slow_branch.forward_sequence(normalised),
                )
                if accelerated is not None:
                    outputs, boundary_logits, fast_state, mid_state, slow_state = accelerated
                else:
                    outputs, boundary_logits, fast_state, mid_state, slow_state = self._compiled_forward_sequence(normalised)
            elif str(self.recurrent_kernel_backend) == "compiled":
                outputs, boundary_logits, fast_state, mid_state, slow_state = self._compiled_forward_sequence(normalised)
            else:
                outputs, boundary_logits, fast_state, mid_state, slow_state = self._forward_sequence_tensors(normalised)
        self.last_boundary_logits = boundary_logits
        next_state = _CoreLayerCacheState(
            fast_state=fast_state.detach(),
            mid_state=mid_state.detach(),
            slow_state=slow_state.detach(),
            fast_buffer=self.fast_branch.buffer_from_sequence(normalised),
            mid_buffer=self.mid_branch.buffer_from_sequence(normalised),
            slow_buffer=self.slow_branch.buffer_from_sequence(normalised),
        )
        return outputs, self.last_boundary_logits, next_state

    def _forward_sequence_tensors(
        self,
        normalised: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = normalised.shape
        fast_inputs = self.fast_branch.forward_sequence(normalised)
        mid_inputs = self.mid_branch.forward_sequence(normalised)
        slow_inputs = self.slow_branch.forward_sequence(normalised)
        fast_state = self._zero_state(batch_size, self.fast_dim, device=normalised.device, dtype=normalised.dtype)
        mid_state = self._zero_state(batch_size, self.mid_dim, device=normalised.device, dtype=normalised.dtype)
        slow_state = self._zero_state(batch_size, self.slow_dim, device=normalised.device, dtype=normalised.dtype)
        outputs = normalised.new_empty((batch_size, seq_len, self.hidden_size))
        boundary_logits = normalised.new_empty((batch_size, seq_len))
        for token_idx in range(seq_len):
            fast_drive = fast_inputs[:, token_idx, :] + self.fast_from_mid(mid_state) + self.fast_from_slow(slow_state)
            fused, fast_state, mid_state, slow_state, boundary_logit, fast_gate = self._token_step(
                token_state=normalised[:, token_idx, :],
                fast_drive=fast_drive,
                mid_drive=mid_inputs[:, token_idx, :],
                slow_drive=slow_inputs[:, token_idx, :],
                fast_prev=fast_state,
                mid_prev=mid_state,
                slow_prev=slow_state,
            )
            outputs[:, token_idx, :] = fused
            boundary_logits[:, token_idx] = boundary_logit
            del fast_gate
        return outputs, boundary_logits, fast_state, mid_state, slow_state

    def _compiled_forward_sequence(
        self,
        normalised: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            return self._forward_sequence_tensors(normalised)
        if self._compiled_forward_sequence_impl is None:
            self._compiled_forward_sequence_impl = compile_fn(
                self._forward_sequence_tensors,
                fullgraph=False,
                dynamic=False,
                mode=str(self.recurrent_compile_mode),
            )
        return self._compiled_forward_sequence_impl(normalised)

    def set_recurrent_kernel_backend(self, backend: str, *, compile_mode: str | None = None) -> None:
        backend_str = str(backend)
        if backend_str not in {"native", "compiled", "sm120_cutlass"}:
            raise ValueError("recurrent kernel backend must be one of {'native', 'compiled', 'sm120_cutlass'}")
        if compile_mode is not None:
            compile_mode_str = str(compile_mode)
            if not compile_mode_str:
                raise ValueError("compile_mode must not be empty")
            if compile_mode_str != self.recurrent_compile_mode:
                self._compiled_forward_sequence_impl = None
            self.recurrent_compile_mode = compile_mode_str
        if backend_str != self.recurrent_kernel_backend:
            self._compiled_forward_sequence_impl = None
        self.recurrent_kernel_backend = backend_str

    def set_runtime_profiler_mode(self, mode: str) -> None:
        mode_str = str(mode)
        if mode_str not in {"off", "nvtx", "benchmark"}:
            raise ValueError("runtime profiler mode must be one of {'off', 'nvtx', 'benchmark'}")
        self.runtime_profiler_mode = mode_str
        self.gfst.set_runtime_profiler_mode(mode_str)

    def set_runtime_profile_recorder(self, recorder: RuntimeProfileRecorder | None) -> None:
        self._runtime_profile_recorder = recorder
        self.gfst.set_runtime_profile_recorder(recorder)

    def _forward_cached(
        self,
        normalised: torch.Tensor,
        *,
        core_state: _CoreLayerCacheState | None,
    ) -> tuple[torch.Tensor, torch.Tensor, _CoreLayerCacheState]:
        if int(normalised.shape[1]) != 1:
            raise ValueError("cached MPUC decode requires seq_len == 1")
        batch_size = int(normalised.shape[0])
        token_state = normalised[:, 0, :]
        fast_prev = self._zero_state(batch_size, self.fast_dim, device=normalised.device, dtype=normalised.dtype) if core_state is None or core_state.fast_state is None else core_state.fast_state.to(device=normalised.device, dtype=normalised.dtype)
        mid_prev = self._zero_state(batch_size, self.mid_dim, device=normalised.device, dtype=normalised.dtype) if core_state is None or core_state.mid_state is None else core_state.mid_state.to(device=normalised.device, dtype=normalised.dtype)
        slow_prev = self._zero_state(batch_size, self.slow_dim, device=normalised.device, dtype=normalised.dtype) if core_state is None or core_state.slow_state is None else core_state.slow_state.to(device=normalised.device, dtype=normalised.dtype)
        fast_input, fast_buffer = self.fast_branch.forward_step(token_state, None if core_state is None else core_state.fast_buffer)
        mid_input, mid_buffer = self.mid_branch.forward_step(token_state, None if core_state is None else core_state.mid_buffer)
        slow_input, slow_buffer = self.slow_branch.forward_step(token_state, None if core_state is None else core_state.slow_buffer)
        fast_drive = fast_input + self.fast_from_mid(mid_prev) + self.fast_from_slow(slow_prev)
        fused, fast_state, mid_state, slow_state, boundary_logit, fast_gate = self._token_step(
            token_state=token_state,
            fast_drive=fast_drive,
            mid_drive=mid_input,
            slow_drive=slow_input,
            fast_prev=fast_prev,
            mid_prev=mid_prev,
            slow_prev=slow_prev,
        )
        self.last_boundary_logits = boundary_logit.unsqueeze(1)
        next_state = _CoreLayerCacheState(
            fast_state=fast_state.detach(),
            mid_state=mid_state.detach(),
            slow_state=slow_state.detach(),
            fast_buffer=None if fast_buffer is None else fast_buffer.detach(),
            mid_buffer=None if mid_buffer is None else mid_buffer.detach(),
            slow_buffer=None if slow_buffer is None else slow_buffer.detach(),
        )
        return fused.unsqueeze(1), self.last_boundary_logits, next_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        *,
        past_key_values: GfstDecodeState,
        core_state: _CoreLayerCacheState | None,
        use_cache: bool,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, _CoreLayerCacheState | None]:
        residual = (hidden_states + residual) if residual is not None else hidden_states
        normalised = self._normalise(residual)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        if bool(use_cache) and int(normalised.shape[1]) == 1:
            tilde_hidden, boundary_logits, next_state = self._forward_cached(normalised, core_state=core_state)
        else:
            tilde_hidden, boundary_logits, sequence_state = self._forward_sequence(normalised)
            next_state = sequence_state if bool(use_cache) else None
        delta, _ = self.gfst(
            hidden_states=tilde_hidden,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=bool(use_cache),
        )
        gfst_stats = self.gfst.last_forward_stats or {}
        self.last_forward_stats = {
            "backbone_residual_rms": _tensor_rms(tilde_hidden),
            "gfst_branch_rms": float(gfst_stats.get("gfst_branch_rms", 0.0)),
            "gate_activation_mean": float(gfst_stats.get("gate_activation_mean", 0.0)),
            "gate_activation_std": float(gfst_stats.get("gate_activation_std", 0.0)),
            "boundary_activation_mean": float(torch.sigmoid(boundary_logits.detach().float()).mean().item()),
        }
        return tilde_hidden + delta, residual, boundary_logits, next_state

    def allocate_inference_cache(self, batch_size, max_seqlen=None, dtype=None):
        del max_seqlen
        device = self.norm.weight.device
        resolved_dtype = dtype if dtype is not None else self.norm.weight.dtype
        return _CoreLayerCacheState(
            fast_state=self._zero_state(int(batch_size), self.fast_dim, device=device, dtype=resolved_dtype),
            mid_state=self._zero_state(int(batch_size), self.mid_dim, device=device, dtype=resolved_dtype),
            slow_state=self._zero_state(int(batch_size), self.slow_dim, device=device, dtype=resolved_dtype),
            fast_buffer=self.fast_branch.allocate_buffer(int(batch_size), device=device, dtype=resolved_dtype),
            mid_buffer=self.mid_branch.allocate_buffer(int(batch_size), device=device, dtype=resolved_dtype),
            slow_buffer=self.slow_branch.allocate_buffer(int(batch_size), device=device, dtype=resolved_dtype),
        )


class _MpucBackbone(nn.Module):
    def __init__(self, config: MpucGfstHmbConfig, retriever: FaissGpuRetriever, *, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        norm_cls: Callable[..., nn.Module]
        if config.rms_norm:
            norm_cls = lambda hidden_size, **kwargs: _make_rms_norm(hidden_size, eps=1e-6, backend="native", **kwargs)
        else:
            norm_cls = nn.LayerNorm
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)
        self.layers = nn.ModuleList(
            [
                MpucGfstHmbBlock(
                    hidden_size=config.d_model,
                    num_layers=config.n_layer,
                    layer_idx=layer_idx,
                    gfst_config=config.gfst_hmb_config,
                    retrieval_config=config.retrieval_config,
                    rope_theta=config.rope_theta,
                    retriever=retriever,
                    mpuc_cfg=config.mpuc_cfg,
                    norm_cls=norm_cls,
                    residual_in_fp32=config.residual_in_fp32,
                    **factory_kwargs,
                )
                for layer_idx in range(config.n_layer)
            ]
        )
        self.norm_f = norm_cls(config.d_model, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen=None, dtype=None):
        return {
            layer_idx: layer.allocate_inference_cache(batch_size, max_seqlen=max_seqlen, dtype=dtype)
            for layer_idx, layer in enumerate(self.layers)
        }


class MpucGfstHmbForCausalLM(nn.Module):
    def __init__(self, config: MpucGfstHmbConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config
        self.gfst_hmb_config = config.gfst_hmb_config
        self.retrieval_config = config.retrieval_config
        self.retriever = FaissGpuRetriever(self.retrieval_config)
        self.backbone = _MpucBackbone(config, self.retriever, device=device, dtype=dtype)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, device=device, dtype=dtype)
        self.exact_readout = ExactRecallReadout(
            hidden_size=config.d_model,
            key_dim=config.gfst_hmb_config.key_dim,
            router_hidden_dim=config.gfst_hmb_config.router_hidden_dim,
            centroid_heads=config.gfst_hmb_config.centroid_heads,
            centroid_key_dim=config.gfst_hmb_config.centroid_key_dim,
            centroid_value_dim=config.gfst_hmb_config.centroid_value_dim,
            cells_per_block=config.gfst_hmb_config.cells_per_block,
            centroid_top_cells_per_head=config.gfst_hmb_config.centroid_top_cells_per_head,
            centroid_selection_policy=config.gfst_hmb_config.centroid_selection_policy,
            centroid_fdr_q=config.gfst_hmb_config.centroid_fdr_q,
            centroid_score_normalization=config.gfst_hmb_config.centroid_score_normalization,
            centroid_background_subtract=config.gfst_hmb_config.centroid_background_subtract,
            centroid_count_penalty=config.gfst_hmb_config.centroid_count_penalty,
            centroid_self_field_weight=config.gfst_hmb_config.centroid_self_field_weight,
            centroid_min_selected_cells_per_head=config.gfst_hmb_config.centroid_min_selected_cells_per_head,
            centroid_selection_temperature=config.gfst_hmb_config.centroid_selection_temperature,
            centroid_gain_entropy_low=config.gfst_hmb_config.centroid_gain_entropy_low,
            centroid_gain_entropy_high=config.gfst_hmb_config.centroid_gain_entropy_high,
            centroid_gain_margin_low=config.gfst_hmb_config.centroid_gain_margin_low,
            centroid_gain_margin_high=config.gfst_hmb_config.centroid_gain_margin_high,
            pointer_value_source=config.gfst_hmb_config.pointer_value_source,
            centroid_recall_value_source=config.gfst_hmb_config.centroid_recall_value_source,
            device=device,
            dtype=dtype,
        )
        self.gradient_checkpointing = False
        self.gfst_feature_backend = "native"
        self.recurrent_kernel_backend = "native"
        self.exact_readout_score_backend = "native"
        self.segmented_reduction_backend = "native"
        self.state_pack_backend = "native"
        self.runtime_profiler_mode = "off"
        self.rms_norm_backend = "native"
        self.use_liger_losses = False
        self._last_forward_signal_stats: dict[str, float] | None = None
        self._last_runtime_profile: dict[str, Any] | None = None
        self._lm_head_copy_grad_hook = None

    @classmethod
    def from_pretrained_teacher(
        cls,
        teacher_model_name_or_path: str,
        *,
        gfst_hmb: dict[str, Any] | GfstHmbConfig | None = None,
        retrieval: dict[str, Any] | FaissGpuRetrievalConfig | None = None,
        copy_token_id: int | None = None,
        **kwargs: Any,
    ) -> "MpucGfstHmbForCausalLM":
        if AutoModelForCausalLM is None:
            raise RuntimeError("transformers is required to load the Qwen teacher")
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name_or_path, **kwargs)
        config = MpucGfstHmbConfig.from_qwen_config(
            teacher.config,
            gfst_hmb=gfst_hmb,
            retrieval=retrieval,
            teacher_model_name=str(teacher_model_name_or_path),
            copy_token_id=copy_token_id,
        )
        torch_dtype = kwargs.get("dtype", kwargs.get("torch_dtype", None))
        model = cls(config=config, dtype=torch_dtype)
        model.initialise_from_teacher(teacher)
        del teacher
        return model

    def initialise_from_teacher(self, teacher: nn.Module) -> None:
        teacher_model = getattr(teacher, "model", None)
        if teacher_model is None:
            return
        with torch.no_grad():
            copy_source_id = int(self.config.pad_token_id if self.config.pad_token_id >= 0 else self.config.eos_token_id)
            if hasattr(teacher_model, "embed_tokens"):
                teacher_embed = teacher_model.embed_tokens.weight
                row_count = min(int(teacher_embed.shape[0]), int(self.backbone.embedding.weight.shape[0]))
                self.backbone.embedding.weight[:row_count].copy_(teacher_embed[:row_count])
                if int(self.config.copy_token_id) < int(self.backbone.embedding.weight.shape[0]):
                    source_index = min(int(copy_source_id), int(teacher_embed.shape[0]) - 1)
                    self.backbone.embedding.weight[int(self.config.copy_token_id)].copy_(teacher_embed[source_index])
            if hasattr(teacher, "lm_head"):
                teacher_lm_head = teacher.lm_head.weight
                row_count = min(int(teacher_lm_head.shape[0]), int(self.lm_head.weight.shape[0]))
                self.lm_head.weight[:row_count].copy_(teacher_lm_head[:row_count])
                if int(self.config.copy_token_id) < int(self.lm_head.weight.shape[0]):
                    source_index = min(int(copy_source_id), int(teacher_lm_head.shape[0]) - 1)
                    self.lm_head.weight[int(self.config.copy_token_id)].copy_(teacher_lm_head[source_index])
            if hasattr(teacher_model, "norm") and hasattr(self.backbone.norm_f, "weight"):
                if tuple(teacher_model.norm.weight.shape) == tuple(self.backbone.norm_f.weight.shape):
                    self.backbone.norm_f.weight.copy_(teacher_model.norm.weight)
            if hasattr(teacher_model, "layers"):
                for student_block, teacher_layer in zip(self.backbone.layers, teacher_model.layers):
                    if hasattr(teacher_layer, "input_layernorm") and hasattr(student_block.norm, "weight"):
                        if tuple(student_block.norm.weight.shape) == tuple(teacher_layer.input_layernorm.weight.shape):
                            student_block.norm.weight.copy_(teacher_layer.input_layernorm.weight)

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        migrated = dict(state_dict)
        target_state = self.state_dict()
        new_init_suffixes = (
            "cell_assign_proj.weight",
            "cell_key_proj.weight",
            "cell_value_proj.weight",
            "router_query_norm.weight",
            "router_query_norm.bias",
            "router_in_proj.weight",
            "router_in_proj.bias",
            "router_out_proj.weight",
            "router_out_proj.bias",
            "centroid_phase_proj.weight",
            "centroid_query_proj.weight",
            "centroid_token_proj.weight",
            "centroid_gamma_m",
            "centroid_gamma_r",
            "centroid_usage_ema",
        )
        for key, value in list(migrated.items()):
            target_value = target_state.get(key)
            if target_value is None:
                migrated.pop(key, None)
                continue
            if key in {"backbone.embedding.weight", "lm_head.weight"} and value.ndim == 2 and target_value.shape[0] == value.shape[0] + 1:
                expanded = target_value.detach().clone()
                expanded.zero_()
                expanded[: value.shape[0]] = value.to(dtype=expanded.dtype)
                source_index = min(int(self.config.pad_token_id if self.config.pad_token_id >= 0 else self.config.eos_token_id), int(value.shape[0]) - 1)
                expanded[int(self.config.copy_token_id)] = value[source_index].to(dtype=expanded.dtype)
                migrated[key] = expanded
        for key, target_value in target_state.items():
            if key in migrated:
                continue
            if key.endswith(new_init_suffixes):
                migrated[key] = target_value.detach().clone()
        return super().load_state_dict(migrated, strict=strict)

    def named_gfst_parameters(self):
        for name, parameter in self.named_parameters():
            if ".gfst." in name:
                yield name, parameter

    def freeze_backbone_except_gfst(self) -> None:
        self._set_lm_head_copy_only_training(False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for _, parameter in self.named_gfst_parameters():
            parameter.requires_grad_(True)

    def freeze_gfst_keep_backbone(self) -> None:
        self._set_lm_head_copy_only_training(False)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(".gfst." not in name)

    def freeze_all_except_exact_readout_lm_head(self) -> None:
        self._set_lm_head_copy_only_training(False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if name.startswith("exact_readout.") or name.startswith("lm_head."):
                parameter.requires_grad_(True)

    def _set_lm_head_copy_only_training(self, enabled: bool) -> None:
        hook = getattr(self, "_lm_head_copy_grad_hook", None)
        if hook is not None:
            hook.remove()
            self._lm_head_copy_grad_hook = None
        if not bool(enabled):
            return
        copy_token_id = int(self.config.copy_token_id)

        def _mask_copy_row_only(grad: torch.Tensor) -> torch.Tensor:
            masked = torch.zeros_like(grad)
            if 0 <= copy_token_id < int(masked.shape[0]):
                masked[copy_token_id] = grad[copy_token_id]
            return masked

        self._lm_head_copy_grad_hook = self.lm_head.weight.register_hook(_mask_copy_row_only)

    def freeze_all_except_exact_readout_only(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if name.startswith("exact_readout.") or name == "lm_head.weight":
                parameter.requires_grad_(True)
        self._set_lm_head_copy_only_training(True)

    def freeze_all_except_exact_readout_strict(self) -> None:
        self._set_lm_head_copy_only_training(False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if name.startswith("exact_readout."):
                parameter.requires_grad_(True)

    def freeze_all_except_exact_readout_plus_query(self) -> None:
        self._set_lm_head_copy_only_training(False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if name.startswith("exact_readout."):
                parameter.requires_grad_(True)
        readout_layer = self._resolve_copy_readout_layer()
        readout_layer.gfst.query_proj.weight.requires_grad_(True)

    def freeze_all_except_exact_readout_plus_query_stack(self) -> None:
        self._set_lm_head_copy_only_training(False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if name.startswith("exact_readout."):
                parameter.requires_grad_(True)
        readout_layer = self._resolve_copy_readout_layer()
        readout_layer.gfst.reduce_proj.weight.requires_grad_(True)
        readout_layer.gfst.query_proj.weight.requires_grad_(True)

    def freeze_all_except_exact_readout_plus_query_stack_no_router(self) -> None:
        self.freeze_all_except_exact_readout_plus_query_stack()
        for name, parameter in self.named_parameters():
            if name.startswith("exact_readout.router_"):
                parameter.requires_grad_(False)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def set_gfst_training_path(self, mode: str) -> None:
        mode_str = str(mode)
        if mode_str not in {"fast", "exact"}:
            raise ValueError("gfst training path must be one of {'fast', 'exact'}")
        for layer in self.backbone.layers:
            if hasattr(layer, "gfst") and isinstance(layer.gfst, GfstHmbSublayer):
                layer.gfst.training_path_mode = mode_str

    def set_gfst_debug_capture(self, enabled: bool) -> None:
        for layer in self.backbone.layers:
            if hasattr(layer, "gfst") and isinstance(layer.gfst, GfstHmbSublayer):
                layer.gfst.set_debug_capture(enabled)

    def reset_temp_local_qr_capture_pool(self, *, limit: int) -> None:
        _temp_local_qr_pool_reset(limit=int(limit))

    def get_temp_local_qr_capture_pool(self) -> list[dict[str, Any]]:
        return _temp_local_qr_pool_records()

    def set_disable_gfst_residual(self, enabled: bool) -> None:
        for layer in self.backbone.layers:
            if hasattr(layer, "gfst") and isinstance(layer.gfst, GfstHmbSublayer):
                layer.gfst.set_disable_residual(enabled)

    def set_gfst_feature_backend(self, backend: str) -> None:
        backend_str = str(backend)
        if backend_str not in {"native", "sm120_exact"}:
            raise ValueError("gfst feature backend must be one of {'native', 'sm120_exact'}")
        self.gfst_feature_backend = backend_str
        for layer in self.backbone.layers:
            if hasattr(layer, "gfst") and isinstance(layer.gfst, GfstHmbSublayer):
                layer.gfst.set_gfst_feature_backend(backend_str)

    def set_gfst_eval_router_mode(
        self,
        mode: str,
        *,
        shortlist_depth: int | None = None,
        facet_count: int = 4,
        facet_aggregation: str = "max",
    ) -> None:
        for layer in self.backbone.layers:
            if hasattr(layer, "gfst") and isinstance(layer.gfst, GfstHmbSublayer):
                layer.gfst.set_eval_router_mode(
                    mode,
                    shortlist_depth=shortlist_depth,
                    facet_count=facet_count,
                    facet_aggregation=facet_aggregation,
                )

    def set_force_store_exact_hidden_states(self, enabled: bool) -> None:
        for layer in self.backbone.layers:
            if hasattr(layer, "gfst") and isinstance(layer.gfst, GfstHmbSublayer):
                layer.gfst.set_force_store_exact_hidden_states(enabled)

    def set_force_centroid_activation(self, enabled: bool) -> None:
        for layer in self.backbone.layers:
            if hasattr(layer, "gfst") and isinstance(layer.gfst, GfstHmbSublayer):
                layer.gfst.set_force_centroid_activation(enabled)

    def debug_recompute_generation_logits_fp32(
        self,
        hidden_states: torch.Tensor,
        *,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        hidden_slice = hidden_states[:, slice_indices, :].to(dtype=torch.float32)
        weight = self.lm_head.weight.detach().to(dtype=torch.float32)
        bias = None if self.lm_head.bias is None else self.lm_head.bias.detach().to(dtype=torch.float32)
        return F.linear(hidden_slice, weight, bias).float()

    def set_centroid_recovery_control(self, control: CentroidRecoveryControl | None) -> None:
        for layer in self.backbone.layers:
            if hasattr(layer, "gfst") and isinstance(layer.gfst, GfstHmbSublayer):
                layer.gfst.set_centroid_recovery_control(control)

    def set_exact_readout_eval_mode(self, mode: str) -> None:
        self.exact_readout.set_eval_pointer_mode(mode)

    def set_recurrent_kernel_backend(self, backend: str, *, compile_mode: str | None = None) -> None:
        self.recurrent_kernel_backend = str(backend)
        for layer in self.backbone.layers:
            layer.set_recurrent_kernel_backend(backend, compile_mode=compile_mode)

    def set_exact_readout_grouped_reduction_backend(self, backend: str) -> None:
        self.segmented_reduction_backend = str(backend)
        self.exact_readout.set_grouped_reduction_backend(backend)

    def set_segmented_reduction_backend(self, backend: str) -> None:
        self.set_exact_readout_grouped_reduction_backend(backend)

    def set_exact_readout_score_backend(self, backend: str) -> None:
        self.exact_readout_score_backend = str(backend)
        self.exact_readout.set_exact_readout_score_backend(backend)

    def set_state_pack_backend(self, backend: str) -> None:
        backend_str = str(backend)
        if backend_str not in {"native", "cuda_packed"}:
            raise ValueError("state pack backend must be one of {'native', 'cuda_packed'}")
        self.state_pack_backend = backend_str

    def set_runtime_profiler_mode(self, mode: str) -> None:
        mode_str = str(mode)
        if mode_str not in {"off", "nvtx", "benchmark"}:
            raise ValueError("runtime profiler mode must be one of {'off', 'nvtx', 'benchmark'}")
        self.runtime_profiler_mode = mode_str
        self.exact_readout.set_runtime_profiler_mode(mode_str)
        for layer in self.backbone.layers:
            layer.set_runtime_profiler_mode(mode_str)

    def set_rms_norm_backend(self, backend: str) -> None:
        backend_str = str(backend)
        if backend_str not in {"native", "liger"}:
            raise ValueError("rms norm backend must be one of {'native', 'liger'}")
        self.rms_norm_backend = backend_str
        if not bool(self.config.rms_norm):
            return
        self.backbone.norm_f = _swap_rms_norm_backend(self.backbone.norm_f, backend_str)
        for layer in self.backbone.layers:
            layer.norm = _swap_rms_norm_backend(layer.norm, backend_str)

    @property
    def last_forward_signal_stats(self) -> dict[str, float] | None:
        return None if self._last_forward_signal_stats is None else dict(self._last_forward_signal_stats)

    @property
    def last_runtime_profile(self) -> dict[str, Any] | None:
        return None if self._last_runtime_profile is None else dict(self._last_runtime_profile)

    def assert_canonical_backbone(self) -> None:
        for layer in self.backbone.layers:
            if not isinstance(layer, MpucGfstHmbBlock):
                raise RuntimeError(f"unexpected non-MPUC GFST block: {type(layer)!r}")

    def assert_no_standard_attention(self) -> None:
        for module in self.modules():
            if hasattr(module, "self_attn"):
                raise RuntimeError("standard transformer attention should not be present in the canonical MPUC student")

    def _resolve_position_ids(
        self,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        position_ids: torch.LongTensor | None,
        past_key_values: GfstDecodeState,
    ) -> torch.LongTensor:
        if position_ids is not None:
            if position_ids.ndim == 1:
                return position_ids.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=torch.long)
            return position_ids.to(device=device, dtype=torch.long)
        return torch.stack(
            [
                torch.arange(
                    past_key_values.samples[batch_idx].seen_tokens,
                    past_key_values.samples[batch_idx].seen_tokens + seq_len,
                    device=device,
                    dtype=torch.long,
                )
                for batch_idx in range(batch_size)
            ],
            dim=0,
        )

    def _resolve_copy_readout_layer(self) -> MpucGfstHmbBlock:
        layer_index = int(self.gfst_hmb_config.copy_readout_layer)
        layers = self.backbone.layers
        if layer_index < 0:
            layer_index = len(layers) + layer_index
        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(f"copy_readout_layer {self.gfst_hmb_config.copy_readout_layer} is out of range")
        return layers[layer_index]

    def _resolve_centroid_write_layer(self) -> MpucGfstHmbBlock:
        layer_index = int(self.gfst_hmb_config.centroid_write_layer)
        layers = self.backbone.layers
        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(f"centroid_write_layer {self.gfst_hmb_config.centroid_write_layer} is out of range")
        return layers[layer_index]

    def get_centroid_usage_ema(self) -> torch.Tensor | None:
        layer = self._resolve_centroid_write_layer()
        return getattr(layer.gfst, "centroid_usage_ema", None)

    def update_centroid_usage_ema(self, batch_mass: torch.Tensor, *, decay: float) -> None:
        ema = self.get_centroid_usage_ema()
        if ema is None:
            return
        with torch.no_grad():
            batch = batch_mass.to(device=ema.device, dtype=ema.dtype)
            total = batch.sum()
            if float(total.item()) > 0.0:
                batch = batch / total
            ema.mul_(float(decay)).add_(batch, alpha=(1.0 - float(decay)))

    def _slice_readout_trace(
        self,
        trace: _ReadoutTrace | None,
        slice_indices: slice,
    ) -> _ReadoutTrace | None:
        if trace is None:
            return None
        return _ReadoutTrace(
            queries=trace.queries[:, slice_indices, :],
            retrieved_block_ids=trace.retrieved_block_ids[:, slice_indices, :],
            retrieved_scores=trace.retrieved_scores[:, slice_indices, :],
            retrieved_prefix_lengths=trace.retrieved_prefix_lengths[:, slice_indices, :],
            coarse_retrieved_block_ids=trace.coarse_retrieved_block_ids[:, slice_indices, :],
            coarse_retrieved_scores=trace.coarse_retrieved_scores[:, slice_indices, :],
            coarse_retrieved_facet_scores=trace.coarse_retrieved_facet_scores[:, slice_indices, :, :],
        )

    def _slice_centroid_trace(
        self,
        trace: _CentroidTrace | None,
        slice_indices: slice,
    ) -> _CentroidTrace | None:
        if trace is None:
            return None
        return _CentroidTrace(
            block_ids=trace.block_ids[:, slice_indices],
            assignment_ids=trace.assignment_ids[:, slice_indices, :],
            assignment_weights=trace.assignment_weights[:, slice_indices, :],
            assignment_logits=trace.assignment_logits[:, slice_indices, :],
            token_keys=trace.token_keys[:, slice_indices, :],
            token_values=trace.token_values[:, slice_indices, :],
            valid_mask=trace.valid_mask[:, slice_indices],
        )

    def _centroid_query_heads(
        self,
        trace: _ReadoutTrace | None,
    ) -> torch.Tensor | None:
        if trace is None:
            return None
        query_dtype = self.exact_readout.centroid_query_proj.weight.dtype
        queries = trace.queries.to(dtype=query_dtype)
        projected = self.exact_readout.centroid_query_proj(queries)
        return projected.reshape(
            int(projected.shape[0]),
            int(projected.shape[1]),
            int(self.gfst_hmb_config.centroid_heads),
            int(self.gfst_hmb_config.centroid_key_dim),
        ).float()

    def _canonical_output(
        self,
        *,
        generation_logits: torch.Tensor,
        past_key_values: GfstDecodeState,
        slice_indices: slice,
        loss: torch.Tensor | None,
        hidden_states: Any,
        boundary_logits: torch.Tensor | None,
    ) -> PaperFaithfulCausalLMOutputWithPast:
        readout_layer = self._resolve_copy_readout_layer()
        centroid_layer = self._resolve_centroid_write_layer()
        readout_trace = self._slice_readout_trace(readout_layer.gfst.last_readout_trace, slice_indices)
        centroid_trace = self._slice_centroid_trace(centroid_layer.gfst.last_centroid_trace, slice_indices)
        readout_outputs = self.exact_readout(
            generation_logits=generation_logits,
            readout_trace=readout_trace,
            state=past_key_values,
            embedding_layer=self.backbone.embedding,
            copy_token_id=int(self.config.copy_token_id),
            block_len=int(self.gfst_hmb_config.block_len),
            cache_capacity=int(self.gfst_hmb_config.exact_cache_blocks),
        )
        return PaperFaithfulCausalLMOutputWithPast(
            loss=loss,
            logits=readout_outputs["logits"],
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            generation_logits=readout_outputs["generation_logits"],
            copy_decision_logits=readout_outputs["copy_decision_logits"],
            pointer_scores=readout_outputs["pointer_scores"],
            retrieval_queries=None if readout_trace is None else readout_trace.queries,
            retrieved_block_indices=readout_outputs["retrieved_block_indices"],
            retrieved_block_scores=readout_outputs["retrieved_block_scores"],
            retrieved_prefix_lengths=readout_outputs["retrieved_prefix_lengths"],
            coarse_retrieved_block_indices=readout_outputs["coarse_retrieved_block_indices"],
            coarse_retrieved_block_scores=readout_outputs["coarse_retrieved_block_scores"],
            coarse_retrieved_facet_scores=readout_outputs["coarse_retrieved_facet_scores"],
            selected_copy_token_ids=readout_outputs["selected_copy_token_ids"],
            pointer_candidate_token_ids=readout_outputs["pointer_candidate_token_ids"],
            pointer_candidate_blocks=readout_outputs["pointer_candidate_blocks"],
            pointer_candidate_positions=readout_outputs["pointer_candidate_positions"],
            pointer_candidate_mask=readout_outputs["pointer_candidate_mask"],
            router_feature_tensor=readout_outputs["router_feature_tensor"],
            centroid_block_ids=None if centroid_trace is None else centroid_trace.block_ids,
            centroid_assignment_ids=None if centroid_trace is None else centroid_trace.assignment_ids,
            centroid_assignment_weights=None if centroid_trace is None else centroid_trace.assignment_weights,
            centroid_assignment_logits=None if centroid_trace is None else centroid_trace.assignment_logits,
            centroid_token_keys=None if centroid_trace is None else centroid_trace.token_keys,
            centroid_token_values=None if centroid_trace is None else centroid_trace.token_values,
            centroid_valid_mask=None if centroid_trace is None else centroid_trace.valid_mask,
            centroid_query_heads=self._centroid_query_heads(readout_trace),
            centroid_selected_block_slots=readout_outputs["centroid_selected_block_slots"],
            centroid_selected_cell_ids=readout_outputs["centroid_selected_cell_ids"],
            centroid_selected_alphas=readout_outputs["centroid_selected_alphas"],
            centroid_selection_entropy=readout_outputs["centroid_selection_entropy"],
            centroid_selection_margin=readout_outputs["centroid_selection_margin"],
            centroid_confidence=readout_outputs["centroid_confidence"],
            centroid_selected_cells_per_head=readout_outputs["centroid_selected_cells_per_head"],
            boundary_logits=None if boundary_logits is None else boundary_logits[:, slice_indices],
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: GfstDecodeState | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> PaperFaithfulCausalLMOutputWithPast:
        del attention_mask, cache_position, kwargs
        if use_cache is None:
            use_cache = True
        if output_hidden_states is None:
            output_hidden_states = False
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if input_ids is None:
            raise ValueError("canonical exact recall requires input_ids; inputs_embeds-only forward is unsupported")

        batch_size, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        hidden_states = self.backbone.embedding(input_ids)

        effective_cache = past_key_values
        runtime_profile_recorder = RuntimeProfileRecorder() if str(self.runtime_profiler_mode) == "benchmark" else None
        if effective_cache is None:
            effective_cache = GfstDecodeState(
                num_layers=len(self.backbone.layers),
                batch_size=batch_size,
                state_pack_backend=self.state_pack_backend,
                runtime_profiler_mode=self.runtime_profiler_mode,
            )
        if not isinstance(effective_cache, GfstDecodeState):
            raise TypeError(f"expected GfstDecodeState, got {type(effective_cache)!r}")
        effective_cache.set_state_pack_backend(self.state_pack_backend)
        effective_cache.set_runtime_profiler_mode(self.runtime_profiler_mode)
        effective_cache.set_runtime_profile_recorder(runtime_profile_recorder)
        self.exact_readout.set_runtime_profile_recorder(runtime_profile_recorder)
        for layer in self.backbone.layers:
            layer.set_runtime_profile_recorder(runtime_profile_recorder)

        effective_cache.ensure_num_layers(len(self.backbone.layers))
        effective_cache.ensure_batch_size(batch_size)
        if use_cache and effective_cache.has_core_state and seq_len != 1:
            raise ValueError("incremental decode with an existing MPUC cache requires seq_len == 1")
        position_ids = self._resolve_position_ids(
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
            position_ids=position_ids,
            past_key_values=effective_cache,
        )
        for batch_idx in range(batch_size):
            effective_cache.prepare_inputs(
                batch_idx,
                positions=position_ids[batch_idx],
                token_ids=input_ids[batch_idx],
            )

        all_hidden_states = () if output_hidden_states else None
        residual = None
        next_core_states: dict[int, _CoreLayerCacheState] = {}
        layer_boundary_logits: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.backbone.layers):
            hidden_states, residual, boundary_logits, next_state = layer(
                hidden_states,
                residual,
                past_key_values=effective_cache,
                core_state=effective_cache.core_layer_states.get(layer_idx),
                use_cache=bool(use_cache),
                position_ids=position_ids,
            )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_boundary_logits.append(boundary_logits)
            if use_cache and next_state is not None:
                next_core_states[layer_idx] = next_state

        residual = (hidden_states + residual) if residual is not None else hidden_states
        norm_weight = getattr(self.backbone.norm_f, "weight", None)
        norm_dtype = hidden_states.dtype if norm_weight is None else norm_weight.dtype
        hidden_states = self.backbone.norm_f(residual.to(dtype=norm_dtype))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_signal_stats = [
            dict(layer.last_forward_stats)
            for layer in self.backbone.layers
            if isinstance(layer, MpucGfstHmbBlock) and layer.last_forward_stats is not None
        ]
        self._last_forward_signal_stats = {
            "hidden_state_rms": _tensor_rms(hidden_states),
            "backbone_residual_rms": float(sum(stats.get("backbone_residual_rms", 0.0) for stats in layer_signal_stats) / max(1, len(layer_signal_stats))),
            "gfst_branch_rms": float(sum(stats.get("gfst_branch_rms", 0.0) for stats in layer_signal_stats) / max(1, len(layer_signal_stats))),
            "gate_activation_mean": float(sum(stats.get("gate_activation_mean", 0.0) for stats in layer_signal_stats) / max(1, len(layer_signal_stats))),
            "gate_activation_std": float(sum(stats.get("gate_activation_std", 0.0) for stats in layer_signal_stats) / max(1, len(layer_signal_stats))),
            "boundary_activation_mean": float(sum(stats.get("boundary_activation_mean", 0.0) for stats in layer_signal_stats) / max(1, len(layer_signal_stats))),
        }

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        generation_logits = self.lm_head(hidden_states[:, slice_indices, :].to(self.lm_head.weight.dtype)).float()
        boundary_logits = torch.stack(layer_boundary_logits, dim=0).mean(dim=0) if layer_boundary_logits else None

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                generation_logits.reshape(-1, int(generation_logits.shape[-1])),
                labels.reshape(-1),
                ignore_index=-100,
            )

        if use_cache:
            effective_cache.sync_from_core_states(next_core_states)
            for batch_idx in range(batch_size):
                effective_cache.advance(batch_idx, seq_len)
        effective_cache.clear_prepared_inputs()
        output = self._canonical_output(
            generation_logits=generation_logits,
            past_key_values=effective_cache,
            slice_indices=slice_indices,
            loss=loss,
            hidden_states=all_hidden_states,
            boundary_logits=boundary_logits,
        )
        self._last_runtime_profile = None if runtime_profile_recorder is None else runtime_profile_recorder.snapshot()
        effective_cache.set_runtime_profile_recorder(None)
        self.exact_readout.set_runtime_profile_recorder(None)
        for layer in self.backbone.layers:
            layer.set_runtime_profile_recorder(None)
        return output
