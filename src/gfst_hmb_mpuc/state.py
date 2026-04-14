from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch

from .perf import RuntimeProfileRecorder, runtime_range
from .runtime_backends import exact_block_materialize

_CENTROID_VALUE_QUANT_SCALE = 32.0
_CENTROID_WEIGHT_QUANT_SCALE = 255.0
_CENTROID_COUNT_QUANT_SCALE = 32.0

try:
    from transformers.cache_utils import Cache as _BaseCache
except ImportError:
    class _BaseCache:
        pass


@dataclass
class GfstBlockSummary:
    basis: Any
    key: Any
    block_index: int
    start_pos: int
    end_pos: int
    token_count: int
    reduced_mean: Any = None


@dataclass
class ExactBlockRecord:
    block_id: int
    start_pos: int
    end_pos: int
    token_ids: tuple[int, ...]
    token_id_tensor: Any = None
    token_cell_id_tensor: Any = None
    token_cell_weight_q_tensor: Any = None
    hidden_states: Any = None
    token_offset_start: int = 0
    token_offset_end: int = 0
    cell_storage_index: int = -1
    cell_keys: Any = None
    cell_values_q: Any = None
    cell_counts: Any = None


@dataclass
class _ExactStoreState:
    completed_blocks: list[ExactBlockRecord] = field(default_factory=list)
    current_block_start: int = 0
    current_block_token_ids: list[int] = field(default_factory=list)
    current_block_token_cell_ids: list[list[int]] = field(default_factory=list)
    current_block_token_cell_weights_q: list[list[int]] = field(default_factory=list)
    current_block_hidden_states: list[Any] = field(default_factory=list)
    current_block_token_count: int = 0
    current_block_membership_width: int = 1
    current_block_token_ids_tensor: Any = None
    current_block_token_cell_ids_tensor: Any = None
    current_block_token_cell_weights_q_tensor: Any = None
    current_block_hidden_states_tensor: Any = None
    exact_hidden_store_mode: str = "tuple"
    packed_token_ids: Any = None
    packed_token_cell_ids: Any = None
    packed_token_cell_weights_q: Any = None
    packed_hidden_states: Any = None
    block_token_offsets: Any = None
    block_cell_keys: Any = None
    block_cell_values_q: Any = None
    block_cell_counts: Any = None
    current_cell_key_sum: Any = None
    current_cell_value_sum: Any = None
    current_cell_counts: Any = None


@dataclass
class _ExactCacheState:
    block_ids_by_recency: list[int] = field(default_factory=list)
    cached_blocks: dict[int, ExactBlockRecord] = field(default_factory=dict)


@dataclass
class _PreparedInputState:
    positions: Any = None
    token_ids: Any = None
    next_index: int = 0


@dataclass
class _GfstTrackerState:
    largest_basis: Any = None
    nested_bases: list[Any] = field(default_factory=list)


@dataclass
class _GfstBlockCompressorState:
    largest_basis: Any = None
    projector_sum: Any = None
    factor_buffer: Any = None
    reduced_sum: Any = None
    current_block_start: int = 0
    current_block_tokens: int = 0
    block_index: int = 0


@dataclass
class _GfstLayerRuntimeState:
    local_tracker: _GfstTrackerState = field(default_factory=_GfstTrackerState)
    block_tracker: _GfstBlockCompressorState = field(default_factory=_GfstBlockCompressorState)
    block_summaries: list[GfstBlockSummary] = field(default_factory=list)
    summary_key_tensor: Any = None
    summary_block_index_tensor: Any = None
    summary_token_count_tensor: Any = None
    summary_reduced_mean_tensor: Any = None
    summary_facet_basis_tensor: Any = None
    summary_basis_tensor: Any = None
    summary_projector_tensor: Any = None
    summary_cache_device: str | None = None
    summary_cache_facet_count: int = 0


@dataclass
class _GfstSampleRuntimeState:
    seen_tokens: int = 0
    layers: list[_GfstLayerRuntimeState] = field(default_factory=list)
    exact_store: _ExactStoreState = field(default_factory=_ExactStoreState)
    exact_cache: _ExactCacheState = field(default_factory=_ExactCacheState)
    prepared_inputs: _PreparedInputState = field(default_factory=_PreparedInputState)


@dataclass
class _CoreLayerCacheState:
    fast_state: Any = None
    mid_state: Any = None
    slow_state: Any = None
    fast_buffer: Any = None
    mid_buffer: Any = None
    slow_buffer: Any = None


class GfstDecodeState(_BaseCache):
    def __init__(
        self,
        num_layers: int,
        batch_size: int = 1,
        *,
        state_pack_backend: str = "native",
        runtime_profiler_mode: str = "off",
    ) -> None:
        self.num_layers = max(0, int(num_layers))
        self.samples = [
            _GfstSampleRuntimeState(layers=[_GfstLayerRuntimeState() for _ in range(self.num_layers)])
            for _ in range(max(1, int(batch_size)))
        ]
        self.state_version = 0
        self.core_layer_states: dict[int, _CoreLayerCacheState] = {}
        self.state_pack_backend = "native"
        self.runtime_profiler_mode = "off"
        self._runtime_profile_recorder: RuntimeProfileRecorder | None = None
        self.set_state_pack_backend(state_pack_backend)
        self.set_runtime_profiler_mode(runtime_profiler_mode)

    def set_state_pack_backend(self, backend: str) -> None:
        backend_str = str(backend)
        if backend_str not in {"native", "cuda_packed"}:
            raise ValueError("state_pack_backend must be one of {'native', 'cuda_packed'}")
        self.state_pack_backend = backend_str

    def set_runtime_profiler_mode(self, mode: str) -> None:
        mode_str = str(mode)
        if mode_str not in {"off", "nvtx", "benchmark"}:
            raise ValueError("runtime_profiler_mode must be one of {'off', 'nvtx', 'benchmark'}")
        self.runtime_profiler_mode = mode_str

    def set_runtime_profile_recorder(self, recorder: RuntimeProfileRecorder | None) -> None:
        self._runtime_profile_recorder = recorder

    def _bump_version(self) -> None:
        self.state_version += 1

    def mark_mutated(self) -> None:
        self._bump_version()

    def ensure_num_layers(self, num_layers: int) -> None:
        target = int(num_layers)
        if target <= self.num_layers:
            return
        for sample in self.samples:
            sample.layers.extend(_GfstLayerRuntimeState() for _ in range(target - self.num_layers))
        self.num_layers = target
        self._bump_version()

    def ensure_batch_size(self, batch_size: int) -> None:
        target = int(batch_size)
        if target <= len(self.samples):
            return
        for _ in range(target - len(self.samples)):
            self.samples.append(_GfstSampleRuntimeState(layers=[_GfstLayerRuntimeState() for _ in range(self.num_layers)]))
        self._bump_version()

    def layer_state(self, layer_idx: int, batch_idx: int) -> _GfstLayerRuntimeState:
        self.ensure_num_layers(layer_idx + 1)
        self.ensure_batch_size(batch_idx + 1)
        return self.samples[batch_idx].layers[layer_idx]

    def advance(self, batch_idx: int, token_count: int) -> None:
        self.samples[batch_idx].seen_tokens += int(token_count)

    def prepare_inputs(self, batch_idx: int, *, positions, token_ids) -> None:
        sample = self.samples[int(batch_idx)]
        sample.prepared_inputs.positions = positions.detach().to(device="cpu", dtype=torch.long).clone()
        sample.prepared_inputs.token_ids = token_ids.detach().to(device="cpu", dtype=torch.long).clone()
        sample.prepared_inputs.next_index = 0

    def clear_prepared_inputs(self) -> None:
        for sample in self.samples:
            sample.prepared_inputs = _PreparedInputState()

    def _append_packed_rows(self, existing: Any, appended: torch.Tensor) -> torch.Tensor:
        payload = appended.detach().clone()
        if existing is None:
            return payload
        return torch.cat((existing, payload), dim=0)

    def _append_offset(self, existing: Any, *, value: int, device: torch.device) -> torch.Tensor:
        next_value = torch.tensor([int(value)], device=device, dtype=torch.int32)
        if existing is None:
            return torch.tensor([0, int(value)], device=device, dtype=torch.int32)
        return torch.cat((existing, next_value), dim=0)

    def _current_block_length(self, store: _ExactStoreState) -> int:
        if str(self.state_pack_backend) == "cuda_packed":
            return int(store.current_block_token_count)
        return len(store.current_block_token_ids)

    def _reset_current_block_buffers(self, store: _ExactStoreState, *, start_pos: int) -> None:
        store.current_block_start = int(start_pos)
        store.current_block_token_ids = []
        store.current_block_token_cell_ids = []
        store.current_block_token_cell_weights_q = []
        store.current_block_hidden_states = []
        store.current_block_token_count = 0

    def _ensure_current_block_packed_buffers(
        self,
        *,
        store: _ExactStoreState,
        device: torch.device,
        block_len: int,
        membership_width: int,
        hidden_state: Any = None,
        exact_hidden_store_mode: str = "tuple",
    ) -> None:
        capacity = max(1, int(block_len))
        width = max(1, int(membership_width))
        token_ids_tensor = store.current_block_token_ids_tensor
        if (
            token_ids_tensor is None
            or token_ids_tensor.device != device
            or int(token_ids_tensor.shape[0]) < capacity
        ):
            replacement = torch.zeros((capacity,), device=device, dtype=torch.int32)
            if token_ids_tensor is not None and int(store.current_block_token_count) > 0:
                keep = min(int(store.current_block_token_count), int(token_ids_tensor.shape[0]), capacity)
                replacement[:keep] = token_ids_tensor[:keep].to(device=device, dtype=torch.int32)
            store.current_block_token_ids_tensor = replacement
        cell_ids_tensor = store.current_block_token_cell_ids_tensor
        cell_weights_tensor = store.current_block_token_cell_weights_q_tensor
        needs_cell_resize = (
            cell_ids_tensor is None
            or cell_ids_tensor.device != device
            or int(cell_ids_tensor.shape[0]) < capacity
            or int(cell_ids_tensor.shape[1]) < width
        )
        if needs_cell_resize:
            replacement_ids = torch.zeros((capacity, width), device=device, dtype=torch.uint8)
            replacement_weights = torch.zeros((capacity, width), device=device, dtype=torch.uint8)
            if cell_ids_tensor is not None and cell_weights_tensor is not None and int(store.current_block_token_count) > 0:
                keep_rows = min(int(store.current_block_token_count), int(cell_ids_tensor.shape[0]), capacity)
                keep_width = min(int(cell_ids_tensor.shape[1]), width)
                replacement_ids[:keep_rows, :keep_width] = cell_ids_tensor[:keep_rows, :keep_width].to(
                    device=device,
                    dtype=torch.uint8,
                )
                replacement_weights[:keep_rows, :keep_width] = cell_weights_tensor[:keep_rows, :keep_width].to(
                    device=device,
                    dtype=torch.uint8,
                )
            store.current_block_token_cell_ids_tensor = replacement_ids
            store.current_block_token_cell_weights_q_tensor = replacement_weights
        store.current_block_membership_width = max(int(store.current_block_membership_width), width)
        if hidden_state is None:
            return
        hidden_tensor = torch.as_tensor(hidden_state, device=device)
        hidden_state_tensor = store.current_block_hidden_states_tensor
        target_dtype = torch.bfloat16 if str(exact_hidden_store_mode) == "packed_bf16" else hidden_tensor.dtype
        hidden_dim = int(hidden_tensor.reshape(-1).shape[0])
        if (
            hidden_state_tensor is None
            or hidden_state_tensor.device != device
            or int(hidden_state_tensor.shape[0]) < capacity
            or int(hidden_state_tensor.shape[1]) != hidden_dim
            or hidden_state_tensor.dtype != target_dtype
        ):
            replacement_hidden = torch.zeros((capacity, hidden_dim), device=device, dtype=target_dtype)
            if hidden_state_tensor is not None and int(store.current_block_token_count) > 0:
                keep = min(int(store.current_block_token_count), int(hidden_state_tensor.shape[0]), capacity)
                replacement_hidden[:keep] = hidden_state_tensor[:keep].to(device=device, dtype=target_dtype)
            store.current_block_hidden_states_tensor = replacement_hidden

    def _append_exact_token_cuda_packed(
        self,
        *,
        store: _ExactStoreState,
        absolute_position: int,
        token_id: int,
        block_len: int,
        hidden_state: Any = None,
        exact_hidden_store_mode: str = "tuple",
        cells_per_block: int = 16,
        centroid_key_dim: int = 32,
        centroid_value_dim: int = 16,
        cell_ids: Any = None,
        cell_weights: Any = None,
        cell_key: Any = None,
        cell_value: Any = None,
    ) -> None:
        device = (
            hidden_state.device
            if hidden_state is not None and hasattr(hidden_state, "device")
            else cell_key.device
            if cell_key is not None and hasattr(cell_key, "device")
            else torch.device("cpu")
        )
        if int(store.current_block_token_count) == 0:
            store.current_block_start = int(absolute_position)
            if hidden_state is None:
                store.current_block_hidden_states_tensor = None
        membership_count = 0
        if cell_ids is not None:
            try:
                membership_count = max(0, int(torch.as_tensor(cell_ids).reshape(-1).numel()))
            except Exception:
                membership_count = len(cell_ids)
        membership_count = max(1, membership_count)
        self._ensure_current_block_packed_buffers(
            store=store,
            device=device,
            block_len=int(block_len),
            membership_width=membership_count,
            hidden_state=hidden_state,
            exact_hidden_store_mode=str(exact_hidden_store_mode or "tuple"),
        )
        token_index = int(store.current_block_token_count)
        token_ids_tensor = store.current_block_token_ids_tensor
        cell_ids_tensor = store.current_block_token_cell_ids_tensor
        cell_weights_tensor = store.current_block_token_cell_weights_q_tensor
        if token_ids_tensor is None or cell_ids_tensor is None or cell_weights_tensor is None:
            raise RuntimeError("packed exact-store buffers were not initialised")
        token_ids_tensor[token_index] = int(token_id)
        if cell_ids is None:
            ids_tensor = torch.zeros((membership_count,), device=device, dtype=torch.long)
        else:
            ids_tensor = torch.as_tensor(cell_ids, device=device, dtype=torch.long).reshape(-1)
            ids_tensor = ids_tensor.clamp(min=0, max=max(0, int(cells_per_block) - 1))
        if cell_weights is None:
            quantized_weights = torch.zeros((membership_count,), device=device, dtype=torch.uint8)
            quantized_weights[0] = int(round(float(_CENTROID_WEIGHT_QUANT_SCALE)))
        else:
            weight_tensor = torch.as_tensor(cell_weights, device=device, dtype=torch.float32).reshape(-1)
            if int(weight_tensor.numel()) != membership_count:
                raise ValueError("cell_ids and cell_weights must have matching lengths")
            quantized_weights = torch.clamp(
                torch.round(weight_tensor * float(_CENTROID_WEIGHT_QUANT_SCALE)),
                min=0.0,
                max=float(_CENTROID_WEIGHT_QUANT_SCALE),
            ).to(dtype=torch.uint8)
        cell_ids_tensor[token_index].zero_()
        cell_weights_tensor[token_index].zero_()
        cell_ids_tensor[token_index, :membership_count] = ids_tensor.to(dtype=torch.uint8)
        cell_weights_tensor[token_index, :membership_count] = quantized_weights
        if hidden_state is not None and store.current_block_hidden_states_tensor is not None:
            hidden_target = store.current_block_hidden_states_tensor[token_index]
            hidden_target.copy_(
                torch.as_tensor(hidden_state, device=device)
                .detach()
                .reshape(-1)
                .to(device=device, dtype=hidden_target.dtype)
            )
        if cell_key is not None and cell_value is not None:
            cell_device = cell_key.device
            self._ensure_cell_buffers(
                store,
                device=cell_device,
                cells_per_block=int(cells_per_block),
                centroid_key_dim=int(centroid_key_dim),
                centroid_value_dim=int(centroid_value_dim),
            )
            key_tensor = cell_key.detach().to(device=cell_device, dtype=torch.float32).reshape(-1)
            value_tensor = cell_value.detach().to(device=cell_device, dtype=torch.float32).reshape(-1)
            ids_tensor = ids_tensor.to(device=cell_device, dtype=torch.long)
            weight_tensor = quantized_weights.to(device=cell_device, dtype=torch.float32) / float(_CENTROID_WEIGHT_QUANT_SCALE)
            positive_mask = weight_tensor.gt(0.0)
            if bool(positive_mask.any().item()):
                ids_tensor = ids_tensor[positive_mask]
                weight_tensor = weight_tensor[positive_mask]
                store.current_cell_key_sum.index_add_(0, ids_tensor, weight_tensor.unsqueeze(-1) * key_tensor.unsqueeze(0))
                store.current_cell_value_sum.index_add_(0, ids_tensor, weight_tensor.unsqueeze(-1) * value_tensor.unsqueeze(0))
                store.current_cell_counts.index_add_(0, ids_tensor, weight_tensor)
        store.current_block_token_count = token_index + 1
        if int(store.current_block_token_count) < int(block_len):
            return
        commit_device = (
            store.current_cell_key_sum.device
            if store.current_cell_key_sum is not None
            else device
        )
        block_token_start = 0 if store.packed_token_ids is None else int(store.packed_token_ids.shape[0])
        committed_count = int(store.current_block_token_count)
        block_tokens_tensor = token_ids_tensor[:committed_count].to(device=commit_device, dtype=torch.int32)
        block_token_cells_tensor = cell_ids_tensor[:committed_count].to(device=commit_device, dtype=torch.uint8)
        block_token_cell_weights_q_tensor = cell_weights_tensor[:committed_count].to(device=commit_device, dtype=torch.uint8)
        store.packed_token_ids = self._append_packed_rows(store.packed_token_ids, block_tokens_tensor)
        store.packed_token_cell_ids = self._append_packed_rows(store.packed_token_cell_ids, block_token_cells_tensor)
        store.packed_token_cell_weights_q = self._append_packed_rows(
            store.packed_token_cell_weights_q,
            block_token_cell_weights_q_tensor,
        )
        block_token_end = int(store.packed_token_ids.shape[0])
        store.block_token_offsets = self._append_offset(store.block_token_offsets, value=block_token_end, device=commit_device)
        stored_hidden_states = None
        hidden_state_tensor = store.current_block_hidden_states_tensor
        if hidden_state_tensor is not None:
            hidden_slice = hidden_state_tensor[:committed_count]
            if str(store.exact_hidden_store_mode) == "packed_bf16":
                store.packed_hidden_states = self._append_packed_rows(
                    store.packed_hidden_states,
                    hidden_slice.to(device=commit_device, dtype=torch.bfloat16),
                )
            else:
                stored_hidden_states = tuple(hidden.detach().clone() for hidden in hidden_slice)
        block_cell_keys, block_cell_values_q, block_cell_counts = self._finalise_block_cells(
            store=store,
            device=commit_device,
            cells_per_block=int(cells_per_block),
            centroid_key_dim=int(centroid_key_dim),
            centroid_value_dim=int(centroid_value_dim),
        )
        cell_storage_index = 0 if store.block_cell_keys is None else int(store.block_cell_keys.shape[0])
        store.block_cell_keys = self._append_packed_rows(store.block_cell_keys, block_cell_keys.unsqueeze(0))
        store.block_cell_values_q = self._append_packed_rows(store.block_cell_values_q, block_cell_values_q.unsqueeze(0))
        store.block_cell_counts = self._append_packed_rows(store.block_cell_counts, block_cell_counts.unsqueeze(0))
        block_id = int(store.current_block_start) // int(block_len)
        record = ExactBlockRecord(
            block_id=block_id,
            start_pos=int(store.current_block_start),
            end_pos=int(absolute_position) + 1,
            token_ids=(),
            hidden_states=stored_hidden_states,
            token_offset_start=int(block_token_start),
            token_offset_end=int(block_token_end),
            cell_storage_index=int(cell_storage_index),
        )
        store.completed_blocks.append(record)
        self._reset_current_block_buffers(store, start_pos=int(absolute_position) + 1)
        self._clear_cell_buffers(store)

    def _ensure_cell_buffers(
        self,
        store: _ExactStoreState,
        *,
        device: torch.device,
        cells_per_block: int,
        centroid_key_dim: int,
        centroid_value_dim: int,
    ) -> None:
        if store.current_cell_key_sum is None:
            store.current_cell_key_sum = torch.zeros(
                (int(cells_per_block), int(centroid_key_dim)),
                device=device,
                dtype=torch.float32,
            )
        if store.current_cell_value_sum is None:
            store.current_cell_value_sum = torch.zeros(
                (int(cells_per_block), int(centroid_value_dim)),
                device=device,
                dtype=torch.float32,
            )
        if store.current_cell_counts is None:
            store.current_cell_counts = torch.zeros((int(cells_per_block),), device=device, dtype=torch.float32)

    def _clear_cell_buffers(self, store: _ExactStoreState) -> None:
        store.current_cell_key_sum = None
        store.current_cell_value_sum = None
        store.current_cell_counts = None

    def _finalise_block_cells(
        self,
        *,
        store: _ExactStoreState,
        device: torch.device,
        cells_per_block: int,
        centroid_key_dim: int,
        centroid_value_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if store.current_cell_key_sum is None or store.current_cell_value_sum is None or store.current_cell_counts is None:
            empty_keys = torch.zeros(
                (int(cells_per_block), int(centroid_key_dim)),
                device=device,
                dtype=torch.float16,
            )
            empty_values = torch.zeros(
                (int(cells_per_block), int(centroid_value_dim)),
                device=device,
                dtype=torch.int8,
            )
            empty_counts = torch.zeros((int(cells_per_block),), device=device, dtype=torch.int16)
            return empty_keys, empty_values, empty_counts
        counts = store.current_cell_counts.to(dtype=torch.float32)
        safe_counts = counts.clamp_min(1.0).unsqueeze(-1)
        mean_keys = store.current_cell_key_sum / safe_counts
        mean_values = store.current_cell_value_sum / safe_counts
        mean_keys = torch.where(
            store.current_cell_counts.unsqueeze(-1).gt(0),
            mean_keys,
            torch.zeros_like(mean_keys),
        )
        mean_values = torch.where(
            store.current_cell_counts.unsqueeze(-1).gt(0),
            mean_values,
            torch.zeros_like(mean_values),
        )
        quantized_values = torch.clamp(
            torch.round(mean_values * float(_CENTROID_VALUE_QUANT_SCALE)),
            min=-127.0,
            max=127.0,
        ).to(dtype=torch.int8)
        quantized_counts = torch.clamp(
            torch.round(counts * float(_CENTROID_COUNT_QUANT_SCALE)),
            min=0.0,
            max=32767.0,
        ).to(dtype=torch.int16)
        return (
            mean_keys.to(dtype=torch.float16),
            quantized_values,
            quantized_counts,
        )

    def _materialize_exact_block(self, store: _ExactStoreState, record: ExactBlockRecord) -> ExactBlockRecord:
        with runtime_range(
            "state.materialize_exact_block",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=store.packed_token_ids.device if torch.is_tensor(store.packed_token_ids) else None,
        ):
            token_tensor = record.token_id_tensor
            if (
                token_tensor is None
                and store.packed_token_ids is not None
                and int(record.token_offset_end) > int(record.token_offset_start)
            ):
                accelerated = None
                if str(self.state_pack_backend) == "cuda_packed":
                    accelerated = exact_block_materialize(
                        store.packed_token_ids,
                        int(record.token_offset_start),
                        int(record.token_offset_end),
                    )
                token_tensor = (
                    accelerated
                    if accelerated is not None
                    else store.packed_token_ids[int(record.token_offset_start) : int(record.token_offset_end)]
                ).to(dtype=torch.long)
            token_cell_tensor = record.token_cell_id_tensor
            if (
                token_cell_tensor is None
                and store.packed_token_cell_ids is not None
                and int(record.token_offset_end) > int(record.token_offset_start)
            ):
                token_cell_tensor = store.packed_token_cell_ids[
                    int(record.token_offset_start) : int(record.token_offset_end)
                ].to(dtype=torch.long)
            token_cell_weight_q_tensor = record.token_cell_weight_q_tensor
            if (
                token_cell_weight_q_tensor is None
                and store.packed_token_cell_weights_q is not None
                and int(record.token_offset_end) > int(record.token_offset_start)
            ):
                token_cell_weight_q_tensor = store.packed_token_cell_weights_q[
                    int(record.token_offset_start) : int(record.token_offset_end)
                ].to(dtype=torch.long)
            hidden_states = record.hidden_states
            if (
                hidden_states is None
                and store.packed_hidden_states is not None
                and int(record.token_offset_end) > int(record.token_offset_start)
            ):
                hidden_states = store.packed_hidden_states[int(record.token_offset_start) : int(record.token_offset_end)]
            cell_keys = record.cell_keys
            cell_values_q = record.cell_values_q
            cell_counts = record.cell_counts
            if int(record.cell_storage_index) >= 0:
                storage_index = int(record.cell_storage_index)
                if cell_keys is None and store.block_cell_keys is not None and storage_index < int(store.block_cell_keys.shape[0]):
                    cell_keys = store.block_cell_keys[storage_index]
                if cell_values_q is None and store.block_cell_values_q is not None and storage_index < int(store.block_cell_values_q.shape[0]):
                    cell_values_q = store.block_cell_values_q[storage_index]
                if cell_counts is None and store.block_cell_counts is not None and storage_index < int(store.block_cell_counts.shape[0]):
                    cell_counts = store.block_cell_counts[storage_index]
            token_ids = record.token_ids
            if not token_ids and token_tensor is not None:
                token_ids = tuple(int(value) for value in token_tensor.to(device="cpu", dtype=torch.long).tolist())
            return ExactBlockRecord(
                block_id=int(record.block_id),
                start_pos=int(record.start_pos),
                end_pos=int(record.end_pos),
                token_ids=token_ids,
                token_id_tensor=token_tensor,
                token_cell_id_tensor=token_cell_tensor,
                token_cell_weight_q_tensor=token_cell_weight_q_tensor,
                hidden_states=hidden_states,
                token_offset_start=int(record.token_offset_start),
                token_offset_end=int(record.token_offset_end),
                cell_storage_index=int(record.cell_storage_index),
                cell_keys=cell_keys,
                cell_values_q=cell_values_q,
                cell_counts=cell_counts,
            )

    def consume_prepared_token(self, batch_idx: int, absolute_position: int) -> int:
        sample = self.samples[int(batch_idx)]
        prepared = sample.prepared_inputs
        if prepared.positions is None or prepared.token_ids is None:
            raise RuntimeError("prepared token IDs are required for canonical exact recall")
        next_index = int(prepared.next_index)
        if next_index >= int(prepared.positions.numel()):
            raise RuntimeError("prepared token stream exhausted before GFST token pass completed")
        expected_position = int(prepared.positions.reshape(-1)[next_index].item())
        if expected_position != int(absolute_position):
            raise RuntimeError(
                f"prepared token position mismatch: expected {expected_position}, got {int(absolute_position)}"
            )
        token_id = int(prepared.token_ids.reshape(-1)[next_index].item())
        prepared.next_index = next_index + 1
        return token_id

    def append_exact_token(
        self,
        *,
        batch_idx: int,
        absolute_position: int,
        token_id: int,
        block_len: int,
        hidden_state: Any = None,
        exact_hidden_store_mode: str = "tuple",
        cells_per_block: int = 16,
        centroid_key_dim: int = 32,
        centroid_value_dim: int = 16,
        cell_ids: Any = None,
        cell_weights: Any = None,
        cell_key: Any = None,
        cell_value: Any = None,
    ) -> None:
        sample = self.samples[int(batch_idx)]
        store = sample.exact_store
        profile_device = None
        if hidden_state is not None and hasattr(hidden_state, "device"):
            profile_device = hidden_state.device
        elif cell_key is not None and hasattr(cell_key, "device"):
            profile_device = cell_key.device
        with runtime_range(
            "state.append_exact_token",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=profile_device,
        ):
            store.exact_hidden_store_mode = str(exact_hidden_store_mode or "tuple")
            if str(self.state_pack_backend) == "cuda_packed":
                self._append_exact_token_cuda_packed(
                    store=store,
                    absolute_position=int(absolute_position),
                    token_id=int(token_id),
                    block_len=int(block_len),
                    hidden_state=hidden_state,
                    exact_hidden_store_mode=store.exact_hidden_store_mode,
                    cells_per_block=int(cells_per_block),
                    centroid_key_dim=int(centroid_key_dim),
                    centroid_value_dim=int(centroid_value_dim),
                    cell_ids=cell_ids,
                    cell_weights=cell_weights,
                    cell_key=cell_key,
                    cell_value=cell_value,
                )
                return
            if not store.current_block_token_ids:
                store.current_block_start = int(absolute_position)
            store.current_block_token_ids.append(int(token_id))
            membership_count = 0
            if cell_ids is not None:
                try:
                    membership_count = max(0, int(torch.as_tensor(cell_ids).reshape(-1).numel()))
                except Exception:
                    membership_count = len(cell_ids)
            membership_count = max(1, membership_count)
            if cell_ids is None:
                ids_tensor = torch.zeros((membership_count,), dtype=torch.long)
                normalized_ids = [0] * membership_count
            else:
                ids_tensor = torch.as_tensor(cell_ids, dtype=torch.long).reshape(-1)
                ids_tensor = ids_tensor.clamp(min=0, max=max(0, int(cells_per_block) - 1))
                normalized_ids = ids_tensor.tolist()
            if cell_weights is None:
                quantized_weights = torch.zeros((membership_count,), dtype=torch.uint8)
                quantized_weights[0] = int(round(float(_CENTROID_WEIGHT_QUANT_SCALE)))
            else:
                weight_tensor = torch.as_tensor(cell_weights, dtype=torch.float32).reshape(-1)
                if int(weight_tensor.numel()) != membership_count:
                    raise ValueError("cell_ids and cell_weights must have matching lengths")
                quantized_weights = torch.clamp(
                    torch.round(weight_tensor * float(_CENTROID_WEIGHT_QUANT_SCALE)),
                    min=0.0,
                    max=float(_CENTROID_WEIGHT_QUANT_SCALE),
                ).to(dtype=torch.uint8)
            normalized_weights = quantized_weights.tolist()
            store.current_block_token_cell_ids.append(normalized_ids)
            store.current_block_token_cell_weights_q.append(normalized_weights)
            if hidden_state is not None:
                store.current_block_hidden_states.append(hidden_state.detach().clone())
            if cell_key is not None and cell_value is not None:
                cell_device = cell_key.device
                self._ensure_cell_buffers(
                    store,
                    device=cell_device,
                    cells_per_block=int(cells_per_block),
                    centroid_key_dim=int(centroid_key_dim),
                    centroid_value_dim=int(centroid_value_dim),
                )
                key_tensor = cell_key.detach().to(device=cell_device, dtype=torch.float32).reshape(-1)
                value_tensor = cell_value.detach().to(device=cell_device, dtype=torch.float32).reshape(-1)
                ids_tensor = ids_tensor.to(device=cell_device, dtype=torch.long)
                weight_tensor = quantized_weights.to(device=cell_device, dtype=torch.float32) / float(_CENTROID_WEIGHT_QUANT_SCALE)
                positive_mask = weight_tensor.gt(0.0)
                if bool(positive_mask.any().item()):
                    ids_tensor = ids_tensor[positive_mask]
                    weight_tensor = weight_tensor[positive_mask]
                    store.current_cell_key_sum.index_add_(0, ids_tensor, weight_tensor.unsqueeze(-1) * key_tensor.unsqueeze(0))
                    store.current_cell_value_sum.index_add_(0, ids_tensor, weight_tensor.unsqueeze(-1) * value_tensor.unsqueeze(0))
                    store.current_cell_counts.index_add_(0, ids_tensor, weight_tensor)
            if len(store.current_block_token_ids) < int(block_len):
                return
            commit_device = (
                store.current_cell_key_sum.device
                if store.current_cell_key_sum is not None
                else hidden_state.device
                if hidden_state is not None and hasattr(hidden_state, "device")
                else torch.device("cpu")
            )
            block_token_start = 0 if store.packed_token_ids is None else int(store.packed_token_ids.shape[0])
            block_tokens_tensor = torch.tensor(store.current_block_token_ids, device=commit_device, dtype=torch.int32)
            block_token_cells_tensor = torch.tensor(store.current_block_token_cell_ids, device=commit_device, dtype=torch.uint8)
            block_token_cell_weights_q_tensor = torch.tensor(
                store.current_block_token_cell_weights_q,
                device=commit_device,
                dtype=torch.uint8,
            )
            store.packed_token_ids = self._append_packed_rows(store.packed_token_ids, block_tokens_tensor)
            store.packed_token_cell_ids = self._append_packed_rows(store.packed_token_cell_ids, block_token_cells_tensor)
            store.packed_token_cell_weights_q = self._append_packed_rows(
                store.packed_token_cell_weights_q,
                block_token_cell_weights_q_tensor,
            )
            block_token_end = int(store.packed_token_ids.shape[0])
            store.block_token_offsets = self._append_offset(store.block_token_offsets, value=block_token_end, device=commit_device)
            stored_hidden_states = None
            if store.current_block_hidden_states:
                hidden_tensor = torch.stack(
                    [
                        torch.as_tensor(hidden, device=commit_device).detach().to(device=commit_device, dtype=torch.bfloat16)
                        for hidden in store.current_block_hidden_states
                    ],
                    dim=0,
                )
                if str(store.exact_hidden_store_mode) == "packed_bf16":
                    store.packed_hidden_states = self._append_packed_rows(store.packed_hidden_states, hidden_tensor)
                else:
                    stored_hidden_states = tuple(hidden.detach().clone() for hidden in store.current_block_hidden_states)
            block_cell_keys, block_cell_values_q, block_cell_counts = self._finalise_block_cells(
                store=store,
                device=commit_device,
                cells_per_block=int(cells_per_block),
                centroid_key_dim=int(centroid_key_dim),
                centroid_value_dim=int(centroid_value_dim),
            )
            cell_storage_index = 0 if store.block_cell_keys is None else int(store.block_cell_keys.shape[0])
            store.block_cell_keys = self._append_packed_rows(store.block_cell_keys, block_cell_keys.unsqueeze(0))
            store.block_cell_values_q = self._append_packed_rows(store.block_cell_values_q, block_cell_values_q.unsqueeze(0))
            store.block_cell_counts = self._append_packed_rows(store.block_cell_counts, block_cell_counts.unsqueeze(0))
            block_id = int(store.current_block_start) // int(block_len)
            record = ExactBlockRecord(
                block_id=block_id,
                start_pos=int(store.current_block_start),
                end_pos=int(absolute_position) + 1,
                token_ids=(),
                hidden_states=stored_hidden_states,
                token_offset_start=int(block_token_start),
                token_offset_end=int(block_token_end),
                cell_storage_index=int(cell_storage_index),
            )
            store.completed_blocks.append(record)
            self._reset_current_block_buffers(store, start_pos=int(absolute_position) + 1)
            self._clear_cell_buffers(store)

    def current_exact_block_id(self, batch_idx: int, block_len: int) -> int | None:
        sample = self.samples[int(batch_idx)]
        store = sample.exact_store
        if self._current_block_length(store) <= 0:
            return None
        return int(store.current_block_start) // int(block_len)

    def current_exact_block_length(self, batch_idx: int) -> int:
        return self._current_block_length(self.samples[int(batch_idx)].exact_store)

    def resolve_exact_block(self, batch_idx: int, block_id: int, block_len: int) -> ExactBlockRecord | None:
        sample = self.samples[int(batch_idx)]
        for record in sample.exact_store.completed_blocks:
            if int(record.block_id) == int(block_id):
                return record
        current_block_id = self.current_exact_block_id(batch_idx, block_len)
        if current_block_id is not None and int(current_block_id) == int(block_id):
            store = sample.exact_store
            current_device = (
                store.current_block_token_ids_tensor.device
                if str(self.state_pack_backend) == "cuda_packed" and torch.is_tensor(store.current_block_token_ids_tensor)
                else store.current_cell_key_sum.device
                if store.current_cell_key_sum is not None
                else torch.device("cpu")
            )
            current_length = self._current_block_length(store)
            cell_keys, cell_values_q, cell_counts = self._finalise_block_cells(
                store=store,
                device=current_device,
                cells_per_block=int(store.current_cell_counts.shape[0]) if store.current_cell_counts is not None else 16,
                centroid_key_dim=int(store.current_cell_key_sum.shape[-1]) if store.current_cell_key_sum is not None else 32,
                centroid_value_dim=int(store.current_cell_value_sum.shape[-1]) if store.current_cell_value_sum is not None else 16,
            )
            if str(self.state_pack_backend) == "cuda_packed":
                token_id_tensor = (
                    store.current_block_token_ids_tensor[:current_length].to(dtype=torch.long)
                    if store.current_block_token_ids_tensor is not None and current_length > 0
                    else None
                )
                token_cell_id_tensor = (
                    store.current_block_token_cell_ids_tensor[:current_length].to(dtype=torch.long)
                    if store.current_block_token_cell_ids_tensor is not None and current_length > 0
                    else None
                )
                token_cell_weight_q_tensor = (
                    store.current_block_token_cell_weights_q_tensor[:current_length].to(dtype=torch.long)
                    if store.current_block_token_cell_weights_q_tensor is not None and current_length > 0
                    else None
                )
                if store.current_block_hidden_states_tensor is not None and current_length > 0:
                    hidden_slice = store.current_block_hidden_states_tensor[:current_length]
                    hidden_states = (
                        hidden_slice.to(dtype=torch.bfloat16)
                        if str(store.exact_hidden_store_mode) == "packed_bf16"
                        else tuple(hidden.detach().clone() for hidden in hidden_slice)
                    )
                else:
                    hidden_states = None
                token_ids = ()
                if token_id_tensor is not None:
                    token_ids = tuple(int(value) for value in token_id_tensor.to(device="cpu", dtype=torch.long).tolist())
                return ExactBlockRecord(
                    block_id=int(block_id),
                    start_pos=int(store.current_block_start),
                    end_pos=int(store.current_block_start) + int(current_length),
                    token_ids=token_ids,
                    token_id_tensor=token_id_tensor,
                    token_cell_id_tensor=token_cell_id_tensor,
                    token_cell_weight_q_tensor=token_cell_weight_q_tensor,
                    hidden_states=hidden_states,
                    cell_keys=cell_keys,
                    cell_values_q=cell_values_q,
                    cell_counts=cell_counts,
                )
            current_device = (
                store.current_cell_key_sum.device
                if store.current_cell_key_sum is not None
                else torch.device("cpu")
            )
            return ExactBlockRecord(
                block_id=int(block_id),
                start_pos=int(store.current_block_start),
                end_pos=int(store.current_block_start) + len(store.current_block_token_ids),
                token_ids=tuple(int(value) for value in store.current_block_token_ids),
                token_id_tensor=torch.tensor(store.current_block_token_ids, dtype=torch.long)
                if store.current_block_token_ids
                else None,
                token_cell_id_tensor=torch.tensor(store.current_block_token_cell_ids, dtype=torch.long)
                if store.current_block_token_cell_ids
                else None,
                token_cell_weight_q_tensor=torch.tensor(store.current_block_token_cell_weights_q, dtype=torch.long)
                if store.current_block_token_cell_weights_q
                else None,
                hidden_states=(
                    torch.stack(
                        [
                            torch.as_tensor(hidden).detach().to(dtype=torch.bfloat16)
                            for hidden in store.current_block_hidden_states
                        ],
                        dim=0,
                    )
                    if store.current_block_hidden_states and str(store.exact_hidden_store_mode) == "packed_bf16"
                    else tuple(store.current_block_hidden_states)
                    if store.current_block_hidden_states
                    else None
                ),
                cell_keys=cell_keys,
                cell_values_q=cell_values_q,
                cell_counts=cell_counts,
            )
        return None

    def load_exact_block(self, *, batch_idx: int, block_id: int, block_len: int, cache_capacity: int) -> ExactBlockRecord | None:
        sample = self.samples[int(batch_idx)]
        cache = sample.exact_cache
        with runtime_range(
            "state.load_exact_block",
            mode=self.runtime_profiler_mode,
            recorder=self._runtime_profile_recorder,
            device=sample.exact_store.packed_token_ids.device if torch.is_tensor(sample.exact_store.packed_token_ids) else None,
        ):
            if int(block_id) in cache.cached_blocks:
                cache.block_ids_by_recency = [value for value in cache.block_ids_by_recency if int(value) != int(block_id)]
                cache.block_ids_by_recency.insert(0, int(block_id))
                return cache.cached_blocks[int(block_id)]
            record = self.resolve_exact_block(batch_idx, block_id, block_len)
            if record is None:
                return None
            record = self._materialize_exact_block(sample.exact_store, record)
            current_block_id = self.current_exact_block_id(batch_idx, block_len)
            if current_block_id is not None and int(current_block_id) == int(block_id):
                return record
            cache.cached_blocks[int(block_id)] = record
            cache.block_ids_by_recency = [value for value in cache.block_ids_by_recency if int(value) != int(block_id)]
            cache.block_ids_by_recency.insert(0, int(block_id))
            while len(cache.block_ids_by_recency) > int(max(1, cache_capacity)):
                evicted = int(cache.block_ids_by_recency.pop())
                cache.cached_blocks.pop(evicted, None)
            return record

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        if not self.samples:
            return 0
        return max(sample.seen_tokens for sample in self.samples)

    def get_mask_sizes(self, cache_position, layer_idx: int) -> tuple[int, int]:
        del layer_idx
        if cache_position is None:
            return self.get_seq_length(), 0
        try:
            last_pos = int(cache_position.reshape(-1)[-1].item())
        except Exception:
            last_pos = self.get_seq_length()
        return max(self.get_seq_length(), last_pos + 1), 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        del layer_idx
        return -1

    def update(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        del args, kwargs
        raise RuntimeError("GfstDecodeState does not store or update standard KV tensors")

    def sync_from_core_states(self, layer_states: dict[int, _CoreLayerCacheState]) -> None:
        self.core_layer_states = {
            int(layer_idx): _CoreLayerCacheState(
                fast_state=None if state.fast_state is None else state.fast_state.detach().clone(),
                mid_state=None if state.mid_state is None else state.mid_state.detach().clone(),
                slow_state=None if state.slow_state is None else state.slow_state.detach().clone(),
                fast_buffer=None if state.fast_buffer is None else state.fast_buffer.detach().clone(),
                mid_buffer=None if state.mid_buffer is None else state.mid_buffer.detach().clone(),
                slow_buffer=None if state.slow_buffer is None else state.slow_buffer.detach().clone(),
            )
            for layer_idx, state in layer_states.items()
        }

    @property
    def has_core_state(self) -> bool:
        return bool(self.core_layer_states)

    def reset(self) -> None:
        for batch_idx in range(len(self.samples)):
            self.samples[batch_idx] = _GfstSampleRuntimeState(
                seen_tokens=0,
                layers=[_GfstLayerRuntimeState() for _ in range(self.num_layers)],
            )
        self.core_layer_states = {}
        self._bump_version()

    def reorder_cache(self, beam_idx) -> None:
        indices = [int(value) for value in beam_idx.reshape(-1).tolist()]
        self.samples = [deepcopy(self.samples[index]) for index in indices]
        if self.core_layer_states:
            selected = beam_idx.to(dtype=torch.long)
            reordered = {}
            for layer_idx, state in self.core_layer_states.items():
                device_state = next(
                    (
                        candidate
                        for candidate in (state.fast_state, state.mid_state, state.slow_state, state.fast_buffer, state.mid_buffer, state.slow_buffer)
                        if candidate is not None
                    ),
                    None,
                )
                if device_state is None:
                    reordered[layer_idx] = deepcopy(state)
                    continue
                device_index = selected.to(device=device_state.device)
                reordered[layer_idx] = _CoreLayerCacheState(
                    fast_state=None if state.fast_state is None else state.fast_state.index_select(0, device_index.to(device=state.fast_state.device)),
                    mid_state=None if state.mid_state is None else state.mid_state.index_select(0, device_index.to(device=state.mid_state.device)),
                    slow_state=None if state.slow_state is None else state.slow_state.index_select(0, device_index.to(device=state.slow_state.device)),
                    fast_buffer=None if state.fast_buffer is None else state.fast_buffer.index_select(0, device_index.to(device=state.fast_buffer.device)),
                    mid_buffer=None if state.mid_buffer is None else state.mid_buffer.index_select(0, device_index.to(device=state.mid_buffer.device)),
                    slow_buffer=None if state.slow_buffer is None else state.slow_buffer.index_select(0, device_index.to(device=state.slow_buffer.device)),
                )
            self.core_layer_states = reordered
        self._bump_version()

    def crop(self, max_length: int) -> None:
        for sample in self.samples:
            sample.seen_tokens = min(sample.seen_tokens, int(max_length))
            sample.prepared_inputs = _PreparedInputState()
            sample.exact_store.completed_blocks = [
                record
                for record in sample.exact_store.completed_blocks
                if int(record.end_pos) <= int(max_length)
            ]
            if sample.exact_store.current_block_start >= int(max_length):
                sample.exact_store.current_block_start = int(max_length)
                sample.exact_store.current_block_token_ids = []
                sample.exact_store.current_block_token_cell_ids = []
                sample.exact_store.current_block_token_cell_weights_q = []
                sample.exact_store.current_block_hidden_states = []
            else:
                keep_count = max(0, int(max_length) - int(sample.exact_store.current_block_start))
                sample.exact_store.current_block_token_ids = sample.exact_store.current_block_token_ids[:keep_count]
                sample.exact_store.current_block_token_cell_ids = sample.exact_store.current_block_token_cell_ids[:keep_count]
                sample.exact_store.current_block_token_cell_weights_q = (
                    sample.exact_store.current_block_token_cell_weights_q[:keep_count]
                )
                sample.exact_store.current_block_hidden_states = sample.exact_store.current_block_hidden_states[:keep_count]
            self._clear_cell_buffers(sample.exact_store)
            valid_block_ids = {int(record.block_id) for record in sample.exact_store.completed_blocks}
            sample.exact_cache.cached_blocks = {
                int(block_id): record
                for block_id, record in sample.exact_cache.cached_blocks.items()
                if int(block_id) in valid_block_ids
            }
            sample.exact_cache.block_ids_by_recency = [
                int(block_id) for block_id in sample.exact_cache.block_ids_by_recency if int(block_id) in valid_block_ids
            ]
        self.core_layer_states = {}
        self._bump_version()

    def batch_repeat_interleave(self, repeats: int) -> None:
        expanded = []
        for sample in self.samples:
            for _ in range(int(repeats)):
                expanded.append(deepcopy(sample))
        self.samples = expanded
        if self.core_layer_states:
            repeated = {}
            for layer_idx, state in self.core_layer_states.items():
                repeated[layer_idx] = _CoreLayerCacheState(
                    fast_state=None if state.fast_state is None else state.fast_state.repeat_interleave(int(repeats), dim=0),
                    mid_state=None if state.mid_state is None else state.mid_state.repeat_interleave(int(repeats), dim=0),
                    slow_state=None if state.slow_state is None else state.slow_state.repeat_interleave(int(repeats), dim=0),
                    fast_buffer=None if state.fast_buffer is None else state.fast_buffer.repeat_interleave(int(repeats), dim=0),
                    mid_buffer=None if state.mid_buffer is None else state.mid_buffer.repeat_interleave(int(repeats), dim=0),
                    slow_buffer=None if state.slow_buffer is None else state.slow_buffer.repeat_interleave(int(repeats), dim=0),
                )
            self.core_layer_states = repeated
        self._bump_version()

    def batch_select_indices(self, indices) -> None:
        picked = [int(value) for value in indices.reshape(-1).tolist()]
        self.samples = [deepcopy(self.samples[index]) for index in picked]
        if self.core_layer_states:
            selected = indices.to(dtype=torch.long)
            reduced = {}
            for layer_idx, state in self.core_layer_states.items():
                device_state = next(
                    (
                        candidate
                        for candidate in (state.fast_state, state.mid_state, state.slow_state, state.fast_buffer, state.mid_buffer, state.slow_buffer)
                        if candidate is not None
                    ),
                    None,
                )
                if device_state is None:
                    reduced[layer_idx] = deepcopy(state)
                    continue
                device_index = selected.to(device=device_state.device)
                reduced[layer_idx] = _CoreLayerCacheState(
                    fast_state=None if state.fast_state is None else state.fast_state.index_select(0, device_index.to(device=state.fast_state.device)),
                    mid_state=None if state.mid_state is None else state.mid_state.index_select(0, device_index.to(device=state.mid_state.device)),
                    slow_state=None if state.slow_state is None else state.slow_state.index_select(0, device_index.to(device=state.slow_state.device)),
                    fast_buffer=None if state.fast_buffer is None else state.fast_buffer.index_select(0, device_index.to(device=state.fast_buffer.device)),
                    mid_buffer=None if state.mid_buffer is None else state.mid_buffer.index_select(0, device_index.to(device=state.mid_buffer.device)),
                    slow_buffer=None if state.slow_buffer is None else state.slow_buffer.index_select(0, device_index.to(device=state.slow_buffer.device)),
                )
            self.core_layer_states = reduced
        self._bump_version()

    @property
    def max_batch_size(self) -> int:
        return len(self.samples)

    @property
    def max_cache_len(self) -> int:
        return self.get_seq_length()

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_initialized(self) -> bool:
        return True

    @property
    def is_sliding(self) -> list[bool]:
        return [False for _ in range(self.num_layers)]

    def __len__(self) -> int:
        return self.num_layers


try:
    import torch
except ImportError:
    torch = None
