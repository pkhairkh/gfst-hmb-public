from __future__ import annotations

import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from .runtime_backends import sm120 as _sm120_runtime

try:  # pragma: no cover - optional accelerator dependency
    from torch_scatter import scatter_add as _scatter_add_impl
    from torch_scatter import scatter_max as _scatter_max_impl
    from torch_scatter import scatter_min as _scatter_min_impl
except ImportError:  # pragma: no cover - exercised via native fallback
    _scatter_add_impl = None
    _scatter_max_impl = None
    _scatter_min_impl = None


def torch_scatter_available() -> bool:
    return all(op is not None for op in (_scatter_add_impl, _scatter_max_impl, _scatter_min_impl))


def grouped_reduction_backend_available(backend: str) -> bool:
    backend_str = str(backend)
    if backend_str == "native":
        return True
    if backend_str == "torch_scatter":
        return torch_scatter_available()
    if backend_str == "cuda_segmented":
        return _sm120_runtime.segmented_reduction_available()
    return False


def _resolve_grouped_reduction_backend(backend: str) -> str:
    backend_str = str(backend)
    if backend_str not in {"native", "torch_scatter", "cuda_segmented"}:
        raise ValueError("grouped reduction backend must be one of {'native', 'torch_scatter', 'cuda_segmented'}")
    if backend_str == "cuda_segmented" and not _sm120_runtime.segmented_reduction_available():
        if torch_scatter_available():
            return "torch_scatter"
        return "native"
    if backend_str == "torch_scatter" and not torch_scatter_available():
        return "native"
    return backend_str


class RuntimeProfileRecorder:
    def __init__(self) -> None:
        self._totals_ms: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._value_totals: dict[str, float] = {}
        self._value_counts: dict[str, int] = {}

    def add(self, name: str, elapsed_ms: float) -> None:
        key = str(name)
        self._totals_ms[key] = self._totals_ms.get(key, 0.0) + float(elapsed_ms)
        self._counts[key] = self._counts.get(key, 0) + 1

    def add_value(self, name: str, value: float) -> None:
        key = str(name)
        self._value_totals[key] = self._value_totals.get(key, 0.0) + float(value)
        self._value_counts[key] = self._value_counts.get(key, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        keys = sorted(set(self._totals_ms) | set(self._value_totals))
        return {
            key: {
                **(
                    {
                        "total_ms": float(self._totals_ms[key]),
                        "count": int(self._counts.get(key, 0)),
                        "avg_ms": float(self._totals_ms[key]) / max(1, int(self._counts.get(key, 0))),
                    }
                    if key in self._totals_ms
                    else {}
                ),
                **(
                    {
                        "value_total": float(self._value_totals[key]),
                        "value_count": int(self._value_counts.get(key, 0)),
                        "value_avg": float(self._value_totals[key]) / max(1, int(self._value_counts.get(key, 0))),
                    }
                    if key in self._value_totals
                    else {}
                ),
            }
            for key in keys
        }


def _maybe_cuda_sync(device: torch.device | None) -> None:
    if device is None or device.type != "cuda":
        return
    torch.cuda.synchronize(device=device)


@contextmanager
def runtime_range(
    name: str,
    *,
    mode: str = "off",
    recorder: RuntimeProfileRecorder | None = None,
    device: torch.device | None = None,
):
    mode_str = str(mode)
    if mode_str == "off":
        yield
        return
    if mode_str == "nvtx":
        if device is not None and device.type == "cuda":
            torch.cuda.nvtx.range_push(str(name))
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
            return
        yield
        return
    if mode_str == "benchmark":
        _maybe_cuda_sync(device)
        start = time.perf_counter()
        try:
            yield
        finally:
            _maybe_cuda_sync(device)
            if recorder is not None:
                recorder.add(str(name), (time.perf_counter() - start) * 1000.0)
        return
    raise ValueError("runtime profiler mode must be one of {'off', 'nvtx', 'benchmark'}")


def collapse_max_scores_by_group(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    backend: str = "native",
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("scores and group_ids must be rank-1 tensors")
    if int(scores.numel()) != int(group_ids.numel()):
        raise ValueError("scores and group_ids must have the same number of elements")
    if int(scores.numel()) == 0:
        return group_ids.new_empty((0,), dtype=torch.long), scores.new_empty((0,), dtype=torch.float32)
    resolved_backend = _resolve_grouped_reduction_backend(backend)
    float_scores = scores.float()
    unique_ids, inverse = torch.unique(group_ids.to(dtype=torch.long), sorted=True, return_inverse=True)
    group_count = int(unique_ids.numel())
    if resolved_backend == "cuda_segmented":
        accelerated = _sm120_runtime.segmented_max_by_group(float_scores, inverse, group_count)
        if accelerated is not None:
            return unique_ids, accelerated
    if resolved_backend == "torch_scatter":
        grouped_scores = _scatter_max_impl(float_scores, inverse, dim=0, dim_size=group_count)[0]
        return unique_ids, grouped_scores
    grouped_scores = float_scores.new_full((group_count,), float("-inf"))
    grouped_scores.scatter_reduce_(0, inverse, float_scores, reduce="amax", include_self=True)
    return unique_ids, grouped_scores


def collapse_logsumexp_scores_and_occurrence(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    occurrence_blocks: torch.Tensor,
    occurrence_positions: torch.Tensor,
    *,
    backend: str = "native",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if any(tensor.ndim != 1 for tensor in (scores, group_ids, occurrence_blocks, occurrence_positions)):
        raise ValueError("collapse inputs must be rank-1 tensors")
    element_count = int(scores.numel())
    if any(int(tensor.numel()) != element_count for tensor in (group_ids, occurrence_blocks, occurrence_positions)):
        raise ValueError("collapse inputs must have matching element counts")
    if element_count == 0:
        empty_long = group_ids.new_empty((0,), dtype=torch.long)
        empty_scores = scores.new_empty((0,), dtype=torch.float32)
        return empty_long, empty_scores, empty_long, empty_long
    resolved_backend = _resolve_grouped_reduction_backend(backend)
    float_scores = scores.float()
    token_ids = group_ids.to(dtype=torch.long)
    block_ids = occurrence_blocks.to(dtype=torch.long)
    positions = occurrence_positions.to(dtype=torch.long)
    unique_ids, inverse = torch.unique(token_ids, sorted=True, return_inverse=True)
    group_count = int(unique_ids.numel())
    if resolved_backend == "cuda_segmented":
        accelerated = _sm120_runtime.segmented_logsumexp_occurrence(
            float_scores,
            inverse,
            block_ids,
            positions,
            group_count=group_count,
        )
        if accelerated is not None:
            grouped_scores, grouped_blocks, grouped_positions = accelerated
            return unique_ids, grouped_scores, grouped_blocks, grouped_positions
    if resolved_backend == "torch_scatter":
        grouped_max = _scatter_max_impl(float_scores, inverse, dim=0, dim_size=group_count)[0]
        stable_scores = torch.exp(float_scores - grouped_max.index_select(0, inverse))
        grouped_sumexp = _scatter_add_impl(stable_scores, inverse, dim=0, dim_size=group_count)
        grouped_scores = grouped_max + grouped_sumexp.clamp_min(1e-8).log()
        flat_indices = torch.arange(element_count, device=float_scores.device, dtype=torch.long)
        sentinel = torch.full_like(flat_indices, element_count)
        best_occurrence_indices = torch.where(
            float_scores.eq(grouped_max.index_select(0, inverse)),
            flat_indices,
            sentinel,
        )
        best_candidate_indices = _scatter_min_impl(best_occurrence_indices, inverse, dim=0, dim_size=group_count)[0]
    else:
        grouped_max = float_scores.new_full((group_count,), float("-inf"))
        grouped_max.scatter_reduce_(0, inverse, float_scores, reduce="amax", include_self=True)
        stable_scores = torch.exp(float_scores - grouped_max.index_select(0, inverse))
        grouped_sumexp = float_scores.new_zeros((group_count,))
        grouped_sumexp.scatter_add_(0, inverse, stable_scores)
        grouped_scores = grouped_max + grouped_sumexp.clamp_min(1e-8).log()
        flat_indices = torch.arange(element_count, device=float_scores.device, dtype=torch.long)
        best_candidate_indices = torch.full((group_count,), element_count, device=float_scores.device, dtype=torch.long)
        best_occurrence_indices = torch.where(
            float_scores.eq(grouped_max.index_select(0, inverse)),
            flat_indices,
            torch.full_like(flat_indices, element_count),
        )
        best_candidate_indices.scatter_reduce_(0, inverse, best_occurrence_indices, reduce="amin", include_self=True)
    grouped_blocks = block_ids.index_select(0, best_candidate_indices)
    grouped_positions = positions.index_select(0, best_candidate_indices)
    return unique_ids, grouped_scores, grouped_blocks, grouped_positions


@lru_cache(maxsize=1)
def get_liger_cross_entropy_cls() -> type[torch.nn.Module] | None:
    try:  # pragma: no cover - optional accelerator dependency
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    except ImportError:  # pragma: no cover - exercised via native fallback
        return None
    return LigerCrossEntropyLoss


@lru_cache(maxsize=1)
def get_liger_rms_norm_cls() -> type[torch.nn.Module] | None:
    try:  # pragma: no cover - optional accelerator dependency
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
    except ImportError:  # pragma: no cover - exercised via native fallback
        return None
    return LigerRMSNorm


@lru_cache(maxsize=4)
def _liger_cross_entropy_loss(ignore_index: int) -> torch.nn.Module | None:
    cls = get_liger_cross_entropy_cls()
    if cls is None:
        return None
    return cls(ignore_index=int(ignore_index), reduction="mean")


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.LongTensor,
    *,
    ignore_index: int = -100,
    backend: str = "native",
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, int(logits.shape[-1]))
    flat_targets = targets.reshape(-1)
    valid_mask = flat_targets.ne(int(ignore_index))
    if not bool(valid_mask.any().item()):
        return logits.new_zeros((), dtype=torch.float32)
    if str(backend) == "liger":
        liger_loss = _liger_cross_entropy_loss(int(ignore_index))
        if liger_loss is not None and flat_logits.is_cuda:
            return liger_loss(flat_logits, flat_targets).to(dtype=torch.float32)
    return F.cross_entropy(flat_logits[valid_mask], flat_targets[valid_mask])
