from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import FaissGpuRetrievalConfig

try:
    import torch
except ImportError:
    torch = None

try:
    import faiss
    import faiss.contrib.torch_utils  # noqa: F401
except ImportError:
    faiss = None


@dataclass
class _FaissIndexRecord:
    index: Any
    state_version: int
    summary_count: int
    device_index: int
    index_type: str


class FaissGpuRetriever:
    def __init__(self, config: FaissGpuRetrievalConfig) -> None:
        self.config = config
        self._resources: dict[int, Any] = {}
        self._index_cache: dict[tuple[int, int, int], _FaissIndexRecord] = {}

    def _resource(self, device_index: int):
        if faiss is None:
            raise RuntimeError("source-built FAISS with Python GPU bindings is required for runtime retrieval")
        resource = self._resources.get(device_index)
        if resource is None:
            resource = faiss.StandardGpuResources()
            temp_memory_bytes = int(self.config.temp_memory_bytes)
            if temp_memory_bytes <= 0 and hasattr(resource, "noTempMemory"):
                resource.noTempMemory()
            elif hasattr(resource, "setTempMemory"):
                resource.setTempMemory(int(max(0, temp_memory_bytes)))
            self._resources[device_index] = resource
        return resource

    def _ensure_runtime_device(self, tensor) -> int:
        if torch is None:
            raise RuntimeError("torch is required for FAISS-GPU retrieval")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for FAISS-GPU retrieval")
        if tensor.device.type != "cuda":
            raise RuntimeError("FAISS-GPU retrieval requires CUDA tensors")
        if tensor.device.index != int(self.config.gpu_device):
            raise RuntimeError(
                f"FAISS-GPU retrieval is pinned to cuda:{self.config.gpu_device}, got {tensor.device}"
            )
        return int(tensor.device.index)

    def _flat_config(self, device_index: int):
        config = faiss.GpuIndexFlatConfig()
        config.device = int(device_index)
        config.useFloat16 = bool(self.config.use_float16)
        return config

    def _ivf_config(self, device_index: int):
        config = faiss.GpuIndexIVFFlatConfig()
        config.device = int(device_index)
        config.useFloat16 = bool(self.config.use_float16)
        return config

    def _make_index(self, device_index: int, dim: int, count: int):
        metric = faiss.METRIC_INNER_PRODUCT
        if self.config.index_type == "ivfflatip" and count >= max(
            self.config.flat_fallback_threshold,
            self.config.nlist * self.config.min_train_points_per_centroid,
        ):
            return faiss.GpuIndexIVFFlat(
                self._resource(device_index),
                dim,
                int(min(self.config.nlist, count)),
                metric,
                self._ivf_config(device_index),
            ), "ivfflatip"
        return faiss.GpuIndexFlatIP(
            self._resource(device_index),
            dim,
            self._flat_config(device_index),
        ), "flatip"

    def _build_record(self, key: tuple[int, int, int], keys_tensor, state_version: int) -> _FaissIndexRecord:
        device_index = self._ensure_runtime_device(keys_tensor)
        count = int(keys_tensor.shape[0])
        dim = int(keys_tensor.shape[1])
        index, index_type = self._make_index(device_index, dim, count)
        if hasattr(index, "setNumProbes"):
            index.setNumProbes(int(self.config.nprobe))
        elif hasattr(index, "nprobe"):
            index.nprobe = int(self.config.nprobe)
        keys = keys_tensor.contiguous().to(dtype=torch.float32)
        if hasattr(index, "train") and not index.is_trained:
            index.train(keys)
        index.add(keys)
        distances, indices = index.search(keys[:1], 1)
        if int(indices.numel()) == 0 or int(indices.reshape(-1)[0].item()) < 0:
            raise RuntimeError("FAISS-GPU probe failed after index construction")
        record = _FaissIndexRecord(
            index=index,
            state_version=int(state_version),
            summary_count=count,
            device_index=device_index,
            index_type=index_type,
        )
        self._index_cache[key] = record
        return record

    def _cache_key(self, state: Any, layer_idx: int, batch_idx: int) -> tuple[int, int, int]:
        return (id(state), int(layer_idx), int(batch_idx))

    def _get_record(self, *, state: Any, layer_idx: int, batch_idx: int, keys_tensor) -> _FaissIndexRecord:
        cache_key = self._cache_key(state, layer_idx, batch_idx)
        record = self._index_cache.get(cache_key)
        summary_count = int(keys_tensor.shape[0])
        if record is None:
            return self._build_record(cache_key, keys_tensor, state.state_version)
        if (
            record.state_version != int(state.state_version)
            or record.summary_count != summary_count
            or record.device_index != int(self.config.gpu_device)
        ):
            return self._build_record(cache_key, keys_tensor, state.state_version)
        return record

    def search(
        self,
        *,
        state: Any,
        layer_idx: int,
        batch_idx: int,
        keys_tensor,
        query_tensor,
        top_k: int,
    ):
        if torch is None:
            raise RuntimeError("torch is required for FAISS-GPU retrieval")
        if int(keys_tensor.shape[0]) == 0:
            return (
                torch.empty(0, device=query_tensor.device, dtype=torch.long),
                torch.empty(0, device=query_tensor.device, dtype=torch.float32),
            )
        self._ensure_runtime_device(query_tensor)
        self._ensure_runtime_device(keys_tensor)
        record = self._get_record(state=state, layer_idx=layer_idx, batch_idx=batch_idx, keys_tensor=keys_tensor)
        query = query_tensor.contiguous().to(dtype=torch.float32)
        if query.ndim == 1:
            query = query.unsqueeze(0)
        distances, indices = record.index.search(query, min(int(top_k), int(keys_tensor.shape[0])))
        index_tensor = indices[0].to(device=query_tensor.device, dtype=torch.long)
        score_tensor = distances[0].to(device=query_tensor.device, dtype=torch.float32)
        valid = index_tensor >= 0
        return index_tensor[valid], score_tensor[valid]

    @staticmethod
    def exact_topk(keys_tensor, query_tensor, top_k: int):
        if query_tensor.ndim == 1:
            scores = keys_tensor @ query_tensor
        else:
            scores = keys_tensor @ query_tensor[0]
        values, indices = torch.topk(scores, k=min(int(top_k), int(keys_tensor.shape[0])), dim=0)
        return indices.to(dtype=torch.long), values.to(dtype=torch.float32)
