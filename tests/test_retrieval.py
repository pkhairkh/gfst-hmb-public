from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("faiss")

from gfst_hmb_mpuc.config import FaissGpuRetrievalConfig
from gfst_hmb_mpuc.retrieval import FaissGpuRetriever
from gfst_hmb_mpuc.state import GfstDecodeState


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FAISS-GPU tests")
def test_faiss_gpu_matches_exact_oracle_on_small_bank() -> None:
    retriever = FaissGpuRetriever(
        FaissGpuRetrievalConfig(
            gpu_device=0,
            index_type="ivfflatip",
            nlist=4,
            nprobe=2,
            flat_fallback_threshold=64,
        )
    )
    state = GfstDecodeState(num_layers=1, batch_size=1)
    keys = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.8, 0.2, 0.0, 0.0],
        ],
        device="cuda:0",
        dtype=torch.float32,
    )
    query = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0", dtype=torch.float32)

    indices, scores = retriever.search(
        state=state,
        layer_idx=0,
        batch_idx=0,
        keys_tensor=keys,
        query_tensor=query,
        top_k=3,
    )
    exact_indices, exact_scores = FaissGpuRetriever.exact_topk(keys, query, 3)

    assert indices.tolist() == exact_indices.tolist()
    assert torch.allclose(scores, exact_scores, atol=1e-5)
    cache_key = (id(state), 0, 0)
    assert retriever._index_cache[cache_key].device_index == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FAISS-GPU tests")
def test_retriever_rebuilds_when_state_version_changes() -> None:
    retriever = FaissGpuRetriever(FaissGpuRetrievalConfig(gpu_device=0, flat_fallback_threshold=64))
    state = GfstDecodeState(num_layers=1, batch_size=1)
    keys = torch.eye(4, device="cuda:0", dtype=torch.float32)
    query = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0", dtype=torch.float32)

    retriever.search(
        state=state,
        layer_idx=0,
        batch_idx=0,
        keys_tensor=keys,
        query_tensor=query,
        top_k=2,
    )
    cache_key = (id(state), 0, 0)
    before = retriever._index_cache[cache_key].state_version
    state.batch_select_indices(torch.tensor([0], dtype=torch.long))
    retriever.search(
        state=state,
        layer_idx=0,
        batch_idx=0,
        keys_tensor=keys,
        query_tensor=query,
        top_k=2,
    )
    after = retriever._index_cache[cache_key].state_version
    assert after > before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FAISS-GPU tests")
def test_retriever_fails_fast_on_noncanonical_device() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires a second CUDA device to validate fail-fast device pinning")

    retriever = FaissGpuRetriever(FaissGpuRetrievalConfig(gpu_device=0, flat_fallback_threshold=64))
    state = GfstDecodeState(num_layers=1, batch_size=1)
    keys = torch.eye(4, device="cuda:1", dtype=torch.float32)
    query = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:1", dtype=torch.float32)

    with pytest.raises(RuntimeError):
        retriever.search(
            state=state,
            layer_idx=0,
            batch_idx=0,
            keys_tensor=keys,
            query_tensor=query,
            top_k=2,
        )


def test_retriever_rejects_cpu_tensors() -> None:
    retriever = FaissGpuRetriever(FaissGpuRetrievalConfig(gpu_device=0, flat_fallback_threshold=64))
    state = GfstDecodeState(num_layers=1, batch_size=1)
    keys = torch.eye(4, dtype=torch.float32)
    query = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    with pytest.raises(RuntimeError):
        retriever.search(
            state=state,
            layer_idx=0,
            batch_idx=0,
            keys_tensor=keys,
            query_tensor=query,
            top_k=2,
        )
