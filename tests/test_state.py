from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gfst_hmb_mpuc.state import GfstDecodeState


def test_state_tracks_sequence_length() -> None:
    state = GfstDecodeState(num_layers=4, batch_size=2)
    assert state.get_seq_length() == 0
    state.advance(0, 5)
    state.advance(1, 3)
    assert state.get_seq_length() == 5


def test_standard_cache_update_is_forbidden() -> None:
    state = GfstDecodeState(num_layers=2, batch_size=1)
    with pytest.raises(RuntimeError):
        state.update(None, None, 0)


def test_state_version_bumps_on_reorder() -> None:
    state = GfstDecodeState(num_layers=2, batch_size=2)
    before = state.state_version
    state.reorder_cache(torch.tensor([1, 0], dtype=torch.long))
    assert state.state_version == before + 1


@pytest.mark.parametrize("state_pack_backend", ["native", "cuda_packed"])
def test_exact_block_hidden_states_roundtrip_from_packed_bf16_store(state_pack_backend: str) -> None:
    state = GfstDecodeState(num_layers=1, batch_size=1, state_pack_backend=state_pack_backend)
    hidden_a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    hidden_b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    state.append_exact_token(
        batch_idx=0,
        absolute_position=0,
        token_id=10,
        block_len=2,
        hidden_state=hidden_a,
        exact_hidden_store_mode="packed_bf16",
    )
    state.append_exact_token(
        batch_idx=0,
        absolute_position=1,
        token_id=11,
        block_len=2,
        hidden_state=hidden_b,
        exact_hidden_store_mode="packed_bf16",
    )

    record = state.load_exact_block(batch_idx=0, block_id=0, block_len=2, cache_capacity=2)
    assert record is not None
    assert torch.is_tensor(record.hidden_states)
    assert record.hidden_states.dtype == torch.bfloat16
    assert tuple(record.hidden_states.shape) == (2, 3)
    assert torch.allclose(record.hidden_states.float(), torch.stack((hidden_a, hidden_b), dim=0), atol=1e-3, rtol=1e-3)


def test_cuda_packed_current_exact_block_roundtrip_before_commit() -> None:
    state = GfstDecodeState(num_layers=1, batch_size=1, state_pack_backend="cuda_packed")
    hidden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    state.append_exact_token(
        batch_idx=0,
        absolute_position=4,
        token_id=17,
        block_len=4,
        hidden_state=hidden,
        exact_hidden_store_mode="packed_bf16",
        cells_per_block=8,
        centroid_key_dim=3,
        centroid_value_dim=2,
        cell_ids=torch.tensor([2, 5], dtype=torch.long),
        cell_weights=torch.tensor([0.75, 0.25], dtype=torch.float32),
        cell_key=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        cell_value=torch.tensor([0.4, 0.5], dtype=torch.float32),
    )

    assert state.current_exact_block_id(0, 4) == 1
    assert state.current_exact_block_length(0) == 1

    record = state.resolve_exact_block(0, 1, 4)
    assert record is not None
    assert record.start_pos == 4
    assert record.end_pos == 5
    assert record.token_ids == (17,)
    assert torch.equal(record.token_id_tensor, torch.tensor([17], dtype=torch.long))
    assert torch.equal(record.token_cell_id_tensor, torch.tensor([[2, 5]], dtype=torch.long))
    assert torch.equal(record.token_cell_weight_q_tensor, torch.tensor([[191, 64]], dtype=torch.long))
    assert torch.is_tensor(record.hidden_states)
    assert record.hidden_states.dtype == torch.bfloat16
    assert torch.allclose(record.hidden_states.float(), hidden.reshape(1, -1), atol=1e-3, rtol=1e-3)
