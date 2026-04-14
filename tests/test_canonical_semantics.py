from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gfst_hmb_mpuc.config import GfstHmbConfig
from gfst_hmb_mpuc.gfst import basis_projector
from gfst_hmb_mpuc.modeling import GfstHmbAttention
from gfst_hmb_mpuc.state import GfstBlockSummary, GfstDecodeState


def _make_config(*, retrieval_temperature: float = 1.0, block_len: int = 2) -> GfstHmbConfig:
    return GfstHmbConfig(
        reduced_dim=4,
        flag_dims=(1, 2),
        block_len=block_len,
        retrieval_depth=2,
        key_dim=2,
        forgetting_rates=(0.1, 0.2),
        retrieval_temperature=retrieval_temperature,
    )


def _build_attention(cfg: GfstHmbConfig) -> GfstHmbAttention:
    attention = GfstHmbAttention(
        hidden_size=4,
        num_layers=1,
        layer_idx=0,
        gfst_config=cfg,
        rope_theta=10000.0,
    )
    with torch.no_grad():
        attention.reduce_proj.weight.copy_(torch.eye(4))
        attention.query_proj.weight.zero_()
        attention.query_proj.weight[:2, :2] = torch.eye(2)
        attention.block_key_proj.weight.zero_()
        attention.block_key_proj.weight[:2, :2] = torch.eye(2)

        attention.feature_gate.weight.zero_()
        attention.feature_gate.bias.zero_()
        attention.feature_gate.weight[0, 4] = 1.0

        attention.feature_value.weight.zero_()
        attention.feature_value.bias.zero_()
        attention.feature_value.weight[0, 4] = 1.0
    return attention


def _prepare_inputs(state: GfstDecodeState, position_ids: torch.LongTensor, *, token_ids: torch.LongTensor | None = None) -> None:
    if token_ids is None:
        token_ids = torch.zeros_like(position_ids)
    for batch_idx in range(int(position_ids.shape[0])):
        state.prepare_inputs(
            batch_idx,
            positions=position_ids[batch_idx],
            token_ids=token_ids[batch_idx],
        )


def _state_with_summaries() -> GfstDecodeState:
    state = GfstDecodeState(num_layers=1, batch_size=1)
    layer_state = state.layer_state(0, 0)
    basis_a = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    basis_b = torch.tensor(
        [[0.70710677, 0.0], [0.70710677, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    layer_state.block_summaries = [
        GfstBlockSummary(basis=basis_a, key=torch.tensor([1.0, 0.0]), block_index=0, start_pos=0, end_pos=2, token_count=2),
        GfstBlockSummary(basis=basis_b, key=torch.tensor([0.9, 0.0]), block_index=1, start_pos=2, end_pos=4, token_count=2),
    ]
    return state


def test_block_summary_depends_only_on_within_block_tokens() -> None:
    attention = _build_attention(_make_config())
    prefix = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    target_block = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )

    state_with_prefix = GfstDecodeState(num_layers=1, batch_size=1)
    position_with_prefix = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    _prepare_inputs(state_with_prefix, position_with_prefix)
    with torch.inference_mode():
        attention(
            hidden_states=torch.cat((prefix, target_block), dim=1),
            position_ids=position_with_prefix,
            past_key_values=state_with_prefix,
            use_cache=True,
        )
    summary_with_prefix = state_with_prefix.layer_state(0, 0).block_summaries[1]

    state_block_only = GfstDecodeState(num_layers=1, batch_size=1)
    position_block_only = torch.tensor([[2, 3]], dtype=torch.long)
    _prepare_inputs(state_block_only, position_block_only)
    with torch.inference_mode():
        attention(
            hidden_states=target_block,
            position_ids=position_block_only,
            past_key_values=state_block_only,
            use_cache=True,
        )
    summary_block_only = state_block_only.layer_state(0, 0).block_summaries[0]

    assert torch.allclose(summary_with_prefix.basis, summary_block_only.basis, atol=1e-5)
    assert torch.allclose(summary_with_prefix.key, summary_block_only.key, atol=1e-5)


def test_retrieval_temperature_changes_global_fusion() -> None:
    def make_temperature_attention(temperature: float) -> GfstHmbAttention:
        attention = GfstHmbAttention(
            hidden_size=4,
            num_layers=1,
            layer_idx=0,
            gfst_config=GfstHmbConfig(
                reduced_dim=4,
                flag_dims=(1,),
                block_len=2,
                retrieval_depth=2,
                key_dim=2,
                forgetting_rates=(0.2,),
                retrieval_temperature=temperature,
            ),
            rope_theta=10000.0,
        )
        with torch.no_grad():
            attention.reduce_proj.weight.copy_(torch.eye(4))
            attention.query_proj.weight.zero_()
            attention.query_proj.weight[:2, :2] = torch.eye(2)
            attention.block_key_proj.weight.zero_()
            attention.block_key_proj.weight[:2, :2] = torch.eye(2)
            attention.feature_gate.weight.zero_()
            attention.feature_gate.bias.zero_()
            attention.feature_gate.weight[0, 2] = 1.0
            attention.feature_value.weight.zero_()
            attention.feature_value.bias.zero_()
            attention.feature_value.weight[0, 2] = 1.0
        return attention

    def make_temperature_state() -> GfstDecodeState:
        state = GfstDecodeState(num_layers=1, batch_size=1)
        layer_state = state.layer_state(0, 0)
        layer_state.block_summaries = [
            GfstBlockSummary(
                basis=torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
                key=torch.tensor([1.0, 0.0], dtype=torch.float32),
                block_index=0,
                start_pos=0,
                end_pos=2,
                token_count=2,
            ),
            GfstBlockSummary(
                basis=torch.tensor([[0.70710677], [0.70710677], [0.0], [0.0]], dtype=torch.float32),
                key=torch.tensor([0.0, 0.0], dtype=torch.float32),
                block_index=1,
                start_pos=2,
                end_pos=4,
                token_count=2,
            ),
        ]
        return state

    cold_attention = make_temperature_attention(0.1)
    hot_attention = make_temperature_attention(10.0)
    hot_attention.load_state_dict(cold_attention.state_dict())

    hidden_states = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    position_ids = torch.tensor([[0]], dtype=torch.long)
    with torch.inference_mode():
        cold_state = make_temperature_state()
        hot_state = make_temperature_state()
        _prepare_inputs(cold_state, position_ids)
        _prepare_inputs(hot_state, position_ids)
        cold_output, _ = cold_attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_values=cold_state,
            use_cache=True,
        )
        hot_output, _ = hot_attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_values=hot_state,
            use_cache=True,
        )

    assert not torch.allclose(cold_output, hot_output, atol=1e-6)


def test_block_memory_grows_by_full_blocks() -> None:
    attention = _build_attention(_make_config(block_len=2))
    hidden_states = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]],
        dtype=torch.float32,
    )
    state = GfstDecodeState(num_layers=1, batch_size=1)
    position_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    _prepare_inputs(state, position_ids)

    with torch.inference_mode():
        attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_values=state,
            use_cache=True,
        )

    layer_state = state.layer_state(0, 0)
    assert len(layer_state.block_summaries) == 2
    assert layer_state.block_tracker.current_block_tokens == 1
    for summary in layer_state.block_summaries:
        assert summary.basis.shape == (4, 2)


def test_basis_storage_reconstructs_projector() -> None:
    summary = _state_with_summaries().layer_state(0, 0).block_summaries[0]
    projector = basis_projector(summary.basis)

    assert projector.shape == (4, 4)
    assert torch.allclose(projector, projector.transpose(0, 1), atol=1e-6)


def test_block_memory_uses_basis_storage_count() -> None:
    cfg = _make_config()
    state = _state_with_summaries()
    summaries = state.layer_state(0, 0).block_summaries

    stored_floats = sum(summary.basis.numel() + summary.key.numel() for summary in summaries)
    expected_per_block = cfg.reduced_dim * cfg.max_flag_dim + cfg.key_dim

    assert stored_floats == len(summaries) * expected_per_block


def test_block_key_uses_canonical_top_eigenvector_half() -> None:
    cfg = _make_config(block_len=2)
    attention = _build_attention(cfg)
    with torch.no_grad():
        attention.block_key_proj.weight.zero_()
        attention.block_key_proj.weight[0, 0] = 1.0
        attention.block_key_proj.weight[1, cfg.reduced_dim + 0] = 1.0

    hidden_states = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.8, 0.1, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    position_ids = torch.tensor([[0, 1]], dtype=torch.long)
    state = GfstDecodeState(num_layers=1, batch_size=1)
    _prepare_inputs(state, position_ids)

    with torch.inference_mode():
        attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_values=state,
            use_cache=True,
        )

    summary = state.layer_state(0, 0).block_summaries[0]
    assert float(summary.key[0].item()) != 0.0
    assert float(summary.key[1].item()) != 0.0
