from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gfst_hmb_mpuc.config import GfstHmbConfig
from gfst_hmb_mpuc.diagnostics.local_qr_divergence import (
    analyze_local_qr_divergence,
    find_earliest_local_feature_divergence,
    reconstruct_local_qr_capture,
)
from gfst_hmb_mpuc.gfst import canonical_qr_batched
from gfst_hmb_mpuc.modeling import GfstHmbSublayer
from gfst_hmb_mpuc.state import GfstDecodeState


def _diag_config() -> GfstHmbConfig:
    return GfstHmbConfig(
        reduced_dim=4,
        flag_dims=(1, 2),
        block_len=4,
        retrieval_depth=2,
        key_dim=4,
        forgetting_rates=(0.05, 0.1),
    )


def test_targeted_local_qr_capture_records_one_selected_event() -> None:
    torch.manual_seed(0)
    config = _diag_config()
    layer = GfstHmbSublayer(hidden_size=8, num_layers=1, layer_idx=0, gfst_config=config)
    layer.train()
    layer.training_path_mode = "fast"
    layer.set_gfst_feature_backend("native")

    hidden_states = torch.randn((1, 3, 8), dtype=torch.float32)
    position_ids = torch.arange(3, dtype=torch.long).unsqueeze(0)
    decode_state = GfstDecodeState(num_layers=1, batch_size=1)
    decode_state.prepare_inputs(
        0,
        positions=position_ids[0],
        token_ids=torch.tensor([3, 5, 7], dtype=torch.long),
    )
    layer._set_local_qr_capture_target(batch_idx=0, token_idx=1)

    _delta, _ = layer(
        hidden_states=hidden_states,
        position_ids=position_ids,
        past_key_values=decode_state,
        use_cache=False,
    )

    capture = layer._get_local_qr_capture()
    assert capture is not None
    assert layer._pending_local_qr_capture is None
    assert capture["metadata"]["layer_idx"] == 0
    assert capture["metadata"]["batch_idx"] == 0
    assert capture["metadata"]["token_idx"] == 1
    assert capture["metadata"]["absolute_position"] == 1
    assert tuple(capture["largest_candidate"].shape) == (4, 2)
    assert tuple(capture["largest_basis_in"].shape) == (4, 2)
    assert tuple(capture["largest_basis_out"].shape) == (4, 2)
    assert tuple(capture["projected"].shape) == (2,)
    assert tuple(capture["local_features"].shape) == (4,)
    assert len(capture["nested_steps"]) == 1
    nested = capture["nested_steps"][0]
    assert nested["nested_index"] == 0
    assert nested["dim"] == 1
    assert tuple(nested["nested_basis_in"].shape) == (2, 1)
    assert tuple(nested["nested_candidate"].shape) == (2, 1)
    assert tuple(nested["nested_basis_out"].shape) == (2, 1)


def test_find_earliest_local_feature_divergence_prefers_earlier_layer_then_token() -> None:
    native = [
        torch.zeros((1, 4, 6), dtype=torch.float32),
        torch.zeros((1, 4, 6), dtype=torch.float32),
    ]
    variant = [
        torch.zeros((1, 4, 6), dtype=torch.float32),
        torch.zeros((1, 4, 6), dtype=torch.float32),
    ]
    variant[0][0, 2, 1] = 1e-4
    variant[1][0, 0, 0] = 1.0

    divergence = find_earliest_local_feature_divergence(native, variant)

    assert divergence is not None
    assert divergence["layer_idx"] == 0
    assert divergence["batch_idx"] == 0
    assert divergence["token_idx"] == 2
    assert divergence["feature_idx"] == 1
    assert divergence["local_dim"] == 3


def _native_qr(candidate: torch.Tensor, eps_sign: float) -> torch.Tensor:
    return canonical_qr_batched(candidate.unsqueeze(0), eps_sign=float(eps_sign)).squeeze(0)


def _rotated_kernel_qr(candidate: torch.Tensor, eps_sign: float) -> torch.Tensor:
    q = _native_qr(candidate, eps_sign)
    if int(q.shape[1]) == 2:
        rotation = torch.tensor(
            [[0.9659258, -0.2588190], [0.2588190, 0.9659258]],
            dtype=q.dtype,
            device=q.device,
        )
        return q @ rotation
    return q


def _base_payload() -> dict[str, object]:
    return {
        "metadata": {
            "layer_idx": 0,
            "batch_idx": 0,
            "token_idx": 0,
            "absolute_position": 0,
            "flag_dims": [1, 2],
            "max_flag_dim": 2,
            "eps": 1e-6,
            "eps_sign": 1e-8,
        },
        "reduced_token": torch.tensor([0.75, -1.25, 0.5, 0.25], dtype=torch.float32),
        "largest_basis_in": torch.eye(4, 2, dtype=torch.float32),
        "largest_candidate": torch.tensor(
            [
                [1.0, 0.2],
                [0.1, 0.95],
                [0.3, 0.1],
                [0.2, -0.05],
            ],
            dtype=torch.float32,
        ),
        "nested_steps": [
            {
                "nested_index": 0,
                "dim": 1,
                "nested_basis_in": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
                "nested_candidate": torch.tensor([[0.9], [0.25]], dtype=torch.float32),
            }
        ],
    }


def _materialize_payload(qr_operator) -> dict[str, object]:
    payload = _base_payload()
    replay = reconstruct_local_qr_capture(payload, qr_operator=qr_operator)
    payload["largest_basis_out"] = replay["largest_basis_out"]
    payload["projected"] = replay["projected"]
    payload["local_features"] = replay["local_features"]
    for nested_payload, nested_replay in zip(payload["nested_steps"], replay["nested_steps"]):
        nested_payload["nested_basis_out"] = nested_replay["nested_basis_out"]
    return payload


def test_replay_analysis_classifies_qr_operator_mismatch_for_same_candidates() -> None:
    native_payload = _materialize_payload(_native_qr)
    variant_payload = _materialize_payload(_rotated_kernel_qr)

    analysis = analyze_local_qr_divergence(
        native_payload,
        variant_payload,
        native_qr_operator=_native_qr,
        kernel_qr_operator=_rotated_kernel_qr,
    )

    assert analysis["classification"] == "qr_operator_mismatch"
    assert analysis["largest"]["candidate_max_abs"] == 0.0
    assert analysis["rollout_local_feature_max_abs"] > 0.0
