from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from gfst_hmb_mpuc.config import GfstHmbConfig
from gfst_hmb_mpuc.diagnostics import temp_local_qr_sweeps as sweeps
from gfst_hmb_mpuc.diagnostics.local_qr_divergence import reconstruct_local_qr_capture
from gfst_hmb_mpuc.gfst import canonical_qr_batched
from gfst_hmb_mpuc.modeling import (
    GfstHmbSublayer,
    _temp_local_qr_pool_records,
    _temp_local_qr_pool_reset,
)
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


def _native_qr(candidate: torch.Tensor, eps_sign: float) -> torch.Tensor:
    return canonical_qr_batched(candidate.unsqueeze(0), eps_sign=float(eps_sign)).squeeze(0)


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


def test_temp_pool_recorder_captures_first_n_native_payloads() -> None:
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

    _temp_local_qr_pool_reset(2)
    try:
        _delta, _ = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_values=decode_state,
            use_cache=False,
        )
        pool = _temp_local_qr_pool_records()
    finally:
        _temp_local_qr_pool_reset(0)

    assert len(pool) == 2
    assert pool[0]["metadata"]["capture_sequence"] == 0
    assert pool[1]["metadata"]["capture_sequence"] == 1
    assert pool[0]["metadata"]["temporary_marker"] == sweeps.TEMP_MARKER
    assert tuple(pool[0]["largest_candidate"].shape) == (4, 2)


def test_select_hardest_payloads_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = []
    for index in range(5):
        payload = copy.deepcopy(_materialize_payload(_native_qr))
        payload["metadata"]["capture_sequence"] = index
        payloads.append(payload)

    def _fake_analysis(payload, **_kwargs):
        capture_sequence = int(payload["metadata"]["capture_sequence"])
        return {
            "prefix_projector_max_abs": float(capture_sequence) / 10.0,
            "reconstructed_local_feature_max_abs": float(capture_sequence) / 100.0,
        }

    monkeypatch.setattr(sweeps, "analyze_local_qr_reflector_mismatch", _fake_analysis)
    selected, scores = sweeps.select_hardest_payloads_from_pool(payloads, max_count=3)

    assert [int(payload["metadata"]["capture_sequence"]) for payload in selected] == [4, 3, 2]
    assert [float(score["prefix_projector_max_abs"]) for score in scores[:3]] == [0.4, 0.3, 0.2]


def test_evaluate_suite_reports_householder_only_metrics_only_for_householder() -> None:
    candidate = torch.tensor(
        [
            [1.0, 0.2],
            [0.1, 0.95],
            [0.3, 0.1],
            [0.2, -0.05],
        ],
        dtype=torch.float32,
    )
    suite = sweeps.make_matrix_suite(candidate, suite_id="matrix_case", prefix_dims=(1, 2))
    householder = sweeps.evaluate_suite(
        suite,
        operator_name="householder_torch_geqrf_householder_product_f32",
        device=torch.device("cpu"),
        repeat_runs=2,
    )
    mgs = sweeps.evaluate_suite(
        suite,
        operator_name="mgs_custom_kernel",
        device=torch.device("cpu"),
        repeat_runs=2,
    )

    assert householder["metrics"]["compact_factors_max_abs"] is not None
    assert householder["metrics"]["tau_max_abs"] is not None
    assert mgs["metrics"]["compact_factors_max_abs"] is None
    assert mgs["metrics"]["tau_max_abs"] is None
    assert sweeps.tensor_byte_hash(candidate) == sweeps.tensor_byte_hash(candidate.clone())


def test_policy_derivation_and_ranking_are_deterministic() -> None:
    baseline = [
        {
            "suite_id": "fixture_largest",
            "metrics": {
                "explicit_q_max_abs": 1e-7,
                "signfixed_q_max_abs": 1e-7,
                "prefix_projector_max_abs": 2e-7,
                "orthogonality_error": 0.0,
                "qr_residual_error": 0.0,
                "reconstructed_local_feature_max_abs": 3e-8,
                "repeatability_delta": 4e-7,
            },
        }
    ]
    policy = sweeps.derive_native_repeatability_policy(baseline)
    assert policy["candidate_repeatability_ceilings"]["repeatability_delta"] >= 6e-7

    records = [
        {
            "operator_name": "householder_torch_qr_f32",
            "suite_id": "fixture_largest",
            "metrics": {
                "explicit_q_max_abs": 1e-7,
                "signfixed_q_max_abs": 1e-7,
                "prefix_projector_max_abs": 2e-7,
                "orthogonality_error": 0.0,
                "qr_residual_error": 0.0,
                "reconstructed_local_feature_max_abs": 1e-8,
                "repeatability_delta": 1e-7,
            },
        },
        {
            "operator_name": "mgs_custom_kernel",
            "suite_id": "fixture_largest",
            "metrics": {
                "explicit_q_max_abs": 4e-7,
                "signfixed_q_max_abs": 4e-7,
                "prefix_projector_max_abs": 8e-7,
                "orthogonality_error": 0.0,
                "qr_residual_error": 0.0,
                "reconstructed_local_feature_max_abs": 2e-7,
                "repeatability_delta": 2e-7,
            },
        },
        {
            "operator_name": "householder_torch_qr_f32",
            "suite_id": "synthetic_rank_deficient_64x32",
            "metrics": {
                "explicit_q_max_abs": 1e-7,
                "signfixed_q_max_abs": 1e-7,
                "prefix_projector_max_abs": 2e-7,
                "orthogonality_error": 0.0,
                "qr_residual_error": 0.0,
                "reconstructed_local_feature_max_abs": None,
                "repeatability_delta": 1e-7,
            },
        },
    ]

    ranked, ineligible = sweeps.rank_operator_records(records, repeatability_policy=policy)

    assert [item["operator_name"] for item in ranked] == ["householder_torch_qr_f32", "mgs_custom_kernel"]
    assert ineligible == []
