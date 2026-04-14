from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gfst_hmb_mpuc.modeling import MpucGfstHmbBlock, MpucGfstHmbForCausalLM
from gfst_hmb_mpuc.state import GfstDecodeState


@pytest.mark.skipif(not os.environ.get("GFST_HMB_MPUC_MODEL_PATH"), reason="GFST_HMB_MPUC_MODEL_PATH is required")
def test_student_forward_and_decode_round_trip() -> None:
    model_path = os.environ["GFST_HMB_MPUC_MODEL_PATH"]
    device_name = os.environ.get("GFST_HMB_SMOKE_DEVICE", "cpu")
    device = torch.device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = MpucGfstHmbForCausalLM.from_pretrained_teacher(model_path, torch_dtype=dtype).to(device=device, dtype=dtype)
    model.eval()
    model.assert_canonical_backbone()
    model.assert_no_standard_attention()

    input_ids = torch.tensor([[1, 2, 3, 4]], device=device, dtype=torch.long)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
    assert isinstance(outputs.past_key_values, GfstDecodeState)
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == 4

    decode_input = torch.tensor([[5]], device=device, dtype=torch.long)
    with torch.inference_mode():
        decoded = model(input_ids=decode_input, use_cache=True, past_key_values=outputs.past_key_values)
    assert isinstance(decoded.past_key_values, GfstDecodeState)
    assert decoded.past_key_values.get_seq_length() >= 5

    for layer in model.backbone.layers:
        assert isinstance(layer, MpucGfstHmbBlock)
        assert not hasattr(layer, "self_attn")
        assert hasattr(layer, "fast_branch")
        assert hasattr(layer, "mid_branch")
        assert hasattr(layer, "slow_branch")
