from __future__ import annotations

from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM
except ImportError as exc:  # pragma: no cover - guarded by remote smoke
    raise RuntimeError("transformers is required to use gfst_hmb_mpuc.teacher") from exc


def load_frozen_teacher(model_name_or_path: str, **kwargs: Any):
    if "torch_dtype" in kwargs and "dtype" not in kwargs:
        kwargs["dtype"] = kwargs.pop("torch_dtype")
    teacher = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
