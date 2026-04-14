from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


def load_tokenizer_with_fallback(model_path: str, **kwargs: Any) -> Any:
    resolved_kwargs = dict(kwargs)
    try:
        return AutoTokenizer.from_pretrained(str(model_path), **resolved_kwargs)
    except (ImportError, ValueError) as exc:
        if bool(resolved_kwargs.get("use_fast") is False):
            raise
        message = str(exc)
        fallback_markers = (
            "backend tokenizer",
            "sentencepiece or tiktoken",
            "protobuf library",
        )
        if not any(marker in message for marker in fallback_markers):
            raise
        retry_kwargs = dict(resolved_kwargs)
        retry_kwargs["use_fast"] = False
        return AutoTokenizer.from_pretrained(str(model_path), **retry_kwargs)
