from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
import gzip
import io
import json
from pathlib import Path
import queue
import random
import threading
import time
from typing import Any, Iterator, Mapping, Sequence
import urllib.request

import torch

from ..config import DatasetStreamConfig

try:
    from datasets import load_dataset
    try:
        from datasets.utils.logging import disable_progress_bar, set_verbosity_error
    except ImportError:  # pragma: no cover - guarded remotely
        disable_progress_bar = None
        set_verbosity_error = None
except ImportError as exc:  # pragma: no cover - guarded remotely
    raise RuntimeError("datasets is required to use gfst_hmb_mpuc.training.data") from exc


def _configure_datasets_logging() -> None:
    if set_verbosity_error is not None:
        set_verbosity_error()
    if disable_progress_bar is not None:
        disable_progress_bar()


def load_text_dataset(
    dataset_config: DatasetStreamConfig,
    *,
    split: str,
    streaming: bool,
):
    _configure_datasets_logging()
    load_kwargs: dict[str, Any] = {
        "path": dataset_config.dataset_name,
        "name": dataset_config.dataset_config,
        "split": split,
        "streaming": bool(streaming),
    }
    if dataset_config.cache_dir:
        load_kwargs["cache_dir"] = str(dataset_config.cache_dir)
    return load_dataset(**load_kwargs)


def _json_or_yaml_loads(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("expected manifest root to be a mapping or list")
    return dict(payload)


def _write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(target)


@dataclass(frozen=True)
class StreamingSourceSpec:
    source_id: str
    family: str
    stream_mode: str
    dataset_name: str = ""
    dataset_config: str = ""
    train_split: str = "train"
    validation_split: str = "validation"
    text_field: str = "text"
    text_fields: tuple[str, ...] = ()
    base_weight: float = 1.0
    license_tag: str = ""
    uri_or_repo: str = ""
    allow_streaming_only: bool = True
    include_field_values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    exclude_field_values: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.source_id):
            raise ValueError("source_id must not be empty")
        if str(self.family) not in {"books", "science", "wiki", "code", "background"}:
            raise ValueError("family must be one of {'books', 'science', 'wiki', 'code', 'background'}")
        if str(self.stream_mode) not in {"hf_streaming", "http_jsonl_gz", "local_jsonl_gz"}:
            raise ValueError("stream_mode must be one of {'hf_streaming', 'http_jsonl_gz', 'local_jsonl_gz'}")
        if float(self.base_weight) <= 0.0:
            raise ValueError("base_weight must be positive")
        if not bool(self.allow_streaming_only):
            raise ValueError("all sources in the streaming corpus manifest must set allow_streaming_only=true")
        if str(self.stream_mode) == "hf_streaming" and not str(self.dataset_name or self.uri_or_repo):
            raise ValueError("hf_streaming sources require dataset_name or uri_or_repo")
        if str(self.stream_mode) in {"http_jsonl_gz", "local_jsonl_gz"} and not str(self.uri_or_repo):
            raise ValueError("jsonl.gz sources require uri_or_repo")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StreamingSourceSpec":
        return cls(
            source_id=str(payload.get("source_id", "") or ""),
            family=str(payload.get("family", "") or ""),
            stream_mode=str(payload.get("stream_mode", "") or ""),
            dataset_name=str(payload.get("dataset_name", "") or ""),
            dataset_config=str(payload.get("dataset_config", "") or ""),
            train_split=str(payload.get("train_split", "train") or "train"),
            validation_split=str(payload.get("validation_split", "validation") or "validation"),
            text_field=str(payload.get("text_field", "text") or "text"),
            text_fields=tuple(str(value) for value in payload.get("text_fields", []) or [] if str(value)),
            base_weight=float(payload.get("base_weight", 1.0)),
            license_tag=str(payload.get("license_tag", "") or ""),
            uri_or_repo=str(payload.get("uri_or_repo", "") or ""),
            allow_streaming_only=bool(payload.get("allow_streaming_only", True)),
            include_field_values={
                str(key): tuple(str(item) for item in (values if isinstance(values, list) else [values]))
                for key, values in dict(payload.get("include_field_values", {}) or {}).items()
            },
            exclude_field_values={
                str(key): tuple(str(item) for item in (values if isinstance(values, list) else [values]))
                for key, values in dict(payload.get("exclude_field_values", {}) or {}).items()
            },
        )


@dataclass(frozen=True)
class StreamWindowRecord:
    sequence_length: int
    input_ids: list[int]
    labels: list[int]
    oracle_target_block_ids: list[int]
    oracle_mask: list[bool]
    source_id: str
    doc_id: str
    window_start: int
    bucket: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "sequence_length": int(self.sequence_length),
            "input_ids": list(self.input_ids),
            "labels": list(self.labels),
            "oracle_target_block_ids": list(self.oracle_target_block_ids),
            "oracle_mask": list(self.oracle_mask),
            "source_id": str(self.source_id),
            "doc_id": str(self.doc_id),
            "window_start": int(self.window_start),
            "bucket": str(self.bucket),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StreamWindowRecord":
        return cls(
            sequence_length=int(payload["sequence_length"]),
            input_ids=[int(value) for value in payload["input_ids"]],
            labels=[int(value) for value in payload["labels"]],
            oracle_target_block_ids=[int(value) for value in payload.get("oracle_target_block_ids", [])],
            oracle_mask=[bool(value) for value in payload.get("oracle_mask", [])],
            source_id=str(payload.get("source_id", "") or ""),
            doc_id=str(payload.get("doc_id", "") or ""),
            window_start=int(payload.get("window_start", 0)),
            bucket=str(payload.get("bucket", "") or ""),
        )


def _load_streaming_source_manifest(path: str | Path) -> list[StreamingSourceSpec]:
    manifest_path = Path(path)
    payload_text = manifest_path.read_text(encoding="utf-8")
    try:
        raw_payload = json.loads(payload_text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("yaml is required to read non-JSON streaming manifests") from exc
        raw_payload = yaml.safe_load(payload_text)
    if isinstance(raw_payload, list):
        entries = raw_payload
    elif isinstance(raw_payload, Mapping):
        entries = raw_payload.get("sources") or raw_payload.get("entries") or []
    else:
        raise ValueError("streaming source manifest must be a mapping or list")
    if not isinstance(entries, list) or not entries:
        raise ValueError("streaming source manifest must contain at least one source entry")
    return [StreamingSourceSpec.from_mapping(dict(entry)) for entry in entries]


def _looks_like_teacher_reasoning_text(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "<think>",
        "</think>",
        "chain-of-thought",
        "reasoning:",
        "let's think step by step",
        "scratchpad:",
        "internal reasoning",
    )
    return any(marker in lowered for marker in markers)


def _iter_text_chunks(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            yield stripped
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _iter_text_chunks(item)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_text_chunks(item)


def _extract_stream_text(row: Mapping[str, Any], source: StreamingSourceSpec) -> str:
    field_names = list(source.text_fields) or [str(source.text_field)]
    chunks: list[str] = []
    for field_name in field_names:
        chunks.extend(_iter_text_chunks(row.get(field_name)))
    if not chunks and "text" not in field_names:
        chunks.extend(_iter_text_chunks(row.get("text")))
    return "\n\n".join(chunk for chunk in chunks if chunk).strip()


def _row_matches_manifest_filters(row: Mapping[str, Any], source: StreamingSourceSpec) -> bool:
    for field_name, allowed_values in source.include_field_values.items():
        allowed = {str(value).strip().lower() for value in allowed_values if str(value).strip()}
        if not allowed:
            continue
        field_value = str(row.get(field_name, "") or "").strip().lower()
        if field_value not in allowed:
            return False
    for field_name, denied_values in source.exclude_field_values.items():
        denied = {str(value).strip().lower() for value in denied_values if str(value).strip()}
        if not denied:
            continue
        field_value = str(row.get(field_name, "") or "").strip().lower()
        if field_value in denied:
            return False
    return True


def _iter_local_jsonl_gz_rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, Mapping):
                yield dict(payload)


def _iter_http_jsonl_gz_rows(url: str) -> Iterator[dict[str, Any]]:
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as gz_handle:
            text_stream = io.TextIOWrapper(gz_handle, encoding="utf-8")
            for line in text_stream:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if isinstance(payload, Mapping):
                    yield dict(payload)


def _jsonl_gz_shards(uri_or_repo: str, *, require_streaming_only: bool) -> list[str]:
    path = Path(uri_or_repo)
    if path.exists():
        if path.is_dir():
            shards = sorted(str(candidate) for candidate in path.glob("*.jsonl.gz"))
            if not shards:
                raise ValueError(f"no .jsonl.gz shards found in {path}")
            return shards
        if path.suffix == ".gz":
            return [str(path)]
        if path.suffix in {".json", ".yaml", ".yml"}:
            payload = _json_or_yaml_loads(path.read_text(encoding="utf-8"))
            entries = payload.get("shards") or payload.get("urls") or payload.get("paths") or []
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"shard manifest {path} must contain 'shards', 'urls', or 'paths'")
            return [str(entry) for entry in entries]
    if require_streaming_only and not str(uri_or_repo).startswith(("http://", "https://")):
        raise ValueError(f"streaming-only source path does not exist: {uri_or_repo}")
    return [str(uri_or_repo)]


def _open_streaming_source(
    source: StreamingSourceSpec,
    *,
    split: str,
    cache_dir: str,
    require_streaming_only: bool,
) -> Iterator[dict[str, Any]]:
    if require_streaming_only and not bool(source.allow_streaming_only):
        raise ValueError(f"source {source.source_id!r} is not marked allow_streaming_only=true")
    if str(source.stream_mode) == "hf_streaming":
        _configure_datasets_logging()
        load_kwargs: dict[str, Any] = {
            "path": str(source.dataset_name or source.uri_or_repo),
            "split": str(split),
            "streaming": True,
        }
        if str(source.dataset_config):
            load_kwargs["name"] = str(source.dataset_config)
        if str(cache_dir):
            load_kwargs["cache_dir"] = str(cache_dir)
        dataset = load_dataset(**load_kwargs)
        for row in dataset:
            if isinstance(row, Mapping):
                yield dict(row)
        return
    shards = _jsonl_gz_shards(str(source.uri_or_repo), require_streaming_only=require_streaming_only)
    for shard in shards:
        if str(source.stream_mode) == "local_jsonl_gz":
            yield from _iter_local_jsonl_gz_rows(Path(shard))
        elif str(source.stream_mode) == "http_jsonl_gz":
            yield from _iter_http_jsonl_gz_rows(str(shard))
        else:  # pragma: no cover - guarded by validation
            raise ValueError(f"unsupported stream_mode: {source.stream_mode}")


def _source_split(source: StreamingSourceSpec, *, kind: str) -> str:
    return str(source.train_split if str(kind) == "train" else source.validation_split)


def _tokenize_stream_document(tokenizer: Any, text: str, *, max_chars: int | None = None) -> list[int]:
    if max_chars is not None and int(max_chars) > 0 and len(text) > int(max_chars):
        text = text[: int(max_chars)]
    token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and token_ids and int(token_ids[-1]) != int(eos_token_id):
        token_ids.append(int(eos_token_id))
    return [int(token) for token in token_ids]


def _window_oracle_targets(
    token_ids: Sequence[int],
    *,
    sequence_length: int,
    block_len: int,
) -> tuple[list[int], list[bool], int]:
    if len(token_ids) != int(sequence_length) + 1:
        raise ValueError("window token_ids length must equal sequence_length + 1")
    oracle_target_block_ids = [-100] * int(sequence_length)
    oracle_mask = [False] * int(sequence_length)
    prior_block_latest: dict[int, int] = {}
    current_block_seen: set[int] = set()
    repeated_anchor_count = 0
    for position in range(int(sequence_length)):
        current_block = int(position // max(1, int(block_len)))
        if position > 0 and (position % max(1, int(block_len)) == 0):
            committed_block = int((position // max(1, int(block_len))) - 1)
            for token_id in current_block_seen:
                prior_block_latest[int(token_id)] = int(committed_block)
            current_block_seen = set()
        anchor_token = int(token_ids[position])
        if anchor_token in prior_block_latest:
            repeated_anchor_count += 1
        gold_token = int(token_ids[position + 1])
        gold_block = prior_block_latest.get(gold_token)
        if gold_block is not None and int(gold_block) < int(current_block):
            oracle_target_block_ids[position] = int(gold_block)
            oracle_mask[position] = True
        current_block_seen.add(anchor_token)
    return oracle_target_block_ids, oracle_mask, repeated_anchor_count


def _bucket_for_window(
    *,
    oracle_mask: Sequence[bool],
    repeated_anchor_count: int,
    dataset_config: DatasetStreamConfig,
) -> str:
    oracle_count = sum(1 for value in oracle_mask if bool(value))
    if oracle_count >= int(dataset_config.memory_positive_min_oracle_positions):
        return "M"
    if (
        repeated_anchor_count >= int(dataset_config.hard_negative_min_repeat_anchors)
        and oracle_count < int(dataset_config.hard_negative_max_oracle_positions)
    ):
        return "H"
    return "G"


def _window_records_from_document(
    *,
    token_ids: Sequence[int],
    source_id: str,
    doc_id: str,
    sequence_length: int,
    block_len: int,
    dataset_config: DatasetStreamConfig,
) -> list[StreamWindowRecord]:
    if len(token_ids) < int(sequence_length) + 1:
        return []
    records: list[StreamWindowRecord] = []
    starts = window_starts(len(token_ids), int(sequence_length), max(1, len(token_ids) // max(1, int(sequence_length))))
    for start in starts:
        window = list(token_ids[start : start + int(sequence_length) + 1])
        if len(window) != int(sequence_length) + 1:
            continue
        oracle_target_block_ids, oracle_mask, repeated_anchor_count = _window_oracle_targets(
            window,
            sequence_length=int(sequence_length),
            block_len=int(block_len),
        )
        bucket = _bucket_for_window(
            oracle_mask=oracle_mask,
            repeated_anchor_count=int(repeated_anchor_count),
            dataset_config=dataset_config,
        )
        records.append(
            StreamWindowRecord(
                sequence_length=int(sequence_length),
                input_ids=list(window[:-1]),
                labels=list(window[1:]),
                oracle_target_block_ids=list(oracle_target_block_ids),
                oracle_mask=list(oracle_mask),
                source_id=str(source_id),
                doc_id=str(doc_id),
                window_start=int(start),
                bucket=str(bucket),
            )
        )
    return records


def _validation_bucket_priority(kind: str) -> tuple[str, ...]:
    if str(kind) == "memory":
        return ("M", "H", "G")
    return ("G", "H", "M")


def write_streaming_eval_pack(
    path: str | Path,
    *,
    pack_name: str,
    records: Sequence[StreamWindowRecord],
) -> Path:
    root = Path(path)
    payload = {
        "schema_version": 1,
        "pack_name": str(pack_name),
        "records": [record.to_payload() for record in records],
    }
    _write_json_atomic(root, payload)
    return root


def load_streaming_eval_pack(path: str | Path) -> dict[int, list[ValidationBatch]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    grouped: dict[int, list[ValidationBatch]] = {}
    for raw_record in payload.get("records", []):
        record = StreamWindowRecord.from_payload(raw_record)
        grouped.setdefault(int(record.sequence_length), []).append(
            ValidationBatch(
                sequence_length=int(record.sequence_length),
                input_ids=torch.tensor(record.input_ids, dtype=torch.long).unsqueeze(0).contiguous(),
                labels=torch.tensor(record.labels, dtype=torch.long).unsqueeze(0).contiguous(),
                oracle_target_block_ids=torch.tensor(record.oracle_target_block_ids, dtype=torch.long).unsqueeze(0).contiguous(),
            )
        )
    if not grouped:
        raise ValueError(f"streaming eval pack {path} does not contain any records")
    return grouped


def _validation_batches_from_stream_records(
    records_by_length: Mapping[int, Sequence[StreamWindowRecord]],
) -> dict[int, list[ValidationBatch]]:
    grouped: dict[int, list[ValidationBatch]] = {}
    for sequence_length, records in records_by_length.items():
        grouped[int(sequence_length)] = [
            ValidationBatch(
                sequence_length=int(record.sequence_length),
                input_ids=torch.tensor(record.input_ids, dtype=torch.long).unsqueeze(0).contiguous(),
                labels=torch.tensor(record.labels, dtype=torch.long).unsqueeze(0).contiguous(),
                oracle_target_block_ids=torch.tensor(record.oracle_target_block_ids, dtype=torch.long).unsqueeze(0).contiguous(),
            )
            for record in records
        ]
    if not grouped:
        raise ValueError("streaming validation did not produce any records")
    return grouped


class StreamingCorpusSampler:
    def __init__(
        self,
        *,
        tokenizer: Any,
        dataset_config: DatasetStreamConfig,
        split: str,
        seed: int,
        sequence_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset_config = dataset_config
        self.split = str(split)
        self.sequence_length = int(sequence_length)
        self.sources = _load_streaming_source_manifest(dataset_config.source_manifest_path)
        self.rng = random.Random(int(seed))
        self._iterators: dict[str, Iterator[dict[str, Any]]] = {}
        self.rows_seen: int = 0
        self.sampled_documents: int = 0
        self.truncated_documents: int = 0
        self.short_documents: int = 0
        self.sampled_source_counts: dict[str, int] = {source.source_id: 0 for source in self.sources}

    def _weighted_source_choice(self) -> StreamingSourceSpec:
        return self.rng.choices(self.sources, weights=[float(source.base_weight) for source in self.sources], k=1)[0]

    def _iterator_for(self, source: StreamingSourceSpec) -> Iterator[dict[str, Any]]:
        iterator = self._iterators.get(source.source_id)
        if iterator is None:
            iterator = _open_streaming_source(
                source,
                split=_source_split(source, kind=self.split),
                cache_dir=str(self.dataset_config.cache_dir),
                require_streaming_only=bool(self.dataset_config.require_streaming_only),
            )
            self._iterators[source.source_id] = iterator
        return iterator

    def next_document(self) -> tuple[StreamingSourceSpec, str, list[int]]:
        while True:
            source = self._weighted_source_choice()
            iterator = self._iterator_for(source)
            collected_token_ids: list[int] = []
            collected_doc_ids: list[str] = []
            collected_chars = 0
            while len(collected_token_ids) < int(self.dataset_config.min_document_tokens):
                try:
                    row = next(iterator)
                except StopIteration:
                    self._iterators.pop(source.source_id, None)
                    break
                self.rows_seen += 1
                if not _row_matches_manifest_filters(row, source):
                    continue
                text = _extract_stream_text(row, source)
                if not text:
                    continue
                if _looks_like_teacher_reasoning_text(text):
                    continue
                remaining_chars = int(self.dataset_config.max_document_chars) - int(collected_chars)
                if remaining_chars <= 0:
                    self.truncated_documents += 1
                    break
                if len(text) > int(remaining_chars):
                    text = text[: int(remaining_chars)]
                    self.truncated_documents += 1
                token_ids = _tokenize_stream_document(
                    self.tokenizer,
                    text,
                    max_chars=int(remaining_chars),
                )
                if not token_ids:
                    continue
                collected_chars += len(text)
                collected_token_ids.extend(int(token) for token in token_ids)
                collected_doc_ids.append(
                    str(row.get("doc_id") or row.get("id") or f"{source.source_id}:{self.rows_seen}")
                )
                if int(collected_chars) >= int(self.dataset_config.max_document_chars):
                    break
            if len(collected_token_ids) < int(self.dataset_config.min_document_tokens):
                self.short_documents += 1
                continue
            self.sampled_documents += 1
            self.sampled_source_counts[source.source_id] = int(self.sampled_source_counts.get(source.source_id, 0) + 1)
            if len(collected_doc_ids) == 1:
                doc_id = str(collected_doc_ids[0])
            else:
                doc_id = f"{source.source_id}:{collected_doc_ids[0]}..{collected_doc_ids[-1]}:{len(collected_doc_ids)}"
            return source, doc_id, collected_token_ids

    def stats_payload(self) -> dict[str, Any]:
        total = max(1, sum(int(value) for value in self.sampled_source_counts.values()))
        source_mix = {
            source_id: {
                "documents": int(count),
                "share": float(count / total),
            }
            for source_id, count in sorted(self.sampled_source_counts.items())
        }
        return {
            "rows_seen": int(self.rows_seen),
            "sampled_documents": int(self.sampled_documents),
            "truncated_documents": int(self.truncated_documents),
            "short_documents": int(self.short_documents),
            "source_mix": source_mix,
        }


def read_training_text(*, text: str, text_file: str) -> str:
    if text_file:
        return Path(text_file).read_text(encoding="utf-8")
    return str(text or "")


def build_token_stream(tokenizer: Any, text: str) -> torch.LongTensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    token_stream = encoded["input_ids"][0].to(dtype=torch.long)
    if int(token_stream.numel()) < 2:
        raise ValueError("training text must produce at least 2 tokens")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and int(token_stream[-1].item()) != int(eos_token_id):
        token_stream = torch.cat((token_stream, torch.tensor([int(eos_token_id)], dtype=torch.long)), dim=0)
    return token_stream


def window_starts(token_count: int, sequence_length: int, windows: int) -> list[int]:
    available = token_count - sequence_length - 1
    if available < 0:
        return []
    starts = list(range(0, available + 1, max(1, sequence_length)))
    if len(starts) <= windows:
        return starts
    if windows <= 1:
        return [starts[0]]
    positions = torch.linspace(0, len(starts) - 1, steps=windows)
    return [starts[int(round(position.item()))] for position in positions]


@dataclass
class ValidationBatch:
    sequence_length: int
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    oracle_target_block_ids: torch.LongTensor | None = None


@dataclass
class PinnedValidationBatchSlot:
    sequence_length: int
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    oracle_target_block_ids: torch.LongTensor | None = None
    pooled: bool = True

    @classmethod
    def allocate(cls, *, sequence_length: int) -> "PinnedValidationBatchSlot":
        def _alloc(shape: tuple[int, ...]) -> torch.LongTensor:
            tensor = torch.empty(shape, dtype=torch.long)
            if bool(torch.cuda.is_available()):
                tensor = tensor.pin_memory()
            return tensor.contiguous()

        return cls(
            sequence_length=int(sequence_length),
            input_ids=_alloc((1, int(sequence_length))),
            labels=_alloc((1, int(sequence_length))),
            oracle_target_block_ids=_alloc((1, int(sequence_length))),
            pooled=True,
        )

    @classmethod
    def from_batch(cls, batch: ValidationBatch) -> "PinnedValidationBatchSlot":
        slot = cls.allocate(sequence_length=int(batch.sequence_length))
        slot.pooled = False
        slot.copy_from(batch)
        return slot

    def copy_from(self, batch: ValidationBatch) -> None:
        if int(batch.sequence_length) != int(self.sequence_length):
            raise ValueError(
                f"validation slot length mismatch: expected {self.sequence_length}, got {batch.sequence_length}"
            )
        self.input_ids.copy_(batch.input_ids, non_blocking=bool(self.input_ids.is_pinned()))
        self.labels.copy_(batch.labels, non_blocking=bool(self.labels.is_pinned()))
        if self.oracle_target_block_ids is not None:
            if batch.oracle_target_block_ids is None:
                self.oracle_target_block_ids.fill_(-100)
            else:
                self.oracle_target_block_ids.copy_(
                    batch.oracle_target_block_ids,
                    non_blocking=bool(self.oracle_target_block_ids.is_pinned()),
                )

    def as_validation_batch(self) -> ValidationBatch:
        oracle_target_block_ids = None
        if self.oracle_target_block_ids is not None:
            oracle_target_block_ids = self.oracle_target_block_ids
        return ValidationBatch(
            sequence_length=int(self.sequence_length),
            input_ids=self.input_ids,
            labels=self.labels,
            oracle_target_block_ids=oracle_target_block_ids,
        )


@dataclass
class TrainingBatch:
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    oracle_target_block_ids: torch.LongTensor | None = None


class BatchSource(ABC):
    @abstractmethod
    def reconfigure(self, *, sequence_length: int, micro_batch_size: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def next_batch(self) -> TrainingBatch:
        raise NotImplementedError

    def set_gradient_accumulation_steps(self, gradient_accumulation_steps: int) -> None:
        return

    def supports_step_addressing(self) -> bool:
        return False

    def batch_for(
        self,
        *,
        step_index: int,
        microstep: int,
        sequence_length: int,
        micro_batch_size: int,
    ) -> TrainingBatch:
        raise NotImplementedError(f"{type(self).__name__} does not support addressed batch access")

    def cycle_length(self) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not expose a deterministic batch cycle")

    def packet_key(
        self,
        *,
        step_index: int,
        microstep: int,
        sequence_length: int,
        micro_batch_size: int,
    ) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not expose deterministic packet keys")


BatchProvider = BatchSource


class ValidationBatchSource(ABC):
    @abstractmethod
    def sequence_lengths(self, *, eval_kind: str) -> tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def cached_batches(self, *, eval_kind: str) -> dict[int, list[ValidationBatch]] | None:
        raise NotImplementedError

    @abstractmethod
    def materialize(self, *, eval_kind: str) -> dict[int, list[ValidationBatch]]:
        raise NotImplementedError

    @abstractmethod
    def iter_batches(self, *, eval_kind: str, sequence_length: int) -> Iterator[ValidationBatch]:
        raise NotImplementedError

    @abstractmethod
    def telemetry(self) -> dict[str, Any]:
        raise NotImplementedError

    def iter_staging_slots(self, *, eval_kind: str, sequence_length: int) -> Iterator[PinnedValidationBatchSlot]:
        for batch in self.iter_batches(eval_kind=str(eval_kind), sequence_length=int(sequence_length)):
            yield PinnedValidationBatchSlot.from_batch(batch)

    def release_staging_slot(
        self,
        *,
        eval_kind: str,
        sequence_length: int,
        slot: PinnedValidationBatchSlot,
    ) -> None:
        del eval_kind, sequence_length, slot
        return

    def close(self) -> None:
        return


class StaticValidationBatchSource(ValidationBatchSource):
    def __init__(
        self,
        *,
        debug_batches: Mapping[int, Sequence[ValidationBatch]],
        full_batches: Mapping[int, Sequence[ValidationBatch]],
    ) -> None:
        self._batches = {
            "debug": {
                int(length): [clone_validation_batch(batch) for batch in batches]
                for length, batches in debug_batches.items()
            },
            "full": {
                int(length): [clone_validation_batch(batch) for batch in batches]
                for length, batches in full_batches.items()
            },
        }

    def sequence_lengths(self, *, eval_kind: str) -> tuple[int, ...]:
        return tuple(sorted(int(length) for length in self._batches[str(eval_kind)].keys()))

    def cached_batches(self, *, eval_kind: str) -> dict[int, list[ValidationBatch]] | None:
        return {
            int(length): [clone_validation_batch(batch) for batch in batches]
            for length, batches in self._batches[str(eval_kind)].items()
        }

    def materialize(self, *, eval_kind: str) -> dict[int, list[ValidationBatch]]:
        cached = self.cached_batches(eval_kind=str(eval_kind))
        if cached is None:
            raise RuntimeError(f"no cached validation batches for eval_kind={eval_kind!r}")
        return cached

    def iter_batches(self, *, eval_kind: str, sequence_length: int) -> Iterator[ValidationBatch]:
        batches = self._batches[str(eval_kind)].get(int(sequence_length), [])
        for batch in batches:
            yield clone_validation_batch(batch)

    def telemetry(self) -> dict[str, Any]:
        return {
            "source": "static",
            "cache_complete": True,
            "windows_emitted": {
                eval_kind: {str(length): len(batches) for length, batches in grouped.items()}
                for eval_kind, grouped in self._batches.items()
            },
        }


_VALIDATION_BATCH_SENTINEL = object()
_STREAMING_VALIDATION_FALLBACK_PATIENCE_DOCS = 32


def _validation_eval_label(eval_kind: str) -> str:
    return "memory" if str(eval_kind) == "debug" else "generic"


def _validation_batch_from_stream_record(record: StreamWindowRecord) -> ValidationBatch:
    return ValidationBatch(
        sequence_length=int(record.sequence_length),
        input_ids=torch.tensor(record.input_ids, dtype=torch.long).unsqueeze(0).contiguous(),
        labels=torch.tensor(record.labels, dtype=torch.long).unsqueeze(0).contiguous(),
        oracle_target_block_ids=torch.tensor(record.oracle_target_block_ids, dtype=torch.long).unsqueeze(0).contiguous(),
    )


@dataclass
class _StreamingValidationTargetState:
    eval_kind: str
    sequence_length: int
    target_count: int
    preference: tuple[str, ...]
    queue_maxsize: int
    cache_batches: bool
    queue: queue.Queue[object] = field(init=False)
    free_slots: queue.Queue["PinnedValidationBatchSlot"] = field(init=False)
    emitted_batches: list[ValidationBatch] = field(default_factory=list)
    emitted_count: int = 0
    released_count: int = 0
    fallback_batches: dict[str, list[ValidationBatch]] = field(
        default_factory=lambda: {"M": [], "H": [], "G": []}
    )
    seen_keys: set[tuple[str, str, int]] = field(default_factory=set)
    docs_since_emit: int = 0
    queue_depth_total: int = 0
    queue_depth_samples: int = 0
    queue_depth_peak: int = 0
    staging_depth_peak: int = 0
    queue_drained: bool = False
    sentinel_emitted: bool = False
    sentinel_seen: bool = False

    def __post_init__(self) -> None:
        self.queue = queue.Queue(maxsize=max(1, int(self.queue_maxsize)))
        self.free_slots = queue.Queue(maxsize=max(1, int(self.queue_maxsize)))
        for _ in range(max(1, int(self.queue_maxsize))):
            self.free_slots.put(PinnedValidationBatchSlot.allocate(sequence_length=int(self.sequence_length)))

    def complete(self) -> bool:
        return int(self.emitted_count) >= int(self.target_count)


class StreamingValidationBatchSource(ValidationBatchSource):
    def __init__(
        self,
        *,
        tokenizer: Any,
        dataset_config: DatasetStreamConfig,
        seed: int,
        evaluation_config: Any,
        host_refill_depth: int = 8,
        cache_batches: bool = True,
    ) -> None:
        if not dataset_config.source_manifest_path:
            raise ValueError("StreamingValidationBatchSource requires dataset_config.source_manifest_path")
        self._dataset_config = dataset_config
        max_requested_sequence_length = max(
            [
                *(int(value) for value in evaluation_config.debug_validation_sequence_lengths),
                *(int(value) for value in evaluation_config.validation_sequence_lengths),
            ]
        )
        self._effective_min_document_tokens = max(2, int(max_requested_sequence_length) + 1)
        sampler_dataset_config = replace(
            dataset_config,
            min_document_tokens=int(self._effective_min_document_tokens),
        )
        self._sampler = StreamingCorpusSampler(
            tokenizer=tokenizer,
            dataset_config=sampler_dataset_config,
            split="validation",
            seed=int(seed),
            sequence_length=int(max_requested_sequence_length),
        )
        self._state_lock = threading.Lock()
        self._done_event = threading.Event()
        self._stop_event = threading.Event()
        self._failure: BaseException | None = None
        self._started_at = time.perf_counter()
        self._first_batch_ready_ms: float | None = None
        self._targets: dict[tuple[str, int], _StreamingValidationTargetState] = {}
        self._sequence_lengths = {"debug": [], "full": []}
        self._host_refill_depth = max(1, int(host_refill_depth))
        self._cache_batches = bool(cache_batches)
        debug_targets = {
            int(sequence_length): int(num_windows)
            for sequence_length, num_windows in zip(
                evaluation_config.debug_validation_sequence_lengths,
                evaluation_config.debug_validation_num_windows,
                strict=True,
            )
        }
        full_targets = {
            int(sequence_length): int(num_windows)
            for sequence_length, num_windows in zip(
                evaluation_config.validation_sequence_lengths,
                evaluation_config.validation_num_windows,
                strict=True,
            )
        }
        for eval_kind, targets in (("debug", debug_targets), ("full", full_targets)):
            for sequence_length, target_count in targets.items():
                self._targets[(str(eval_kind), int(sequence_length))] = _StreamingValidationTargetState(
                    eval_kind=str(eval_kind),
                    sequence_length=int(sequence_length),
                    target_count=int(target_count),
                    preference=_validation_bucket_priority(_validation_eval_label(str(eval_kind))),
                    queue_maxsize=int(self._host_refill_depth),
                    cache_batches=bool(self._cache_batches),
                )
                self._sequence_lengths[str(eval_kind)].append(int(sequence_length))
        self._thread = threading.Thread(
            target=self._worker_main,
            name="streaming-validation-producer",
            daemon=True,
        )
        self._thread.start()

    def sequence_lengths(self, *, eval_kind: str) -> tuple[int, ...]:
        return tuple(int(length) for length in self._sequence_lengths[str(eval_kind)])

    def cached_batches(self, *, eval_kind: str) -> dict[int, list[ValidationBatch]] | None:
        if not self._done_event.is_set():
            return None
        self._raise_if_failed()
        if not bool(self._cache_batches):
            return None
        with self._state_lock:
            return {
                int(sequence_length): list(state.emitted_batches)
                for (kind, sequence_length), state in self._targets.items()
                if str(kind) == str(eval_kind)
            }

    def materialize(self, *, eval_kind: str) -> dict[int, list[ValidationBatch]]:
        self._done_event.wait()
        self._raise_if_failed()
        cached = self.cached_batches(eval_kind=str(eval_kind))
        if cached is None:
            if not bool(self._cache_batches):
                raise RuntimeError(
                    f"streaming validation materialize for eval_kind={eval_kind!r} requires cache_batches=True"
                )
            raise RuntimeError(f"streaming validation cache for eval_kind={eval_kind!r} did not complete")
        return cached

    def iter_batches(self, *, eval_kind: str, sequence_length: int) -> Iterator[ValidationBatch]:
        for slot in self.iter_staging_slots(eval_kind=str(eval_kind), sequence_length=int(sequence_length)):
            try:
                yield slot.as_validation_batch()
            finally:
                self.release_staging_slot(
                    eval_kind=str(eval_kind),
                    sequence_length=int(sequence_length),
                    slot=slot,
                )

    def iter_staging_slots(self, *, eval_kind: str, sequence_length: int) -> Iterator[PinnedValidationBatchSlot]:
        state = self._state_for(eval_kind=str(eval_kind), sequence_length=int(sequence_length))
        with self._state_lock:
            use_cached_only = bool(self._cache_batches and state.queue_drained and self._done_event.is_set())
            cached_batches = list(state.emitted_batches) if use_cached_only else []
        if use_cached_only:
            for batch in cached_batches:
                yield PinnedValidationBatchSlot.from_batch(batch)
            return
        while True:
            item = state.queue.get()
            self._sample_queue_depth(state)
            if item is _VALIDATION_BATCH_SENTINEL:
                with self._state_lock:
                    state.queue_drained = True
                    state.sentinel_seen = True
                self._raise_if_failed()
                break
            if not isinstance(item, PinnedValidationBatchSlot):
                raise RuntimeError(f"unexpected validation queue item type: {type(item).__name__}")
            yield item

    def release_staging_slot(
        self,
        *,
        eval_kind: str,
        sequence_length: int,
        slot: PinnedValidationBatchSlot,
    ) -> None:
        if not bool(slot.pooled):
            return
        state = self._state_for(eval_kind=str(eval_kind), sequence_length=int(sequence_length))
        with self._state_lock:
            state.released_count = int(state.released_count) + 1
        state.free_slots.put(slot)

    def telemetry(self) -> dict[str, Any]:
        sampler_payload = self._sampler.stats_payload()
        with self._state_lock:
            queue_depth_current = {
                f"{kind}:{length}": int(state.queue.qsize())
                for (kind, length), state in self._targets.items()
            }
            queue_depth_peak = {
                f"{kind}:{length}": int(state.queue_depth_peak)
                for (kind, length), state in self._targets.items()
            }
            queue_depth_avg = {
                f"{kind}:{length}": (
                    float(state.queue_depth_total / max(1, state.queue_depth_samples))
                    if state.queue_depth_samples > 0
                    else 0.0
                )
                for (kind, length), state in self._targets.items()
            }
            staging_depth_current = {
                f"{kind}:{length}": int(state.queue_maxsize - state.free_slots.qsize())
                for (kind, length), state in self._targets.items()
            }
            staging_depth_peak = {
                f"{kind}:{length}": int(state.staging_depth_peak)
                for (kind, length), state in self._targets.items()
            }
            emitted = {
                kind: {
                    str(length): int(state.emitted_count)
                    for (target_kind, length), state in self._targets.items()
                    if str(target_kind) == str(kind)
                }
                for kind in ("debug", "full")
            }
            released = {
                kind: {
                    str(length): int(state.released_count)
                    for (target_kind, length), state in self._targets.items()
                    if str(target_kind) == str(kind)
                }
                for kind in ("debug", "full")
            }
            targeted = {
                kind: {
                    str(length): int(state.target_count)
                    for (target_kind, length), state in self._targets.items()
                    if str(target_kind) == str(kind)
                }
                for kind in ("debug", "full")
            }
        return {
            "source": "streaming",
            "cache_enabled": bool(self._cache_batches),
            "cache_complete": bool(self._cache_batches and self._done_event.is_set() and self._failure is None),
            "stream_complete": bool(self._done_event.is_set() and self._failure is None),
            "failed": bool(self._failure is not None),
            "failure": None if self._failure is None else str(self._failure),
            "first_batch_ready_ms": self._first_batch_ready_ms,
            "effective_min_document_tokens": int(self._effective_min_document_tokens),
            "rows_seen": int(sampler_payload["rows_seen"]),
            "documents_sampled": int(sampler_payload["sampled_documents"]),
            "truncated_documents": int(sampler_payload["truncated_documents"]),
            "short_documents": int(sampler_payload["short_documents"]),
            "source_mix": dict(sampler_payload["source_mix"]),
            "windows_targeted": targeted,
            "windows_emitted": emitted,
            "windows_released": released,
            "queue_depth_current": queue_depth_current,
            "queue_depth_peak": queue_depth_peak,
            "queue_depth_avg": queue_depth_avg,
            "staging_depth_current": staging_depth_current,
            "staging_depth_peak": staging_depth_peak,
        }

    def close(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _state_for(self, *, eval_kind: str, sequence_length: int) -> _StreamingValidationTargetState:
        key = (str(eval_kind), int(sequence_length))
        try:
            return self._targets[key]
        except KeyError as exc:
            raise KeyError(f"unknown validation target eval_kind={eval_kind!r} sequence_length={sequence_length}") from exc

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("streaming validation producer failed") from self._failure

    def _sample_queue_depth(self, state: _StreamingValidationTargetState) -> None:
        depth = int(state.queue.qsize())
        state.queue_depth_total += int(depth)
        state.queue_depth_samples += 1
        state.queue_depth_peak = max(int(state.queue_depth_peak), int(depth))

    def _emit_batch(self, state: _StreamingValidationTargetState, batch: ValidationBatch) -> bool:
        with self._state_lock:
            if state.complete():
                return False
            state.emitted_count = int(state.emitted_count) + 1
            if bool(state.cache_batches):
                state.emitted_batches.append(clone_validation_batch(batch))
            state.docs_since_emit = 0
            if self._first_batch_ready_ms is None:
                self._first_batch_ready_ms = float((time.perf_counter() - self._started_at) * 1000.0)
        slot = state.free_slots.get()
        with self._state_lock:
            used_depth = max(0, int(state.queue_maxsize) - int(state.free_slots.qsize()))
            state.staging_depth_peak = max(int(state.staging_depth_peak), int(used_depth))
        slot.copy_from(batch)
        state.queue.put(slot)
        with self._state_lock:
            self._sample_queue_depth(state)
        return True

    def _emit_sentinel(self, state: _StreamingValidationTargetState) -> None:
        with self._state_lock:
            if state.sentinel_emitted:
                return
            state.sentinel_emitted = True
        state.queue.put(_VALIDATION_BATCH_SENTINEL)
        with self._state_lock:
            self._sample_queue_depth(state)

    def _store_candidate(
        self,
        state: _StreamingValidationTargetState,
        *,
        record: StreamWindowRecord,
        batch: ValidationBatch,
    ) -> None:
        record_key = (str(record.source_id), str(record.doc_id), int(record.window_start))
        with self._state_lock:
            if record_key in state.seen_keys:
                return
            state.seen_keys.add(record_key)
        if str(record.bucket) == str(state.preference[0]):
            self._emit_batch(state, batch)
            return
        with self._state_lock:
            state.fallback_batches[str(record.bucket)].append(clone_validation_batch(batch))

    def _promote_fallback_batches(self, state: _StreamingValidationTargetState) -> bool:
        promoted = False
        for bucket in state.preference[1:]:
            while True:
                with self._state_lock:
                    if state.complete():
                        return promoted
                    if not state.fallback_batches[str(bucket)]:
                        break
                    batch = state.fallback_batches[str(bucket)].pop(0)
                if self._emit_batch(state, clone_validation_batch(batch)):
                    promoted = True
        return promoted

    def _targets_complete(self) -> bool:
        return all(state.complete() for state in self._targets.values())

    def _flush_completion(self) -> None:
        for state in self._targets.values():
            if not state.complete():
                if not self._promote_fallback_batches(state) or not state.complete():
                    raise RuntimeError("unable to mine enough live streaming validation windows from the configured sources")
            self._emit_sentinel(state)

    def _worker_main(self) -> None:
        attempts = 0
        max_attempts = max(
            2048,
            32 * sum(int(state.target_count) for state in self._targets.values()),
        )
        try:
            while not self._targets_complete() and not self._stop_event.is_set():
                source, doc_id, token_ids = self._sampler.next_document()
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError("unable to mine enough live streaming validation windows from the configured sources")
                candidates_by_length: dict[int, list[StreamWindowRecord]] = {}
                for sequence_length in {
                    int(length)
                    for (_kind, length), state in self._targets.items()
                    if not state.complete()
                }:
                    candidates_by_length[int(sequence_length)] = _window_records_from_document(
                        token_ids=token_ids,
                        source_id=source.source_id,
                        doc_id=doc_id,
                        sequence_length=int(sequence_length),
                        block_len=int(self._dataset_config.block_len),
                        dataset_config=self._dataset_config,
                    )
                emitted_any_this_doc = False
                for (eval_kind, sequence_length), state in self._targets.items():
                    if state.complete():
                        continue
                    candidates = candidates_by_length.get(int(sequence_length), [])
                    emitted_for_state = False
                    for record in candidates:
                        batch = _validation_batch_from_stream_record(record)
                        before_count = int(state.emitted_count)
                        self._store_candidate(state, record=record, batch=batch)
                        if int(state.emitted_count) > int(before_count):
                            emitted_for_state = True
                            emitted_any_this_doc = True
                    with self._state_lock:
                        if not state.complete():
                            state.docs_since_emit = 0 if emitted_for_state else int(state.docs_since_emit) + 1
                    if (
                        not state.complete()
                        and int(state.docs_since_emit) >= int(_STREAMING_VALIDATION_FALLBACK_PATIENCE_DOCS)
                    ):
                        if self._promote_fallback_batches(state):
                            emitted_for_state = True
                            emitted_any_this_doc = True
                            with self._state_lock:
                                state.docs_since_emit = 0
                    if state.complete():
                        self._emit_sentinel(state)
                if emitted_any_this_doc:
                    attempts = 0
            self._flush_completion()
        except BaseException as exc:
            self._failure = exc
            for state in self._targets.values():
                self._emit_sentinel(state)
        finally:
            self._done_event.set()


def _maybe_pin(tensor: torch.Tensor) -> torch.Tensor:
    if not bool(torch.cuda.is_available()):
        return tensor.contiguous()
    return tensor.pin_memory().contiguous()


def clone_training_batch(batch: TrainingBatch) -> TrainingBatch:
    oracle_target_block_ids = None
    if batch.oracle_target_block_ids is not None:
        oracle_target_block_ids = batch.oracle_target_block_ids.detach().clone().contiguous()
    return TrainingBatch(
        input_ids=batch.input_ids.detach().clone().contiguous(),
        labels=batch.labels.detach().clone().contiguous(),
        oracle_target_block_ids=oracle_target_block_ids,
    )


def clone_validation_batch(batch: ValidationBatch) -> ValidationBatch:
    oracle_target_block_ids = None
    if batch.oracle_target_block_ids is not None:
        oracle_target_block_ids = batch.oracle_target_block_ids.detach().clone().contiguous()
    return ValidationBatch(
        sequence_length=int(batch.sequence_length),
        input_ids=batch.input_ids.detach().clone().contiguous(),
        labels=batch.labels.detach().clone().contiguous(),
        oracle_target_block_ids=oracle_target_block_ids,
    )


def pin_training_batch(batch: TrainingBatch) -> TrainingBatch:
    oracle_target_block_ids = None
    if batch.oracle_target_block_ids is not None:
        oracle_target_block_ids = _maybe_pin(batch.oracle_target_block_ids.detach().clone())
    return TrainingBatch(
        input_ids=_maybe_pin(batch.input_ids.detach().clone()),
        labels=_maybe_pin(batch.labels.detach().clone()),
        oracle_target_block_ids=oracle_target_block_ids,
    )


def pin_validation_batch(batch: ValidationBatch) -> ValidationBatch:
    oracle_target_block_ids = None
    if batch.oracle_target_block_ids is not None:
        oracle_target_block_ids = _maybe_pin(batch.oracle_target_block_ids.detach().clone())
    return ValidationBatch(
        sequence_length=int(batch.sequence_length),
        input_ids=_maybe_pin(batch.input_ids.detach().clone()),
        labels=_maybe_pin(batch.labels.detach().clone()),
        oracle_target_block_ids=oracle_target_block_ids,
    )


def training_batch_to_device(batch: TrainingBatch, *, device: torch.device) -> TrainingBatch:
    oracle_target_block_ids = None
    if batch.oracle_target_block_ids is not None:
        oracle_target_block_ids = batch.oracle_target_block_ids.to(
            device=device,
            dtype=torch.long,
            non_blocking=(device.type == "cuda"),
        ).contiguous()
    return TrainingBatch(
        input_ids=batch.input_ids.to(device=device, dtype=torch.long, non_blocking=(device.type == "cuda")).contiguous(),
        labels=batch.labels.to(device=device, dtype=torch.long, non_blocking=(device.type == "cuda")).contiguous(),
        oracle_target_block_ids=oracle_target_block_ids,
    )


def validation_batch_to_device(batch: ValidationBatch, *, device: torch.device) -> ValidationBatch:
    oracle_target_block_ids = None
    if batch.oracle_target_block_ids is not None:
        oracle_target_block_ids = batch.oracle_target_block_ids.to(
            device=device,
            dtype=torch.long,
            non_blocking=(device.type == "cuda"),
        ).contiguous()
    return ValidationBatch(
        sequence_length=int(batch.sequence_length),
        input_ids=batch.input_ids.to(device=device, dtype=torch.long, non_blocking=(device.type == "cuda")).contiguous(),
        labels=batch.labels.to(device=device, dtype=torch.long, non_blocking=(device.type == "cuda")).contiguous(),
        oracle_target_block_ids=oracle_target_block_ids,
    )


class TextBatchProvider(BatchSource):
    def __init__(
        self,
        *,
        tokenizer: Any,
        text: str,
        seed: int,
        sequence_length: int,
        micro_batch_size: int,
    ) -> None:
        self.token_stream = build_token_stream(tokenizer, text)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.sequence_length = int(sequence_length)
        self.micro_batch_size = int(micro_batch_size)

    def reconfigure(self, *, sequence_length: int, micro_batch_size: int) -> None:
        self.sequence_length = int(sequence_length)
        self.micro_batch_size = int(micro_batch_size)

    def next_batch(self) -> TrainingBatch:
        max_start = int(self.token_stream.numel()) - ((int(self.sequence_length) * int(self.micro_batch_size)) + 1)
        if max_start <= 0:
            start = 0
        else:
            start = int(
                torch.randint(
                    low=0,
                    high=max_start + 1,
                    size=(1,),
                    generator=self.generator,
                    dtype=torch.long,
                ).item()
            )
        windows = []
        for batch_idx in range(int(self.micro_batch_size)):
            offset = start + (batch_idx * int(self.sequence_length))
            end = offset + int(self.sequence_length) + 1
            if end > int(self.token_stream.numel()):
                offset = max(0, int(self.token_stream.numel()) - int(self.sequence_length) - 1)
                end = offset + int(self.sequence_length) + 1
            windows.append(self.token_stream[offset:end])
        batch = torch.stack(windows, dim=0).to(dtype=torch.long)
        return TrainingBatch(
            input_ids=batch[:, :-1].contiguous(),
            labels=batch[:, 1:].contiguous(),
        )


class FixedWindowBatchProvider(BatchSource):
    def __init__(
        self,
        *,
        train_batches: Sequence[dict[str, Any]],
    ) -> None:
        if not train_batches:
            raise ValueError("fixed-window diagnostics require at least one train batch")
        self._batches = [
            (
                torch.tensor(batch["input_ids"], dtype=torch.long),
                torch.tensor(batch["labels"], dtype=torch.long),
                None if batch.get("oracle_target_block_ids") is None else torch.tensor(batch["oracle_target_block_ids"], dtype=torch.long),
            )
            for batch in train_batches
        ]
        self.sequence_length = int(self._batches[0][0].shape[-1])
        self.micro_batch_size = int(self._batches[0][0].shape[0])
        self._index = 0

    def reconfigure(self, *, sequence_length: int, micro_batch_size: int) -> None:
        if int(sequence_length) != int(self.sequence_length):
            raise ValueError(
                f"fixed windows require sequence_length={self.sequence_length}, got {sequence_length}"
            )
        if int(micro_batch_size) != int(self.micro_batch_size):
            raise ValueError(
                f"fixed windows require micro_batch_size={self.micro_batch_size}, got {micro_batch_size}"
            )

    def next_batch(self) -> TrainingBatch:
        input_ids, labels, oracle_target_block_ids = self._batches[self._index]
        self._index = int((self._index + 1) % len(self._batches))
        return TrainingBatch(
            input_ids=input_ids.to(dtype=torch.long).contiguous(),
            labels=labels.to(dtype=torch.long).contiguous(),
            oracle_target_block_ids=None if oracle_target_block_ids is None else oracle_target_block_ids.to(dtype=torch.long).contiguous(),
        )


class StreamingBatchProvider(BatchSource):
    def __init__(
        self,
        *,
        tokenizer: Any,
        dataset_config: DatasetStreamConfig,
        seed: int,
        sequence_length: int,
        micro_batch_size: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset_config = dataset_config
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.micro_batch_size = int(micro_batch_size)
        self._dataset = None
        self._iterator: Iterator[dict[str, Any]] | None = None
        self._buffer: list[int] = []
        self._epoch = 0
        self._reset_dataset()

    def _reset_dataset(self) -> None:
        dataset = load_text_dataset(
            self.dataset_config,
            split=self.dataset_config.train_split,
            streaming=bool(self.dataset_config.streaming),
        )
        if self.dataset_config.streaming:
            dataset = dataset.shuffle(
                buffer_size=int(self.dataset_config.shuffle_buffer_size),
                seed=int(self.seed + self._epoch),
            )
        self._dataset = dataset
        self._iterator = iter(dataset)
        self._buffer = []
        self._epoch += 1

    def reconfigure(self, *, sequence_length: int, micro_batch_size: int) -> None:
        self.sequence_length = int(sequence_length)
        self.micro_batch_size = int(micro_batch_size)
        self._reset_dataset()

    def _fill_buffer(self) -> None:
        if self._iterator is None:
            self._reset_dataset()
        assert self._iterator is not None
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        while len(self._buffer) < ((int(self.sequence_length) + 1) * int(self.micro_batch_size)):
            try:
                row = next(self._iterator)
            except StopIteration:
                self._reset_dataset()
                assert self._iterator is not None
                row = next(self._iterator)
            text = str(row.get(self.dataset_config.text_field, ""))
            if not text.strip():
                continue
            token_ids = self.tokenizer(text, add_special_tokens=True)["input_ids"]
            if not token_ids:
                continue
            if eos_token_id is not None and int(token_ids[-1]) != int(eos_token_id):
                token_ids.append(int(eos_token_id))
            self._buffer.extend(int(token) for token in token_ids)

    def next_batch(self) -> TrainingBatch:
        windows: list[torch.Tensor] = []
        while len(windows) < int(self.micro_batch_size):
            self._fill_buffer()
            if len(self._buffer) < int(self.sequence_length) + 1:
                continue
            window = torch.tensor(self._buffer[: int(self.sequence_length) + 1], dtype=torch.long)
            windows.append(window)
            del self._buffer[: int(self.sequence_length)]
        batch = torch.stack(windows, dim=0).to(dtype=torch.long)
        return TrainingBatch(
            input_ids=batch[:, :-1].contiguous(),
            labels=batch[:, 1:].contiguous(),
        )


class MultiSourceStreamingBatchProvider(BatchSource):
    def __init__(
        self,
        *,
        tokenizer: Any,
        dataset_config: DatasetStreamConfig,
        seed: int,
        sequence_length: int,
        micro_batch_size: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset_config = dataset_config
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.micro_batch_size = int(micro_batch_size)
        self._rng = random.Random(int(seed))
        self._sampler = StreamingCorpusSampler(
            tokenizer=tokenizer,
            dataset_config=dataset_config,
            split="train",
            seed=int(seed),
            sequence_length=int(sequence_length),
        )
        self._buffers: dict[str, list[StreamWindowRecord]] = {"M": [], "H": [], "G": []}
        self._windows_emitted_by_bucket: dict[str, int] = {"M": 0, "H": 0, "G": 0}
        self._stats_flush_counter = 0
        self._flush_stats(force=True)

    def _flush_stats(self, *, force: bool = False) -> None:
        if not self.dataset_config.stream_stats_path:
            return
        self._stats_flush_counter += 1
        if not force and int(self._stats_flush_counter % 4) != 0:
            return
        payload = {
            **self._sampler.stats_payload(),
            "windows_emitted_by_bucket": {
                bucket: int(count) for bucket, count in sorted(self._windows_emitted_by_bucket.items())
            },
            "buffer_sizes": {
                bucket: int(len(records)) for bucket, records in sorted(self._buffers.items())
            },
            "sequence_length": int(self.sequence_length),
            "micro_batch_size": int(self.micro_batch_size),
        }
        _write_json_atomic(self.dataset_config.stream_stats_path, payload)

    def _fill_buffers(
        self,
        *,
        preferred_bucket: str | None = None,
        min_total_windows: int = 1,
    ) -> None:
        max_buffer = int(self.dataset_config.bucket_candidate_buffer_size)
        attempts = 0
        while True:
            total_buffered = sum(len(records) for records in self._buffers.values())
            if preferred_bucket is None:
                if total_buffered >= int(min_total_windows):
                    break
            elif len(self._buffers[str(preferred_bucket)]) > 0 and total_buffered >= int(min_total_windows):
                break
            elif total_buffered >= int(min_total_windows) and attempts > 0:
                break
            if attempts >= 4096:
                if any(len(records) > 0 for records in self._buffers.values()):
                    break
                raise RuntimeError("unable to mine enough streaming windows for the requested bucket mix")
            source, doc_id, token_ids = self._sampler.next_document()
            records = _window_records_from_document(
                token_ids=token_ids,
                source_id=source.source_id,
                doc_id=doc_id,
                sequence_length=int(self.sequence_length),
                block_len=int(self.dataset_config.block_len),
                dataset_config=self.dataset_config,
            )
            attempts += 1
            if not records:
                continue
            self._rng.shuffle(records)
            for record in records:
                bucket = str(record.bucket)
                bucket_records = self._buffers[bucket]
                if len(bucket_records) >= max_buffer:
                    continue
                bucket_records.append(record)
            self._flush_stats()

    def _choose_bucket(self) -> str:
        return self._rng.choices(
            ("M", "H", "G"),
            weights=[float(value) for value in self.dataset_config.train_bucket_weights],
            k=1,
        )[0]

    def reconfigure(self, *, sequence_length: int, micro_batch_size: int) -> None:
        self.sequence_length = int(sequence_length)
        self.micro_batch_size = int(micro_batch_size)
        self._rng = random.Random(int(self.seed + sequence_length + micro_batch_size))
        self._sampler = StreamingCorpusSampler(
            tokenizer=self.tokenizer,
            dataset_config=self.dataset_config,
            split="train",
            seed=int(self.seed + sequence_length + micro_batch_size),
            sequence_length=int(sequence_length),
        )
        self._buffers = {"M": [], "H": [], "G": []}
        self._windows_emitted_by_bucket = {"M": 0, "H": 0, "G": 0}
        self._flush_stats(force=True)

    def next_batch(self) -> TrainingBatch:
        input_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        oracle_rows: list[torch.Tensor] = []
        for _ in range(int(self.micro_batch_size)):
            preferred_bucket = self._choose_bucket()
            self._fill_buffers(min_total_windows=1)
            record = None
            if self._buffers[preferred_bucket]:
                record = self._buffers[preferred_bucket].pop(0)
            else:
                for fallback_bucket in ("M", "H", "G"):
                    if self._buffers[fallback_bucket]:
                        record = self._buffers[fallback_bucket].pop(0)
                        break
            if record is None:
                raise RuntimeError("streaming corpus provider could not produce a training window")
            self._windows_emitted_by_bucket[str(record.bucket)] = int(
                self._windows_emitted_by_bucket.get(str(record.bucket), 0) + 1
            )
            input_rows.append(torch.tensor(record.input_ids, dtype=torch.long))
            label_rows.append(torch.tensor(record.labels, dtype=torch.long))
            oracle_rows.append(torch.tensor(record.oracle_target_block_ids, dtype=torch.long))
        self._flush_stats(force=True)
        return TrainingBatch(
            input_ids=torch.stack(input_rows, dim=0).contiguous(),
            labels=torch.stack(label_rows, dim=0).contiguous(),
            oracle_target_block_ids=torch.stack(oracle_rows, dim=0).contiguous(),
        )


def _usable_token_ids(tokenizer: Any, *, minimum: int = 16) -> list[int]:
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    special_ids = {
        int(value)
        for value in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "bos_token_id", None),
        )
        if value is not None and int(value) >= 0
    }
    candidates = [token_id for token_id in range(vocab_size) if token_id not in special_ids]
    if len(candidates) < int(minimum):
        raise ValueError("tokenizer vocabulary is too small for synthetic needle-recall mode")
    return candidates


def _synthetic_needle_window(
    *,
    tokenizer: Any,
    sequence_length: int,
    block_len: int,
    sample_index: int,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    usable_ids = _usable_token_ids(tokenizer)
    query_token = usable_ids[0]
    filler_ids = usable_ids[1:]
    window = torch.empty((int(sequence_length) + 1,), dtype=torch.long)
    oracle_target_block_ids = torch.full((int(sequence_length),), -100, dtype=torch.long)
    for position in range(int(window.numel())):
        window[position] = int(filler_ids[(sample_index + position) % len(filler_ids)])

    effective_block_len = max(4, int(block_len))
    if int(sequence_length) >= 2 * int(effective_block_len):
        first_block_start = 0
        second_block_start = int(effective_block_len)
        max_pairs = max(1, min((int(effective_block_len) - 2) // 2, (int(sequence_length) - int(second_block_start) - 2) // 3))
        for pair_idx in range(max_pairs):
            key_token = int(filler_ids[(sample_index + (pair_idx * 2)) % len(filler_ids)])
            value_token = int(filler_ids[(sample_index + (pair_idx * 2) + 1) % len(filler_ids)])
            memory_pos = first_block_start + (pair_idx * 2)
            query_pos = second_block_start + (pair_idx * 3)
            if query_pos + 2 >= int(window.numel()):
                break
            window[memory_pos] = key_token
            window[memory_pos + 1] = value_token
            window[query_pos] = int(query_token)
            window[query_pos + 1] = key_token
            window[query_pos + 2] = value_token
            oracle_target_block_ids[query_pos + 1] = 0
        return window.contiguous(), oracle_target_block_ids.contiguous()

    prefix_pairs = max(1, int(sequence_length) // 6)
    query_start = min(int(sequence_length) // 2, int(sequence_length) - 3)
    for pair_idx in range(prefix_pairs):
        key_token = int(filler_ids[(sample_index + (pair_idx * 2)) % len(filler_ids)])
        value_token = int(filler_ids[(sample_index + (pair_idx * 2) + 1) % len(filler_ids)])
        memory_pos = pair_idx * 2
        query_pos = query_start + (pair_idx * 3)
        if memory_pos + 1 >= int(window.numel()) or query_pos + 2 >= int(window.numel()):
            break
        window[memory_pos] = key_token
        window[memory_pos + 1] = value_token
        window[query_pos] = int(query_token)
        window[query_pos + 1] = key_token
        window[query_pos + 2] = value_token
        oracle_target_block_ids[query_pos + 1] = 0
    return window.contiguous(), oracle_target_block_ids.contiguous()


class SyntheticNeedleBatchProvider(BatchSource):
    def __init__(
        self,
        *,
        tokenizer: Any,
        seed: int,
        sequence_length: int,
        micro_batch_size: int,
        block_len: int,
        num_batches: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.micro_batch_size = int(micro_batch_size)
        self.block_len = int(block_len)
        self.num_batches = int(num_batches)
        self.gradient_accumulation_steps = 1
        self._index = 0

    def reconfigure(self, *, sequence_length: int, micro_batch_size: int) -> None:
        self.sequence_length = int(sequence_length)
        self.micro_batch_size = int(micro_batch_size)
        self._index = 0

    def set_gradient_accumulation_steps(self, gradient_accumulation_steps: int) -> None:
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))

    def supports_step_addressing(self) -> bool:
        return True

    def _batch_from_batch_ordinal(
        self,
        *,
        batch_ordinal: int,
        sequence_length: int,
        micro_batch_size: int,
    ) -> TrainingBatch:
        ordinal = int(batch_ordinal) % max(1, int(self.num_batches))
        input_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        oracle_rows: list[torch.Tensor] = []
        for batch_idx in range(int(micro_batch_size)):
            sample_index = int(self.seed + (ordinal * int(micro_batch_size)) + batch_idx)
            window, oracle_target_block_ids = _synthetic_needle_window(
                tokenizer=self.tokenizer,
                sequence_length=int(sequence_length),
                block_len=int(self.block_len),
                sample_index=sample_index,
            )
            input_rows.append(window[:-1].clone())
            label_rows.append(window[1:].clone())
            oracle_rows.append(oracle_target_block_ids)
        input_batch = torch.stack(input_rows, dim=0).to(dtype=torch.long)
        label_batch = torch.stack(label_rows, dim=0).to(dtype=torch.long)
        oracle_batch = torch.stack(oracle_rows, dim=0).to(dtype=torch.long)
        return TrainingBatch(
            input_ids=input_batch.contiguous(),
            labels=label_batch.contiguous(),
            oracle_target_block_ids=oracle_batch.contiguous(),
        )

    def batch_for(
        self,
        *,
        step_index: int,
        microstep: int,
        sequence_length: int,
        micro_batch_size: int,
    ) -> TrainingBatch:
        batch_ordinal = (int(step_index) * int(self.gradient_accumulation_steps)) + int(microstep)
        return self._batch_from_batch_ordinal(
            batch_ordinal=int(batch_ordinal),
            sequence_length=int(sequence_length),
            micro_batch_size=int(micro_batch_size),
        )

    def cycle_length(self) -> int:
        return max(1, int(self.num_batches))

    def packet_key(
        self,
        *,
        step_index: int,
        microstep: int,
        sequence_length: int,
        micro_batch_size: int,
    ) -> dict[str, Any]:
        batch_ordinal = (int(step_index) * int(self.gradient_accumulation_steps)) + int(microstep)
        return {
            "provider": "synthetic_needle",
            "seed": int(self.seed),
            "block_len": int(self.block_len),
            "num_batches": int(self.num_batches),
            "cycle_length": int(self.cycle_length()),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "step_mod": int(batch_ordinal) % int(self.cycle_length()),
            "microstep": int(microstep),
            "sequence_length": int(sequence_length),
            "micro_batch_size": int(micro_batch_size),
        }

    def next_batch(self) -> TrainingBatch:
        batch = self._batch_from_batch_ordinal(
            batch_ordinal=int(self._index),
            sequence_length=int(self.sequence_length),
            micro_batch_size=int(self.micro_batch_size),
        )
        self._index = int((self._index + 1) % max(1, int(self.num_batches)))
        return batch


def build_streaming_eval_packs(
    *,
    tokenizer: Any,
    dataset_config: DatasetStreamConfig,
    seed: int,
) -> tuple[Path, Path]:
    if not dataset_config.source_manifest_path:
        raise ValueError("build_streaming_eval_packs requires dataset_config.source_manifest_path")
    if not dataset_config.dev_memory_fixed_windows_path or not dataset_config.dev_generic_fixed_windows_path:
        raise ValueError("build_streaming_eval_packs requires both dev_memory_fixed_windows_path and dev_generic_fixed_windows_path")
    memory_path = Path(dataset_config.dev_memory_fixed_windows_path)
    generic_path = Path(dataset_config.dev_generic_fixed_windows_path)
    if memory_path.exists() and generic_path.exists():
        return memory_path, generic_path

    sampler = StreamingCorpusSampler(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        split="validation",
        seed=int(seed),
        sequence_length=int(dataset_config.train_sequence_length),
    )
    memory_sequence_length = int(dataset_config.debug_validation_sequence_lengths[0])
    generic_sequence_length = int(dataset_config.validation_sequence_lengths[0])
    target_memory = int(dataset_config.debug_validation_num_windows[0])
    target_generic = int(dataset_config.validation_num_windows[0])
    memory_records: list[StreamWindowRecord] = []
    generic_records: list[StreamWindowRecord] = []
    while len(memory_records) < target_memory or len(generic_records) < target_generic:
        source, doc_id, token_ids = sampler.next_document()
        if len(memory_records) < target_memory:
            memory_candidates = _window_records_from_document(
                token_ids=token_ids,
                source_id=source.source_id,
                doc_id=doc_id,
                sequence_length=int(memory_sequence_length),
                block_len=int(dataset_config.block_len),
                dataset_config=dataset_config,
            )
            for record in memory_candidates:
                if str(record.bucket) == "M" and len(memory_records) < target_memory:
                    memory_records.append(record)
                if len(memory_records) >= target_memory:
                    break
        if len(generic_records) < target_generic:
            generic_candidates = _window_records_from_document(
                token_ids=token_ids,
                source_id=source.source_id,
                doc_id=doc_id,
                sequence_length=int(generic_sequence_length),
                block_len=int(dataset_config.block_len),
                dataset_config=dataset_config,
            )
            for record in generic_candidates:
                if str(record.bucket) == "G" and len(generic_records) < target_generic:
                    generic_records.append(record)
                if len(generic_records) >= target_generic:
                    break
    write_streaming_eval_pack(memory_path, pack_name="Dev-M", records=memory_records)
    write_streaming_eval_pack(generic_path, pack_name="Dev-G", records=generic_records)
    return memory_path, generic_path


def build_streaming_validation_batches(
    *,
    tokenizer: Any,
    dataset_config: DatasetStreamConfig,
    seed: int,
    evaluation_config: Any,
) -> tuple[dict[int, list[ValidationBatch]], dict[int, list[ValidationBatch]]]:
    if not dataset_config.source_manifest_path:
        raise ValueError("build_streaming_validation_batches requires dataset_config.source_manifest_path")

    sampler = StreamingCorpusSampler(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        split="validation",
        seed=int(seed),
        sequence_length=int(dataset_config.train_sequence_length),
    )
    debug_targets = {
        int(sequence_length): int(num_windows)
        for sequence_length, num_windows in zip(
            evaluation_config.debug_validation_sequence_lengths,
            evaluation_config.debug_validation_num_windows,
            strict=True,
        )
    }
    full_targets = {
        int(sequence_length): int(num_windows)
        for sequence_length, num_windows in zip(
            evaluation_config.validation_sequence_lengths,
            evaluation_config.validation_num_windows,
            strict=True,
        )
    }
    debug_fallback_records: dict[int, dict[str, list[StreamWindowRecord]]] = {
        int(length): {"M": [], "H": [], "G": []} for length in debug_targets
    }
    full_fallback_records: dict[int, dict[str, list[StreamWindowRecord]]] = {
        int(length): {"M": [], "H": [], "G": []} for length in full_targets
    }
    debug_records: dict[int, list[StreamWindowRecord]] = {int(length): [] for length in debug_targets}
    full_records: dict[int, list[StreamWindowRecord]] = {int(length): [] for length in full_targets}

    def _targets_satisfied() -> bool:
        debug_ready = all(
            sum(len(records) for records in debug_fallback_records[length].values()) >= debug_targets[length]
            for length in debug_targets
        )
        full_ready = all(
            sum(len(records) for records in full_fallback_records[length].values()) >= full_targets[length]
            for length in full_targets
        )
        return bool(debug_ready and full_ready)

    attempts = 0
    max_attempts = max(2048, 32 * (sum(debug_targets.values()) + sum(full_targets.values())))
    while not _targets_satisfied():
        source, doc_id, token_ids = sampler.next_document()
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("unable to mine enough live streaming validation windows from the configured sources")
        for sequence_length, target_windows in debug_targets.items():
            if sum(len(records) for records in debug_fallback_records[sequence_length].values()) >= target_windows:
                continue
            candidates = _window_records_from_document(
                token_ids=token_ids,
                source_id=source.source_id,
                doc_id=doc_id,
                sequence_length=int(sequence_length),
                block_len=int(dataset_config.block_len),
                dataset_config=dataset_config,
            )
            for record in candidates:
                debug_fallback_records[sequence_length][str(record.bucket)].append(record)
                if str(record.bucket) == "M" and len(debug_records[sequence_length]) < target_windows:
                    debug_records[sequence_length].append(record)
        for sequence_length, target_windows in full_targets.items():
            if sum(len(records) for records in full_fallback_records[sequence_length].values()) >= target_windows:
                continue
            candidates = _window_records_from_document(
                token_ids=token_ids,
                source_id=source.source_id,
                doc_id=doc_id,
                sequence_length=int(sequence_length),
                block_len=int(dataset_config.block_len),
                dataset_config=dataset_config,
            )
            for record in candidates:
                full_fallback_records[sequence_length][str(record.bucket)].append(record)
                if str(record.bucket) == "G" and len(full_records[sequence_length]) < target_windows:
                    full_records[sequence_length].append(record)

    for sequence_length, target_windows in debug_targets.items():
        if len(debug_records[sequence_length]) >= target_windows:
            continue
        seen_keys = {
            (record.source_id, record.doc_id, int(record.window_start))
            for record in debug_records[sequence_length]
        }
        for bucket in _validation_bucket_priority("memory"):
            for record in debug_fallback_records[sequence_length][bucket]:
                key = (record.source_id, record.doc_id, int(record.window_start))
                if key in seen_keys:
                    continue
                debug_records[sequence_length].append(record)
                seen_keys.add(key)
                if len(debug_records[sequence_length]) >= target_windows:
                    break
            if len(debug_records[sequence_length]) >= target_windows:
                break
        if len(debug_records[sequence_length]) < target_windows:
            raise RuntimeError("unable to mine enough live streaming validation windows from the configured sources")

    for sequence_length, target_windows in full_targets.items():
        if len(full_records[sequence_length]) >= target_windows:
            continue
        seen_keys = {
            (record.source_id, record.doc_id, int(record.window_start))
            for record in full_records[sequence_length]
        }
        for bucket in _validation_bucket_priority("generic"):
            for record in full_fallback_records[sequence_length][bucket]:
                key = (record.source_id, record.doc_id, int(record.window_start))
                if key in seen_keys:
                    continue
                full_records[sequence_length].append(record)
                seen_keys.add(key)
                if len(full_records[sequence_length]) >= target_windows:
                    break
            if len(full_records[sequence_length]) >= target_windows:
                break
        if len(full_records[sequence_length]) < target_windows:
            raise RuntimeError("unable to mine enough live streaming validation windows from the configured sources")

    return _validation_batches_from_stream_records(debug_records), _validation_batches_from_stream_records(full_records)


def validation_token_stream_from_dataset(
    *,
    tokenizer: Any,
    dataset_config: DatasetStreamConfig,
) -> torch.LongTensor:
    dataset = load_text_dataset(
        dataset_config,
        split=dataset_config.validation_split,
        streaming=False,
    )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    tokens: list[int] = []
    for row in dataset:
        text = str(row.get(dataset_config.text_field, ""))
        if not text.strip():
            continue
        token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if not token_ids:
            continue
        if eos_token_id is not None and int(token_ids[-1]) != int(eos_token_id):
            token_ids.append(int(eos_token_id))
        tokens.extend(int(token) for token in token_ids)
    if len(tokens) < 2:
        raise ValueError("validation dataset did not produce enough tokens")
    return torch.tensor(tokens, dtype=torch.long)


def build_validation_batch_windows(
    *,
    token_stream: torch.LongTensor,
    sequence_lengths: Sequence[int],
    num_windows: Sequence[int],
) -> dict[int, list[ValidationBatch]]:
    batches: dict[int, list[ValidationBatch]] = {}
    for sequence_length, windows in zip(sequence_lengths, num_windows, strict=True):
        starts = window_starts(int(token_stream.numel()), int(sequence_length), int(windows))
        if not starts:
            raise ValueError(f"validation token stream is too short for sequence length {sequence_length}")
        entries: list[ValidationBatch] = []
        for start in starts:
            window = token_stream[start : start + int(sequence_length) + 1]
            batch = window.unsqueeze(0).to(dtype=torch.long)
            entries.append(
                ValidationBatch(
                    sequence_length=int(sequence_length),
                    input_ids=batch[:, :-1].contiguous(),
                    labels=batch[:, 1:].contiguous(),
                )
            )
        batches[int(sequence_length)] = entries
    return batches


def build_validation_batches(
    *,
    tokenizer: Any,
    dataset_config: DatasetStreamConfig,
    evaluation_config,
    text_fallback: str,
    host_refill_depth: int = 8,
    cache_batches: bool = True,
) -> tuple[dict[int, list[ValidationBatch]], dict[int, list[ValidationBatch]]]:
    validation_source = build_validation_source(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        evaluation_config=evaluation_config,
        text_fallback=text_fallback,
        host_refill_depth=int(host_refill_depth),
        cache_batches=bool(cache_batches),
    )
    try:
        return (
            validation_source.materialize(eval_kind="debug"),
            validation_source.materialize(eval_kind="full"),
        )
    finally:
        validation_source.close()


def build_validation_source(
    *,
    tokenizer: Any,
    dataset_config: DatasetStreamConfig,
    evaluation_config: Any,
    text_fallback: str,
    host_refill_depth: int = 8,
    cache_batches: bool = True,
) -> ValidationBatchSource:
    if dataset_config.dev_memory_fixed_windows_path and dataset_config.dev_generic_fixed_windows_path:
        memory_path = Path(dataset_config.dev_memory_fixed_windows_path)
        generic_path = Path(dataset_config.dev_generic_fixed_windows_path)
        if memory_path.exists() and generic_path.exists():
            return StaticValidationBatchSource(
                debug_batches=load_streaming_eval_pack(memory_path),
                full_batches=load_streaming_eval_pack(generic_path),
            )
    if dataset_config.source_manifest_path:
        return StreamingValidationBatchSource(
            tokenizer=tokenizer,
            dataset_config=dataset_config,
            seed=0,
            evaluation_config=evaluation_config,
            host_refill_depth=int(host_refill_depth),
            cache_batches=bool(cache_batches),
        )
    if text_fallback:
        token_stream = build_token_stream(tokenizer, text_fallback)
    else:
        token_stream = validation_token_stream_from_dataset(tokenizer=tokenizer, dataset_config=dataset_config)
    debug_batches = build_validation_batch_windows(
        token_stream=token_stream,
        sequence_lengths=evaluation_config.debug_validation_sequence_lengths,
        num_windows=evaluation_config.debug_validation_num_windows,
    )
    full_batches = build_validation_batch_windows(
        token_stream=token_stream,
        sequence_lengths=evaluation_config.validation_sequence_lengths,
        num_windows=evaluation_config.validation_num_windows,
    )
    return StaticValidationBatchSource(
        debug_batches=debug_batches,
        full_batches=full_batches,
    )


def build_synthetic_needle_validation_batches(
    *,
    tokenizer: Any,
    block_len: int,
    sequence_lengths: Sequence[int],
    num_windows: Sequence[int],
    seed: int,
) -> dict[int, list[ValidationBatch]]:
    batches: dict[int, list[ValidationBatch]] = {}
    running_index = int(seed)
    for sequence_length, windows in zip(sequence_lengths, num_windows, strict=True):
        entries: list[ValidationBatch] = []
        for _ in range(int(windows)):
            window, oracle_target_block_ids = _synthetic_needle_window(
                tokenizer=tokenizer,
                sequence_length=int(sequence_length),
                block_len=int(block_len),
                sample_index=running_index,
            )
            input_ids = window[:-1].clone()
            labels = window[1:].clone()
            entries.append(
                ValidationBatch(
                    sequence_length=int(sequence_length),
                    input_ids=input_ids.unsqueeze(0).to(dtype=torch.long).contiguous(),
                    labels=labels.unsqueeze(0).to(dtype=torch.long).contiguous(),
                    oracle_target_block_ids=oracle_target_block_ids.unsqueeze(0).to(dtype=torch.long).contiguous(),
                )
            )
            running_index += 1
        batches[int(sequence_length)] = entries
    return batches
