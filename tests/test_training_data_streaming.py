from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from gfst_hmb_mpuc.config import DatasetStreamConfig
from gfst_hmb_mpuc.training.data import (
    MultiSourceStreamingBatchProvider,
    StreamingValidationBatchSource,
    StreamWindowRecord,
    _extract_stream_text,
    _bucket_for_window,
    _load_streaming_source_manifest,
    _row_matches_manifest_filters,
    _window_oracle_targets,
    load_streaming_eval_pack,
    write_streaming_eval_pack,
)


class _WhitespaceTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def __call__(self, text: str, *, add_special_tokens: bool = True, return_tensors=None):
        token_ids: list[int] = []
        for token in text.split():
            token_ids.append(self._vocab.setdefault(token, len(self._vocab) + 1))
        if add_special_tokens:
            token_ids.append(int(self.eos_token_id))
        return {"input_ids": token_ids}


def _write_jsonl_gz(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
    return path


def test_streaming_source_manifest_loads_local_jsonl_gz(tmp_path: Path) -> None:
    shard_path = _write_jsonl_gz(
        tmp_path / "docs.jsonl.gz",
        [{"doc_id": "doc-1", "text": "alpha beta gamma delta"}],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "books_local",
                        "family": "books",
                        "stream_mode": "local_jsonl_gz",
                        "uri_or_repo": str(shard_path),
                        "base_weight": 1.0,
                        "allow_streaming_only": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    entries = _load_streaming_source_manifest(manifest_path)

    assert len(entries) == 1
    assert entries[0].source_id == "books_local"
    assert entries[0].stream_mode == "local_jsonl_gz"
    assert entries[0].uri_or_repo == str(shard_path)


def test_streaming_source_manifest_supports_text_fields_and_value_filters(tmp_path: Path) -> None:
    shard_path = _write_jsonl_gz(tmp_path / "docs.jsonl.gz", [{"doc_id": "doc-1", "text": "alpha beta gamma delta"}])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "code_local",
                        "family": "code",
                        "stream_mode": "local_jsonl_gz",
                        "uri_or_repo": str(shard_path),
                        "base_weight": 1.0,
                        "text_field": "content",
                        "text_fields": ["front", "body"],
                        "include_field_values": {"license": ["mit", "apache-2.0"]},
                        "exclude_field_values": {"path": ["vendor/generated.py"]},
                        "allow_streaming_only": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    entries = _load_streaming_source_manifest(manifest_path)

    assert len(entries) == 1
    assert entries[0].text_field == "content"
    assert entries[0].text_fields == ("front", "body")
    assert entries[0].include_field_values["license"] == ("mit", "apache-2.0")
    assert entries[0].exclude_field_values["path"] == ("vendor/generated.py",)


def test_extract_stream_text_joins_structured_fields_and_filters_manifest_values(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "science_local",
                        "family": "science",
                        "stream_mode": "local_jsonl_gz",
                        "uri_or_repo": str(tmp_path / "science.jsonl.gz"),
                        "base_weight": 1.0,
                        "text_fields": ["front", "methods", "results"],
                        "include_field_values": {"license": ["mit", "apache-2.0"]},
                        "allow_streaming_only": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = _load_streaming_source_manifest(manifest_path)
    source = entries[0]
    accepted_row = {
        "license": "mit",
        "path": "src/main.py",
        "front": ["Title", "Abstract"],
        "methods": ["Method A", "Method B"],
        "results": ["Result A"],
    }
    rejected_row = {
        "license": "gpl-3.0",
        "path": "src/main.py",
        "front": ["Title"],
        "methods": ["Method A"],
        "results": ["Result A"],
    }

    assert _row_matches_manifest_filters(accepted_row, source) is True
    assert _row_matches_manifest_filters(rejected_row, source) is False
    assert _extract_stream_text(accepted_row, source) == "Title\n\nAbstract\n\nMethod A\n\nMethod B\n\nResult A"


def test_window_oracle_targets_and_bucketization() -> None:
    window = [11, 12, 13, 14, 11, 12, 13, 14, 11]
    oracle_target_block_ids, oracle_mask, repeated_anchor_count = _window_oracle_targets(
        window,
        sequence_length=8,
        block_len=4,
    )
    dataset_config = DatasetStreamConfig(
        train_sequence_length=8,
        block_len=4,
        debug_validation_sequence_lengths=(8,),
        debug_validation_num_windows=(2,),
        validation_sequence_lengths=(8,),
        validation_num_windows=(2,),
        memory_positive_min_oracle_positions=2,
        hard_negative_min_repeat_anchors=2,
    )

    bucket = _bucket_for_window(
        oracle_mask=oracle_mask,
        repeated_anchor_count=repeated_anchor_count,
        dataset_config=dataset_config,
    )

    assert oracle_target_block_ids[4] == 0
    assert oracle_mask[4] is True
    assert repeated_anchor_count >= 4
    assert bucket == "M"


def test_streaming_eval_pack_roundtrip(tmp_path: Path) -> None:
    pack_path = tmp_path / "dev_m.json"
    write_streaming_eval_pack(
        pack_path,
        pack_name="Dev-M",
        records=[
            StreamWindowRecord(
                sequence_length=8,
                input_ids=[1, 2, 3, 4, 1, 2, 3, 4],
                labels=[2, 3, 4, 1, 2, 3, 4, 0],
                oracle_target_block_ids=[-100, -100, -100, -100, 0, 0, 0, 0],
                oracle_mask=[False, False, False, False, True, True, True, True],
                source_id="books_local",
                doc_id="doc-1",
                window_start=0,
                bucket="M",
            )
        ],
    )

    grouped = load_streaming_eval_pack(pack_path)

    assert list(grouped.keys()) == [8]
    batch = grouped[8][0]
    assert tuple(batch.input_ids.shape) == (1, 8)
    assert tuple(batch.labels.shape) == (1, 8)
    assert batch.oracle_target_block_ids is not None
    assert int(batch.oracle_target_block_ids[0, 4].item()) == 0


def test_multi_source_streaming_batch_provider_writes_stream_stats(tmp_path: Path) -> None:
    shard_path = _write_jsonl_gz(
        tmp_path / "docs.jsonl.gz",
        [
            {"doc_id": "mem-1", "text": "a b c d a b c d a b c d a b c d"},
            {"doc_id": "gen-1", "text": "q w e r t y u i o p s d f g h j"},
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    stats_path = tmp_path / "stream_stats.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "mixed_local",
                        "family": "books",
                        "stream_mode": "local_jsonl_gz",
                        "uri_or_repo": str(shard_path),
                        "base_weight": 1.0,
                        "text_field": "text",
                        "allow_streaming_only": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tokenizer = _WhitespaceTokenizer()
    dataset_config = DatasetStreamConfig(
        source_manifest_path=str(manifest_path),
        require_streaming_only=True,
        stream_stats_path=str(stats_path),
        train_sequence_length=8,
        block_len=4,
        debug_validation_sequence_lengths=(8,),
        debug_validation_num_windows=(2,),
        validation_sequence_lengths=(8,),
        validation_num_windows=(2,),
        min_document_tokens=8,
        memory_positive_min_oracle_positions=2,
        hard_negative_min_repeat_anchors=2,
        train_bucket_weights=(0.6, 0.2, 0.2),
    )

    provider = MultiSourceStreamingBatchProvider(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        seed=7,
        sequence_length=8,
        micro_batch_size=1,
    )
    batch = provider.next_batch()

    assert tuple(batch.input_ids.shape) == (1, 8)
    assert batch.oracle_target_block_ids is not None
    assert stats_path.exists()
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["sampled_documents"] >= 1
    assert "mixed_local" in payload["source_mix"]


def test_multi_source_streaming_batch_provider_falls_back_when_preferred_bucket_is_unavailable(tmp_path: Path) -> None:
    shard_path = _write_jsonl_gz(
        tmp_path / "docs.jsonl.gz",
        [
            {"doc_id": "mem-1", "text": "a b c d a b c d a b c d a b c d"},
            {"doc_id": "mem-2", "text": "k l m n k l m n k l m n k l m n"},
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "memory_only_local",
                        "family": "books",
                        "stream_mode": "local_jsonl_gz",
                        "uri_or_repo": str(shard_path),
                        "base_weight": 1.0,
                        "text_field": "text",
                        "allow_streaming_only": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tokenizer = _WhitespaceTokenizer()
    dataset_config = DatasetStreamConfig(
        source_manifest_path=str(manifest_path),
        require_streaming_only=True,
        train_sequence_length=8,
        block_len=4,
        debug_validation_sequence_lengths=(8,),
        debug_validation_num_windows=(2,),
        validation_sequence_lengths=(8,),
        validation_num_windows=(2,),
        min_document_tokens=8,
        memory_positive_min_oracle_positions=2,
        hard_negative_min_repeat_anchors=2,
        train_bucket_weights=(1e-6, 1e-6, 1.0),
    )

    provider = MultiSourceStreamingBatchProvider(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        seed=7,
        sequence_length=8,
        micro_batch_size=1,
    )
    provider._choose_bucket = lambda: "G"  # type: ignore[method-assign]

    batch = provider.next_batch()

    assert tuple(batch.input_ids.shape) == (1, 8)
    assert batch.oracle_target_block_ids is not None


def test_streaming_validation_source_emits_cached_live_batches_without_writing_packs(tmp_path: Path) -> None:
    shard_path = _write_jsonl_gz(
        tmp_path / "docs.jsonl.gz",
        [
            {"doc_id": "mem-1", "text": "a b c d a b c d a b c d a b c d"},
            {"doc_id": "gen-1", "text": "q w e r t y u i o p s d f g h j"},
            {"doc_id": "mix-1", "text": "n o p q n o p q x y z k l m n o"},
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "mixed_local",
                        "family": "books",
                        "stream_mode": "local_jsonl_gz",
                        "uri_or_repo": str(shard_path),
                        "base_weight": 1.0,
                        "text_field": "text",
                        "allow_streaming_only": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tokenizer = _WhitespaceTokenizer()
    dataset_config = DatasetStreamConfig(
        source_manifest_path=str(manifest_path),
        require_streaming_only=True,
        train_sequence_length=8,
        block_len=4,
        debug_validation_sequence_lengths=(8,),
        debug_validation_num_windows=(2,),
        validation_sequence_lengths=(8,),
        validation_num_windows=(2,),
        min_document_tokens=8,
        memory_positive_min_oracle_positions=2,
        hard_negative_min_repeat_anchors=2,
    )
    evaluation_config = SimpleNamespace(
        debug_validation_sequence_lengths=(8,),
        debug_validation_num_windows=(2,),
        validation_sequence_lengths=(8,),
        validation_num_windows=(2,),
    )

    source = StreamingValidationBatchSource(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        seed=7,
        evaluation_config=evaluation_config,
    )
    try:
        debug_batches = list(source.iter_batches(eval_kind="debug", sequence_length=8))
        full_batches = list(source.iter_batches(eval_kind="full", sequence_length=8))
        cached_debug = source.materialize(eval_kind="debug")
        cached_full = source.materialize(eval_kind="full")
        telemetry = source.telemetry()
    finally:
        source.close()

    assert len(debug_batches) == 2
    assert len(full_batches) == 2
    assert len(cached_debug[8]) == 2
    assert len(cached_full[8]) == 2
    if torch.cuda.is_available():
        assert debug_batches[0].input_ids.is_pinned()
        assert full_batches[0].input_ids.is_pinned()
    assert telemetry["cache_complete"] is True
    assert telemetry["windows_emitted"]["debug"]["8"] == 2
    assert telemetry["windows_emitted"]["full"]["8"] == 2
    assert telemetry["queue_depth_peak"]["debug:8"] <= 3
    assert telemetry["queue_depth_peak"]["full:8"] <= 3
    assert telemetry["staging_depth_peak"]["debug:8"] <= 8
    assert telemetry["staging_depth_peak"]["full:8"] <= 8
    assert sorted(path.name for path in tmp_path.iterdir()) == ["docs.jsonl.gz", "manifest.json"]


def test_streaming_validation_source_promotes_fallback_windows_when_memory_bucket_is_unavailable(tmp_path: Path) -> None:
    shard_path = _write_jsonl_gz(
        tmp_path / "docs.jsonl.gz",
        [
            {"doc_id": "hard-1", "text": "a b c d a b c d e f g h i j k l"},
            {"doc_id": "hard-2", "text": "m n o p m n o p q r s t u v w x"},
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "hard_only_local",
                        "family": "books",
                        "stream_mode": "local_jsonl_gz",
                        "uri_or_repo": str(shard_path),
                        "base_weight": 1.0,
                        "text_field": "text",
                        "allow_streaming_only": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tokenizer = _WhitespaceTokenizer()
    dataset_config = DatasetStreamConfig(
        source_manifest_path=str(manifest_path),
        require_streaming_only=True,
        train_sequence_length=8,
        block_len=4,
        debug_validation_sequence_lengths=(8,),
        debug_validation_num_windows=(1,),
        validation_sequence_lengths=(8,),
        validation_num_windows=(1,),
        min_document_tokens=8,
        memory_positive_min_oracle_positions=5,
        hard_negative_min_repeat_anchors=2,
        hard_negative_max_oracle_positions=0,
    )
    evaluation_config = SimpleNamespace(
        debug_validation_sequence_lengths=(8,),
        debug_validation_num_windows=(1,),
        validation_sequence_lengths=(8,),
        validation_num_windows=(1,),
    )

    source = StreamingValidationBatchSource(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        seed=3,
        evaluation_config=evaluation_config,
    )
    try:
        debug_batches = source.materialize(eval_kind="debug")
        full_batches = source.materialize(eval_kind="full")
    finally:
        source.close()

    assert len(debug_batches[8]) == 1
    assert len(full_batches[8]) == 1


def test_streaming_validation_source_completes_without_cache_batches(tmp_path: Path) -> None:
    shard_path = _write_jsonl_gz(
        tmp_path / "docs.jsonl.gz",
        [
            {"doc_id": "doc-1", "text": "a b c d a b c d a b c d a b c d"},
            {"doc_id": "doc-2", "text": "q w e r t y u i q w e r t y u i"},
            {"doc_id": "doc-3", "text": "z x c v b n m l z x c v b n m l"},
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "mixed_local",
                        "family": "books",
                        "stream_mode": "local_jsonl_gz",
                        "uri_or_repo": str(shard_path),
                        "base_weight": 1.0,
                        "text_field": "text",
                        "allow_streaming_only": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tokenizer = _WhitespaceTokenizer()
    dataset_config = DatasetStreamConfig(
        source_manifest_path=str(manifest_path),
        require_streaming_only=True,
        train_sequence_length=8,
        block_len=4,
        debug_validation_sequence_lengths=(8,),
        debug_validation_num_windows=(1,),
        validation_sequence_lengths=(8,),
        validation_num_windows=(2,),
        min_document_tokens=8,
        memory_positive_min_oracle_positions=2,
        hard_negative_min_repeat_anchors=2,
    )
    evaluation_config = SimpleNamespace(
        debug_validation_sequence_lengths=(8,),
        debug_validation_num_windows=(1,),
        validation_sequence_lengths=(8,),
        validation_num_windows=(2,),
    )

    source = StreamingValidationBatchSource(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        seed=13,
        evaluation_config=evaluation_config,
        host_refill_depth=2,
        cache_batches=False,
    )
    try:
        slots = list(source.iter_staging_slots(eval_kind="full", sequence_length=8))
        for slot in slots:
            source.release_staging_slot(eval_kind="full", sequence_length=8, slot=slot)
        telemetry = source.telemetry()
        with pytest.raises(RuntimeError, match="cache_batches=True"):
            source.materialize(eval_kind="full")
    finally:
        source.close()

    assert len(slots) == 2
    assert telemetry["cache_enabled"] is False
    assert telemetry["cache_complete"] is False
    assert telemetry["stream_complete"] is True
    assert telemetry["windows_targeted"]["full"]["8"] == 2
    assert telemetry["windows_emitted"]["full"]["8"] == 2
    assert telemetry["windows_released"]["full"]["8"] == 2


def test_streaming_validation_source_reuses_staging_slots(tmp_path: Path) -> None:
    shard_path = _write_jsonl_gz(
        tmp_path / "docs.jsonl.gz",
        [
            {"doc_id": "doc-1", "text": "a b c d a b c d a b c d a b c d"},
            {"doc_id": "doc-2", "text": "q w e r t y u i q w e r t y u i"},
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "reuse_local",
                        "family": "books",
                        "stream_mode": "local_jsonl_gz",
                        "uri_or_repo": str(shard_path),
                        "base_weight": 1.0,
                        "text_field": "text",
                        "allow_streaming_only": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tokenizer = _WhitespaceTokenizer()
    dataset_config = DatasetStreamConfig(
        source_manifest_path=str(manifest_path),
        require_streaming_only=True,
        train_sequence_length=8,
        block_len=4,
        debug_validation_sequence_lengths=(8,),
        debug_validation_num_windows=(1,),
        validation_sequence_lengths=(8,),
        validation_num_windows=(2,),
        min_document_tokens=8,
        memory_positive_min_oracle_positions=2,
        hard_negative_min_repeat_anchors=2,
    )
    evaluation_config = SimpleNamespace(
        debug_validation_sequence_lengths=(8,),
        debug_validation_num_windows=(1,),
        validation_sequence_lengths=(8,),
        validation_num_windows=(2,),
    )

    source = StreamingValidationBatchSource(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        seed=11,
        evaluation_config=evaluation_config,
        host_refill_depth=1,
    )
    try:
        iterator = source.iter_staging_slots(eval_kind="full", sequence_length=8)
        first_slot = next(iterator)
        first_slot_id = id(first_slot)
        source.release_staging_slot(eval_kind="full", sequence_length=8, slot=first_slot)
        second_slot = next(iterator)
        second_slot_id = id(second_slot)
        source.release_staging_slot(eval_kind="full", sequence_length=8, slot=second_slot)
        telemetry = source.telemetry()
    finally:
        source.close()

    assert first_slot_id == second_slot_id
    assert telemetry["staging_depth_peak"]["full:8"] == 1
    assert telemetry["staging_depth_current"]["full:8"] in (0, 1)
