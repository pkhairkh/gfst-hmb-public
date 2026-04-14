from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_dashboard_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "training_dashboard.py"
    spec = importlib.util.spec_from_file_location("training_dashboard", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _build_run(run_dir: Path) -> None:
    _write_jsonl(
        run_dir / "events" / "train.jsonl",
        [
            {
                "schema_version": 2,
                "event_type": "train_step",
                "run_id": "demo_run",
                "step": 1,
                "stage": "A",
                "phase": 1,
                "mode": "distill",
                "teacher_overlap_mode": "gpu_pipeline",
                "gpu_slot_count": 3,
                "gpu_slot_depth_peak": 2,
                "slot_depth_avg": 2.2,
                "host_refill_depth_peak": 6,
                "host_pinned_mb": 32.0,
                "gpu_slot_pool_mb": 96.0,
                "next_token_loss": 8.2,
                "distill_loss": 1.8,
                "geometry_boundary_loss": 0.0,
                "router_loss": 0.4,
                "pointer_loss": 1.1,
                "retrieval_loss": 0.0,
                "copy_active_fraction": 0.95,
                "total_loss": 10.0,
                "tokens_per_second": 1234.5,
                "teacher_used": True,
                "teacher_logits_mb": 24.0,
                "teacher_hidden_mb": 0.0,
                "cuda_allocated_mb": 12345.0,
                "cuda_reserved_mb": 16384.0,
                "cuda_peak_allocated_mb": 17000.0,
                "gpu_util_avg_pct": 71.0,
                "gpu_util_peak_pct": 82.0,
                "gpu_power_avg_w": 205.0,
                "gpu_power_peak_w": 225.0,
                "gpu_vram_used_mb_avg": 14000.0,
                "gpu_util_ema_pct": 74.0,
                "gpu_power_ema_w": 210.0,
                "active_sequence_length": 128,
                "active_micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "lr": 0.0004,
                "step_wall_ms": 1500.0,
            },
            {
                "schema_version": 2,
                "event_type": "train_step",
                "run_id": "demo_run",
                "step": 2,
                "stage": "A",
                "phase": 1,
                "mode": "distill",
                "teacher_overlap_mode": "gpu_pipeline",
                "gpu_slot_count": 3,
                "gpu_slot_depth_peak": 3,
                "slot_depth_avg": 2.4,
                "host_refill_depth_peak": 6,
                "host_pinned_mb": 32.0,
                "gpu_slot_pool_mb": 96.0,
                "next_token_loss": 7.2,
                "distill_loss": 0.8,
                "geometry_boundary_loss": 0.1,
                "router_loss": 0.3,
                "pointer_loss": 0.9,
                "retrieval_loss": 0.0,
                "copy_active_fraction": 0.97,
                "total_loss": 8.1,
                "tokens_per_second": 1334.5,
                "teacher_used": True,
                "teacher_logits_mb": 24.0,
                "teacher_hidden_mb": 12.0,
                "cuda_allocated_mb": 12345.0,
                "cuda_reserved_mb": 16384.0,
                "cuda_peak_allocated_mb": 17000.0,
                "gpu_util_avg_pct": 75.0,
                "gpu_util_peak_pct": 88.0,
                "gpu_power_avg_w": 211.0,
                "gpu_power_peak_w": 230.0,
                "gpu_vram_used_mb_avg": 14500.0,
                "gpu_util_ema_pct": 76.0,
                "gpu_power_ema_w": 212.0,
                "active_sequence_length": 128,
                "active_micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "lr": 0.0004,
                "step_wall_ms": 1100.0,
            },
        ],
    )
    _write_jsonl(
        run_dir / "events" / "eval.jsonl",
        [
            {
                "schema_version": 2,
                "event_type": "eval",
                "run_id": "demo_run",
                "step": 2,
                "eval_kind": "full",
                "val_ce": 6.4,
                "teacher_gap": 0.9,
                "decode_round_trip_ok": True,
                "student_ce_by_length": {"64": 6.7},
            }
        ],
    )
    _write_jsonl(
        run_dir / "events" / "system.jsonl",
        [
            {
                "schema_version": 2,
                "event_type": "run_config",
                "run_id": "demo_run",
                "config": {
                    "steps": 10,
                    "optimizer": {"optimizer_name": "adamw_only"},
                    "model": {
                        "train_scope": "all",
                        "gfst_update_rule": "residual_rank_one_v1",
                        "summary_layout_version": 1,
                        "retrieval_key_version": "mean_top1_v1",
                    },
                },
                "model_metadata": {
                    "gfst_update_rule": "residual_rank_one_v1",
                    "summary_layout_version": 1,
                    "retrieval_key_version": "mean_top1_v1",
                },
                "recovery_state": {
                    "warm_start_mode": "weights_only",
                    "source_run_id": "prior_run",
                    "source_checkpoint_step": 42,
                },
            },
            {
                "schema_version": 2,
                "event_type": "bootstrap_loaded",
                "run_id": "demo_run",
            },
            {
                "schema_version": 2,
                "event_type": "step_timing",
                "run_id": "demo_run",
                "step": 1,
                "step_wall_ms": 1500.0,
                "student_forward_ms": 400.0,
                "teacher_overlap_mode": "gpu_pipeline",
                "gpu_slot_count": 3,
                "gpu_slot_depth_peak": 2,
                "gpu_slot_depth_avg": 2.2,
                "host_refill_depth_peak": 6,
                "gpu_stage_launch_ms": 30.0,
                "gpu_stage_wait_ms": 5.0,
                "teacher_prefetch_launch_ms": 120.0,
                "teacher_prefetch_wait_ms": 40.0,
                "student_wait_on_teacher_ms": 40.0,
                "student_wait_on_stage_ms": 5.0,
                "teacher_logits_mb": 24.0,
                "teacher_hidden_mb": 12.0,
                "host_pinned_mb": 32.0,
                "gpu_slot_pool_mb": 96.0,
                "cuda_allocated_mb": 12345.0,
                "cuda_reserved_mb": 16384.0,
                "cuda_peak_allocated_mb": 17000.0,
                "gpu_util_avg_pct": 71.0,
                "gpu_util_peak_pct": 82.0,
                "gpu_power_avg_w": 205.0,
                "gpu_power_peak_w": 225.0,
                "gpu_vram_used_mb_avg": 14000.0,
                "gpu_util_ema_pct": 74.0,
                "gpu_power_ema_w": 210.0,
                "backward_ms": 700.0,
                "optimizer_ms": 100.0,
            },
            {
                "schema_version": 2,
                "event_type": "step_timing",
                "run_id": "demo_run",
                "step": 2,
                "step_wall_ms": 1100.0,
                "student_forward_ms": 300.0,
                "teacher_overlap_mode": "gpu_pipeline",
                "gpu_slot_count": 3,
                "gpu_slot_depth_peak": 3,
                "gpu_slot_depth_avg": 2.4,
                "host_refill_depth_peak": 6,
                "gpu_stage_launch_ms": 30.0,
                "gpu_stage_wait_ms": 5.0,
                "teacher_prefetch_launch_ms": 120.0,
                "teacher_prefetch_wait_ms": 40.0,
                "student_wait_on_teacher_ms": 40.0,
                "student_wait_on_stage_ms": 5.0,
                "teacher_logits_mb": 24.0,
                "teacher_hidden_mb": 12.0,
                "host_pinned_mb": 32.0,
                "gpu_slot_pool_mb": 96.0,
                "cuda_allocated_mb": 12345.0,
                "cuda_reserved_mb": 16384.0,
                "cuda_peak_allocated_mb": 17000.0,
                "gpu_util_avg_pct": 75.0,
                "gpu_util_peak_pct": 88.0,
                "gpu_power_avg_w": 211.0,
                "gpu_power_peak_w": 230.0,
                "gpu_vram_used_mb_avg": 14500.0,
                "gpu_util_ema_pct": 76.0,
                "gpu_power_ema_w": 212.0,
                "backward_ms": 500.0,
                "optimizer_ms": 100.0,
            },
            {
                "schema_version": 2,
                "event_type": "eval_start",
                "run_id": "demo_run",
                "step": 2,
                "eval_kind": "full",
            },
            {
                "schema_version": 2,
                "event_type": "eval_done",
                "run_id": "demo_run",
                "step": 2,
                "eval_kind": "full",
                "eval_ms": 5000.0,
            },
        ],
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "demo_run",
                "config": {
                    "steps": 10,
                    "model": {
                        "gfst_update_rule": "residual_rank_one_v1",
                        "summary_layout_version": 1,
                        "retrieval_key_version": "mean_top1_v1",
                    },
                },
                "metadata": {
                    "gfst_update_rule": "residual_rank_one_v1",
                    "summary_layout_version": 1,
                    "retrieval_key_version": "mean_top1_v1",
                    "warm_start_mode": "weights_only",
                    "source_run_id": "prior_run",
                    "source_checkpoint_step": 42,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"best-checkpoint")
    (run_dir / "checkpoints" / "latest.pt.meta.json").write_text("{}", encoding="utf-8")
    (run_dir / "checkpoints" / "best.pt.meta.json").write_text("{}", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "demo_run",
                "gfst_update_rule": "residual_rank_one_v1",
                "summary_layout_version": 1,
                "retrieval_key_version": "mean_top1_v1",
                "warm_start_mode": "weights_only",
                "source_run_id": "prior_run",
                "source_checkpoint_step": 42,
                "best_val_ce": 6.1,
                "best_teacher_gap": 0.8,
                "best_step": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text(
        "\n".join(
            [
                "Using the latest cached version of the dataset since Salesforce/wikitext couldn't be found on the Hugging Face Hub (offline mode is enabled).",
                "trainer started",
                "trainer healthy",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _build_canonical_run(run_dir: Path) -> None:
    _write_jsonl(
        run_dir / "events" / "train.jsonl",
        [
            {
                "schema_version": 2,
                "event_type": "train_step",
                "run_id": "canonical_run",
                "step": 3,
                "training_mode": "canonical",
                "stage": "A",
                "phase": 1,
                "mode": "canonical",
                "gpu_slot_count": 0,
                "gpu_slot_depth_peak": 0,
                "slot_depth_avg": 0.0,
                "next_token_loss": 6.8,
                "router_loss": 0.2,
                "pointer_loss": 0.4,
                "retrieval_loss": 0.0,
                "retrieval_contrastive_loss": 0.9,
                "copy_active_fraction": 0.55,
                "total_loss": 8.3,
                "tokens_per_second": 222.0,
                "teacher_used": False,
                "cuda_allocated_mb": 4096.0,
                "cuda_reserved_mb": 6144.0,
                "cuda_peak_allocated_mb": 6400.0,
                "gpu_util_avg_pct": 66.0,
                "gpu_util_peak_pct": 91.0,
                "gpu_power_avg_w": 180.0,
                "gpu_power_peak_w": 220.0,
                "gpu_vram_used_mb_avg": 6000.0,
                "active_sequence_length": 4096,
                "active_micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
            }
        ],
    )
    _write_jsonl(
        run_dir / "events" / "eval.jsonl",
        [
            {
                "schema_version": 2,
                "event_type": "eval",
                "run_id": "canonical_run",
                "step": 3,
                "training_mode": "canonical",
                "eval_kind": "full",
                "val_ce": 5.2,
                "retrieval_contrastive_loss": 0.7,
                "top1_block_recall_by_length": {"4096": 0.5},
                "top_r_block_recall_by_length": {"4096": 0.8},
                "decode_round_trip_ok": True,
                "student_ce_by_length": {"4096": 5.2},
            }
        ],
    )
    _write_jsonl(
        run_dir / "events" / "system.jsonl",
        [
            {
                "schema_version": 2,
                "event_type": "run_config",
                "run_id": "canonical_run",
                "config": {
                    "training_mode": "canonical",
                    "steps": 20,
                    "optimizer": {"optimizer_name": "adamw_only"},
                    "model": {
                        "train_scope": "all",
                        "gfst_update_rule": "residual_rank_one_v1",
                        "summary_layout_version": 1,
                        "retrieval_key_version": "mean_top1_v1",
                    },
                },
            },
            {
                "schema_version": 2,
                "event_type": "step_timing",
                "run_id": "canonical_run",
                "step": 3,
                "training_mode": "canonical",
                "step_wall_ms": 2000.0,
                "student_forward_ms": 700.0,
                "backward_ms": 1000.0,
                "optimizer_ms": 100.0,
                "gpu_util_avg_pct": 66.0,
                "gpu_util_peak_pct": 91.0,
                "gpu_power_avg_w": 180.0,
                "gpu_power_peak_w": 220.0,
                "gpu_vram_used_mb_avg": 6000.0,
                "cuda_allocated_mb": 4096.0,
                "cuda_reserved_mb": 6144.0,
                "cuda_peak_allocated_mb": 6400.0,
                "gpu_slot_pool_mb": 0.0,
                "host_pinned_mb": 0.0,
            },
        ],
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "canonical_run",
                "training_mode": "canonical",
                "config": {
                    "steps": 20,
                    "training_mode": "canonical",
                    "model": {
                        "gfst_update_rule": "residual_rank_one_v1",
                        "summary_layout_version": 1,
                        "retrieval_key_version": "mean_top1_v1",
                    },
                },
                "metadata": {
                    "gfst_update_rule": "residual_rank_one_v1",
                    "summary_layout_version": 1,
                    "retrieval_key_version": "mean_top1_v1",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "latest.pt").write_bytes(b"latest")
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"best")
    (run_dir / "checkpoints" / "latest.pt.meta.json").write_text("{}", encoding="utf-8")
    (run_dir / "checkpoints" / "best.pt.meta.json").write_text("{}", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "canonical_run",
                "training_mode": "canonical",
                "gfst_update_rule": "residual_rank_one_v1",
                "summary_layout_version": 1,
                "retrieval_key_version": "mean_top1_v1",
                "best_val_ce": 5.1,
                "best_step": 3,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text("canonical trainer healthy\n", encoding="utf-8")


def test_run_data_store_incremental_tail_does_not_duplicate_rows(tmp_path) -> None:
    module = _load_dashboard_module()
    run_dir = tmp_path / "demo_run"
    _write_jsonl(run_dir / "events" / "train.jsonl", [{"step": 0, "event_type": "train_step", "run_id": "demo"}])
    _write_jsonl(run_dir / "events" / "eval.jsonl", [])
    _write_jsonl(run_dir / "events" / "system.jsonl", [])

    store = module.RunDataStore(run_dir, max_rows=8)
    first = store.snapshot()
    assert [row["step"] for row in first.train_rows] == [0]

    _write_jsonl(
        run_dir / "events" / "train.jsonl",
        [
            {"step": 0, "event_type": "train_step", "run_id": "demo"},
            {"step": 1, "event_type": "train_step", "run_id": "demo"},
        ],
    )
    second = store.snapshot()
    assert [row["step"] for row in second.train_rows] == [0, 1]


def test_dashboard_view_model_normalizes_runtime_fields(tmp_path) -> None:
    module = _load_dashboard_module()
    run_dir = tmp_path / "demo_run"
    _build_run(run_dir)

    vm = module.DashboardViewModel.from_snapshot(module.RunDataStore(run_dir).snapshot())

    assert vm.run_name == "demo_run"
    assert vm.status in {"IDLE", "RUNNING", "STALE", "DONE"}
    assert vm.overlap_mode == "gpu_pipeline"
    assert vm.queue_size == "3"
    assert vm.host_refill == "6"
    assert vm.best_ce == "6.10"
    assert vm.best_gap == "0.80"
    assert any(line[0] == "memory" for line in vm.runtime_lines)
    assert any(line[0] == "gfst.update" and "residual_rank_one_v1" in line[1] for line in vm.state_lines)
    assert any(line[0] == "recovery" and "prior_run@42" in line[1] for line in vm.state_lines)
    assert any(line[0] == "gpu.step" for line in vm.state_lines)
    assert any(line[0] == "artifacts" and "meta yes" in line[1] for line in vm.state_lines)
    assert any(line.prefix == "boot" for line in vm.event_lines)
    assert "trainer healthy" in vm.log_lines[-1]


def test_dashboard_view_model_switches_to_canonical_runtime(tmp_path) -> None:
    module = _load_dashboard_module()
    run_dir = tmp_path / "canonical_run"
    _build_canonical_run(run_dir)

    vm = module.DashboardViewModel.from_snapshot(module.RunDataStore(run_dir).snapshot())

    assert vm.mode == "canonical"
    assert vm.overlap_mode == "-"
    assert vm.teacher_state == "-"
    assert vm.best_gap == "-"
    assert any(line[0] == "pipeline" and "teacher=off" in line[1] for line in vm.runtime_lines)
    assert any(line[0] == "recall" and "top1" in line[1] and "topR" in line[1] for line in vm.runtime_lines)
    assert any(line[0] == "best" and "gap" not in line[1] for line in vm.state_lines)


def test_dashboard_view_model_uses_replay_step_labels(tmp_path) -> None:
    module = _load_dashboard_module()
    run_dir = tmp_path / "replay_run"
    _write_jsonl(
        run_dir / "events" / "train.jsonl",
        [
            {
                "schema_version": 2,
                "event_type": "centroid_replay_step",
                "run_id": "replay_run",
                "replay_step": 24,
                "total_loss": 12.5,
                "tokens_per_second": 42.0,
                "stage": "R",
                "mode": "replay",
            }
        ],
    )
    _write_jsonl(
        run_dir / "events" / "system.jsonl",
        [
            {
                "schema_version": 2,
                "event_type": "run_config",
                "run_id": "replay_run",
                "config": {"steps": 250},
            },
            {
                "schema_version": 2,
                "event_type": "step_timing",
                "run_id": "replay_run",
                "step": 24,
                "step_wall_ms": 1500.0,
            },
        ],
    )
    _write_jsonl(run_dir / "events" / "eval.jsonl", [])
    (run_dir / "run_manifest.json").write_text(json.dumps({"schema_version": 2, "run_id": "replay_run", "config": {"steps": 250}}), encoding="utf-8")
    (run_dir / "run.log").write_text("replay trainer healthy\n", encoding="utf-8")

    vm = module.DashboardViewModel.from_snapshot(module.RunDataStore(run_dir).snapshot())

    assert vm.step == 24
    assert vm.progress_completed == 25
    assert vm.recent_steps[-1].step == "R24"


def test_dashboard_view_model_handles_missing_step_timing_rows(tmp_path) -> None:
    module = _load_dashboard_module()
    run_dir = tmp_path / "demo_run"
    _build_run(run_dir)

    system_rows = [
        row
        for row in (
            json.loads(line)
            for line in (run_dir / "events" / "system.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("event_type") != "step_timing"
    ]
    _write_jsonl(run_dir / "events" / "system.jsonl", system_rows)

    vm = module.DashboardViewModel.from_snapshot(module.RunDataStore(run_dir).snapshot())

    assert vm.latest_timing["step_wall_ms"] == pytest.approx(1100.0)
    assert vm.queue_size == "3"
    assert vm.host_refill == "6"
    assert vm.timing_rows[0][0] == "student.fwd"
    assert vm.recent_steps[-1].wall == "1.10s"


def test_training_dashboard_snapshot_render(tmp_path) -> None:
    pytest.importorskip("rich")
    run_dir = tmp_path / "demo_run"
    _build_run(run_dir)

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "training_dashboard.py"
    completed = subprocess.run(
        [sys.executable, str(script_path), str(run_dir), "--snapshot", "--color", "never"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Run Status" in completed.stdout
    assert "KPI Strip" in completed.stdout
    assert "System / Runtime" in completed.stdout
    assert "Active / Recent Steps" in completed.stdout
    assert "Timing / Trends" in completed.stdout
    assert "Alerts / Events / Log" in completed.stdout
    assert "Run State / Artifacts" in completed.stdout
    assert "Control / Watch" in completed.stdout
    assert "demo_run" in completed.stdout
    assert "Using the latest cached version of the dataset since" not in completed.stdout


def test_training_dashboard_textual_app_mounts(tmp_path) -> None:
    pytest.importorskip("rich")
    pytest.importorskip("textual")
    module = _load_dashboard_module()
    run_dir = tmp_path / "demo_run"
    _build_run(run_dir)

    async def _run() -> None:
        app = module.TrainingDashboardApp(module.RunDataStore(run_dir), refresh=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#status") is not None
            assert app.query_one("#runtime") is not None
            assert app.query_one("#events") is not None

    asyncio.run(_run())
