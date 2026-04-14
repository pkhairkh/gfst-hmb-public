from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from ..training.artifacts import validate_checkpoint
from ..training.config import load_training_run_config
from ..training.data import SyntheticNeedleBatchProvider, training_batch_to_device
from ..training.engine import _centroid_auxiliary_losses, _centroid_replay_focus_mask
from .training_recipe_eval import _load_checkpoint_models, resolve_checkpoint_path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _resolve_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _scatter_write_assignments(
    *,
    assignment_ids: torch.Tensor,
    assignment_weights: torch.Tensor,
    cells_per_block: int,
) -> torch.Tensor:
    full = torch.zeros(
        (*assignment_ids.shape[:-1], int(cells_per_block)),
        device=assignment_weights.device,
        dtype=torch.float32,
    )
    valid = assignment_ids.ge(0)
    if bool(valid.any().item()):
        safe_ids = assignment_ids.clamp(0, max(0, int(cells_per_block) - 1))
        full.scatter_add_(
            -1,
            safe_ids,
            torch.where(valid, assignment_weights, torch.zeros_like(assignment_weights)),
        )
    return full


def _mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / float(len(values))) if values else 0.0


def _sorted_rank_means(values: list[torch.Tensor]) -> list[float]:
    if not values:
        return []
    stacked = torch.stack(values, dim=0)
    return [float(x) for x in stacked.mean(dim=0).detach().cpu().tolist()]


def _write_side_block_stats(
    *,
    block_ids: torch.Tensor,
    full_assignments: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    zero = {
        "block_count": 0,
        "hard_active_cells_per_block": 0.0,
        "soft_active_cells_per_block": 0.0,
        "paper_cardinality_loss": 0.0,
        "token_support_per_token": 0.0,
        "mass_entropy": 0.0,
        "top1_mass": 0.0,
        "top2_mass": 0.0,
        "top4_mass": 0.0,
        "tail_mass_after_top4": 0.0,
        "mass_rank_profile": [],
    }
    if not bool(valid_mask.any().item()):
        return zero

    cells_per_block = int(full_assignments.shape[-1])
    support_temperature = max(1e-3, float(threshold) * 0.2)
    block_counts: list[float] = []
    soft_counts: list[float] = []
    paper_card_losses: list[float] = []
    token_support_counts: list[float] = []
    mass_entropy: list[float] = []
    top1_mass: list[float] = []
    top2_mass: list[float] = []
    top4_mass: list[float] = []
    tail_mass: list[float] = []
    rank_profiles: list[torch.Tensor] = []

    support_per_token = full_assignments.gt(1e-8).sum(dim=-1).to(dtype=torch.float32)
    token_support_counts.extend(float(x) for x in support_per_token[valid_mask].detach().cpu().tolist())

    for batch_idx in range(int(block_ids.shape[0])):
        sample_valid = valid_mask[batch_idx]
        if not bool(sample_valid.any().item()):
            continue
        sample_block_ids = block_ids[batch_idx][sample_valid]
        sample_assign = full_assignments[batch_idx][sample_valid]
        unique_blocks = torch.unique(sample_block_ids[sample_block_ids.ge(0)])
        for block_id in unique_blocks.tolist():
            block_mask = sample_block_ids.eq(int(block_id))
            if not bool(block_mask.any().item()):
                continue
            cell_mass = sample_assign[block_mask].sum(dim=0)
            total_mass = cell_mass.sum()
            if float(total_mass.item()) <= 0.0:
                continue
            normalized_mass = cell_mass / total_mass.clamp_min(1e-6)
            sorted_mass = torch.sort(normalized_mass, descending=True).values
            hard_count = normalized_mass.ge(float(threshold)).sum().item()
            soft_count = torch.sigmoid((normalized_mass - float(threshold)) / support_temperature).sum().item()
            p = normalized_mass.clamp_min(1e-8)
            ent = float((-(p * p.log()).sum() / math.log(float(max(2, cells_per_block)))).item())
            block_counts.append(float(hard_count))
            soft_counts.append(float(soft_count))
            paper_card_losses.append(float(max(0.0, float(hard_count) - 3.5) ** 2))
            mass_entropy.append(ent)
            top1_mass.append(float(sorted_mass[0].item()))
            top2_mass.append(float(sorted_mass[:2].sum().item()))
            top4_mass.append(float(sorted_mass[:4].sum().item()))
            tail_mass.append(float(sorted_mass[4:].sum().item()))
            rank_profiles.append(sorted_mass)

    if not block_counts:
        return zero
    return {
        "block_count": int(len(block_counts)),
        "hard_active_cells_per_block": _mean_or_zero(block_counts),
        "soft_active_cells_per_block": _mean_or_zero(soft_counts),
        "paper_cardinality_loss": _mean_or_zero(paper_card_losses),
        "token_support_per_token": _mean_or_zero(token_support_counts),
        "mass_entropy": _mean_or_zero(mass_entropy),
        "top1_mass": _mean_or_zero(top1_mass),
        "top2_mass": _mean_or_zero(top2_mass),
        "top4_mass": _mean_or_zero(top4_mass),
        "tail_mass_after_top4": _mean_or_zero(tail_mass),
        "mass_rank_profile": _sorted_rank_means(rank_profiles),
    }


def _read_side_stats(
    *,
    selected_block_slots: torch.Tensor | None,
    selected_cell_ids: torch.Tensor | None,
    selected_alphas: torch.Tensor | None,
    selected_cells_per_head: torch.Tensor | None,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    zero = {
        "positions": 0,
        "selected_cells_per_head_mean": 0.0,
        "selected_entries_per_position": 0.0,
        "unique_block_cells_per_position": 0.0,
        "unique_blocks_per_position": 0.0,
        "block_mass_entropy": 0.0,
    }
    if (
        selected_block_slots is None
        or selected_cell_ids is None
        or selected_alphas is None
        or selected_cells_per_head is None
        or not bool(valid_mask.any().item())
    ):
        return zero

    head_counts = selected_cells_per_head[valid_mask].to(dtype=torch.float32)
    selected_entries_per_position: list[float] = []
    unique_block_cells_per_position: list[float] = []
    unique_blocks_per_position: list[float] = []
    block_mass_entropy: list[float] = []
    positions = 0

    for batch_idx in range(int(valid_mask.shape[0])):
        position_indices = torch.nonzero(valid_mask[batch_idx], as_tuple=False).reshape(-1)
        for pos_idx in position_indices.tolist():
            pos_slots = selected_block_slots[batch_idx, pos_idx]
            pos_cells = selected_cell_ids[batch_idx, pos_idx]
            pos_alphas = selected_alphas[batch_idx, pos_idx].to(dtype=torch.float32)
            valid = pos_slots.ge(0) & pos_cells.ge(0) & pos_alphas.gt(0)
            if not bool(valid.any().item()):
                continue
            positions += 1
            pair_tensor = torch.stack((pos_slots[valid], pos_cells[valid]), dim=-1)
            unique_pairs, inverse = torch.unique(pair_tensor, dim=0, return_inverse=True)
            pair_mass = torch.zeros((int(unique_pairs.shape[0]),), device=pos_alphas.device, dtype=torch.float32)
            pair_mass.index_add_(0, inverse, pos_alphas[valid])
            unique_block_cells_per_position.append(float(unique_pairs.shape[0]))
            unique_blocks, block_inverse = torch.unique(unique_pairs[:, 0], return_inverse=True)
            unique_blocks_per_position.append(float(unique_blocks.shape[0]))
            selected_entries_per_position.append(float(valid.sum().item()))
            block_mass = torch.zeros((int(unique_blocks.shape[0]),), device=pos_alphas.device, dtype=torch.float32)
            block_mass.index_add_(0, block_inverse, pair_mass)
            if int(unique_blocks.shape[0]) > 1:
                p = block_mass / block_mass.sum().clamp_min(1e-6)
                block_mass_entropy.append(
                    float((-(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum() / math.log(float(unique_blocks.shape[0]))).item())
                )
            else:
                block_mass_entropy.append(0.0)

    if positions == 0:
        return zero
    return {
        "positions": int(positions),
        "selected_cells_per_head_mean": float(head_counts.mean().detach().cpu().item()),
        "selected_entries_per_position": _mean_or_zero(selected_entries_per_position),
        "unique_block_cells_per_position": _mean_or_zero(unique_block_cells_per_position),
        "unique_blocks_per_position": _mean_or_zero(unique_blocks_per_position),
        "block_mass_entropy": _mean_or_zero(block_mass_entropy),
    }


def _summarize_masked_quantities(
    *,
    student_out: Any,
    labels: torch.Tensor,
    device: torch.device,
    assignment_temperature: float,
    train_support_k: int,
    block_support_threshold: float,
    target_active_cells_per_block: float,
    focus_mask: torch.Tensor | None,
    student: Any | None,
) -> dict[str, Any]:
    block_ids = getattr(student_out, "centroid_block_ids", None)
    assignment_ids = getattr(student_out, "centroid_assignment_ids", None)
    assignment_weights = getattr(student_out, "centroid_assignment_weights", None)
    valid_mask = getattr(student_out, "centroid_valid_mask", None)
    selected_block_slots = getattr(student_out, "centroid_selected_block_slots", None)
    selected_cell_ids = getattr(student_out, "centroid_selected_cell_ids", None)
    selected_alphas = getattr(student_out, "centroid_selected_alphas", None)
    selected_cells_per_head = getattr(student_out, "centroid_selected_cells_per_head", None)
    cells_per_block = int(getattr(student.gfst_hmb_config, "cells_per_block", 0)) if student is not None else 0

    if (
        block_ids is None
        or assignment_ids is None
        or assignment_weights is None
        or valid_mask is None
        or cells_per_block <= 0
    ):
        return {
            "valid_fraction": 0.0,
            "focus_fraction": 0.0,
            "write": {},
            "proxy": {},
            "read": {},
        }

    block_ids = block_ids.to(device=device, dtype=torch.long)
    assignment_ids = assignment_ids.to(device=device, dtype=torch.long)
    assignment_weights = assignment_weights.to(device=device, dtype=torch.float32)
    valid_mask = valid_mask.to(device=device, dtype=torch.bool)
    if focus_mask is not None:
        masked_valid = valid_mask & focus_mask.to(device=device, dtype=torch.bool)
        focus_fraction = (
            float((masked_valid.float().sum() / valid_mask.float().sum().clamp_min(1.0)).detach().cpu().item())
            if bool(valid_mask.any().item())
            else 0.0
        )
    else:
        masked_valid = valid_mask
        focus_fraction = 1.0 if bool(valid_mask.any().item()) else 0.0

    full_assignments = _scatter_write_assignments(
        assignment_ids=assignment_ids,
        assignment_weights=assignment_weights,
        cells_per_block=cells_per_block,
    )
    write_stats = _write_side_block_stats(
        block_ids=block_ids,
        full_assignments=full_assignments,
        valid_mask=masked_valid,
        threshold=float(block_support_threshold),
    )
    write_stats_at_010 = _write_side_block_stats(
        block_ids=block_ids,
        full_assignments=full_assignments,
        valid_mask=masked_valid,
        threshold=0.10,
    )
    write_stats_at_005 = _write_side_block_stats(
        block_ids=block_ids,
        full_assignments=full_assignments,
        valid_mask=masked_valid,
        threshold=0.05,
    )
    proxy_stats = _centroid_auxiliary_losses(
        student_out=student_out,
        device=device,
        cells_per_block=cells_per_block,
        assignment_temperature=float(assignment_temperature),
        support_k=int(train_support_k),
        block_support_threshold=float(block_support_threshold),
        target_active_cells_per_block=float(target_active_cells_per_block),
        focus_mask=focus_mask,
        student=None,
    )
    read_stats = _read_side_stats(
        selected_block_slots=None if selected_block_slots is None else selected_block_slots.to(device=device, dtype=torch.long),
        selected_cell_ids=None if selected_cell_ids is None else selected_cell_ids.to(device=device, dtype=torch.long),
        selected_alphas=None if selected_alphas is None else selected_alphas.to(device=device, dtype=torch.float32),
        selected_cells_per_head=None if selected_cells_per_head is None else selected_cells_per_head.to(device=device, dtype=torch.float32),
        valid_mask=masked_valid,
    )
    return {
        "valid_fraction": float(masked_valid.float().mean().detach().cpu().item()) if int(masked_valid.numel()) > 0 else 0.0,
        "focus_fraction": float(focus_fraction),
        "write": write_stats,
        "write_at_0_10": write_stats_at_010,
        "write_at_0_05": write_stats_at_005,
        "proxy": {
            "active_cells_per_block": float(proxy_stats["active_cells_per_block"].detach().cpu().item()),
            "assignment_entropy": float(proxy_stats["assignment_entropy"].detach().cpu().item()),
            "cardinality_loss": float(proxy_stats["cardinality_loss"].detach().cpu().item()),
            "occupancy_balance_loss": float(proxy_stats["occupancy_balance_loss"].detach().cpu().item()),
            "coherence_loss": float(proxy_stats["coherence_loss"].detach().cpu().item()),
            "compactness_loss": float(proxy_stats["compactness_loss"].detach().cpu().item()),
            "selection_entropy": float(proxy_stats["selection_entropy"].detach().cpu().item()),
            "selection_margin": float(proxy_stats["selection_margin"].detach().cpu().item()),
            "confidence": float(proxy_stats["confidence"].detach().cpu().item()),
            "selected_cells_per_head_mean": float(proxy_stats["selected_cells_per_head_mean"].detach().cpu().item()),
        },
        "read": read_stats,
    }


def _aggregate_run_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {}

    def mean_path(*path: str) -> float:
        values: list[float] = []
        for summary in summaries:
            node: Any = summary
            for key in path:
                node = node.get(key, {})
            if isinstance(node, (float, int)):
                values.append(float(node))
        return _mean_or_zero(values)

    def mean_rank_profile(path0: str, path1: str, path2: str) -> list[float]:
        profiles: list[torch.Tensor] = []
        for summary in summaries:
            node = summary.get(path0, {}).get(path1, {}).get(path2, [])
            if node:
                profiles.append(torch.tensor(node, dtype=torch.float32))
        return _sorted_rank_means(profiles)

    def summarize_write(scope: str, key: str) -> dict[str, Any]:
        return {
            "hard_active_cells_per_block": mean_path(scope, key, "hard_active_cells_per_block"),
            "soft_active_cells_per_block": mean_path(scope, key, "soft_active_cells_per_block"),
            "paper_cardinality_loss": mean_path(scope, key, "paper_cardinality_loss"),
            "token_support_per_token": mean_path(scope, key, "token_support_per_token"),
            "mass_entropy": mean_path(scope, key, "mass_entropy"),
            "top1_mass": mean_path(scope, key, "top1_mass"),
            "top2_mass": mean_path(scope, key, "top2_mass"),
            "top4_mass": mean_path(scope, key, "top4_mass"),
            "tail_mass_after_top4": mean_path(scope, key, "tail_mass_after_top4"),
            "mass_rank_profile": mean_rank_profile(scope, key, "mass_rank_profile"),
        }

    return {
        "probe_count": int(len(summaries)),
        "all_valid": {
            "valid_fraction": mean_path("all_valid", "valid_fraction"),
            "write": summarize_write("all_valid", "write"),
            "write_at_0_10": summarize_write("all_valid", "write_at_0_10"),
            "write_at_0_05": summarize_write("all_valid", "write_at_0_05"),
            "proxy": {
                "active_cells_per_block": mean_path("all_valid", "proxy", "active_cells_per_block"),
                "assignment_entropy": mean_path("all_valid", "proxy", "assignment_entropy"),
                "cardinality_loss": mean_path("all_valid", "proxy", "cardinality_loss"),
                "occupancy_balance_loss": mean_path("all_valid", "proxy", "occupancy_balance_loss"),
                "coherence_loss": mean_path("all_valid", "proxy", "coherence_loss"),
                "compactness_loss": mean_path("all_valid", "proxy", "compactness_loss"),
            },
            "read": {
                "selected_cells_per_head_mean": mean_path("all_valid", "read", "selected_cells_per_head_mean"),
                "selected_entries_per_position": mean_path("all_valid", "read", "selected_entries_per_position"),
                "unique_block_cells_per_position": mean_path("all_valid", "read", "unique_block_cells_per_position"),
                "unique_blocks_per_position": mean_path("all_valid", "read", "unique_blocks_per_position"),
                "block_mass_entropy": mean_path("all_valid", "read", "block_mass_entropy"),
            },
        },
        "focus_only": {
            "valid_fraction": mean_path("focus_only", "valid_fraction"),
            "focus_fraction": mean_path("focus_only", "focus_fraction"),
            "write": summarize_write("focus_only", "write"),
            "write_at_0_10": summarize_write("focus_only", "write_at_0_10"),
            "write_at_0_05": summarize_write("focus_only", "write_at_0_05"),
            "proxy": {
                "active_cells_per_block": mean_path("focus_only", "proxy", "active_cells_per_block"),
                "assignment_entropy": mean_path("focus_only", "proxy", "assignment_entropy"),
                "cardinality_loss": mean_path("focus_only", "proxy", "cardinality_loss"),
                "occupancy_balance_loss": mean_path("focus_only", "proxy", "occupancy_balance_loss"),
                "coherence_loss": mean_path("focus_only", "proxy", "coherence_loss"),
                "compactness_loss": mean_path("focus_only", "proxy", "compactness_loss"),
            },
            "read": {
                "selected_cells_per_head_mean": mean_path("focus_only", "read", "selected_cells_per_head_mean"),
                "selected_entries_per_position": mean_path("focus_only", "read", "selected_entries_per_position"),
                "unique_block_cells_per_position": mean_path("focus_only", "read", "unique_block_cells_per_position"),
                "unique_blocks_per_position": mean_path("focus_only", "read", "unique_blocks_per_position"),
                "block_mass_entropy": mean_path("focus_only", "read", "block_mass_entropy"),
            },
        },
    }


def _run_audit(
    *,
    run_dir: Path,
    checkpoint_name: str,
    device: torch.device,
    dtype: torch.dtype,
    probe_steps: list[int],
) -> dict[str, Any]:
    config_path = run_dir / "train_config.yaml"
    run_config = load_training_run_config(config_path)
    checkpoint_path = resolve_checkpoint_path(run_dir, checkpoint_name).resolve()
    checkpoint = validate_checkpoint(checkpoint_path, require_extended_metadata=False)
    active_micro_batch_size = max(1, int(checkpoint.payload.get("active_micro_batch_size", 1)))
    gradient_accumulation_steps = max(1, int(checkpoint.payload.get("active_gradient_accumulation_steps", 1)))
    active_sequence_length = max(1, int(checkpoint.payload.get("active_sequence_length", run_config.min_sequence_length)))
    tokenizer, student, _teacher = _load_checkpoint_models(
        run_config=run_config,
        checkpoint=checkpoint,
        device=device,
        dtype=dtype,
        load_teacher=False,
    )
    provider = SyntheticNeedleBatchProvider(
        tokenizer=tokenizer,
        seed=int(run_config.seed),
        sequence_length=int(active_sequence_length),
        micro_batch_size=int(active_micro_batch_size),
        block_len=int(run_config.model.block_len),
        num_batches=int(run_config.data.synthetic_train_batches),
    )
    provider.set_gradient_accumulation_steps(gradient_accumulation_steps)

    step_summaries: list[dict[str, Any]] = []
    with torch.inference_mode():
        for step_index in probe_steps:
            batch = provider.batch_for(
                step_index=int(step_index),
                microstep=0,
                sequence_length=int(active_sequence_length),
                micro_batch_size=int(active_micro_batch_size),
            )
            device_batch = training_batch_to_device(batch, device=device)
            student_out = student(
                input_ids=device_batch.input_ids,
                use_cache=False,
                output_hidden_states=False,
            )
            focus_mask = _centroid_replay_focus_mask(
                student_out=student_out,
                labels=device_batch.labels,
                pointer_margin_threshold=float(run_config.centroid_replay.centroid_replay_pointer_margin_threshold),
                entropy_threshold=float(run_config.centroid_replay.centroid_replay_entropy_threshold),
                confidence_threshold=float(run_config.centroid_replay.centroid_replay_confidence_threshold),
            )
            all_valid = _summarize_masked_quantities(
                student_out=student_out,
                labels=device_batch.labels,
                device=device,
                assignment_temperature=float(run_config.model.centroid_assignment_temperature),
                train_support_k=int(run_config.model.centroid_train_support_k),
                block_support_threshold=float(run_config.model.centroid_block_support_threshold),
                target_active_cells_per_block=float(run_config.model.centroid_target_active_cells_per_block),
                focus_mask=None,
                student=student,
            )
            focus_only = _summarize_masked_quantities(
                student_out=student_out,
                labels=device_batch.labels,
                device=device,
                assignment_temperature=float(run_config.model.centroid_assignment_temperature),
                train_support_k=int(run_config.model.centroid_train_support_k),
                block_support_threshold=float(run_config.model.centroid_block_support_threshold),
                target_active_cells_per_block=float(run_config.model.centroid_target_active_cells_per_block),
                focus_mask=focus_mask,
                student=student,
            )
            step_summaries.append(
                {
                    "probe_step": int(step_index),
                    "all_valid": all_valid,
                    "focus_only": focus_only,
                }
            )

    return {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_step": int(checkpoint.payload.get("step", -1)),
        "probe_steps": [int(x) for x in probe_steps],
        "config": {
            "sequence_length": int(active_sequence_length),
            "micro_batch_size": int(active_micro_batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "block_len": int(run_config.model.block_len),
            "cells_per_block": int(run_config.model.cells_per_block),
            "centroid_assignment_top_k": int(run_config.model.centroid_assignment_top_k),
            "centroid_train_support_k": int(run_config.model.centroid_train_support_k),
            "centroid_min_selected_cells_per_head": int(run_config.model.centroid_min_selected_cells_per_head),
            "centroid_block_support_threshold": float(run_config.model.centroid_block_support_threshold),
            "centroid_target_active_cells_per_block": float(run_config.model.centroid_target_active_cells_per_block),
        },
        "summary": _aggregate_run_summaries(step_summaries),
        "per_probe": step_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit centroid quantities from frozen checkpoints.")
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory to audit. Repeat for multiple runs.")
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint key or path within each run directory.")
    parser.add_argument("--probe-steps", default="0,1,2,3", help="Comma-separated synthetic probe step indices.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    probe_steps = [int(part.strip()) for part in str(args.probe_steps).split(",") if part.strip()]
    device = torch.device(str(args.device))
    dtype = _resolve_dtype(str(args.dtype))

    payload = {
        "probe_steps": probe_steps,
        "runs": [
            _run_audit(
                run_dir=Path(run_dir).resolve(),
                checkpoint_name=str(args.checkpoint),
                device=device,
                dtype=dtype,
                probe_steps=probe_steps,
            )
            for run_dir in args.run_dir
        ],
    }
    rendered = json.dumps(payload, indent=2, default=_json_default)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
