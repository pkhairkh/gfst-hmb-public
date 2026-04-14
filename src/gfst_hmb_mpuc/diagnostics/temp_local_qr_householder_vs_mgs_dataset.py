from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import torch

from ..gfst import basis_projector, canonicalize_qr_column_signs


TEMP_MARKER = "TEMP_REMOVE_AFTER_SM120_EXACT_LOCAL_QR_PARITY"


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def _scalar_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().to(dtype=torch.float32).item())
    return float(value)


def _grad_summary(grad: torch.Tensor | None) -> dict[str, float | bool | None]:
    if grad is None:
        return {
            "present": False,
            "max_abs": None,
            "l2_norm": None,
            "mean_abs": None,
        }
    grad_float = grad.detach().float()
    return {
        "present": True,
        "max_abs": float(grad_float.abs().max().item()) if int(grad_float.numel()) > 0 else 0.0,
        "l2_norm": float(torch.linalg.vector_norm(grad_float.reshape(-1)).item()) if int(grad_float.numel()) > 0 else 0.0,
        "mean_abs": float(grad_float.abs().mean().item()) if int(grad_float.numel()) > 0 else 0.0,
    }


def _prefix_projector_losses(
    reference_q: torch.Tensor,
    candidate_q: torch.Tensor,
    *,
    prefix_dims: tuple[int, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, float], float]:
    losses: dict[str, torch.Tensor] = {}
    values: dict[str, float] = {}
    total = torch.zeros((), device=candidate_q.device, dtype=candidate_q.dtype)
    max_cols = min(int(reference_q.shape[1]), int(candidate_q.shape[1]))
    for dim in prefix_dims:
        prefix_dim = int(dim)
        if prefix_dim <= 0 or prefix_dim > max_cols:
            continue
        ref_projector = basis_projector(reference_q[:, :prefix_dim])
        cand_projector = basis_projector(candidate_q[:, :prefix_dim])
        loss = (cand_projector - ref_projector).square().sum()
        losses[str(prefix_dim)] = loss
        values[str(prefix_dim)] = float(loss.detach().item())
        total = total + loss
    return losses, values, float(total.detach().item())


def modified_gram_schmidt_q(
    matrix: torch.Tensor,
    *,
    eps_sign: float,
    reorthogonalize: bool = False,
    eps_norm: float = 1e-12,
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"expected rank-2 matrix, got shape {tuple(matrix.shape)}")
    rows, cols = matrix.shape
    q = torch.zeros((rows, cols), device=matrix.device, dtype=matrix.dtype)
    eps_tensor = torch.tensor(float(eps_norm), device=matrix.device, dtype=matrix.dtype)
    for col_idx in range(cols):
        vector = matrix[:, col_idx]
        for prev_idx in range(col_idx):
            basis = q[:, prev_idx]
            vector = vector - torch.dot(basis, vector) * basis
        if reorthogonalize:
            for prev_idx in range(col_idx):
                basis = q[:, prev_idx]
                vector = vector - torch.dot(basis, vector) * basis
        norm = torch.linalg.vector_norm(vector)
        fallback = torch.zeros_like(vector)
        q[:, col_idx] = torch.where(
            norm > eps_tensor,
            vector / norm.clamp_min(float(eps_norm)),
            fallback,
        )
    return canonicalize_qr_column_signs(q, eps_sign=float(eps_sign))


def householder_reduced_q_with_trace(
    matrix: torch.Tensor,
    *,
    eps_sign: float,
    eps_norm: float = 1e-12,
) -> dict[str, Any]:
    if matrix.ndim != 2:
        raise ValueError(f"expected rank-2 matrix, got shape {tuple(matrix.shape)}")
    rows, cols = matrix.shape
    if cols > rows:
        raise ValueError(f"expected rows >= cols for reduced QR, got shape {tuple(matrix.shape)}")
    working = matrix.clone()
    q_total = torch.eye(rows, device=matrix.device, dtype=matrix.dtype)
    steps: list[dict[str, Any]] = []
    eps_tensor = torch.tensor(float(eps_norm), device=matrix.device, dtype=matrix.dtype)

    for step_idx in range(cols):
        x = working[step_idx:, step_idx]
        if x.requires_grad:
            x.retain_grad()
        x_norm = torch.linalg.vector_norm(x)
        if x_norm.requires_grad:
            x_norm.retain_grad()
        sign = torch.where(
            x[0].lt(0),
            -torch.ones((), device=matrix.device, dtype=matrix.dtype),
            torch.ones((), device=matrix.device, dtype=matrix.dtype),
        )
        alpha = sign * x_norm
        if alpha.requires_grad:
            alpha.retain_grad()
        e1 = torch.zeros_like(x)
        e1[0] = 1.0
        v_raw = x + alpha * e1
        if v_raw.requires_grad:
            v_raw.retain_grad()
        v_norm = torch.linalg.vector_norm(v_raw)
        if v_norm.requires_grad:
            v_norm.retain_grad()
        fallback = torch.zeros_like(v_raw)
        fallback[0] = 1.0
        v_unit = torch.where(
            v_norm > eps_tensor,
            v_raw / v_norm.clamp_min(float(eps_norm)),
            fallback,
        )
        if v_unit.requires_grad:
            v_unit.retain_grad()

        panel_before = working[step_idx:, step_idx:]
        if panel_before.requires_grad:
            panel_before.retain_grad()
        panel_after = panel_before - 2.0 * v_unit.unsqueeze(-1) * (v_unit @ panel_before).unsqueeze(0)
        if panel_after.requires_grad:
            panel_after.retain_grad()
        next_working = working.clone()
        next_working[step_idx:, step_idx:] = panel_after
        working = next_working

        q_block_before = q_total[:, step_idx:]
        if q_block_before.requires_grad:
            q_block_before.retain_grad()
        q_block_after = q_block_before - 2.0 * (q_block_before @ v_unit).unsqueeze(-1) * v_unit.unsqueeze(0)
        if q_block_after.requires_grad:
            q_block_after.retain_grad()
        next_q_total = q_total.clone()
        next_q_total[:, step_idx:] = q_block_after
        q_total = next_q_total

        steps.append(
            {
                "step_index": int(step_idx),
                "x": x,
                "x_norm": x_norm,
                "alpha": alpha,
                "v_raw": v_raw,
                "v_norm": v_norm,
                "v_unit": v_unit,
                "panel_before": panel_before,
                "panel_after": panel_after,
                "q_block_before": q_block_before,
                "q_block_after": q_block_after,
            }
        )

    raw_q = q_total[:, :cols]
    if raw_q.requires_grad:
        raw_q.retain_grad()
    signfixed_q = canonicalize_qr_column_signs(raw_q, eps_sign=float(eps_sign))
    if signfixed_q.requires_grad:
        signfixed_q.retain_grad()
    return {
        "raw_q": raw_q,
        "signfixed_q": signfixed_q,
        "r": working[:cols, :],
        "steps": steps,
    }


def compare_mgs_against_householder(
    matrix: torch.Tensor,
    *,
    prefix_dims: tuple[int, ...],
    eps_sign: float,
    reorthogonalize: bool = False,
    eps_norm: float = 1e-12,
) -> dict[str, Any]:
    if matrix.ndim != 2:
        raise ValueError(f"expected rank-2 matrix, got shape {tuple(matrix.shape)}")
    matrix_float = matrix.detach().to(device=matrix.device, dtype=torch.float32).contiguous()
    mgs_q = modified_gram_schmidt_q(
        matrix_float,
        eps_sign=float(eps_sign),
        reorthogonalize=bool(reorthogonalize),
        eps_norm=float(eps_norm),
    )
    hh_input = matrix_float.clone().requires_grad_(True)
    householder = householder_reduced_q_with_trace(
        hh_input,
        eps_sign=float(eps_sign),
        eps_norm=float(eps_norm),
    )
    raw_q = householder["raw_q"]
    signfixed_q = householder["signfixed_q"]

    raw_q_loss = (raw_q - mgs_q).square().sum()
    signfixed_q_loss = (signfixed_q - mgs_q).square().sum()
    prefix_loss_terms, prefix_loss_values, prefix_loss_total_value = _prefix_projector_losses(
        mgs_q,
        signfixed_q,
        prefix_dims=prefix_dims,
    )
    prefix_projector_total_loss = sum(prefix_loss_terms.values(), torch.zeros((), device=hh_input.device, dtype=hh_input.dtype))
    primary_loss = signfixed_q_loss + prefix_projector_total_loss

    operation_order = (
        "x",
        "x_norm",
        "alpha",
        "v_raw",
        "v_norm",
        "v_unit",
        "panel_before",
        "panel_after",
        "q_block_before",
        "q_block_after",
    )
    step_reports: list[dict[str, Any]] = []
    for step in householder["steps"]:
        operation_reports: dict[str, Any] = {}
        dominant_operation = None
        dominant_grad = -math.inf
        for op_name in operation_order:
            tensor = step[op_name]
            grad = None
            if torch.is_tensor(tensor) and bool(getattr(tensor, "requires_grad", False)):
                grad = torch.autograd.grad(primary_loss, tensor, retain_graph=True, allow_unused=True)[0]
            summary = _grad_summary(grad)
            operation_reports[op_name] = {
                "shape": list(tensor.shape) if torch.is_tensor(tensor) else [],
                "primary_loss_grad": summary,
            }
            max_abs = summary["max_abs"]
            if max_abs is not None and float(max_abs) > float(dominant_grad):
                dominant_grad = float(max_abs)
                dominant_operation = op_name
        step_reports.append(
            {
                "step_index": int(step["step_index"]),
                "dominant_operation": dominant_operation,
                "dominant_primary_loss_grad_max_abs": None if dominant_operation is None else float(dominant_grad),
                "operations": operation_reports,
            }
        )

    input_grad = torch.autograd.grad(primary_loss, hh_input, allow_unused=True)[0]
    raw_q_diff = (raw_q - mgs_q).abs()
    signfixed_q_diff = (signfixed_q - mgs_q).abs()

    max_prefix_delta = 0.0
    for dim in prefix_dims:
        prefix_dim = int(dim)
        if prefix_dim <= 0 or prefix_dim > int(signfixed_q.shape[1]):
            continue
        ref_projector = basis_projector(mgs_q[:, :prefix_dim])
        cand_projector = basis_projector(signfixed_q[:, :prefix_dim])
        max_prefix_delta = max(max_prefix_delta, float((cand_projector - ref_projector).abs().max().item()))

    return {
        "mgs_q": _cpu_clone(mgs_q),
        "householder_raw_q": _cpu_clone(raw_q),
        "householder_signfixed_q": _cpu_clone(signfixed_q),
        "householder_r": _cpu_clone(householder["r"]),
        "metrics": {
            "raw_q_fro_loss": float(raw_q_loss.detach().item()),
            "signfixed_q_fro_loss": float(signfixed_q_loss.detach().item()),
            "prefix_projector_total_loss": float(prefix_projector_total_loss.detach().item()),
            "prefix_projector_losses": {key: float(value) for key, value in prefix_loss_values.items()},
            "raw_q_max_abs": float(raw_q_diff.max().item()) if int(raw_q_diff.numel()) > 0 else 0.0,
            "signfixed_q_max_abs": float(signfixed_q_diff.max().item()) if int(signfixed_q_diff.numel()) > 0 else 0.0,
            "prefix_projector_max_abs": float(max_prefix_delta),
            "primary_loss": float(primary_loss.detach().item()),
        },
        "input_grad_summary": _grad_summary(input_grad),
        "householder_step_reports": step_reports,
    }


def load_payload_pool_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path).resolve()
    bundle = torch.load(bundle_path, map_location="cpu")
    marker = str(bundle.get("temporary_marker", ""))
    if marker != TEMP_MARKER:
        raise RuntimeError(f"payload pool bundle {bundle_path} is missing temporary marker {TEMP_MARKER!r}")
    return bundle


def _case_from_matrix(
    matrix: torch.Tensor,
    *,
    case_id: str,
    source: str,
    prefix_dims: tuple[int, ...],
    eps_sign: float,
    stage_name: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": str(case_id),
        "source": str(source),
        "stage_name": str(stage_name),
        "prefix_dims": tuple(int(dim) for dim in prefix_dims),
        "eps_sign": float(eps_sign),
        "matrix": _cpu_clone(matrix),
        "metadata": {} if metadata is None else copy.deepcopy(metadata),
    }


def fixture_and_payload_cases(
    *,
    seam_fixture_bundle: dict[str, Any],
    selected_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fixture_payload = copy.deepcopy(seam_fixture_bundle["native_capture"])
    flag_dims = tuple(int(dim) for dim in fixture_payload["metadata"]["flag_dims"])
    cases = [
        _case_from_matrix(
            fixture_payload["largest_candidate"],
            case_id="fixture_largest",
            source="fixture",
            stage_name="largest",
            prefix_dims=tuple(dim for dim in flag_dims if dim <= int(fixture_payload["metadata"]["max_flag_dim"])),
            eps_sign=float(fixture_payload["metadata"]["eps_sign"]),
            metadata=fixture_payload["metadata"],
        )
    ]
    for nested_step in sorted(fixture_payload.get("nested_steps", []), key=lambda item: int(item["nested_index"])):
        dim = int(nested_step["dim"])
        cases.append(
            _case_from_matrix(
                nested_step["nested_candidate"],
                case_id=f"fixture_nested_{int(nested_step['nested_index'])}",
                source="fixture",
                stage_name=f"nested_{int(nested_step['nested_index'])}",
                prefix_dims=tuple(prefix for prefix in flag_dims if prefix <= dim),
                eps_sign=float(fixture_payload["metadata"]["eps_sign"]),
                metadata={**fixture_payload["metadata"], "dim": dim, "nested_index": int(nested_step["nested_index"])},
            )
        )

    for payload_idx, payload in enumerate(selected_payloads):
        payload_copy = copy.deepcopy(payload)
        payload_flag_dims = tuple(int(dim) for dim in payload_copy["metadata"]["flag_dims"])
        capture_sequence = int(payload_copy["metadata"].get("capture_sequence", payload_idx))
        cases.append(
            _case_from_matrix(
                payload_copy["largest_candidate"],
                case_id=f"payload_{capture_sequence}_largest",
                source="payload_pool",
                stage_name="largest",
                prefix_dims=tuple(dim for dim in payload_flag_dims if dim <= int(payload_copy["metadata"]["max_flag_dim"])),
                eps_sign=float(payload_copy["metadata"]["eps_sign"]),
                metadata=payload_copy["metadata"],
            )
        )
        for nested_step in sorted(payload_copy.get("nested_steps", []), key=lambda item: int(item["nested_index"])):
            dim = int(nested_step["dim"])
            cases.append(
                _case_from_matrix(
                    nested_step["nested_candidate"],
                    case_id=f"payload_{capture_sequence}_nested_{int(nested_step['nested_index'])}",
                    source="payload_pool",
                    stage_name=f"nested_{int(nested_step['nested_index'])}",
                    prefix_dims=tuple(prefix for prefix in payload_flag_dims if prefix <= dim),
                    eps_sign=float(payload_copy["metadata"]["eps_sign"]),
                    metadata={**payload_copy["metadata"], "dim": dim, "nested_index": int(nested_step["nested_index"])},
                )
            )
    return cases


def synthetic_cases(*, seed: int) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    base = torch.randn((64, 32), generator=generator, dtype=torch.float32)
    near_collinear = base.clone()
    near_collinear[:, 1:] = near_collinear[:, :1] + 1e-4 * near_collinear[:, 1:]
    left = torch.randn((64, 12), generator=generator, dtype=torch.float32)
    right = torch.randn((12, 32), generator=generator, dtype=torch.float32)
    rank_deficient = left @ right
    return [
        _case_from_matrix(
            near_collinear,
            case_id="synthetic_near_collinear_64x32",
            source="synthetic",
            stage_name="matrix",
            prefix_dims=(8, 16, 32),
            eps_sign=1e-8,
            metadata={"seed": int(seed)},
        ),
        _case_from_matrix(
            rank_deficient,
            case_id="synthetic_rank_deficient_64x32",
            source="synthetic",
            stage_name="matrix",
            prefix_dims=(8, 16, 32),
            eps_sign=1e-8,
            metadata={"seed": int(seed)},
        ),
    ]


def json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
