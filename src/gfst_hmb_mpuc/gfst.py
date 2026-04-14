from __future__ import annotations

import torch

RESIDUAL_RANK_ONE_GFST_UPDATE_RULE = "residual_rank_one_v1"
_DEFAULT_EIG_CLUSTER_ATOL = 1e-9
_DEFAULT_EIG_CLUSTER_RTOL = 1e-7


def orthonormal_columns(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"expected a rank-2 matrix, got shape {tuple(matrix.shape)}")
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


def canonical_qr(matrix: torch.Tensor, *, eps_sign: float = 1e-8) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"expected a rank-2 matrix, got shape {tuple(matrix.shape)}")
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    if int(q.numel()) == 0:
        return q
    return canonicalize_qr_column_signs(q, eps_sign=eps_sign)


def canonical_qr_batched(matrices: torch.Tensor, *, eps_sign: float = 1e-8) -> torch.Tensor:
    if matrices.ndim < 3:
        raise ValueError(f"expected rank >= 3 for batched QR, got shape {tuple(matrices.shape)}")
    q, _ = torch.linalg.qr(matrices, mode="reduced")
    if int(q.numel()) == 0:
        return q
    return canonicalize_qr_column_signs(q, eps_sign=eps_sign)


def canonicalize_qr_column_signs(q: torch.Tensor, *, eps_sign: float = 1e-8) -> torch.Tensor:
    if q.ndim < 2:
        raise ValueError(f"expected rank >= 2 for QR sign canonicalization, got shape {tuple(q.shape)}")
    magnitude_mask = q.abs() > float(eps_sign)
    first_indices = torch.argmax(magnitude_mask.to(dtype=torch.int64), dim=-2)
    first_values = torch.gather(q, -2, first_indices.unsqueeze(-2)).squeeze(-2)
    sign = torch.where(
        magnitude_mask.any(dim=-2) & first_values.lt(0),
        -torch.ones_like(first_values),
        torch.ones_like(first_values),
    )
    return q * sign.unsqueeze(-2)


def init_basis(rows: int, cols: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(rows, cols, device=device, dtype=dtype)


def update_basis(
    basis: torch.Tensor,
    value: torch.Tensor,
    alpha: float,
    *,
    update_rule: str = RESIDUAL_RANK_ONE_GFST_UPDATE_RULE,
    eps_update: float = 1e-6,
    eps_sign: float = 1e-8,
) -> torch.Tensor:
    candidate = update_basis_candidate(
        basis,
        value,
        alpha,
        update_rule=update_rule,
        eps_update=eps_update,
    )
    return canonical_qr(candidate, eps_sign=eps_sign)


def update_basis_batched(
    basis: torch.Tensor,
    value: torch.Tensor,
    alpha: float,
    *,
    update_rule: str = RESIDUAL_RANK_ONE_GFST_UPDATE_RULE,
    eps_update: float = 1e-6,
    eps_sign: float = 1e-8,
) -> torch.Tensor:
    candidate = update_basis_candidate_batched(
        basis,
        value,
        alpha,
        update_rule=update_rule,
        eps_update=eps_update,
    )
    return canonical_qr_batched(candidate, eps_sign=eps_sign)


def update_basis_candidate(
    basis: torch.Tensor,
    value: torch.Tensor,
    alpha: float,
    *,
    update_rule: str = RESIDUAL_RANK_ONE_GFST_UPDATE_RULE,
    eps_update: float = 1e-6,
) -> torch.Tensor:
    weights = basis.transpose(0, 1) @ value
    if str(update_rule) != RESIDUAL_RANK_ONE_GFST_UPDATE_RULE:
        raise ValueError(f"unsupported gfst update rule {update_rule!r}; expected {RESIDUAL_RANK_ONE_GFST_UPDATE_RULE!r}")
    projection = basis @ weights
    residual = value - projection
    value_norm = torch.linalg.vector_norm(value)
    scaled_residual = residual / torch.sqrt(value_norm.square() + float(eps_update) ** 2)
    novelty_column = torch.zeros_like(weights)
    novelty_column[-1] = 1.0
    return (
        (1.0 - alpha) * basis
        + alpha * torch.outer(value, weights)
        + alpha * torch.outer(scaled_residual, novelty_column)
    )


def update_basis_candidate_batched(
    basis: torch.Tensor,
    value: torch.Tensor,
    alpha: float,
    *,
    update_rule: str = RESIDUAL_RANK_ONE_GFST_UPDATE_RULE,
    eps_update: float = 1e-6,
) -> torch.Tensor:
    if basis.ndim != 3:
        raise ValueError(f"expected rank-3 basis tensor, got shape {tuple(basis.shape)}")
    if value.ndim != 2:
        raise ValueError(f"expected rank-2 value tensor, got shape {tuple(value.shape)}")
    weights = torch.matmul(basis.transpose(-2, -1), value.unsqueeze(-1)).squeeze(-1)
    if str(update_rule) != RESIDUAL_RANK_ONE_GFST_UPDATE_RULE:
        raise ValueError(f"unsupported gfst update rule {update_rule!r}; expected {RESIDUAL_RANK_ONE_GFST_UPDATE_RULE!r}")
    projection = torch.matmul(basis, weights.unsqueeze(-1)).squeeze(-1)
    residual = value - projection
    value_norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    scaled_residual = residual / torch.sqrt(value_norm.square() + float(eps_update) ** 2)
    novelty_column = torch.zeros_like(weights)
    novelty_column[..., -1] = 1.0
    return (
        (1.0 - alpha) * basis
        + alpha * value.unsqueeze(-1) * weights.unsqueeze(-2)
        + alpha * scaled_residual.unsqueeze(-1) * novelty_column.unsqueeze(-2)
    )


def basis_projector(basis: torch.Tensor) -> torch.Tensor:
    return basis @ basis.transpose(0, 1)


def _sign_fix_vector(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    flattened = vector.reshape(-1)
    magnitude_mask = flattened.abs() > float(eps)
    if not bool(magnitude_mask.any()):
        return vector
    first_index = torch.argmax(magnitude_mask.to(dtype=torch.int64), dim=0)
    sign = -1.0 if bool(flattened[first_index].lt(0)) else 1.0
    return vector * sign


def _append_mgs_column(
    columns: list[torch.Tensor],
    candidate: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor | None:
    vector = candidate.clone()
    for column in columns:
        vector = vector - torch.dot(column, vector) * column
    norm = torch.linalg.norm(vector)
    if float(norm.item()) <= float(eps):
        return None
    return vector / norm


def _canonical_basis_from_cluster_space(
    cluster_space: torch.Tensor,
    *,
    target_rank: int,
    eps: float = 1e-9,
    eps_sign: float = 1e-8,
) -> torch.Tensor:
    if cluster_space.ndim != 2:
        raise ValueError(f"expected rank-2 cluster space, got shape {tuple(cluster_space.shape)}")
    if int(target_rank) <= 0:
        return torch.zeros(
            (int(cluster_space.shape[0]), 0),
            device=cluster_space.device,
            dtype=cluster_space.dtype,
        )
    projector = cluster_space @ cluster_space.transpose(0, 1)
    diagonal = torch.diag(projector)
    axis_order = sorted(
        range(int(projector.shape[0])),
        key=lambda axis_idx: (-float(diagonal[axis_idx].item()), int(axis_idx)),
    )
    columns: list[torch.Tensor] = []
    for axis_idx in axis_order:
        if len(columns) >= int(target_rank):
            break
        candidate = projector[:, int(axis_idx)]
        column = _append_mgs_column(columns, candidate, eps=float(eps))
        if column is not None:
            columns.append(column)
    if len(columns) < int(target_rank):
        for column_idx in range(int(cluster_space.shape[1])):
            if len(columns) >= int(target_rank):
                break
            column = _append_mgs_column(columns, cluster_space[:, int(column_idx)], eps=float(eps))
            if column is not None:
                columns.append(column)
    if len(columns) < int(target_rank):
        raise RuntimeError(
            f"canonical eigenspace construction produced only {len(columns)} columns for requested rank {int(target_rank)}"
        )
    basis = torch.stack(columns[: int(target_rank)], dim=1)
    return canonicalize_qr_column_signs(basis, eps_sign=float(eps_sign))


def _descending_top_eig_cluster_spans(
    eigvals: torch.Tensor,
    *,
    rank: int,
    atol: float = _DEFAULT_EIG_CLUSTER_ATOL,
    rtol: float = _DEFAULT_EIG_CLUSTER_RTOL,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    selected_rank = max(0, min(int(rank), int(eigvals.numel())))
    if selected_rank == 0:
        return eigvals.new_zeros((0,), dtype=torch.float64), []
    descending = torch.flip(eigvals[-selected_rank:].to(dtype=torch.float64), dims=(0,))
    spans: list[tuple[int, int]] = []
    start = 0
    total = int(descending.numel())
    while start < total:
        end = int(start)
        while end + 1 < total and bool(
            torch.isclose(
                descending[end],
                descending[end + 1],
                atol=float(atol),
                rtol=float(rtol),
            ).item()
        ):
            end += 1
        spans.append((int(start), int(end)))
        start = end + 1
    return descending, spans


def _canonical_nested_top_eig_basis_from_eigpairs(
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    *,
    rank: int,
    output_dtype: torch.dtype,
    atol: float = _DEFAULT_EIG_CLUSTER_ATOL,
    rtol: float = _DEFAULT_EIG_CLUSTER_RTOL,
    eps: float = 1e-9,
    eps_sign: float = 1e-8,
) -> torch.Tensor:
    if eigvals.ndim != 1:
        raise ValueError(f"expected rank-1 eigvals, got shape {tuple(eigvals.shape)}")
    if eigvecs.ndim != 2:
        raise ValueError(f"expected rank-2 eigvecs, got shape {tuple(eigvecs.shape)}")
    basis_rank = max(0, min(int(rank), int(eigvecs.shape[0]), int(eigvecs.shape[1])))
    if basis_rank == 0:
        return torch.zeros((int(eigvecs.shape[0]), 0), device=eigvecs.device, dtype=output_dtype)

    math_eigvals = eigvals.to(dtype=torch.float64)
    math_eigvecs = eigvecs.to(dtype=torch.float64)
    remaining = int(basis_rank)
    cluster_end = int(math_eigvals.shape[0]) - 1
    cluster_bases: list[torch.Tensor] = []

    while cluster_end >= 0 and remaining > 0:
        cluster_start = int(cluster_end)
        while cluster_start > 0 and bool(
            torch.isclose(
                math_eigvals[cluster_start - 1],
                math_eigvals[cluster_start],
                atol=float(atol),
                rtol=float(rtol),
            ).item()
        ):
            cluster_start -= 1
        cluster_space = math_eigvecs[:, cluster_start : cluster_end + 1]
        take_count = min(int(cluster_space.shape[1]), remaining)
        cluster_bases.append(
            _canonical_basis_from_cluster_space(
                cluster_space,
                target_rank=int(take_count),
                eps=float(eps),
                eps_sign=float(eps_sign),
            )
        )
        remaining -= int(take_count)
        cluster_end = cluster_start - 1

    if remaining > 0:
        raise RuntimeError(f"insufficient eigenspace columns for requested rank {int(rank)}")

    basis = torch.cat(cluster_bases, dim=1)
    return canonicalize_qr_column_signs(basis.to(dtype=output_dtype), eps_sign=float(eps_sign))


def _canonical_prefix_projectors_from_basis(
    basis: torch.Tensor,
    *,
    rank_boundaries: tuple[int, ...] | list[int],
    output_dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    if basis.ndim != 2:
        raise ValueError(f"expected rank-2 basis tensor, got shape {tuple(basis.shape)}")
    sorted_boundaries = sorted(
        {
            int(boundary)
            for boundary in rank_boundaries
            if int(boundary) > 0 and int(boundary) <= int(basis.shape[1])
        }
    )
    prefix_projectors: dict[int, torch.Tensor] = {}
    for boundary in sorted_boundaries:
        prefix = basis_projector(basis[:, : int(boundary)])
        prefix_projectors[int(boundary)] = (0.5 * (prefix + prefix.transpose(0, 1))).to(dtype=output_dtype)
    return prefix_projectors


def canonical_prefix_projectors_from_eigpairs(
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    *,
    rank_boundaries: tuple[int, ...] | list[int],
    output_dtype: torch.dtype,
    atol: float = _DEFAULT_EIG_CLUSTER_ATOL,
    rtol: float = _DEFAULT_EIG_CLUSTER_RTOL,
    eps: float = 1e-9,
    eps_sign: float = 1e-8,
) -> dict[int, torch.Tensor]:
    if eigvals.ndim != 1:
        raise ValueError(f"expected rank-1 eigvals, got shape {tuple(eigvals.shape)}")
    if eigvecs.ndim != 2:
        raise ValueError(f"expected rank-2 eigvecs, got shape {tuple(eigvecs.shape)}")
    if int(eigvals.shape[0]) != int(eigvecs.shape[1]):
        raise ValueError(
            f"expected eigvals/eigvecs to align, got shapes {tuple(eigvals.shape)} and {tuple(eigvecs.shape)}"
        )
    sorted_boundaries = sorted(
        {
            int(boundary)
            for boundary in rank_boundaries
            if int(boundary) > 0 and int(boundary) <= min(int(eigvecs.shape[0]), int(eigvecs.shape[1]))
        }
    )
    if not sorted_boundaries:
        return {}

    max_boundary = int(sorted_boundaries[-1])
    full_basis = _canonical_nested_top_eig_basis_from_eigpairs(
        eigvals,
        eigvecs,
        rank=int(max_boundary),
        output_dtype=output_dtype,
        atol=float(atol),
        rtol=float(rtol),
        eps=float(eps),
        eps_sign=float(eps_sign),
    )
    return _canonical_prefix_projectors_from_basis(
        full_basis,
        rank_boundaries=sorted_boundaries,
        output_dtype=output_dtype,
    )


def _symmetric_eigpairs_with_fallback(sketch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    symmetric = 0.5 * (sketch + sketch.transpose(0, 1))
    math_sketch = symmetric.to(dtype=torch.float64)
    try:
        return torch.linalg.eigh(math_sketch)
    except RuntimeError:
        jitter = 1e-5 * torch.eye(math_sketch.shape[0], device=math_sketch.device, dtype=math_sketch.dtype)
        stabilized = math_sketch + jitter
        try:
            return torch.linalg.eigh(stabilized)
        except RuntimeError:
            left_singular_vectors, singular_values, _right_singular_vectors = torch.linalg.svd(
                stabilized,
                full_matrices=False,
            )
            return torch.flip(singular_values, dims=(0,)), torch.flip(left_singular_vectors, dims=(1,))


def top_eig_summary(
    sketch: torch.Tensor,
    *,
    rank: int,
    rank_boundaries: tuple[int, ...] | list[int] = (),
    atol: float = _DEFAULT_EIG_CLUSTER_ATOL,
    rtol: float = _DEFAULT_EIG_CLUSTER_RTOL,
    eps: float = 1e-9,
    eps_sign: float = 1e-8,
) -> dict[str, torch.Tensor | dict[int, torch.Tensor]]:
    eigvals, eigvecs = _symmetric_eigpairs_with_fallback(sketch)
    max_boundary = max((int(boundary) for boundary in rank_boundaries), default=0)
    target_rank = max(int(rank), int(max_boundary), 1)
    full_basis = _canonical_nested_top_eig_basis_from_eigpairs(
        eigvals,
        eigvecs,
        rank=int(target_rank),
        output_dtype=sketch.dtype,
        atol=float(atol),
        rtol=float(rtol),
        eps=float(eps),
        eps_sign=float(eps_sign),
    )
    basis_rank = max(0, min(int(rank), int(full_basis.shape[1])))
    basis = full_basis[:, :basis_rank]
    if int(full_basis.shape[1]) > 0:
        top_vector = full_basis[:, 0]
    else:
        top_vector = torch.zeros((int(sketch.shape[0]),), device=sketch.device, dtype=sketch.dtype)
    prefix_projectors = _canonical_prefix_projectors_from_basis(
        full_basis,
        rank_boundaries=rank_boundaries,
        output_dtype=sketch.dtype,
    )
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "full_basis": full_basis,
        "basis": basis,
        "top_vector": top_vector,
        "prefix_projectors": prefix_projectors,
    }


def top_eig_summary_from_eigpairs_batched(
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    *,
    rank: int,
    rank_boundaries: tuple[int, ...] | list[int] = (),
    output_dtype: torch.dtype,
    vector_output_dtype: torch.dtype | None = None,
    atol: float = _DEFAULT_EIG_CLUSTER_ATOL,
    rtol: float = _DEFAULT_EIG_CLUSTER_RTOL,
    eps: float = 1e-9,
    eps_sign: float = 1e-8,
) -> dict[str, torch.Tensor]:
    if eigvals.ndim != 2:
        raise ValueError(f"expected rank-2 eigvals, got shape {tuple(eigvals.shape)}")
    if eigvecs.ndim != 3:
        raise ValueError(f"expected rank-3 eigvecs, got shape {tuple(eigvecs.shape)}")
    if int(eigvals.shape[0]) != int(eigvecs.shape[0]) or int(eigvals.shape[1]) != int(eigvecs.shape[2]):
        raise ValueError(
            f"expected batched eigvals/eigvecs to align, got shapes {tuple(eigvals.shape)} and {tuple(eigvecs.shape)}"
        )
    batch_size = int(eigvals.shape[0])
    rows = int(eigvecs.shape[1])
    max_boundary = max((int(boundary) for boundary in rank_boundaries), default=0)
    target_rank = max(int(rank), int(max_boundary), 1)
    basis_rank = max(0, min(int(rank), int(target_rank), rows))
    vector_dtype = output_dtype if vector_output_dtype is None else vector_output_dtype
    full_basis_rows: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        full_basis_rows.append(
            _canonical_nested_top_eig_basis_from_eigpairs(
                eigvals[batch_idx],
                eigvecs[batch_idx],
                rank=int(target_rank),
                output_dtype=output_dtype,
                atol=float(atol),
                rtol=float(rtol),
                eps=float(eps),
                eps_sign=float(eps_sign),
            )
        )
    if full_basis_rows:
        full_basis = torch.stack(full_basis_rows, dim=0)
    else:
        full_basis = torch.zeros((0, rows, 0), device=eigvecs.device, dtype=output_dtype)
    basis = full_basis[:, :, :basis_rank]
    if int(full_basis.shape[-1]) > 0:
        top_vector = full_basis[:, :, 0].to(dtype=vector_dtype)
    else:
        top_vector = torch.zeros((batch_size, rows), device=eigvecs.device, dtype=vector_dtype)
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "full_basis": full_basis,
        "basis": basis,
        "top_vector": top_vector,
    }


def canonical_top_eig_basis_from_eigpairs(
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    *,
    rank: int,
    output_dtype: torch.dtype,
    atol: float = _DEFAULT_EIG_CLUSTER_ATOL,
    rtol: float = _DEFAULT_EIG_CLUSTER_RTOL,
    eps: float = 1e-9,
    eps_sign: float = 1e-8,
) -> torch.Tensor:
    return _canonical_nested_top_eig_basis_from_eigpairs(
        eigvals,
        eigvecs,
        rank=int(rank),
        output_dtype=output_dtype,
        atol=float(atol),
        rtol=float(rtol),
        eps=float(eps),
        eps_sign=float(eps_sign),
    )


def top_eig_basis(sketch: torch.Tensor, rank: int) -> torch.Tensor:
    summary = top_eig_summary(sketch, rank=int(rank))
    basis = summary["basis"]
    if not isinstance(basis, torch.Tensor):
        raise RuntimeError("top_eig_summary returned an invalid basis")
    return basis


def top_eig_basis_from_factor(factor: torch.Tensor, rank: int, *, eps: float = 1e-8) -> torch.Tensor:
    del eps
    if factor.ndim != 2:
        raise ValueError(f"expected a rank-2 factor matrix, got shape {tuple(factor.shape)}")
    rows = int(factor.shape[0])
    basis_rank = max(0, min(int(rank), rows))
    if int(factor.shape[1]) == 0 or basis_rank == 0:
        return init_basis(rows, basis_rank, device=factor.device, dtype=factor.dtype)
    dense = factor @ factor.transpose(0, 1)
    return top_eig_basis(dense.to(dtype=factor.dtype), rank=basis_rank)


def canonical_top_eigenvector(sketch: torch.Tensor, atol: float = 1e-9, rtol: float = 1e-7) -> torch.Tensor:
    symmetric = 0.5 * (sketch + sketch.transpose(0, 1))
    math_sketch = symmetric.to(dtype=torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(math_sketch)
    top_eval = eigvals[-1]
    degenerate_mask = torch.isclose(eigvals, top_eval, atol=float(atol), rtol=float(rtol))
    top_count = int(degenerate_mask.sum().item())
    if top_count <= 1:
        vector = eigvecs[:, -1]
        return _sign_fix_vector(vector.to(dtype=sketch.dtype))

    top_space = eigvecs[:, degenerate_mask]
    projector = top_space @ top_space.transpose(0, 1)
    diagonal = torch.diag(projector)
    axis_index = int(torch.argmax(diagonal).item())
    probe = torch.zeros(projector.shape[0], device=projector.device, dtype=projector.dtype)
    probe[axis_index] = 1.0
    vector = projector @ probe
    norm = torch.linalg.norm(vector)
    if float(norm.item()) <= float(atol):
        vector = top_space[:, 0]
        norm = torch.linalg.norm(vector)
    vector = vector / torch.clamp(norm, min=float(atol))
    return _sign_fix_vector(vector.to(dtype=sketch.dtype))


def rotate_half(vector: torch.Tensor) -> torch.Tensor:
    half = vector.shape[-1] // 2
    return torch.cat((-vector[..., half:], vector[..., :half]), dim=-1)


def apply_rope(vector: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    if vector.shape[-1] % 2 != 0:
        raise ValueError(f"expected an even final dimension for RoPE, got {vector.shape[-1]}")

    math_vector = vector.float()
    half_dim = math_vector.shape[-1] // 2
    inv_freq = theta ** (-torch.arange(half_dim, device=math_vector.device, dtype=torch.float32) / float(half_dim))
    angles = positions.to(device=math_vector.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    rotated = (math_vector * cos) + (rotate_half(math_vector) * sin)
    return rotated.to(dtype=vector.dtype)


def alignment_components(vector: torch.Tensor, projectors: list[torch.Tensor], eps: float) -> torch.Tensor:
    denom = torch.dot(vector, vector) + eps
    cumulative = []
    for projector in projectors:
        cumulative.append(torch.dot(vector, projector @ vector) / denom)
    cumulative_tensor = torch.stack(cumulative)
    previous = torch.cat((torch.zeros_like(cumulative_tensor[:1]), cumulative_tensor[:-1]), dim=0)
    increments = cumulative_tensor - previous
    return torch.stack((cumulative_tensor, increments), dim=-1).reshape(-1)


def alignment_components_from_bases(vector: torch.Tensor, bases: list[torch.Tensor], eps: float) -> torch.Tensor:
    denom = torch.dot(vector, vector) + eps
    cumulative = []
    for basis in bases:
        weights = basis.transpose(0, 1) @ vector
        cumulative.append(weights.square().sum() / denom)
    cumulative_tensor = torch.stack(cumulative)
    previous = torch.cat((torch.zeros_like(cumulative_tensor[:1]), cumulative_tensor[:-1]), dim=0)
    increments = cumulative_tensor - previous
    return torch.stack((cumulative_tensor, increments), dim=-1).reshape(-1)


def alignment_components_from_bases_batch(
    vectors: torch.Tensor,
    bases: list[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError(f"expected rank-2 vectors tensor, got shape {tuple(vectors.shape)}")
    denom = vectors.square().sum(dim=-1) + float(eps)
    cumulative = []
    for basis in bases:
        weights = torch.einsum("bdk,bd->bk", basis, vectors)
        cumulative.append(weights.square().sum(dim=-1) / denom)
    cumulative_tensor = torch.stack(cumulative, dim=-1)
    previous = torch.cat((torch.zeros_like(cumulative_tensor[:, :1]), cumulative_tensor[:, :-1]), dim=-1)
    increments = cumulative_tensor - previous
    return torch.stack((cumulative_tensor, increments), dim=-1).reshape(int(vectors.shape[0]), -1)


def alignment_components_from_full_basis_batch(
    vectors: torch.Tensor,
    full_basis: torch.Tensor,
    rank_boundaries: tuple[int, ...] | list[int],
    eps: float,
) -> torch.Tensor:
    if full_basis.ndim != 3:
        raise ValueError(f"expected rank-3 full_basis tensor, got shape {tuple(full_basis.shape)}")
    ordered_boundaries = [
        int(boundary)
        for boundary in rank_boundaries
        if int(boundary) > 0 and int(boundary) <= int(full_basis.shape[-1])
    ]
    if not ordered_boundaries:
        return torch.zeros((int(vectors.shape[0]), 0), device=vectors.device, dtype=torch.float32)
    return alignment_components_from_bases_batch(
        vectors,
        [full_basis[:, :, : int(boundary)] for boundary in ordered_boundaries],
        eps,
    )


def alignment_components_from_full_basis_projector_batch(
    vectors: torch.Tensor,
    full_basis: torch.Tensor,
    rank_boundaries: tuple[int, ...] | list[int],
    eps: float,
) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError(f"expected rank-2 vectors tensor, got shape {tuple(vectors.shape)}")
    if full_basis.ndim != 3:
        raise ValueError(f"expected rank-3 full_basis tensor, got shape {tuple(full_basis.shape)}")
    if int(vectors.shape[0]) != int(full_basis.shape[0]):
        raise ValueError(
            f"expected vectors/full_basis batch sizes to match, got {tuple(vectors.shape)} and {tuple(full_basis.shape)}"
        )
    ordered_boundaries = [
        int(boundary)
        for boundary in rank_boundaries
        if int(boundary) > 0 and int(boundary) <= int(full_basis.shape[-1])
    ]
    if not ordered_boundaries:
        return torch.zeros((int(vectors.shape[0]), 0), device=vectors.device, dtype=torch.float32)
    denom = vectors.square().sum(dim=-1) + float(eps)
    cumulative: list[torch.Tensor] = []
    for boundary in ordered_boundaries:
        prefix_basis = full_basis[:, :, : int(boundary)]
        prefix_projector = torch.matmul(prefix_basis, prefix_basis.transpose(-2, -1))
        prefix_projector = 0.5 * (prefix_projector + prefix_projector.transpose(-2, -1))
        projected = torch.einsum("bij,bj->bi", prefix_projector, vectors)
        cumulative.append((vectors * projected).sum(dim=-1) / denom)
    cumulative_tensor = torch.stack(cumulative, dim=-1)
    previous = torch.cat((torch.zeros_like(cumulative_tensor[:, :1]), cumulative_tensor[:, :-1]), dim=-1)
    increments = cumulative_tensor - previous
    return torch.stack((cumulative_tensor, increments), dim=-1).reshape(int(vectors.shape[0]), -1)


def alignment_components_debug(
    vector: torch.Tensor,
    projectors: list[torch.Tensor],
    eps: float,
    *,
    boundaries: tuple[int, ...] | list[int] | None = None,
) -> dict[str, object]:
    ordered_boundaries = (
        [int(boundary) for boundary in boundaries]
        if boundaries is not None
        else [int(index + 1) for index in range(len(projectors))]
    )
    if len(ordered_boundaries) != len(projectors):
        raise ValueError("boundaries length must match projectors length")
    denom = torch.dot(vector, vector) + eps
    cumulative: list[torch.Tensor] = []
    projected_vectors: dict[str, torch.Tensor] = {}
    prefix_projectors: dict[str, torch.Tensor] = {}
    prefix_energy_numerators: dict[str, torch.Tensor] = {}
    for boundary, projector in zip(ordered_boundaries, projectors):
        projected = projector @ vector
        numerator = torch.dot(vector, projected)
        cumulative.append(numerator / denom)
        key = str(int(boundary))
        projected_vectors[key] = projected
        prefix_projectors[key] = projector
        prefix_energy_numerators[key] = numerator
    cumulative_tensor = torch.stack(cumulative)
    previous = torch.cat((torch.zeros_like(cumulative_tensor[:1]), cumulative_tensor[:-1]), dim=0)
    increments = cumulative_tensor - previous
    output = torch.stack((cumulative_tensor, increments), dim=-1).reshape(-1)
    return {
        "ordered_boundaries": [int(boundary) for boundary in ordered_boundaries],
        "denom": denom,
        "prefix_projectors": prefix_projectors,
        "projected_vectors": projected_vectors,
        "prefix_energy_numerators": prefix_energy_numerators,
        "cumulative": cumulative_tensor,
        "increments": increments,
        "output": output,
    }


def alignment_components_from_bases_debug(
    vector: torch.Tensor,
    bases: list[torch.Tensor],
    eps: float,
    *,
    boundaries: tuple[int, ...] | list[int] | None = None,
) -> dict[str, object]:
    ordered_boundaries = (
        [int(boundary) for boundary in boundaries]
        if boundaries is not None
        else [int(index + 1) for index in range(len(bases))]
    )
    if len(ordered_boundaries) != len(bases):
        raise ValueError("boundaries length must match bases length")
    denom = torch.dot(vector, vector) + eps
    cumulative: list[torch.Tensor] = []
    prefix_bases: dict[str, torch.Tensor] = {}
    prefix_weights: dict[str, torch.Tensor] = {}
    prefix_energy_numerators: dict[str, torch.Tensor] = {}
    for boundary, basis in zip(ordered_boundaries, bases):
        weights = basis.transpose(0, 1) @ vector
        numerator = weights.square().sum()
        cumulative.append(numerator / denom)
        key = str(int(boundary))
        prefix_bases[key] = basis
        prefix_weights[key] = weights
        prefix_energy_numerators[key] = numerator
    cumulative_tensor = torch.stack(cumulative)
    previous = torch.cat((torch.zeros_like(cumulative_tensor[:1]), cumulative_tensor[:-1]), dim=0)
    increments = cumulative_tensor - previous
    output = torch.stack((cumulative_tensor, increments), dim=-1).reshape(-1)
    return {
        "ordered_boundaries": [int(boundary) for boundary in ordered_boundaries],
        "denom": denom,
        "prefix_bases": prefix_bases,
        "prefix_weights": prefix_weights,
        "prefix_energy_numerators": prefix_energy_numerators,
        "cumulative": cumulative_tensor,
        "increments": increments,
        "output": output,
    }


def alignment_components_from_full_basis_debug(
    vector: torch.Tensor,
    full_basis: torch.Tensor,
    rank_boundaries: tuple[int, ...] | list[int],
    eps: float,
) -> dict[str, object]:
    if full_basis.ndim != 2:
        raise ValueError(f"expected rank-2 full_basis tensor, got shape {tuple(full_basis.shape)}")
    ordered_boundaries = [
        int(boundary)
        for boundary in rank_boundaries
        if int(boundary) > 0 and int(boundary) <= int(full_basis.shape[-1])
    ]
    if not ordered_boundaries:
        return {
            "ordered_boundaries": [],
            "full_basis": full_basis,
            "denom": torch.dot(vector, vector) + eps,
            "prefix_bases": {},
            "prefix_weights": {},
            "prefix_energy_numerators": {},
            "cumulative": torch.zeros((0,), device=vector.device, dtype=vector.dtype),
            "increments": torch.zeros((0,), device=vector.device, dtype=vector.dtype),
            "output": torch.zeros((0,), device=vector.device, dtype=vector.dtype),
        }
    debug = alignment_components_from_bases_debug(
        vector,
        [full_basis[:, : int(boundary)] for boundary in ordered_boundaries],
        eps,
        boundaries=ordered_boundaries,
    )
    debug["full_basis"] = full_basis
    return debug
