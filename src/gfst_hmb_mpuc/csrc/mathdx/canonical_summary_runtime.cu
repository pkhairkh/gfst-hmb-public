#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <tuple>

namespace {

constexpr int kMaxCanonicalDim = 64;
constexpr int kMaxCanonicalCutoffs = 64;

void _check_cuda_double_tensor(const torch::Tensor& tensor, const char* name, int64_t expected_dim) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kDouble, name, " must have dtype float64");
  TORCH_CHECK(tensor.dim() == expected_dim, name, " must have rank ", expected_dim);
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void _check_cuda_long_tensor(const torch::Tensor& tensor, const char* name, int64_t expected_dim) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must have dtype int64");
  TORCH_CHECK(tensor.dim() == expected_dim, name, " must have rank ", expected_dim);
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__device__ bool _canonical_isclose(double lhs, double rhs, double atol, double rtol) {
  const double diff = fabs(lhs - rhs);
  const double scale = fmax(fabs(lhs), fabs(rhs));
  const double threshold = fmax(atol, scale * rtol);
  return diff <= threshold;
}

__device__ int _first_significant_index(const double* values, int size, double eps_sign) {
  for (int idx = 0; idx < size; ++idx) {
    if (fabs(values[idx]) > eps_sign) {
      return idx;
    }
  }
  return -1;
}

__device__ void _sign_fix_column(double* values, int size, double eps_sign) {
  const int first_idx = _first_significant_index(values, size, eps_sign);
  if (first_idx >= 0 && values[first_idx] < 0.0) {
    for (int idx = 0; idx < size; ++idx) {
      values[idx] = -values[idx];
    }
  }
}

__device__ bool _append_mgs_column_device(
    const double* candidate,
    double* basis,
    int rows,
    int leading_dim,
    int existing_cols,
    double eps,
    double eps_sign) {
  double working[kMaxCanonicalDim];
  for (int row = 0; row < rows; ++row) {
    working[row] = candidate[row];
  }
  for (int col = 0; col < existing_cols; ++col) {
    double dot = 0.0;
    for (int row = 0; row < rows; ++row) {
      dot += basis[row * leading_dim + col] * working[row];
    }
    for (int row = 0; row < rows; ++row) {
      working[row] -= dot * basis[row * leading_dim + col];
    }
  }
  double norm_sq = 0.0;
  for (int row = 0; row < rows; ++row) {
    norm_sq += working[row] * working[row];
  }
  const double norm = sqrt(norm_sq);
  if (norm <= eps) {
    return false;
  }
  for (int row = 0; row < rows; ++row) {
    working[row] /= norm;
  }
  _sign_fix_column(working, rows, eps_sign);
  for (int row = 0; row < rows; ++row) {
    basis[row * leading_dim + existing_cols] = working[row];
  }
  return true;
}

__global__ void gfst_canonical_summary_kernel(
    const double* eigvals,
    const double* eigvecs,
    const double* reduced_tokens,
    const int64_t* cutoffs,
    float* basis_out,
    float* features_out,
    float* top_vec_out,
    int* info,
    int64_t batch_count,
    int dim,
    int cutoff_count,
    int target_rank,
    double eps,
    double atol,
    double rtol,
    double eps_cluster,
    double eps_sign) {
  const int64_t batch_idx = static_cast<int64_t>(blockIdx.x);
  if (batch_idx >= batch_count || threadIdx.x != 0) {
    return;
  }
  if (dim > kMaxCanonicalDim || cutoff_count > kMaxCanonicalCutoffs) {
    info[batch_idx] = 1;
    return;
  }

  const int64_t eigval_stride = static_cast<int64_t>(dim);
  const int64_t eigvec_stride = static_cast<int64_t>(dim) * static_cast<int64_t>(dim);
  const int64_t feature_stride = static_cast<int64_t>(cutoff_count) * 2;
  const int64_t basis_stride = static_cast<int64_t>(dim) * static_cast<int64_t>(target_rank);

  const double* eigvals_row = eigvals + batch_idx * eigval_stride;
  const double* eigvecs_row = eigvecs + batch_idx * eigvec_stride;
  const double* token_row = reduced_tokens + batch_idx * eigval_stride;
  float* basis_row = basis_out + batch_idx * basis_stride;
  float* feature_row = features_out + batch_idx * feature_stride;
  float* top_vec_row = top_vec_out + batch_idx * eigval_stride;

  double basis_local[kMaxCanonicalDim * kMaxCanonicalDim];
  for (int idx = 0; idx < dim * target_rank; ++idx) {
    basis_local[idx] = 0.0;
  }

  int col = 0;
  int cluster_end = dim - 1;
  while (cluster_end >= 0 && col < target_rank) {
    int cluster_start = cluster_end;
    while (cluster_start > 0 &&
           _canonical_isclose(eigvals_row[cluster_start - 1], eigvals_row[cluster_start], atol, rtol)) {
      cluster_start -= 1;
    }
    const int cluster_size = cluster_end - cluster_start + 1;
    const int cluster_target_rank = min(cluster_size, target_rank - col);
    const int cluster_col_start = col;
    double cluster_basis[kMaxCanonicalDim * kMaxCanonicalDim];
    for (int idx = 0; idx < dim * kMaxCanonicalDim; ++idx) {
      cluster_basis[idx] = 0.0;
    }
    int cluster_col_count = 0;

    double diag_values[kMaxCanonicalDim];
    int axis_order[kMaxCanonicalDim];
    for (int axis = 0; axis < dim; ++axis) {
      double diag = 0.0;
      for (int cluster_idx = 0; cluster_idx < cluster_size; ++cluster_idx) {
        const int eig_col = cluster_start + cluster_idx;
        const double value = eigvecs_row[axis * dim + eig_col];
        diag += value * value;
      }
      diag_values[axis] = diag;
      axis_order[axis] = axis;
    }
    for (int left = 0; left < dim; ++left) {
      int best = left;
      for (int right = left + 1; right < dim; ++right) {
        const int best_axis = axis_order[best];
        const int cand_axis = axis_order[right];
        if (diag_values[cand_axis] > diag_values[best_axis] ||
            (diag_values[cand_axis] == diag_values[best_axis] && cand_axis < best_axis)) {
          best = right;
        }
      }
      const int tmp = axis_order[left];
      axis_order[left] = axis_order[best];
      axis_order[best] = tmp;
    }

    for (int order_idx = 0; order_idx < dim && cluster_col_count < cluster_target_rank; ++order_idx) {
      const int anchor_axis = axis_order[order_idx];
      double candidate[kMaxCanonicalDim];
      for (int row = 0; row < dim; ++row) {
        double accum = 0.0;
        for (int cluster_idx = 0; cluster_idx < cluster_size; ++cluster_idx) {
          const int eig_col = cluster_start + cluster_idx;
          accum += eigvecs_row[row * dim + eig_col] * eigvecs_row[anchor_axis * dim + eig_col];
        }
        candidate[row] = accum;
      }
      if (_append_mgs_column_device(
              candidate,
              cluster_basis,
              dim,
              kMaxCanonicalDim,
              cluster_col_count,
              eps_cluster,
              eps_sign)) {
        ++cluster_col_count;
        ++col;
      }
    }

    if (cluster_col_count < cluster_target_rank) {
      for (int cluster_idx = 0; cluster_idx < cluster_size && cluster_col_count < cluster_target_rank; ++cluster_idx) {
        double candidate[kMaxCanonicalDim];
        const int eig_col = cluster_start + cluster_idx;
        for (int row = 0; row < dim; ++row) {
          candidate[row] = eigvecs_row[row * dim + eig_col];
        }
        if (_append_mgs_column_device(
                candidate,
                cluster_basis,
                dim,
                kMaxCanonicalDim,
                cluster_col_count,
                eps_cluster,
                eps_sign)) {
          ++cluster_col_count;
          ++col;
        }
      }
    }

    for (int row = 0; row < dim; ++row) {
      for (int cluster_col = 0; cluster_col < cluster_target_rank; ++cluster_col) {
        basis_local[row * target_rank + cluster_col_start + cluster_col] =
            cluster_basis[row * kMaxCanonicalDim + cluster_col];
      }
    }

    cluster_end = cluster_start - 1;
  }

  if (col < target_rank) {
    info[batch_idx] = 2;
    return;
  }

  for (int basis_col = 0; basis_col < target_rank; ++basis_col) {
    double column_values[kMaxCanonicalDim];
    for (int row = 0; row < dim; ++row) {
      column_values[row] = basis_local[row * target_rank + basis_col];
    }
    _sign_fix_column(column_values, dim, eps_sign);
    for (int row = 0; row < dim; ++row) {
      basis_local[row * target_rank + basis_col] = column_values[row];
    }
  }

  double norm2_x = eps;
  for (int row = 0; row < dim; ++row) {
    norm2_x += token_row[row] * token_row[row];
  }
  double previous = 0.0;
  for (int cutoff_idx = 0; cutoff_idx < cutoff_count; ++cutoff_idx) {
    const int cutoff = static_cast<int>(cutoffs[cutoff_idx]);
    double cumulative = 0.0;
    for (int basis_col = 0; basis_col < cutoff; ++basis_col) {
      double dot = 0.0;
      for (int row = 0; row < dim; ++row) {
        dot += basis_local[row * target_rank + basis_col] * token_row[row];
      }
      cumulative += dot * dot;
    }
    cumulative /= norm2_x;
    feature_row[cutoff_idx * 2] = static_cast<float>(cumulative);
    feature_row[cutoff_idx * 2 + 1] = static_cast<float>(cumulative - previous);
    previous = cumulative;
  }

  for (int row = 0; row < dim; ++row) {
    top_vec_row[row] = static_cast<float>(basis_local[row * target_rank]);
  }
  for (int row = 0; row < dim; ++row) {
    for (int basis_col = 0; basis_col < target_rank; ++basis_col) {
      basis_row[row * target_rank + basis_col] = static_cast<float>(basis_local[row * target_rank + basis_col]);
    }
  }
  info[batch_idx] = 0;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gfst_canonical_summary_forward(
    const torch::Tensor& eigvals,
    const torch::Tensor& eigvecs,
    const torch::Tensor& reduced_tokens,
    const torch::Tensor& cutoffs,
    int64_t target_rank,
    double eps,
    double atol,
    double rtol) {
  _check_cuda_double_tensor(eigvals, "eigvals", 2);
  _check_cuda_double_tensor(eigvecs, "eigvecs", 3);
  TORCH_CHECK(reduced_tokens.is_cuda(), "reduced_tokens must be a CUDA tensor");
  TORCH_CHECK(
      reduced_tokens.scalar_type() == at::kFloat || reduced_tokens.scalar_type() == at::kDouble,
      "reduced_tokens must have dtype float32 or float64");
  TORCH_CHECK(reduced_tokens.dim() == 2, "reduced_tokens must have rank 2");
  TORCH_CHECK(reduced_tokens.is_contiguous(), "reduced_tokens must be contiguous");
  _check_cuda_long_tensor(cutoffs, "cutoffs", 1);

  TORCH_CHECK(eigvals.size(0) == eigvecs.size(0), "eigvals and eigvecs must share the batch dimension");
  TORCH_CHECK(eigvecs.size(1) == eigvecs.size(2), "eigvecs must be square per batch");
  TORCH_CHECK(eigvals.size(1) == eigvecs.size(2), "eigvals and eigvecs must align on the eig dimension");
  TORCH_CHECK(reduced_tokens.size(0) == eigvals.size(0), "reduced_tokens and eigvals must share the batch dimension");
  TORCH_CHECK(reduced_tokens.size(1) == eigvals.size(1), "reduced_tokens and eigvals must share the reduced dimension");
  TORCH_CHECK(target_rank >= 1, "target_rank must be positive");
  TORCH_CHECK(eps > 0.0, "eps must be positive");

  const int dim = static_cast<int>(eigvals.size(1));
  const int cutoff_count = static_cast<int>(cutoffs.size(0));
  TORCH_CHECK(dim <= kMaxCanonicalDim, "canonical summary only supports dims up to ", kMaxCanonicalDim);
  TORCH_CHECK(cutoff_count <= kMaxCanonicalCutoffs, "canonical summary only supports up to ", kMaxCanonicalCutoffs, " cutoffs");

  int max_cutoff = 0;
  auto cutoffs_cpu = cutoffs.cpu();
  const int64_t* cutoffs_ptr = cutoffs_cpu.data_ptr<int64_t>();
  int64_t previous_cutoff = 0;
  for (int idx = 0; idx < cutoff_count; ++idx) {
    const int64_t cutoff = cutoffs_ptr[idx];
    TORCH_CHECK(cutoff > 0, "cutoffs must be positive");
    TORCH_CHECK(cutoff <= dim, "cutoffs must not exceed the eig dimension");
    TORCH_CHECK(cutoff >= previous_cutoff, "cutoffs must be sorted ascending");
    previous_cutoff = cutoff;
    max_cutoff = std::max(max_cutoff, static_cast<int>(cutoff));
  }
  const int rank = std::min(dim, std::max(static_cast<int>(target_rank), max_cutoff));

  c10::cuda::CUDAGuard device_guard(eigvals.device());
  auto reduced_tokens_math =
      reduced_tokens.scalar_type() == at::kDouble ? reduced_tokens : reduced_tokens.to(torch::kDouble);
  auto basis = torch::empty(
      {eigvals.size(0), eigvals.size(1), rank},
      torch::TensorOptions().device(eigvals.device()).dtype(torch::kFloat));
  auto features = torch::empty(
      {eigvals.size(0), static_cast<int64_t>(cutoff_count) * 2},
      torch::TensorOptions().device(eigvals.device()).dtype(torch::kFloat));
  auto top_vec = torch::empty(
      {eigvals.size(0), eigvals.size(1)},
      torch::TensorOptions().device(eigvals.device()).dtype(torch::kFloat));
  auto info = torch::zeros(
      {eigvals.size(0)},
      torch::TensorOptions().device(eigvals.device()).dtype(torch::kInt));

  const auto stream = at::cuda::getDefaultCUDAStream();
  gfst_canonical_summary_kernel<<<static_cast<unsigned int>(eigvals.size(0)), 1, 0, stream>>>(
      eigvals.data_ptr<double>(),
      eigvecs.data_ptr<double>(),
      reduced_tokens_math.data_ptr<double>(),
      cutoffs.data_ptr<int64_t>(),
      basis.data_ptr<float>(),
      features.data_ptr<float>(),
      top_vec.data_ptr<float>(),
      info.data_ptr<int>(),
      eigvals.size(0),
      dim,
      cutoff_count,
      rank,
      eps,
      atol,
      rtol,
      1e-9,
      1e-8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto info_host = info.cpu();
  const int* info_ptr = info_host.data_ptr<int>();
  for (int64_t batch_idx = 0; batch_idx < info_host.numel(); ++batch_idx) {
    TORCH_CHECK(info_ptr[batch_idx] == 0, "canonical summary kernel failed with info=", info_ptr[batch_idx]);
  }
  return std::make_tuple(basis, features, top_vec);
}
