#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include "cublasdx.hpp"
#include "cusolverdx.hpp"
#include "cusolverdx/detail/shared_memory.hpp"

namespace cdx_shared_memory = cusolverdx::shared_memory;

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kSmArch = 1200;

#define CUDA_CHECK(EXPR) TORCH_CHECK((EXPR) == cudaSuccess, "CUDA call failed: ", cudaGetErrorString(EXPR))
#define CUBLAS_CHECK(EXPR)                                        \
  do {                                                            \
    const cublasStatus_t _status = (EXPR);                        \
    TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS,                 \
                "cuBLAS call failed: ",                           \
                static_cast<int>(_status));                       \
  } while (false)
#define CUSOLVER_CHECK(EXPR)                                      \
  do {                                                            \
    const cusolverStatus_t _status = (EXPR);                      \
    TORCH_CHECK(_status == CUSOLVER_STATUS_SUCCESS,               \
                "cuSOLVER call failed: ",                         \
                static_cast<int>(_status));                       \
  } while (false)

template<typename Scalar, int M>
struct HeevTraits {
  using Base = decltype(
      cusolverdx::Size<M>() +
      cusolverdx::Precision<Scalar>() +
      cusolverdx::Type<cusolverdx::type::real>() +
      cusolverdx::Function<cusolverdx::heev>() +
      cusolverdx::FillMode<cusolverdx::fill_mode::upper>() +
      cusolverdx::Arrangement<cusolverdx::arrangement::row_major>() +
      cusolverdx::Job<cusolverdx::job::overwrite_vectors>() +
      cusolverdx::SM<kSmArch>() +
      cusolverdx::Block());
  using Solver = decltype(Base() + cusolverdx::BatchesPerBlock<1>());
  using precision_type = typename Solver::a_precision;
  static constexpr auto block_dim = Solver::block_dim;
  static constexpr int shared_memory_size = Solver::shared_memory_size;
  static constexpr int lda = Solver::lda;
};

__global__ void gfst_hmb_mathdx_probe_kernel(int64_t* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = 1;
  }
}

__global__ void gfst_global_sketch_accum_kernel(
    const float* summary_projectors,
    const int64_t* final_indices,
    const float* weights,
    float* output,
    int64_t selected_count,
    int64_t projector_stride,
    int64_t element_count) {
  const int64_t linear_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t element_index = linear_index; element_index < element_count; element_index += stride) {
    float accum = 0.0f;
    for (int64_t selected_idx = 0; selected_idx < selected_count; ++selected_idx) {
      const int64_t projector_index = final_indices[selected_idx];
      accum += weights[selected_idx] * summary_projectors[projector_index * projector_stride + element_index];
    }
    output[element_index] = accum;
  }
}

__global__ void scale_tensor_kernel(
    const float* input,
    float* output,
    float scale,
    int64_t element_count) {
  const int64_t linear_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t element_index = linear_index; element_index < element_count; element_index += stride) {
    output[element_index] = input[element_index] * scale;
  }
}

template <typename Scalar>
__global__ void fill_pointer_array_kernel(
    Scalar* base,
    int64_t stride,
    Scalar** pointers,
    int64_t count) {
  const int64_t linear_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = linear_index; idx < count; idx += step) {
    pointers[idx] = base + idx * stride;
  }
}

template<typename Scalar, int M>
__global__ void gfst_heev_kernel(
    const Scalar* input,
    Scalar* eigenvectors,
    typename HeevTraits<Scalar, M>::precision_type* eigenvalues,
    int* info,
    int64_t batch_count) {
  using Traits = HeevTraits<Scalar, M>;
  using Solver = typename Traits::Solver;
  using Precision = typename Traits::precision_type;
  constexpr int lda_smem = Traits::lda;
  constexpr int matrix_smem_size = lda_smem * M;

  const int64_t batch_idx = static_cast<int64_t>(blockIdx.x);
  if (batch_idx >= batch_count) {
    return;
  }

  extern __shared__ __align__(16) unsigned char shared_mem[];
  auto [a_smem, lambda_smem, workspace_smem] =
      cdx_shared_memory::slice<Scalar, Precision, Scalar>(
          shared_mem,
          alignof(Scalar),
          matrix_smem_size,
          alignof(Precision),
          M,
          alignof(Scalar));

  const Scalar* input_matrix = input + batch_idx * static_cast<int64_t>(M) * static_cast<int64_t>(M);
  Scalar* output_vectors = eigenvectors + batch_idx * static_cast<int64_t>(M) * static_cast<int64_t>(M);
  Precision* output_values = eigenvalues + batch_idx * static_cast<int64_t>(M);

  const int thread_id =
      threadIdx.x + Solver::block_dim.x * (threadIdx.y + Solver::block_dim.y * threadIdx.z);

  for (int idx = thread_id; idx < matrix_smem_size; idx += Solver::max_threads_per_block) {
    a_smem[idx] = Scalar(0);
  }
  __syncthreads();

  for (int idx = thread_id; idx < M * M; idx += Solver::max_threads_per_block) {
    const int row = idx / M;
    const int col = idx % M;
    a_smem[row * lda_smem + col] = input_matrix[idx];
  }
  __syncthreads();

  Solver().execute(a_smem, lda_smem, lambda_smem, workspace_smem, info + batch_idx);
  __syncthreads();

  for (int idx = thread_id; idx < M * M; idx += Solver::max_threads_per_block) {
    const int row = idx / M;
    const int col = idx % M;
    output_vectors[idx] = a_smem[row * lda_smem + col];
  }
  for (int idx = thread_id; idx < M; idx += Solver::max_threads_per_block) {
    output_values[idx] = lambda_smem[idx];
  }
}

int _block_count_for(int64_t element_count) {
  if (element_count <= 0) {
    return 1;
  }
  const int64_t raw = (element_count + static_cast<int64_t>(kThreadsPerBlock) - 1) /
                      static_cast<int64_t>(kThreadsPerBlock);
  return static_cast<int>(raw > 4096 ? 4096 : raw);
}

void _check_cuda_float_tensor(const torch::Tensor& tensor, const char* name, int64_t expected_dim) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must have dtype float32");
  TORCH_CHECK(tensor.dim() == expected_dim, name, " must have rank ", expected_dim);
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void _check_cuda_long_tensor(const torch::Tensor& tensor, const char* name, int64_t expected_dim) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must have dtype int64");
  TORCH_CHECK(tensor.dim() == expected_dim, name, " must have rank ", expected_dim);
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void _check_cuda_real_matrix_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kDouble,
      name,
      " must have dtype float32 or float64");
  TORCH_CHECK(tensor.dim() == 2 || tensor.dim() == 3, name, " must have rank 2 or 3");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.size(-1) == tensor.size(-2), name, " must be square");
}

class ScopedCusolverDnHandle {
 public:
  ScopedCusolverDnHandle() {
    CUSOLVER_CHECK(cusolverDnCreate(&handle_));
  }

  ~ScopedCusolverDnHandle() {
    if (handle_ != nullptr) {
      (void)cusolverDnDestroy(handle_);
    }
  }

  cusolverDnHandle_t get() const {
    return handle_;
  }

 private:
  cusolverDnHandle_t handle_ = nullptr;
};

class ScopedCublasHandle {
 public:
  ScopedCublasHandle() {
    CUBLAS_CHECK(cublasCreate(&handle_));
  }

  ~ScopedCublasHandle() {
    if (handle_ != nullptr) {
      (void)cublasDestroy(handle_);
    }
  }

  cublasHandle_t get() const {
    return handle_;
  }

 private:
  cublasHandle_t handle_ = nullptr;
};

struct LocalQrCompact {
  torch::Tensor factors_row_major;
  torch::Tensor tau;
};

LocalQrCompact _launch_local_qr_compact_forward(const torch::Tensor& candidate_matrices) {
  c10::cuda::CUDAGuard device_guard(candidate_matrices.device());
  const auto batch_count = candidate_matrices.size(0);
  const auto row_count = static_cast<int>(candidate_matrices.size(1));
  const auto col_count = static_cast<int>(candidate_matrices.size(2));
  const auto tau_count = std::min(row_count, col_count);
  auto factors_col_major = candidate_matrices.transpose(1, 2).contiguous();
  auto tau = torch::empty(
      {batch_count, static_cast<int64_t>(tau_count)},
      torch::TensorOptions().device(candidate_matrices.device()).dtype(torch::kFloat));
  auto info = torch::zeros(
      {batch_count},
      torch::TensorOptions().device(candidate_matrices.device()).dtype(torch::kInt));
  auto workspace_probe = torch::empty(
      {std::max(row_count, col_count)},
      torch::TensorOptions().device(candidate_matrices.device()).dtype(torch::kFloat));
  int lwork_geqrf = 0;
  int lwork_orgqr = 0;
  ScopedCusolverDnHandle handle;
  const auto stream = at::cuda::getDefaultCUDAStream();
  CUSOLVER_CHECK(cusolverDnSetStream(handle.get(), stream.stream()));
  CUSOLVER_CHECK(
      cusolverDnSgeqrf_bufferSize(
          handle.get(),
          row_count,
          col_count,
          workspace_probe.data_ptr<float>(),
          row_count,
          &lwork_geqrf));
  CUSOLVER_CHECK(
      cusolverDnSorgqr_bufferSize(
          handle.get(),
          row_count,
          col_count,
          col_count,
          workspace_probe.data_ptr<float>(),
          row_count,
          workspace_probe.data_ptr<float>(),
          &lwork_orgqr));
  const int lwork = std::max(lwork_geqrf, lwork_orgqr);
  auto workspace = torch::empty(
      {static_cast<int64_t>(lwork)},
      torch::TensorOptions().device(candidate_matrices.device()).dtype(torch::kFloat));

  auto* factor_ptr = factors_col_major.data_ptr<float>();
  auto* tau_ptr = tau.data_ptr<float>();
  auto* info_ptr = info.data_ptr<int>();
  auto* workspace_ptr = workspace.data_ptr<float>();
  const auto matrix_stride = static_cast<int64_t>(row_count) * static_cast<int64_t>(col_count);
  for (int64_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
    float* factor_matrix = factor_ptr + batch_idx * matrix_stride;
    float* tau_matrix = tau_ptr + batch_idx * static_cast<int64_t>(tau_count);
    int* info_matrix = info_ptr + batch_idx;
    CUSOLVER_CHECK(
        cusolverDnSgeqrf(
            handle.get(),
            row_count,
            col_count,
            factor_matrix,
            row_count,
            tau_matrix,
            workspace_ptr,
            lwork_geqrf,
            info_matrix));
  }

  auto info_cpu = info.cpu();
  const int* info_values = info_cpu.data_ptr<int>();
  for (int64_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
    TORCH_CHECK(info_values[batch_idx] == 0, "cuSOLVER dense QR failed with info=", info_values[batch_idx]);
  }

  return {
      factors_col_major.transpose(1, 2).contiguous(),
      tau,
  };
}

LocalQrCompact _launch_local_qr_compact_cublas_batched_forward(const torch::Tensor& candidate_matrices) {
  c10::cuda::CUDAGuard device_guard(candidate_matrices.device());
  const auto batch_count = candidate_matrices.size(0);
  const auto row_count = static_cast<int>(candidate_matrices.size(1));
  const auto col_count = static_cast<int>(candidate_matrices.size(2));
  const auto tau_count = std::min(row_count, col_count);
  auto factors_col_major = candidate_matrices.transpose(1, 2).contiguous();
  auto tau = torch::empty(
      {batch_count, static_cast<int64_t>(tau_count)},
      torch::TensorOptions().device(candidate_matrices.device()).dtype(torch::kFloat));
  auto info = torch::zeros(
      {batch_count},
      torch::TensorOptions().device(candidate_matrices.device()).dtype(torch::kInt));
  const auto matrix_stride = static_cast<int64_t>(row_count) * static_cast<int64_t>(col_count);
  const auto tau_stride = static_cast<int64_t>(tau_count);
  const auto stream = at::cuda::getDefaultCUDAStream();
  std::vector<float*> factor_ptrs(static_cast<size_t>(batch_count));
  std::vector<float*> tau_ptrs(static_cast<size_t>(batch_count));
  auto* factor_base = factors_col_major.data_ptr<float>();
  auto* tau_base = tau.data_ptr<float>();
  for (int64_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
    factor_ptrs[static_cast<size_t>(batch_idx)] = factor_base + batch_idx * matrix_stride;
    tau_ptrs[static_cast<size_t>(batch_idx)] = tau_base + batch_idx * tau_stride;
  }

  ScopedCublasHandle handle;
  CUBLAS_CHECK(cublasSetStream(handle.get(), stream.stream()));
  CUBLAS_CHECK(
      cublasSgeqrfBatched(
          handle.get(),
          row_count,
          col_count,
          factor_ptrs.data(),
          row_count,
          tau_ptrs.data(),
          info.data_ptr<int>(),
          static_cast<int>(batch_count)));

  auto info_cpu = info.cpu();
  const int* info_values = info_cpu.data_ptr<int>();
  for (int64_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
    TORCH_CHECK(info_values[batch_idx] == 0, "cuBLAS batched QR failed with info=", info_values[batch_idx]);
  }

  return {
      factors_col_major.transpose(1, 2).contiguous(),
      tau,
  };
}

torch::Tensor _expand_local_qr_explicit_q(
    const torch::Tensor& compact_factors,
    const torch::Tensor& tau) {
  c10::cuda::CUDAGuard device_guard(compact_factors.device());
  const auto batch_count = compact_factors.size(0);
  const auto row_count = static_cast<int>(compact_factors.size(1));
  const auto col_count = static_cast<int>(compact_factors.size(2));
  auto factors_col_major = compact_factors.transpose(1, 2).contiguous();
  auto tau_contiguous = tau.contiguous();
  auto info = torch::zeros(
      {batch_count},
      torch::TensorOptions().device(compact_factors.device()).dtype(torch::kInt));
  auto workspace_probe = torch::empty(
      {std::max(row_count, col_count)},
      torch::TensorOptions().device(compact_factors.device()).dtype(torch::kFloat));
  int lwork_orgqr = 0;
  ScopedCusolverDnHandle handle;
  const auto stream = at::cuda::getDefaultCUDAStream();
  CUSOLVER_CHECK(cusolverDnSetStream(handle.get(), stream.stream()));
  CUSOLVER_CHECK(
      cusolverDnSorgqr_bufferSize(
          handle.get(),
          row_count,
          col_count,
          col_count,
          workspace_probe.data_ptr<float>(),
          row_count,
          workspace_probe.data_ptr<float>(),
          &lwork_orgqr));
  auto workspace = torch::empty(
      {static_cast<int64_t>(lwork_orgqr)},
      torch::TensorOptions().device(compact_factors.device()).dtype(torch::kFloat));
  auto* factor_ptr = factors_col_major.data_ptr<float>();
  auto* tau_ptr = tau_contiguous.data_ptr<float>();
  auto* info_ptr = info.data_ptr<int>();
  auto* workspace_ptr = workspace.data_ptr<float>();
  const auto matrix_stride = static_cast<int64_t>(row_count) * static_cast<int64_t>(col_count);
  const auto tau_stride = static_cast<int64_t>(std::min(row_count, col_count));
  for (int64_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
    float* factor_matrix = factor_ptr + batch_idx * matrix_stride;
    float* tau_matrix = tau_ptr + batch_idx * tau_stride;
    int* info_matrix = info_ptr + batch_idx;
    CUSOLVER_CHECK(
        cusolverDnSorgqr(
            handle.get(),
            row_count,
            col_count,
            col_count,
            factor_matrix,
            row_count,
            tau_matrix,
            workspace_ptr,
            lwork_orgqr,
            info_matrix));
  }
  auto info_cpu = info.cpu();
  const int* info_values = info_cpu.data_ptr<int>();
  for (int64_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
    TORCH_CHECK(info_values[batch_idx] == 0, "cuSOLVER dense ORGQR failed with info=", info_values[batch_idx]);
  }
  return factors_col_major.transpose(1, 2).contiguous();
}

torch::Tensor _launch_local_qr_forward(const torch::Tensor& candidate_matrices, double /*eps_sign*/) {
  const auto compact = _launch_local_qr_compact_forward(candidate_matrices);
  return _expand_local_qr_explicit_q(compact.factors_row_major, compact.tau);
}

__global__ void gfst_local_qr_fixed_order_compact_kernel(
    const float* input,
    float* compact_factors,
    float* tau,
    int* info,
    int64_t batch_count,
    int row_count,
    int col_count) {
  const int64_t batch_idx = static_cast<int64_t>(blockIdx.x);
  if (batch_idx >= batch_count || threadIdx.x != 0) {
    return;
  }
  const int tau_count = row_count < col_count ? row_count : col_count;
  const int64_t matrix_stride = static_cast<int64_t>(row_count) * static_cast<int64_t>(col_count);
  const int64_t tau_stride = static_cast<int64_t>(tau_count);
  const float* input_matrix = input + batch_idx * matrix_stride;
  float* factor_matrix = compact_factors + batch_idx * matrix_stride;
  float* tau_matrix = tau + batch_idx * tau_stride;
  for (int idx = 0; idx < row_count * col_count; ++idx) {
    factor_matrix[idx] = input_matrix[idx];
  }
  for (int k = 0; k < tau_count; ++k) {
    float norm_sq = 0.0f;
    for (int row = k; row < row_count; ++row) {
      const float value = factor_matrix[row * col_count + k];
      norm_sq += value * value;
    }
    const float norm = sqrtf(norm_sq);
    if (norm == 0.0f) {
      tau_matrix[k] = 0.0f;
      continue;
    }
    const float alpha = factor_matrix[k * col_count + k];
    const float sign = alpha < 0.0f ? -1.0f : 1.0f;
    const float beta = -sign * norm;
    const float denom = alpha - beta;
    if (fabsf(denom) <= 0.0f) {
      tau_matrix[k] = 0.0f;
      continue;
    }
    const float tau_value = (beta - alpha) / beta;
    factor_matrix[k * col_count + k] = beta;
    for (int row = k + 1; row < row_count; ++row) {
      factor_matrix[row * col_count + k] /= denom;
    }
    for (int col = k + 1; col < col_count; ++col) {
      float dot = factor_matrix[k * col_count + col];
      for (int row = k + 1; row < row_count; ++row) {
        dot += factor_matrix[row * col_count + k] * factor_matrix[row * col_count + col];
      }
      const float weight = tau_value * dot;
      factor_matrix[k * col_count + col] -= weight;
      for (int row = k + 1; row < row_count; ++row) {
        factor_matrix[row * col_count + col] -= factor_matrix[row * col_count + k] * weight;
      }
    }
    tau_matrix[k] = tau_value;
  }
  info[batch_idx] = 0;
}

__global__ void gfst_local_qr_fixed_order_explicit_q_kernel(
    const float* compact_factors,
    const float* tau,
    float* explicit_q,
    int* info,
    int64_t batch_count,
    int row_count,
    int col_count) {
  const int64_t batch_idx = static_cast<int64_t>(blockIdx.x);
  if (batch_idx >= batch_count || threadIdx.x != 0) {
    return;
  }
  const int tau_count = row_count < col_count ? row_count : col_count;
  const int64_t matrix_stride = static_cast<int64_t>(row_count) * static_cast<int64_t>(col_count);
  const int64_t tau_stride = static_cast<int64_t>(tau_count);
  const float* factor_matrix = compact_factors + batch_idx * matrix_stride;
  const float* tau_matrix = tau + batch_idx * tau_stride;
  float* q_matrix = explicit_q + batch_idx * matrix_stride;
  for (int row = 0; row < row_count; ++row) {
    for (int col = 0; col < col_count; ++col) {
      q_matrix[row * col_count + col] = row == col ? 1.0f : 0.0f;
    }
  }
  for (int k = tau_count - 1; k >= 0; --k) {
    const float tau_value = tau_matrix[k];
    if (tau_value == 0.0f) {
      continue;
    }
    for (int col = 0; col < col_count; ++col) {
      float dot = q_matrix[k * col_count + col];
      for (int row = k + 1; row < row_count; ++row) {
        dot += factor_matrix[row * col_count + k] * q_matrix[row * col_count + col];
      }
      const float weight = tau_value * dot;
      q_matrix[k * col_count + col] -= weight;
      for (int row = k + 1; row < row_count; ++row) {
        q_matrix[row * col_count + col] -= factor_matrix[row * col_count + k] * weight;
      }
    }
  }
  info[batch_idx] = 0;
}

LocalQrCompact _launch_local_qr_compact_fixed_order_forward(const torch::Tensor& candidate_matrices) {
  c10::cuda::CUDAGuard device_guard(candidate_matrices.device());
  const auto batch_count = candidate_matrices.size(0);
  const auto row_count = static_cast<int>(candidate_matrices.size(1));
  const auto col_count = static_cast<int>(candidate_matrices.size(2));
  const auto tau_count = std::min(row_count, col_count);
  auto compact_factors = torch::empty_like(candidate_matrices);
  auto tau = torch::empty(
      {batch_count, static_cast<int64_t>(tau_count)},
      torch::TensorOptions().device(candidate_matrices.device()).dtype(torch::kFloat));
  auto info = torch::zeros(
      {batch_count},
      torch::TensorOptions().device(candidate_matrices.device()).dtype(torch::kInt));
  const auto stream = at::cuda::getDefaultCUDAStream();
  gfst_local_qr_fixed_order_compact_kernel<<<static_cast<unsigned int>(batch_count), 1, 0, stream>>>(
      candidate_matrices.data_ptr<float>(),
      compact_factors.data_ptr<float>(),
      tau.data_ptr<float>(),
      info.data_ptr<int>(),
      batch_count,
      row_count,
      col_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto info_cpu = info.cpu();
  const int* info_values = info_cpu.data_ptr<int>();
  for (int64_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
    TORCH_CHECK(info_values[batch_idx] == 0, "fixed-order QR failed with info=", info_values[batch_idx]);
  }

  return {compact_factors, tau};
}

torch::Tensor _expand_local_qr_explicit_q_fixed_order(
    const torch::Tensor& compact_factors,
    const torch::Tensor& tau) {
  c10::cuda::CUDAGuard device_guard(compact_factors.device());
  const auto batch_count = compact_factors.size(0);
  const auto row_count = static_cast<int>(compact_factors.size(1));
  const auto col_count = static_cast<int>(compact_factors.size(2));
  auto explicit_q = torch::empty_like(compact_factors);
  auto info = torch::zeros(
      {batch_count},
      torch::TensorOptions().device(compact_factors.device()).dtype(torch::kInt));
  const auto stream = at::cuda::getDefaultCUDAStream();
  gfst_local_qr_fixed_order_explicit_q_kernel<<<static_cast<unsigned int>(batch_count), 1, 0, stream>>>(
      compact_factors.data_ptr<float>(),
      tau.data_ptr<float>(),
      explicit_q.data_ptr<float>(),
      info.data_ptr<int>(),
      batch_count,
      row_count,
      col_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto info_cpu = info.cpu();
  const int* info_values = info_cpu.data_ptr<int>();
  for (int64_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
    TORCH_CHECK(info_values[batch_idx] == 0, "fixed-order explicit Q failed with info=", info_values[batch_idx]);
  }
  return explicit_q;
}

torch::Tensor _launch_local_qr_fixed_order_forward(const torch::Tensor& candidate_matrices) {
  const auto compact = _launch_local_qr_compact_fixed_order_forward(candidate_matrices);
  return _expand_local_qr_explicit_q_fixed_order(compact.factors_row_major, compact.tau);
}

template<typename Scalar, int M>
std::tuple<torch::Tensor, torch::Tensor> _launch_heev_forward(const torch::Tensor& matrices) {
  c10::cuda::CUDAGuard device_guard(matrices.device());
  const bool single_matrix = matrices.dim() == 2;
  auto batched = single_matrix ? matrices.unsqueeze(0) : matrices;
  auto eigenvectors = torch::empty_like(batched);
  auto eigenvalues = torch::empty(
      {batched.size(0), static_cast<int64_t>(M)},
      torch::TensorOptions().device(batched.device()).dtype(batched.scalar_type()));
  auto info = torch::zeros(
      {batched.size(0)},
      torch::TensorOptions().device(batched.device()).dtype(torch::kInt));
  const auto stream = at::cuda::getDefaultCUDAStream();

  CUDA_CHECK(cudaFuncSetAttribute(
      gfst_heev_kernel<Scalar, M>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      HeevTraits<Scalar, M>::shared_memory_size));
  gfst_heev_kernel<Scalar, M><<<static_cast<unsigned int>(batched.size(0)), HeevTraits<Scalar, M>::block_dim, HeevTraits<Scalar, M>::shared_memory_size, stream>>>(
      batched.data_ptr<Scalar>(),
      eigenvectors.data_ptr<Scalar>(),
      eigenvalues.data_ptr<typename HeevTraits<Scalar, M>::precision_type>(),
      info.data_ptr<int>(),
      batched.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto info_cpu = info.cpu();
  const int* info_ptr = info_cpu.data_ptr<int>();
  for (int64_t batch_idx = 0; batch_idx < info_cpu.numel(); ++batch_idx) {
    TORCH_CHECK(info_ptr[batch_idx] == 0, "cuSOLVERDx HEEV failed with info=", info_ptr[batch_idx]);
  }

  if (single_matrix) {
    return std::make_tuple(eigenvalues.squeeze(0), eigenvectors.squeeze(0));
  }
  return std::make_tuple(eigenvalues, eigenvectors);
}

}  // namespace

torch::Tensor mathdx_probe_forward() {
  auto out = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
  gfst_hmb_mathdx_probe_kernel<<<1, 1>>>(out.data_ptr<int64_t>());
  return out;
}

torch::Tensor gfst_local_qr_sm120_forward(
    const torch::Tensor& candidate_matrices,
    double eps_sign) {
  _check_cuda_float_tensor(candidate_matrices, "candidate_matrices", 3);
  TORCH_CHECK(candidate_matrices.size(1) >= candidate_matrices.size(2), "candidate_matrices must be tall or square");
  return _launch_local_qr_forward(candidate_matrices, eps_sign);
}

std::tuple<torch::Tensor, torch::Tensor> gfst_local_qr_sm120_compact_debug_forward(
    const torch::Tensor& candidate_matrices) {
  _check_cuda_float_tensor(candidate_matrices, "candidate_matrices", 3);
  TORCH_CHECK(candidate_matrices.size(1) >= candidate_matrices.size(2), "candidate_matrices must be tall or square");
  const auto compact = _launch_local_qr_compact_forward(candidate_matrices);
  return std::make_tuple(compact.factors_row_major, compact.tau);
}

std::tuple<torch::Tensor, torch::Tensor> gfst_local_qr_sm120_cublas_batched_compact_debug_forward(
    const torch::Tensor& candidate_matrices) {
  _check_cuda_float_tensor(candidate_matrices, "candidate_matrices", 3);
  TORCH_CHECK(candidate_matrices.size(1) >= candidate_matrices.size(2), "candidate_matrices must be tall or square");
  const auto compact = _launch_local_qr_compact_cublas_batched_forward(candidate_matrices);
  return std::make_tuple(compact.factors_row_major, compact.tau);
}

torch::Tensor gfst_local_qr_sm120_explicit_q_debug_forward(
    const torch::Tensor& compact_factors,
    const torch::Tensor& tau) {
  _check_cuda_float_tensor(compact_factors, "compact_factors", 3);
  _check_cuda_float_tensor(tau, "tau", 2);
  TORCH_CHECK(
      compact_factors.size(0) == tau.size(0),
      "compact_factors and tau must have matching batch dimension");
  TORCH_CHECK(
      tau.size(1) == std::min(compact_factors.size(1), compact_factors.size(2)),
      "tau has incorrect reflector dimension");
  TORCH_CHECK(compact_factors.size(1) >= compact_factors.size(2), "compact_factors must be tall or square");
  return _expand_local_qr_explicit_q(compact_factors, tau);
}

torch::Tensor gfst_local_qr_sm120_fixed_order_forward(
    const torch::Tensor& candidate_matrices) {
  _check_cuda_float_tensor(candidate_matrices, "candidate_matrices", 3);
  TORCH_CHECK(candidate_matrices.size(1) >= candidate_matrices.size(2), "candidate_matrices must be tall or square");
  return _launch_local_qr_fixed_order_forward(candidate_matrices);
}

std::tuple<torch::Tensor, torch::Tensor> gfst_local_qr_sm120_fixed_order_compact_debug_forward(
    const torch::Tensor& candidate_matrices) {
  _check_cuda_float_tensor(candidate_matrices, "candidate_matrices", 3);
  TORCH_CHECK(candidate_matrices.size(1) >= candidate_matrices.size(2), "candidate_matrices must be tall or square");
  const auto compact = _launch_local_qr_compact_fixed_order_forward(candidate_matrices);
  return std::make_tuple(compact.factors_row_major, compact.tau);
}

torch::Tensor gfst_global_sketch_accum_forward(
    const torch::Tensor& summary_projectors,
    const torch::Tensor& final_indices,
    const torch::Tensor& weights) {
  _check_cuda_float_tensor(summary_projectors, "summary_projectors", 3);
  _check_cuda_long_tensor(final_indices, "final_indices", 1);
  _check_cuda_float_tensor(weights, "weights", 1);
  TORCH_CHECK(summary_projectors.size(1) == summary_projectors.size(2), "summary_projectors must contain square projectors");
  TORCH_CHECK(final_indices.size(0) == weights.size(0), "final_indices and weights must have the same length");
  TORCH_CHECK(summary_projectors.size(0) > 0, "summary_projectors must not be empty");

  c10::cuda::CUDAGuard device_guard(summary_projectors.device());
  const auto projector_dim = summary_projectors.size(1);
  const auto projector_stride = projector_dim * projector_dim;
  auto output = torch::zeros(
      {projector_dim, projector_dim},
      torch::TensorOptions().device(summary_projectors.device()).dtype(torch::kFloat));
  const auto element_count = output.numel();
  const auto blocks = _block_count_for(element_count);
  gfst_global_sketch_accum_kernel<<<blocks, kThreadsPerBlock, 0, at::cuda::getDefaultCUDAStream()>>>(
      summary_projectors.data_ptr<float>(),
      final_indices.data_ptr<int64_t>(),
      weights.data_ptr<float>(),
      output.data_ptr<float>(),
      final_indices.size(0),
      projector_stride,
      element_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> gfst_heev_sm120_forward(const torch::Tensor& matrix) {
  _check_cuda_real_matrix_tensor(matrix, "matrix");
  const auto dim = static_cast<int>(matrix.size(-1));
  if (matrix.scalar_type() == at::kFloat) {
    if (dim == 4) {
      return _launch_heev_forward<float, 4>(matrix);
    }
    if (dim == 16) {
      return _launch_heev_forward<float, 16>(matrix);
    }
    if (dim == 32) {
      return _launch_heev_forward<float, 32>(matrix);
    }
    if (dim == 64) {
      return _launch_heev_forward<float, 64>(matrix);
    }
  } else if (matrix.scalar_type() == at::kDouble) {
    if (dim == 4) {
      return _launch_heev_forward<double, 4>(matrix);
    }
    if (dim == 16) {
      return _launch_heev_forward<double, 16>(matrix);
    }
    if (dim == 32) {
      return _launch_heev_forward<double, 32>(matrix);
    }
    if (dim == 64) {
      return _launch_heev_forward<double, 64>(matrix);
    }
  }
  TORCH_CHECK(false, "unsupported GFST HEEV dimension ", dim);
}

std::tuple<torch::Tensor, torch::Tensor> gfst_block_average_forward(
    const torch::Tensor& projector_sum,
    const torch::Tensor& reduced_sum,
    int64_t token_count) {
  _check_cuda_float_tensor(projector_sum, "projector_sum", 2);
  _check_cuda_float_tensor(reduced_sum, "reduced_sum", 1);
  TORCH_CHECK(projector_sum.size(0) == projector_sum.size(1), "projector_sum must be square");
  TORCH_CHECK(projector_sum.size(0) == reduced_sum.size(0), "projector_sum and reduced_sum must share the reduced dimension");
  TORCH_CHECK(token_count > 0, "token_count must be positive");

  c10::cuda::CUDAGuard device_guard(projector_sum.device());
  const float scale = 1.0f / static_cast<float>(token_count);
  auto average_projector = torch::empty_like(projector_sum);
  auto reduced_mean = torch::empty_like(reduced_sum);

  const auto matrix_elements = average_projector.numel();
  const auto vector_elements = reduced_mean.numel();
  const auto stream = at::cuda::getDefaultCUDAStream();
  scale_tensor_kernel<<<_block_count_for(matrix_elements), kThreadsPerBlock, 0, stream>>>(
      projector_sum.data_ptr<float>(),
      average_projector.data_ptr<float>(),
      scale,
      matrix_elements);
  scale_tensor_kernel<<<_block_count_for(vector_elements), kThreadsPerBlock, 0, stream>>>(
      reduced_sum.data_ptr<float>(),
      reduced_mean.data_ptr<float>(),
      scale,
      vector_elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(average_projector, reduced_mean);
}
