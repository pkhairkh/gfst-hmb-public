#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <math_constants.h>
#include <cuda_runtime.h>

#include <limits>
#include <tuple>

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void segmented_max_kernel(
    const float* scores,
    const int64_t* inverse,
    float* grouped,
    int64_t element_count) {
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x);
  __shared__ float partial[kThreadsPerBlock];
  float local_max = -CUDART_INF_F;
  for (int64_t idx = threadIdx.x; idx < element_count; idx += blockDim.x) {
    if (inverse[idx] == group_idx) {
      local_max = fmaxf(local_max, scores[idx]);
    }
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    grouped[group_idx] = partial[0];
  }
}

__global__ void segmented_logsumexp_occurrence_kernel(
    const float* scores,
    const int64_t* inverse,
    const int64_t* occurrence_blocks,
    const int64_t* occurrence_positions,
    float* grouped_scores,
    int64_t* grouped_blocks,
    int64_t* grouped_positions,
    int64_t element_count) {
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x);
  __shared__ float max_partial[kThreadsPerBlock];
  __shared__ float sum_partial[kThreadsPerBlock];
  __shared__ int64_t best_partial[kThreadsPerBlock];

  float local_max = -CUDART_INF_F;
  for (int64_t idx = threadIdx.x; idx < element_count; idx += blockDim.x) {
    if (inverse[idx] == group_idx) {
      local_max = fmaxf(local_max, scores[idx]);
    }
  }
  max_partial[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      max_partial[threadIdx.x] = fmaxf(max_partial[threadIdx.x], max_partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float group_max = max_partial[0];

  float local_sum = 0.0f;
  int64_t local_best = element_count;
  if (isfinite(group_max)) {
    for (int64_t idx = threadIdx.x; idx < element_count; idx += blockDim.x) {
      if (inverse[idx] != group_idx) {
        continue;
      }
      const float value = scores[idx];
      local_sum += expf(value - group_max);
      if (value == group_max && idx < local_best) {
        local_best = idx;
      }
    }
  }
  sum_partial[threadIdx.x] = local_sum;
  best_partial[threadIdx.x] = local_best;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_partial[threadIdx.x] += sum_partial[threadIdx.x + stride];
      const int64_t rhs_best = best_partial[threadIdx.x + stride];
      if (rhs_best < best_partial[threadIdx.x]) {
        best_partial[threadIdx.x] = rhs_best;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    grouped_scores[group_idx] = group_max + logf(fmaxf(sum_partial[0], 1e-8f));
    const int64_t best_index = best_partial[0];
    if (best_index >= 0 && best_index < element_count) {
      grouped_blocks[group_idx] = occurrence_blocks[best_index];
      grouped_positions[group_idx] = occurrence_positions[best_index];
    } else {
      grouped_blocks[group_idx] = -1;
      grouped_positions[group_idx] = -1;
    }
  }
}

__global__ void materialize_bytes_kernel(
    const unsigned char* packed,
    unsigned char* output,
    int64_t start_byte,
    int64_t byte_count) {
  const int64_t linear_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = linear_index; idx < byte_count; idx += stride) {
    output[idx] = packed[start_byte + idx];
  }
}

int _grid_dim_for(int64_t element_count) {
  if (element_count <= 0) {
    return 1;
  }
  const int64_t raw = (element_count + static_cast<int64_t>(kThreadsPerBlock) - 1) /
                      static_cast<int64_t>(kThreadsPerBlock);
  return static_cast<int>(raw > 4096 ? 4096 : raw);
}

void _check_cuda_float_vector(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must have dtype float32");
  TORCH_CHECK(tensor.dim() == 1, name, " must have rank 1");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void _check_cuda_long_vector(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must have dtype int64");
  TORCH_CHECK(tensor.dim() == 1, name, " must have rank 1");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor segmented_max_forward(
    const torch::Tensor& scores,
    const torch::Tensor& inverse,
    int64_t group_count) {
  _check_cuda_float_vector(scores, "scores");
  _check_cuda_long_vector(inverse, "inverse");
  TORCH_CHECK(scores.numel() == inverse.numel(), "scores and inverse must have the same number of elements");
  TORCH_CHECK(group_count >= 0, "group_count must be nonnegative");

  c10::cuda::CUDAGuard device_guard(scores.device());
  auto grouped =
      torch::full({group_count}, -std::numeric_limits<float>::infinity(), scores.options().dtype(torch::kFloat32));
  if (group_count == 0) {
    return grouped;
  }
  segmented_max_kernel<<<static_cast<unsigned int>(group_count), kThreadsPerBlock, 0, at::cuda::getDefaultCUDAStream()>>>(
      scores.data_ptr<float>(),
      inverse.data_ptr<int64_t>(),
      grouped.data_ptr<float>(),
      scores.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grouped;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> segmented_logsumexp_occurrence_forward(
    const torch::Tensor& scores,
    const torch::Tensor& inverse,
    const torch::Tensor& occurrence_blocks,
    const torch::Tensor& occurrence_positions,
    int64_t group_count) {
  _check_cuda_float_vector(scores, "scores");
  _check_cuda_long_vector(inverse, "inverse");
  _check_cuda_long_vector(occurrence_blocks, "occurrence_blocks");
  _check_cuda_long_vector(occurrence_positions, "occurrence_positions");
  TORCH_CHECK(scores.numel() == inverse.numel(), "scores and inverse must have the same number of elements");
  TORCH_CHECK(scores.numel() == occurrence_blocks.numel(), "scores and occurrence_blocks must have the same number of elements");
  TORCH_CHECK(scores.numel() == occurrence_positions.numel(), "scores and occurrence_positions must have the same number of elements");
  TORCH_CHECK(group_count >= 0, "group_count must be nonnegative");

  c10::cuda::CUDAGuard device_guard(scores.device());
  auto grouped_scores =
      torch::full({group_count}, -std::numeric_limits<float>::infinity(), scores.options().dtype(torch::kFloat32));
  auto grouped_blocks = torch::full({group_count}, -1, inverse.options().dtype(torch::kLong));
  auto grouped_positions = torch::full({group_count}, -1, inverse.options().dtype(torch::kLong));
  if (group_count == 0) {
    return std::make_tuple(grouped_scores, grouped_blocks, grouped_positions);
  }
  segmented_logsumexp_occurrence_kernel<<<static_cast<unsigned int>(group_count), kThreadsPerBlock, 0, at::cuda::getDefaultCUDAStream()>>>(
      scores.data_ptr<float>(),
      inverse.data_ptr<int64_t>(),
      occurrence_blocks.data_ptr<int64_t>(),
      occurrence_positions.data_ptr<int64_t>(),
      grouped_scores.data_ptr<float>(),
      grouped_blocks.data_ptr<int64_t>(),
      grouped_positions.data_ptr<int64_t>(),
      scores.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(grouped_scores, grouped_blocks, grouped_positions);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mpuc_recurrent_scan_forward(
    const torch::Tensor& normalized,
    const torch::Tensor& fast_inputs,
    const torch::Tensor& mid_inputs,
    const torch::Tensor& slow_inputs) {
  auto batch = normalized.size(0);
  auto seq = normalized.size(1);
  auto hidden = normalized.size(2);
  auto outputs = torch::zeros({batch, seq, hidden}, normalized.options());
  auto boundary = torch::zeros({batch, seq}, normalized.options().dtype(torch::kFloat32));
  auto fast_state = torch::zeros({batch, fast_inputs.size(2)}, normalized.options());
  auto mid_state = torch::zeros({batch, mid_inputs.size(2)}, normalized.options());
  auto slow_state = torch::zeros({batch, slow_inputs.size(2)}, normalized.options());
  return std::make_tuple(outputs, boundary, fast_state, mid_state, slow_state);
}

torch::Tensor exact_block_materialize_forward(const torch::Tensor& packed, int64_t start, int64_t end) {
  TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
  TORCH_CHECK(packed.dim() == 1, "packed must be rank 1");
  TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
  TORCH_CHECK(start >= 0, "start must be nonnegative");
  TORCH_CHECK(end >= start, "end must be >= start");
  TORCH_CHECK(end <= packed.size(0), "end must be <= packed.size(0)");

  c10::cuda::CUDAGuard device_guard(packed.device());
  auto output = torch::empty({end - start}, packed.options());
  const int64_t itemsize = static_cast<int64_t>(packed.element_size());
  const int64_t byte_count = (end - start) * itemsize;
  const int64_t start_byte = start * itemsize;
  materialize_bytes_kernel<<<_grid_dim_for(byte_count), kThreadsPerBlock, 0, at::cuda::getDefaultCUDAStream()>>>(
      reinterpret_cast<const unsigned char*>(packed.data_ptr()),
      reinterpret_cast<unsigned char*>(output.data_ptr()),
      start_byte,
      byte_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
