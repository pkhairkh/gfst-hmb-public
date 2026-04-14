#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

namespace {

template <typename T>
T* ptr_cast(void* ptr) {
  return reinterpret_cast<T*>(ptr);
}

template <typename T>
T const* ptr_cast(void const* ptr) {
  return reinterpret_cast<T const*>(ptr);
}

void check_inputs(
    torch::Tensor const& token_features,
    torch::Tensor const& within_block_weight) {
  TORCH_CHECK(token_features.is_cuda(), "exact_block_score_forward expects CUDA token_features");
  TORCH_CHECK(within_block_weight.is_cuda(), "exact_block_project_forward expects CUDA within_block_weight");
  TORCH_CHECK(token_features.scalar_type() == torch::kBFloat16, "exact_block_score_forward expects BF16 token_features");
  TORCH_CHECK(within_block_weight.scalar_type() == torch::kBFloat16, "exact_block_project_forward expects BF16 within_block_weight");
  TORCH_CHECK(token_features.dim() == 2, "exact_block_score_forward expects token_features to be rank 2");
  TORCH_CHECK(within_block_weight.dim() == 2, "exact_block_project_forward expects within_block_weight to be rank 2");
  TORCH_CHECK(token_features.size(1) == within_block_weight.size(1), "exact_block_project_forward dimension mismatch");
  TORCH_CHECK(token_features.is_contiguous(), "exact_block_score_forward expects contiguous token_features");
  TORCH_CHECK(within_block_weight.is_contiguous(), "exact_block_project_forward expects contiguous within_block_weight");
}

}  // namespace

torch::Tensor exact_block_project_forward(
    const torch::Tensor& token_features,
    const torch::Tensor& within_block_weight) {
  check_inputs(token_features, within_block_weight);

  c10::cuda::CUDAGuard device_guard(token_features.device());
  auto current_stream = at::cuda::getCurrentCUDAStream(token_features.get_device()).stream();

  const int64_t m = token_features.size(0);
  const int64_t k = token_features.size(1);
  const int64_t n = within_block_weight.size(0);
  if (m == 0 || k == 0 || n == 0) {
    return torch::zeros({m, n}, token_features.options().dtype(torch::kBFloat16));
  }

  using ElementInput = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  // CUTLASS's native SM120 builder currently rejects BF16 dense GEMM, so we
  // use the mature BF16 TensorOp path and compile it for SM120 targets.
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      8,
      ElementAccumulator,
      ElementAccumulator>;

  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int Stages = 3;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInput,
      LayoutA,
      ElementInput,
      LayoutB,
      ElementOutput,
      LayoutC,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Stages,
      AlignmentA,
      AlignmentB>;

  TORCH_CHECK((n % 8) == 0, "exact_block_project_forward requires key_dim aligned to 8 elements");

  auto source = torch::zeros({m, n}, token_features.options().dtype(torch::kBFloat16));
  auto output = torch::zeros_like(source);

  typename Gemm::Arguments arguments{
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)},
      {ptr_cast<ElementInput const>(token_features.data_ptr()), static_cast<int>(k)},
      {ptr_cast<ElementInput const>(within_block_weight.data_ptr()), static_cast<int>(k)},
      {ptr_cast<ElementOutput const>(source.data_ptr()), static_cast<int>(n)},
      {ptr_cast<ElementOutput>(output.data_ptr()), static_cast<int>(n)},
      {1.0f, 0.0f},
  };

  auto const workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::zeros(
      {static_cast<int64_t>(workspace_size)},
      token_features.options().dtype(torch::kUInt8));
  void* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;

  Gemm gemm_op;
  auto const can_implement = gemm_op.can_implement(arguments);
  TORCH_CHECK(can_implement == cutlass::Status::kSuccess, "CUTLASS exact_block_score can_implement failed");

  auto const status = gemm_op(arguments, workspace_ptr, current_stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS exact_block_score launch failed");

  auto const last_error = cudaGetLastError();
  TORCH_CHECK(last_error == cudaSuccess, "CUTLASS exact_block_score CUDA error: ", cudaGetErrorString(last_error));

  return output;
}
