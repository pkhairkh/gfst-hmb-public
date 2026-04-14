#include <torch/extension.h>

#include <tuple>

torch::Tensor segmented_max_forward(
    const torch::Tensor& scores,
    const torch::Tensor& inverse,
    int64_t group_count);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> segmented_logsumexp_occurrence_forward(
    const torch::Tensor& scores,
    const torch::Tensor& inverse,
    const torch::Tensor& occurrence_blocks,
    const torch::Tensor& occurrence_positions,
    int64_t group_count);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mpuc_recurrent_scan_forward(
    const torch::Tensor& normalized,
    const torch::Tensor& fast_inputs,
    const torch::Tensor& mid_inputs,
    const torch::Tensor& slow_inputs);

torch::Tensor exact_block_project_forward(
    const torch::Tensor& token_features,
    const torch::Tensor& within_block_weight);

torch::Tensor exact_block_materialize_forward(const torch::Tensor& packed, int64_t start, int64_t end);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("segmented_max_forward", &segmented_max_forward, "Segmented max forward");
  m.def(
      "segmented_logsumexp_occurrence_forward",
      &segmented_logsumexp_occurrence_forward,
      "Segmented logsumexp with occurrence forward");
  m.def("mpuc_recurrent_scan_forward", &mpuc_recurrent_scan_forward, "MPUC recurrent scan forward");
  m.def("exact_block_project_forward", &exact_block_project_forward, "Exact block project forward");
  m.def("exact_block_materialize_forward", &exact_block_materialize_forward, "Exact block materialize forward");
}
