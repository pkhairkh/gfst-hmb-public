#include <torch/extension.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

#ifndef GFST_HMB_MATHDX_PTXAS_OPT_LEVEL_STR
#define GFST_HMB_MATHDX_PTXAS_OPT_LEVEL_STR "DEFAULT"
#endif

#ifndef GFST_HMB_MATHDX_DEVICE_LTO_ENABLED
#define GFST_HMB_MATHDX_DEVICE_LTO_ENABLED 1
#endif

#ifndef GFST_HMB_MATHDX_IPO_ENABLED
#define GFST_HMB_MATHDX_IPO_ENABLED 1
#endif

torch::Tensor mathdx_probe_forward();
torch::Tensor gfst_local_qr_sm120_forward(
    const torch::Tensor& candidate_matrices,
    double eps_sign);
std::tuple<torch::Tensor, torch::Tensor> gfst_local_qr_sm120_compact_debug_forward(
    const torch::Tensor& candidate_matrices);
std::tuple<torch::Tensor, torch::Tensor> gfst_local_qr_sm120_cublas_batched_compact_debug_forward(
    const torch::Tensor& candidate_matrices);
torch::Tensor gfst_local_qr_sm120_explicit_q_debug_forward(
    const torch::Tensor& compact_factors,
    const torch::Tensor& tau);
torch::Tensor gfst_local_qr_sm120_fixed_order_forward(
    const torch::Tensor& candidate_matrices);
std::tuple<torch::Tensor, torch::Tensor> gfst_local_qr_sm120_fixed_order_compact_debug_forward(
    const torch::Tensor& candidate_matrices);
torch::Tensor gfst_global_sketch_accum_forward(
    const torch::Tensor& summary_projectors,
    const torch::Tensor& final_indices,
    const torch::Tensor& weights);
std::tuple<torch::Tensor, torch::Tensor> gfst_heev_sm120_forward(const torch::Tensor& matrix);
std::tuple<torch::Tensor, torch::Tensor> gfst_block_average_forward(
    const torch::Tensor& projector_sum,
    const torch::Tensor& reduced_sum,
    int64_t token_count);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gfst_canonical_summary_forward(
    const torch::Tensor& eigvals,
    const torch::Tensor& eigvecs,
    const torch::Tensor& reduced_tokens,
    const torch::Tensor& cutoffs,
    int64_t target_rank,
    double eps,
    double atol,
    double rtol);

py::dict dependency_report() {
    py::dict payload;
    payload["module"] = "gfst_hmb_mpuc_mathdx_ops";
    payload["cublasdx_header_compiled"] = true;
    payload["cusolverdx_header_compiled"] = true;
    payload["gfst_local_qr_sm120_compiled"] = true;
    payload["gfst_local_qr_sm120_compact_debug_compiled"] = true;
    payload["gfst_local_qr_sm120_cublas_batched_compact_debug_compiled"] = true;
    payload["gfst_local_qr_sm120_explicit_q_debug_compiled"] = true;
    payload["gfst_local_qr_sm120_fixed_order_compiled"] = true;
    payload["gfst_local_qr_sm120_fixed_order_compact_debug_compiled"] = true;
    payload["gfst_global_sketch_accum_forward_compiled"] = true;
    payload["gfst_heev_sm120_compiled"] = true;
    payload["gfst_block_average_forward_compiled"] = true;
    payload["gfst_canonical_summary_forward_compiled"] = true;
    payload["ptxas_opt_level"] = GFST_HMB_MATHDX_PTXAS_OPT_LEVEL_STR;
    payload["device_lto_enabled"] = static_cast<bool>(GFST_HMB_MATHDX_DEVICE_LTO_ENABLED);
    payload["ipo_enabled"] = static_cast<bool>(GFST_HMB_MATHDX_IPO_ENABLED);
    return payload;
}

PYBIND11_MODULE(gfst_hmb_mpuc_mathdx_ops, m) {
    m.def("dependency_report", &dependency_report, "MathDx dependency report");
    m.def("mathdx_probe_forward", &mathdx_probe_forward, "MathDx probe forward");
    m.def("gfst_local_qr_sm120_forward", &gfst_local_qr_sm120_forward, "GFST exact local QR forward");
    m.def(
        "gfst_local_qr_sm120_compact_debug_forward",
        &gfst_local_qr_sm120_compact_debug_forward,
        "TEMP_REMOVE_AFTER_SM120_EXACT_LOCAL_QR_PARITY GFST local QR compact debug forward");
    m.def(
        "gfst_local_qr_sm120_cublas_batched_compact_debug_forward",
        &gfst_local_qr_sm120_cublas_batched_compact_debug_forward,
        "TEMP_REMOVE_AFTER_SM120_EXACT_LOCAL_QR_PARITY GFST local QR cuBLAS batched compact debug forward");
    m.def(
        "gfst_local_qr_sm120_explicit_q_debug_forward",
        &gfst_local_qr_sm120_explicit_q_debug_forward,
        "TEMP_REMOVE_AFTER_SM120_EXACT_LOCAL_QR_PARITY GFST local QR explicit-Q debug forward");
    m.def(
        "gfst_local_qr_sm120_fixed_order_forward",
        &gfst_local_qr_sm120_fixed_order_forward,
        "TEMP_REMOVE_AFTER_SM120_EXACT_LOCAL_QR_PARITY GFST local QR fixed-order explicit-Q forward");
    m.def(
        "gfst_local_qr_sm120_fixed_order_compact_debug_forward",
        &gfst_local_qr_sm120_fixed_order_compact_debug_forward,
        "TEMP_REMOVE_AFTER_SM120_EXACT_LOCAL_QR_PARITY GFST local QR fixed-order compact debug forward");
    m.def("gfst_global_sketch_accum_forward", &gfst_global_sketch_accum_forward, "GFST exact global sketch accumulation");
    m.def("gfst_heev_sm120_forward", &gfst_heev_sm120_forward, "GFST exact symmetric eig forward");
    m.def("gfst_block_average_forward", &gfst_block_average_forward, "GFST exact block projector average");
    m.def("gfst_canonical_summary_forward", &gfst_canonical_summary_forward, "GFST exact canonical summary from eigpairs");
}
