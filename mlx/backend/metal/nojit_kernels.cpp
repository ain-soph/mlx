// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

#ifdef MLX_METAL_NO_NAX
namespace metal {
const char* gemm_nax() {
  return "";
}
const char* steel_gemm_fused_nax() {
  return "";
}
const char* steel_gemm_splitk_nax() {
  return "";
}
} // namespace metal
#endif // MLX_METAL_NO_NAX

MTL::ComputePipelineState* get_arange_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_unary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype,
    Dtype,
    const char*) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_binary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype,
    Dtype,
    const char*) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_binary_two_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype,
    Dtype,
    const char*) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_ternary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype,
    const char*) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_copy_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_dynamic_copy_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_softmax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_logsumexp_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_cross_entropy_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_scan_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool,
    bool,
    const std::string&,
    const array&,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_sort_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&,
    int,
    int) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_searchsorted_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_mb_sort_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&,
    int,
    int) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_reduce_init_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string&,
    const std::string&,
    const Dtype&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_reduce_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string&,
    const std::string&,
    const Dtype&,
    const Dtype&,
    const std::string&,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_steel_gemm_fused_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_fused_mixed_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& a,
    const array& b,
    const array& out,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm()
                  << metal::steel_gemm_fused()
                  << get_template_definition(
                         lib_name,
                         "gemm",
                         get_type_string(out.dtype()),
                         bm,
                         bn,
                         bk,
                         wm,
                         wn,
                         transpose_a,
                         transpose_b,
                         "float",
                         get_type_string(a.dtype()),
                         get_type_string(b.dtype()));
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int,
    bool,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_mixed_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& a,
    const array& b,
    const array& out,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool mn_aligned,
    bool k_aligned) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm()
                  << metal::steel_gemm_splitk()
                  << get_template_definition(
                         lib_name,
                         "gemm_splitk",
                         get_type_string(out.dtype()),
                         get_type_string(out.dtype()),
                         bm,
                         bn,
                         bk,
                         wm,
                         wn,
                         transpose_a,
                         transpose_b,
                         mn_aligned,
                         k_aligned,
                         get_type_string(a.dtype()),
                         get_type_string(b.dtype()));
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_accum_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_steel_gemm_masked_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const std::optional<array>& mask_out,
    const std::optional<array>& mask_op,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int,
    bool,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_steel_gemm_gather_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int,
    bool) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_segmented_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_gemv_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    bool,
    int,
    int,
    int,
    int,
    int,
    int,
    bool,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_gemv_mixed_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& mat,
    const array& vec,
    const array& out,
    bool transpose_mat,
    int bm,
    int bn,
    int sm,
    int sn,
    int tm,
    int tn,
    bool nc,
    bool axpby) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::gemv()
                  << get_template_definition(
                         lib_name,
                         transpose_mat ? "gemv_t" : "gemv",
                         get_type_string(out.dtype()),
                         bm,
                         bn,
                         sm,
                         sn,
                         tm,
                         tn,
                         nc ? 1 : 0,
                         axpby ? 1 : 0,
                         get_type_string(mat.dtype()),
                         get_type_string(vec.dtype()));
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_gemv_gather_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    bool,
    int,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_gemv_masked_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const std::optional<array>&,
    const std::optional<array>&,
    bool,
    int,
    int,
    int,
    int,
    int,
    int,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_gemv_wide_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_gemv_wide_mixed_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& mat,
    const array& vec,
    const array& out,
    int vecs_per_tg,
    int k_lanes) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::gemv()
                  << get_template_definition(
                         lib_name,
                         "gemv_wide",
                         get_type_string(out.dtype()),
                         vecs_per_tg,
                         k_lanes,
                         get_type_string(mat.dtype()),
                         get_type_string(vec.dtype()));
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_gemv_wide_gather_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_conv_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    int,
    int,
    int,
    int,
    int,
    int,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_steel_conv_3d_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    int,
    int,
    int,
    int,
    int,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_steel_conv_general_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_fft_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const std::string&) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_fft_twiddle_kernel(
    metal::Device& d,
    const std::string&,
    const std::string&) {
  return d.get_kernel("generate_bluestein_twiddles");
}

MTL::ComputePipelineState* get_quantized_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string&,
    const std::string&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_gather_qmm_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    const std::string&,
    int,
    int,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_fused_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_fused_nax_mixed_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& a,
    const array& b,
    const array& out,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool relaxed) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm_nax()
                  << metal::steel_gemm_fused_nax()
                  << get_template_definition(
                         lib_name,
                         "gemm",
                         get_type_string(a.dtype()),
                         bm,
                         bn,
                         bk,
                         wm,
                         wn,
                         transpose_a,
                         transpose_b,
                         "float",
                         get_type_string(b.dtype()),
                         get_type_string(out.dtype()),
                         relaxed ? "true" : "false");
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_gather_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int,
    bool) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_nax_mixed_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& a,
    const array& b,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool relaxed) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm_nax()
                  << metal::steel_gemm_splitk_nax()
                  << get_template_definition(
                         lib_name,
                         "gemm_splitk_nax",
                         get_type_string(a.dtype()),
                         bm,
                         bn,
                         bk,
                         wm,
                         wn,
                         transpose_a,
                         transpose_b,
                         "float",
                         get_type_string(b.dtype()),
                         relaxed ? "true" : "false");
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_segmented_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_thin_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const std::string&) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_qmm_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string&,
    const std::string&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_gather_qmm_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    const std::string&,
    int,
    int,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_attention_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    int,
    int,
    int,
    int,
    int,
    int,
    const array&) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_attention_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    int,
    int,
    int,
    int,
    int,
    int,
    const array&,
    bool) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_attention_split_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_attention_split_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_gated_delta_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_gated_delta_vjp_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}
MTL::ComputePipelineState* get_gated_delta_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    int,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

MTL::ComputePipelineState* get_gated_delta_vjp_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    int,
    int,
    int,
    int,
    int,
    int,
    bool) {
  return d.get_kernel(kernel_name, hash_name, func_consts);
}

} // namespace mlx::core
