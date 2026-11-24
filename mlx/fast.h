// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>
#include <variant>

#include "mlx/api.h"
#include "mlx/backend/common/metal_kernel.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

MLX_API array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s = {});

MLX_API array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

/** Fused cross entropy with class indices as targets. */
MLX_API array
cross_entropy(const array& logits, const array& targets, StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
MLX_API array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    bool force_fused = false,
    StreamOrDevice s = {});

MLX_API std::vector<array> gated_delta_update(
    const array& queries,
    const array& keys,
    const array& values,
    const array& gates,
    const array& beta_,
    const std::optional<array>& initial_state = std::nullopt,
    const std::optional<array>& mask = std::nullopt,
    StreamOrDevice s = {});

using TemplateArg = std::variant<int, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

using CustomKernelFunction = std::function<std::vector<array>(
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::vector<std::pair<std::string, TemplateArg>>,
    std::optional<float>,
    bool,
    StreamOrDevice)>;

MLX_API CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    bool atomic_outputs = false,
    const CompileOptions& compile_options = {});

MLX_API CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0);

MLX_API std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory = 0,
    std::optional<float> init_value = std::nullopt,
    bool ensure_row_contiguous = false,
    StreamOrDevice s = {});

/**
 * ConvRot W8A8 (the ComfyUI "int8_convrot" format): a weight W [N, K] is stored
 * as int8 rows of W H with one float32 scale per row, where H is block-diagonal
 * with 256x256 Hadamard blocks along K (K a multiple of 256). At run time the
 * activations get the same rotation and a per-row int8 quantization, so the
 * rotations cancel: y = q(x H) q(W H)^T * x_scale * w_scale ~ x W^T.
 */

/**
 * Rotate the rows of x [M, K] by H and quantize each row to int8. Returns
 * (int8 [M', K], float32 [M]) where M' is M rounded up to a multiple of 128:
 * the extra rows are unspecified and let the TensorOps GEMM read whole tiles.
 * Pass the pair to convrot_matmul_quantized to reuse one quantization for
 * several weights that read the same input (e.g. Q, K and V).
 */
MLX_API std::pair<array, array> convrot_quantize(
    const array& x,
    StreamOrDevice s = {});

/**
 * Quantize a float weight [N, K] to ConvRot int8 rows and float32 row scales.
 * Uses a float32 matrix product for the rotation (the reference math; on a GPU
 * with TensorOps that product follows MLX_ENABLE_TF32 like any float32 matmul).
 */
MLX_API std::pair<array, array> convrot_quantize_weight(
    const array& w,
    StreamOrDevice s = {});

/**
 * y = x W^T for a ConvRot int8 weight w [N, K] with float32 scales [N];
 * x [..., K] in any float type, y [..., N] in x's type. On a GPU with TensorOps
 * (M5) the activations are rotated and quantized in one kernel and the product
 * runs on the INT8 TensorOps path; elsewhere it is computed in float.
 */
MLX_API array convrot_matmul(
    const array& x,
    const array& w,
    const array& scales,
    StreamOrDevice s = {});

/**
 * As convrot_matmul, from a convrot_quantize result; y [M, N] in `dtype`.
 * tiles = (BM, BN, BK) overrides the GEMM tile shape (every tiling gives the
 * same bytes: the int32 sums are exact). BM must not exceed 128, the row
 * padding of convrot_quantize, and xq must have a multiple of BM rows.
 */
MLX_API array convrot_matmul_quantized(
    const array& xq,
    const array& xs,
    const array& w,
    const array& scales,
    Dtype dtype,
    std::optional<std::tuple<int, int, int>> tiles = std::nullopt,
    StreamOrDevice s = {});

/**
 * x H for x [..., K] (K a multiple of 256) in x's type: the ConvRot activation
 * rotation without quantization. On the GPU the rotation runs in float32 and
 * rounds once to x's type.
 */
MLX_API array convrot_rotate(const array& x, StreamOrDevice s = {});

/**
 * ConvRot W8A16: y = (x H) W^T for a ConvRot int8 weight w [N, K] with float32
 * scales [N], keeping the activations in x's floating point type (no int8
 * activation quantization; more accurate than convrot_matmul). x [..., K],
 * y [..., N] in x's type. rotated = true means x is already x H (from
 * convrot_rotate), e.g. to share one rotation between several weights.
 * On NAX GPUs (M5) with half / bfloat x the int8 weight rows go straight into
 * the NAX GEMM (converted in registers, no dequantized copy); elsewhere the
 * product is computed in float32. split_k: K partitions over the GPU grid
 * (float32 partial sums added in a fixed order); nullopt picks them by shape,
 * 1 keeps one pass over K (the float sums run in K order: the same bytes as a
 * sequential-K TensorOps GEMM).
 */
MLX_API array convrot_matmul_w8a16(
    const array& x,
    const array& w,
    const array& scales,
    bool rotated = false,
    std::optional<int> split_k = std::nullopt,
    StreamOrDevice s = {});

} // namespace mlx::core::fast
