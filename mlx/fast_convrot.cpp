// ConvRot W8A8 ("int8_convrot"): Hadamard-256 rotation + per-row int8
// weights and activations. The GEMM kernel is mlx-serve's (David Dalcu, MIT
// license, https://github.com/daviddalcu/mlx-serve) as used by h3-image-mlx,
// with the edge tiles kept inside their tensors.

#include <sstream>
#include <stdexcept>
#include <tuple>

#include "mlx/backend/metal/metal.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

constexpr int kGroup = 256;
constexpr int kRowPad = 128; // the largest GEMM tile height

// One threadgroup per row, one thread per position in a 256-wide group.
// H256 = H4 (x) H4 (x) H4 (x) H4 with H4 = ones - 2 * antidiagonal, so each
// base-4 digit is a 4-point butterfly. The row is read once, kept in
// registers, and written once as int8.
constexpr const char* kRotateQuantize = R"(
    const uint t = thread_position_in_threadgroup.x;
    const ushort lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint K = x_shape[1];
    const uint groups = K / 256;
    threadgroup float buf[256];
    threadgroup float red[8];
    float v[MAX_GROUPS];
    float amax = 0.0f;
    device const T* xr = x + (size_t)row * K;
    for (uint g = 0; g < groups; ++g) {
        float a = float(xr[g * 256 + t]);
        for (uint s = 1; s < 256; s *= 4) {
            buf[t] = a;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint k = (t / s) % 4, base = t - k * s;
            a = buf[base] + buf[base + s] + buf[base + 2 * s] + buf[base + 3 * s] - 2.0f * buf[base + (3 - k) * s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        a *= 0.0625f;
        v[g] = a;
        amax = max(amax, fabs(a));
    }
    amax = simd_max(amax);
    if (lane == 0) red[simdgroup_index_in_threadgroup] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = red[0];
    for (uint i = 1; i < 8; ++i) amax = max(amax, red[i]);
    const float scale = max(amax / 127.0f, 1e-20f);
    device int8_t* qr = xq + (size_t)row * K;
    for (uint g = 0; g < groups; ++g) qr[g * 256 + t] = int8_t(clamp(rint(v[g] / scale), -127.0f, 127.0f));
    if (t == 0) xs[row] = scale;
)";

constexpr const char* kGemmHeader = R"(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
)";

// xq has Mx >= M rows (padded to a multiple of BM), so every A slice lies
// inside the tensor. The last column tile is moved left to end at N, so every
// B slice lies inside wq (needs N >= BN); the epilogue stores only the columns
// from n_lo on.
constexpr const char* kGemmBody = R"(
const int K = xq_shape[1];
const int Mx = xq_shape[0];
const int M = xs_shape[0];
const int N = wq_shape[0];
const int m0 = int(threadgroup_position_in_grid.y) * BM;
const int n_lo = int(threadgroup_position_in_grid.x) * BN;
const int n0 = min(n_lo, N - BN);
constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, Mx));
tensor<device int8_t, dextents<int, 2>, tensor_inline> B((device int8_t*)wq, dextents<int, 2>(K, N));
auto a0 = A.template slice<BK, BM>(0, m0);
auto b0 = B.template slice<BK, BN>(0, n0);
auto cT = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0;
for (int k0 = 0; k0 < K; k0 += BK) {
  auto a = A.template slice<BK, BM>(k0, m0);
  auto b = B.template slice<BK, BN>(k0, n0);
  op.run(a, b, cT);
}
auto dT = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), T>();
for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
  auto ids = cT.get_multidimensional_index(i);
  const int n = min(n0 + ids[0], N - 1), m = min(m0 + ids[1], M - 1);
  dT[i] = T(float(cT[i]) * xs[m] * ws[n]);
}
if (n0 == n_lo && m0 + BM <= M) {
  tensor<device T, dextents<int, 2>, tensor_inline> D((device T*)y, dextents<int, 2>(N, M));
  auto dS = D.template slice<BN, BM>(n0, m0);
  dT.store(dS);
} else {
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    auto ids = cT.get_multidimensional_index(i);
    const int n = n0 + ids[0], m = m0 + ids[1];
    if (m < M && n >= n_lo && n < N) y[(size_t)m * N + n] = dT[i];
  }
}
)";

void check_k(const char* name, int k) {
  if (k % kGroup != 0) {
    std::ostringstream msg;
    msg << "[" << name << "] The last dimension must be a multiple of "
        << kGroup << " but is " << k << ".";
    throw std::invalid_argument(msg.str());
  }
}

/** The 256x256 Hadamard block (orthonormal, symmetric), float32. */
const array& hadamard256() {
  static array h = [] {
    std::vector<float> h4 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    for (int i = 0; i < 4; ++i) {
      h4[i * 4 + (3 - i)] = -1;
    }
    std::vector<float> m = {1.0f};
    int n = 1;
    for (int r = 0; r < 4; ++r) { // m = m (x) H4
      std::vector<float> out(n * 4 * n * 4);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
              out[(i * 4 + a) * n * 4 + (j * 4 + b)] = m[i * n + j] * h4[a * 4 + b];
      m = std::move(out);
      n *= 4;
    }
    for (auto& v : m) {
      v /= 16.0f;
    }
    return array(m.begin(), {kGroup, kGroup}, float32);
  }();
  return h;
}

/** Reference math: rotate rows of x [M, K] in float32 and quantize per row. */
std::pair<array, array> quantize_reference(const array& x, StreamOrDevice s) {
  int m = x.shape(0), k = x.shape(1);
  auto rotated = reshape(
      matmul(reshape(astype(x, float32, s), {-1, kGroup}, s), hadamard256(), s),
      {m, k},
      s);
  auto scale = maximum(
      divide(max(abs(rotated, s), -1, false, s), array(127.0f), s),
      array(1e-20f),
      s);
  auto q = astype(
      clip(
          round(divide(rotated, expand_dims(scale, -1, s), s), 0, s),
          array(-127.0f),
          array(127.0f),
          s),
      int8,
      s);
  return {q, scale};
}

bool use_kernels(StreamOrDevice s) {
  auto stream = to_stream(s);
  return stream.device == Device::gpu && metal::is_available() &&
      metal::is_tensor_ops_available();
}

} // namespace

std::pair<array, array> convrot_quantize(const array& x, StreamOrDevice s) {
  if (x.ndim() != 2) {
    throw std::invalid_argument("[convrot_quantize] x must be 2-D [M, K].");
  }
  check_k("convrot_quantize", x.shape(1));
  int m = x.shape(0), k = x.shape(1);
  int padded = (m + kRowPad - 1) / kRowPad * kRowPad;
  if (!use_kernels(s)) {
    auto [q, scale] = quantize_reference(x, s);
    if (padded != m) {
      q = pad(q, {{0, padded - m}, {0, 0}}, array(0, int8), "constant", s);
    }
    return {q, scale};
  }
  static auto kernel = metal_kernel(
      "convrot_rotate_quantize", {"x"}, {"xq", "xs"}, kRotateQuantize);
  int groups = k / kGroup, max_groups = 1;
  while (max_groups < groups) {
    max_groups <<= 1;
  }
  auto out = kernel(
      {x},
      {{padded, k}, {m}},
      {int8, float32},
      {kGroup, m, 1},
      {kGroup, 1, 1},
      {{"T", x.dtype()}, {"MAX_GROUPS", max_groups}},
      std::nullopt,
      false,
      s);
  return {out[0], out[1]};
}

std::pair<array, array> convrot_quantize_weight(
    const array& w,
    StreamOrDevice s) {
  if (w.ndim() != 2) {
    throw std::invalid_argument(
        "[convrot_quantize_weight] w must be 2-D [N, K].");
  }
  check_k("convrot_quantize_weight", w.shape(1));
  return quantize_reference(w, s);
}

array convrot_matmul_quantized(
    const array& xq,
    const array& xs,
    const array& w,
    const array& scales,
    Dtype dtype,
    std::optional<std::tuple<int, int, int>> tiles,
    StreamOrDevice s) {
  if (w.dtype() != int8 || w.ndim() != 2 || scales.ndim() != 1 ||
      scales.shape(0) != w.shape(0) || xq.ndim() != 2 ||
      xq.shape(1) != w.shape(1)) {
    throw std::invalid_argument(
        "[convrot_matmul] Expected int8 w [N, K], scales [N] and xq [M, K].");
  }
  int m = xs.shape(0), n = w.shape(0);
  int bm = m < 8192 ? 128 : 64;
  int bn = 8192 / bm, bk = 64;
  if (tiles) {
    std::tie(bm, bn, bk) = *tiles;
    if (bm <= 0 || bn <= 0 || bk <= 0 || bm > kRowPad || xq.shape(0) % bm != 0 ||
        xq.shape(1) % bk != 0) {
      throw std::invalid_argument(
          "[convrot_matmul] tiles must be positive, BM <= 128 dividing the "
          "rows of xq, and BK dividing K.");
    }
  }
  if (!use_kernels(s) || n < bn) {
    // Float product: no TensorOps, or narrower than one column tile
    auto acc = matmul(
        astype(slice(xq, {0, 0}, {m, xq.shape(1)}, s), float32, s),
        transpose(astype(w, float32, s), s),
        s);
    acc = multiply(
        multiply(acc, expand_dims(xs, -1, s), s),
        expand_dims(scales, 0, s),
        s);
    return astype(acc, dtype, s);
  }
  static auto kernel = metal_kernel(
      "convrot_w8a8_gemm",
      {"xq", "wq", "xs", "ws"},
      {"y"},
      kGemmBody,
      kGemmHeader);
  return kernel(
      {xq, w, xs, scales},
      {{m, n}},
      {dtype},
      {(n + bn - 1) / bn * 128, (m + bm - 1) / bm, 1},
      {128, 1, 1},
      {{"BM", bm}, {"BN", bn}, {"BK", bk}, {"T", dtype}},
      std::nullopt,
      false,
      s)[0];
}

array convrot_matmul(
    const array& x,
    const array& w,
    const array& scales,
    StreamOrDevice s) {
  if (x.ndim() < 1 || !issubdtype(x.dtype(), floating)) {
    throw std::invalid_argument(
        "[convrot_matmul] x must be a floating point array.");
  }
  auto shape = x.shape();
  int k = shape.back();
  auto x2 = reshape(x, {-1, k}, s);
  auto [xq, xs] = convrot_quantize(x2, s);
  auto y = convrot_matmul_quantized(xq, xs, w, scales, x.dtype(), std::nullopt, s);
  shape.back() = w.shape(0);
  return reshape(y, shape, s);
}

} // namespace mlx::core::fast
