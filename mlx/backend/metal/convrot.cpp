// Copyright © 2026 ain-soph (MLX fork).
//
// ConvRot W8A16 GEMM on NAX (M5) GPUs: y = (x H) (q s)^T with the rotated
// activations kept in half / bfloat, int8 weight rows q [N, K] and float32 row
// scales s [N]. Kernels: kernels/steel/gemm/kernels/steel_gemm_convrot_nax.h.

#include <climits>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

namespace {

struct ConvRotConfig {
  bool thin; // thin kernel (few rows) or tiled kernel
  int t[5]; // thin: um un sm sn ks; tiled: bm bn bk wm wn
  int partitions; // K partitions over the grid
  int swizzle = -1; // tiled: log2 of the swizzle tile, -1 = default
};

// MLX_CONVROT_W8A16=thin,UM,UN,SM,SN,KS,P or nax,BM,BN,BK,WM,WN,P[,SWIZZLE]
// picks the kernel (for tuning; it must be one of the instantiated shapes).
bool config_from_env(ConvRotConfig& c) {
  const char* v = std::getenv("MLX_CONVROT_W8A16");
  if (v == nullptr || *v == '\0') {
    return false;
  }
  std::string s(v);
  std::istringstream in(s);
  std::string kind;
  std::getline(in, kind, ',');
  if (kind != "thin" && kind != "nax") {
    throw std::invalid_argument("[convrot_matmul_w8a16] bad MLX_CONVROT_W8A16");
  }
  c.thin = kind == "thin";
  std::string item;
  for (int i = 0; i < 6; i++) {
    if (!std::getline(in, item, ',')) {
      throw std::invalid_argument(
          "[convrot_matmul_w8a16] bad MLX_CONVROT_W8A16");
    }
    (i < 5 ? c.t[i] : c.partitions) = std::stoi(item);
  }
  if (!c.thin && std::getline(in, item, ',')) {
    c.swizzle = std::stoi(item);
  }
  bool bad = c.partitions < 1 || c.swizzle < -1 || c.swizzle > 4;
  for (int i = 0; i < 5; i++) {
    bad |= c.t[i] < 1;
  }
  if (bad) {
    throw std::invalid_argument("[convrot_matmul_w8a16] bad MLX_CONVROT_W8A16");
  }
  return true;
}

// Float32 partial sums of a K split ([P, M, N]) stay below this size.
constexpr size_t kMaxPartialBytes = size_t(1) << 30;

// Tuned on M5 Ultra for the H3 text encoder down_proj (N 5120, K 25600,
// M 32 ... 16595; upstream/mlx/convrot-w8a16-20261124 in llm-serving).
// Smaller GPUs (devc 's' = Max, others) need fewer threadgroups (untested).
// split_k: K partitions asked by the caller (0 = by shape). One partition
// uses the tiled kernel for every M: its float sums run in K order.
ConvRotConfig choose_config(int M, int N, int K, char devc, int split_k) {
  ConvRotConfig c;
  if (config_from_env(c)) {
    return c;
  }
  const int div = devc == 'd' ? 1 : (devc == 's' ? 2 : 4);
  int p = 1;
  if (split_k == 1) {
    // Every simdgroup sums its 32 x 32 block over K in order (same bytes for
    // both tiles); 64 x 64 gives more threadgroups up to ~2048 rows.
    c.thin = false;
    if (M <= 2048) {
      c.t[0] = 64, c.t[1] = 64, c.t[2] = 512, c.t[3] = 2, c.t[4] = 2;
    } else {
      c.t[0] = 128, c.t[1] = 64, c.t[2] = 512, c.t[3] = 4, c.t[4] = 2;
    }
  } else if (M <= 64) {
    // Few rows: weight-bandwidth bound. 32 x 32 blocks per simdgroup, KS
    // groups split K inside the threadgroup and P partitions over the grid
    // until about 1280 simdgroups stream the weight (partitions >= 2048).
    const int ks = M <= 32 ? 4 : 2;
    c.thin = true;
    c.t[0] = 2, c.t[1] = 2, c.t[2] = 1, c.t[3] = 1, c.t[4] = ks;
    const int tiles = ((M + 31) / 32) * ((N + 31) / 32);
    while (p < 8 && tiles * p * ks < 1280 / div && K / (2 * p) >= 2048) {
      p *= 2;
    }
  } else {
    // 128 x 64 tiles of 8 simdgroups (32 x 32 each). K is split in at least
    // 4 partitions, more while there are < 2048 threadgroups (partitions keep
    // >= 1536 of K). The grid runs partition by partition, so all row tiles
    // reread one partition's weight slice while it is hot: on the down_proj
    // this beats one partition with any swizzle (16595 rows: 44.5 ms with 4
    // partitions, 50.6 / 52.8 / 59.5 / 70.4 ms with one and swizzle 0-3).
    c.thin = false;
    c.t[0] = 128, c.t[1] = 64, c.t[2] = 512, c.t[3] = 4, c.t[4] = 2;
    const int tiles = ((M + c.t[0] - 1) / c.t[0]) * ((N + c.t[1] - 1) / c.t[1]);
    p = std::max(4, (2048 / div + tiles - 1) / tiles);
    p = std::min({p, 16, std::max(1, K / 1536)});
  }
  if (split_k > 1) {
    p = split_k;
  } else if (p > 1) {
    const size_t mn = size_t(M) * size_t(N) * sizeof(float);
    p = int(std::min<size_t>(p, std::max<size_t>(1, kMaxPartialBytes / mn)));
  }
  c.partitions = p;
  return c;
}

array ensure_row_contiguous(const array& x, const Stream& s) {
  if (x.flags().row_contiguous) {
    return x;
  }
  array x_copy = contiguous_copy_gpu(x, s);
  metal::get_command_encoder(s).add_temporary(x_copy);
  return x_copy;
}

} // namespace

void ConvRotW8A16::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto x = ensure_row_contiguous(inputs[0], s);
  auto w = ensure_row_contiguous(inputs[1], s);
  auto scales = ensure_row_contiguous(inputs[2], s);
  auto& out = outputs[0];

  const int M = x.shape(0), K = x.shape(1), N = w.shape(0);
  out.set_data(allocator::malloc(out.nbytes()));
  if (M == 0 || N == 0) {
    return;
  }
  if (K == 0) {
    fill_gpu(array(0, out.dtype()), out, s);
    return;
  }

  auto c = choose_config(M, N, K, d.get_architecture().back(), split_k_);
  // The partition stride is an int in the kernel parameters.
  if (size_t(M) * size_t(N) > size_t(INT_MAX)) {
    c.partitions = 1;
  }
  const int align = c.thin ? 64 * c.t[4] : c.t[2];
  int partition = (K + c.partitions - 1) / c.partitions;
  partition = (partition + align - 1) / align * align;
  const int partitions = (K + partition - 1) / partition;
  const bool partial = partitions > 1;

  array acc = out;
  if (partial) {
    acc = array({partitions, M, N}, float32, nullptr, {});
    acc.set_data(allocator::malloc(acc.nbytes()));
  }

  auto& enc = metal::get_command_encoder(s);
  const std::string tname = type_to_name(x);
  const std::string oname = partial ? std::string("float32") : tname;
  std::ostringstream kname;
  std::string func;
  if (c.thin) {
    kname << "convrot_w8a16_thin_nax_" << tname << "_" << oname << "_um"
          << c.t[0] << "_un" << c.t[1] << "_sm" << c.t[2] << "_sn" << c.t[3]
          << "_ks" << c.t[4];
    func = "convrot_w8a16_thin_nax";
  } else {
    kname << "convrot_w8a16_nax_" << tname << "_" << oname << "_bm" << c.t[0]
          << "_bn" << c.t[1] << "_bk" << c.t[2] << "_wm" << c.t[3] << "_wn"
          << c.t[4];
    func = "convrot_w8a16_nax";
  }
  auto template_def = get_template_definition(
      kname.str(),
      func,
      get_type_string(x.dtype()),
      partial ? std::string("float") : get_type_string(x.dtype()),
      c.t[0],
      c.t[1],
      c.t[2],
      c.t[3],
      c.t[4]);
  auto kernel = get_steel_gemm_convrot_nax_kernel(d, kname.str(), template_def);
  enc.set_compute_pipeline_state(kernel);

  if (c.thin) {
    const int tsm = 16 * c.t[0], tsn = 16 * c.t[1];
    const int sm = c.t[2], sn = c.t[3], ks = c.t[4];
    const int tm = (M + tsm * sm - 1) / (tsm * sm);
    const int tn = (N + tsn * sn - 1) / (tsn * sn);
    const int k_iters = ((partition + 63) / 64 + ks - 1) / ks;
    mlx::steel::GEMMSpiltKParams params{
        M, N, K, K, K, N, tn, tm, partitions, partial ? M * N : 0, partition, 0, k_iters};
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_input_array(scales, 2);
    enc.set_output_array(acc, 3);
    enc.set_bytes(params, 4);
    enc.dispatch_threadgroups(
        MTL::Size(tm * tn, partitions, 1), MTL::Size(32 * sm * sn * ks, 1, 1));
  } else {
    const int bm = c.t[0], bn = c.t[1], bk = c.t[2], wm = c.t[3], wn = c.t[4];
    const int tm = (M + bm - 1) / bm, tn = (N + bn - 1) / bn;
    // No swizzle by default: measured best on the down_proj at 96 ... 16595
    // rows (swizzle 1 is within 1-2 %, 2 and 3 are slower).
    const int swizzle_log = std::max(c.swizzle, 0);
    const int tile = 1 << swizzle_log;
    const int tm_swizzled = (tm + tile - 1) / tile;
    const int tn_swizzled = tn * tile;
    mlx::steel::GEMMSpiltKParams params{
        M,
        N,
        K,
        K,
        K,
        N,
        tn,
        tm,
        partitions,
        partial ? M * N : 0,
        partition,
        swizzle_log,
        partition / bk};
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_input_array(scales, 2);
    enc.set_output_array(acc, 3);
    enc.set_bytes(params, 4);
    enc.dispatch_threadgroups(
        MTL::Size(tn_swizzled * tm_swizzled * partitions, 1, 1),
        MTL::Size(32, wn, wm));
  }

  if (partial) {
    auto accum_name = "convrot_w8a16_accum_" + tname;
    auto accum = get_steel_gemm_convrot_nax_kernel(
        d,
        accum_name,
        get_template_definition(
            accum_name, "convrot_w8a16_accum", get_type_string(x.dtype())));
    enc.set_compute_pipeline_state(accum);
    const int stride = M * N;
    enc.set_input_array(acc, 0);
    enc.set_input_array(scales, 1);
    enc.set_output_array(out, 2);
    enc.set_bytes(partitions, 3);
    enc.set_bytes(stride, 4);
    enc.set_bytes(N, 5);
    enc.dispatch_threads(MTL::Size(N, M, 1), get_block_dims(N, M, 1));
    enc.add_temporary(acc);
  }
}

} // namespace mlx::core::fast
