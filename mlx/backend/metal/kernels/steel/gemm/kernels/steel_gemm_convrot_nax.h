// Copyright © 2026 ain-soph (MLX fork). Based on the Apple NAX GEMM kernels.

// ConvRot W8A16: y[m, n] = s[n] * sum_k x[m, k] q[n, k], with x the
// activations already rotated into the weights' Hadamard basis (T = half or
// bfloat), q the int8 weight rows [N, K] and s their float32 scales.
//
// Both kernels below can split K over partitions (grid). Partitions write
// float32 partial sums [P, M, N] (OutT = float) that convrot_w8a16_accum adds
// in partition order and scales, so the result does not depend on scheduling.
// With one partition (OutT = T) they scale and store the output directly.

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// Tiled kernel: the NAX split-K GEMM structure with int8 B. Each simdgroup
// loads its A and B fragments straight from device memory into registers;
// the int8 weights are converted to T while loading (exact for |q| <= 127).
///////////////////////////////////////////////////////////////////////////////

// clang-format off
template <
    typename T,
    typename OutT,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void convrot_w8a16_nax(
    const device T* A [[buffer(0)]],
    const device int8_t* B [[buffer(1)]],
    const device float* S [[buffer(2)]],
    device OutT* C [[buffer(3)]],
    const constant GEMMSpiltKParams* params [[buffer(4)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on
  constexpr bool kPartial = metal::is_same_v<OutT, float>;

  const int linear_tid = tid.x;

  // Swizzled tile order, as in gemm_splitk_nax
  const int tn_swizzled = params->tiles_n << params->swizzle_log;
  const int tm_swizzled =
      (params->tiles_m + (1 << params->swizzle_log) - 1) >> params->swizzle_log;
  const int tiles_per_partition = tn_swizzled * tm_swizzled;

  const int tid_z = linear_tid / tiles_per_partition;
  const int xy_flat = linear_tid % tiles_per_partition;
  const int grid_x = xy_flat % tn_swizzled;
  const int grid_y = xy_flat / tn_swizzled;
  const int tid_y = (grid_y << params->swizzle_log) +
      (grid_x & ((1 << params->swizzle_log) - 1));
  const int tid_x = grid_x >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int k_start = params->split_k_partition_size * tid_z;

  // A [M, K] row-major, B [N, K] row-major (transposed operand)
  A += size_t(c_row) * params->lda + size_t(k_start);
  B += size_t(c_col) * params->ldb + size_t(k_start);
  C += (kPartial ? size_t(params->split_k_partition_stride) * tid_z : 0) +
      size_t(c_row) * params->ldc + size_t(c_col);
  S += c_col;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const bool align_M = (params->M % BM) == 0;
  const bool align_N = (params->N % BN) == 0;

  const int sgp_sm_int =
      align_M ? int(SM) : min(int(SM), params->M - (c_row + tm));
  const short sgp_sm = short(sgp_sm_int);
  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);

  const int sgp_sn_int =
      align_N ? int(SN) : min(int(SN), params->N - (c_col + tn));
  const short sgp_sn = short(sgp_sn_int);
  const bool is_unaligned_sn = align_N ? false : (sgp_sn != SN);

  A += size_t(tm) * params->lda;
  B += size_t(tn) * params->ldb;
  C += size_t(tm) * params->ldc + tn;
  S += tn;

  NAXTile<float, TM, TN> Dtile;

  const int partition_k_size =
      min(params->split_k_partition_size, params->K - k_start);
  const int partition_k_iters = partition_k_size / BK;
  const bool partition_k_aligned = (partition_k_size % BK) == 0;

  dispatch_bool(partition_k_aligned, [&](auto kAlignedK) {
    dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(!is_unaligned_sn, [&](auto kAlignedN) {
        Dtile = gemm_loop<
            T,
            SM,
            SN,
            SK,
            BK,
            false,
            true,
            kAlignedM.value,
            kAlignedN.value,
            kAlignedK.value,
            float,
            int8_t>(
            A,
            B,
            params->lda,
            params->ldb,
            partition_k_size,
            partition_k_iters,
            sgp_sm,
            sgp_sn);
      });
    });
  });

  if (sgp_sm <= 0 || sgp_sn <= 0) {
    return;
  }

  if constexpr (!kPartial) {
    // Row scales of the weights = column scales of the output
    using Frag = typename NAXTile<float, TM, TN>::NAXFrag_t;
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < TN; nn++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < Frag::kElemsPerFrag; i++) {
        const short col = min(short(nn * 16 + Frag::get_coord(i).x), short(sgp_sn - 1));
        const float s = S[col];
        STEEL_PRAGMA_UNROLL
        for (short mm = 0; mm < TM; mm++) {
          Dtile.frag_at(mm, nn)[i] *= s;
        }
      }
    }
  }

  if (!is_unaligned_sm && !is_unaligned_sn) {
    Dtile.store(C, int(params->ldc));
  } else {
    Dtile.store_safe(C, int(params->ldc), short2(sgp_sn, sgp_sm));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Thin kernel (few rows): gemm_thin_nax with int8 B. Each simdgroup computes
// one (16 UM) x (16 UN) block, keeps a K chunk of A in registers and MPP reads
// the int8 B chunk from device memory (bfloat/half x int8 -> float). KS groups
// of simdgroups split K inside the threadgroup; the grid's y dimension splits
// K over partitions.
///////////////////////////////////////////////////////////////////////////////

// clang-format off
template <
    typename T,
    typename OutT,
    int UM,
    int UN,
    int SM,
    int SN,
    int KS>
[[kernel, max_total_threads_per_threadgroup(SM * SN * KS * 32)]] void convrot_w8a16_thin_nax(
    const device T* A [[buffer(0)]],
    const device int8_t* B [[buffer(1)]],
    const device float* S [[buffer(2)]],
    device OutT* D [[buffer(3)]],
    const constant GEMMSpiltKParams* params [[buffer(4)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on
  constexpr bool kPartial = metal::is_same_v<OutT, float>;
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using a_tensor_t = metal::tensor<device T, extents_t, metal::tensor_inline>;
  using b_tensor_t =
      metal::tensor<device int8_t, extents_t, metal::tensor_inline>;
  using d_tensor_t =
      metal::tensor<device OutT, extents_t, metal::tensor_inline>;

  constexpr int TSM = 16 * UM;
  constexpr int TSN = 16 * UN;
  constexpr int BK = 64;

  const int M = params->M;
  const int N = params->N;
  const int lda = params->lda;
  const int ldb = params->ldb;

  const int k_start = params->split_k_partition_size * int(tid.y);
  const int K = min(params->split_k_partition_size, params->K - k_start);

  // Threadgroups go along M first, so groups that run together share B.
  const int tgp = tid.x;
  const int tile_m = tgp % params->tiles_m;
  const int tile_n = tgp / params->tiles_m;

  const int k_group = simd_group_id / (SM * SN);
  const int sg = simd_group_id % (SM * SN);
  int m_off = (tile_m * SM + sg % SM) * TSM;
  int n_off = (tile_n * SN + sg / SM) * TSN;

  // A block at the edge moves back inside if the matrix is large enough. It
  // then writes again some outputs of a neighbor block, with equal values.
  const bool active = m_off < M && n_off < N;
  if (m_off + TSM > M && M >= TSM) {
    m_off = M - TSM;
  }
  if (n_off + TSN > N && N >= TSN) {
    n_off = N - TSN;
  }
  const bool inside = m_off + TSM <= M && n_off + TSN <= N;
  const int rows = active ? M - m_off : 0;
  const int cols = active ? N - n_off : 0;

  A += int64_t(m_off) * lda + k_start;
  B += int64_t(n_off) * ldb + k_start;
  D += (kPartial ? int64_t(params->split_k_partition_stride) * tid.y : 0) +
      int64_t(m_off) * params->ldc + n_off;

  a_tensor_t tA((device T*)A, extents_t(K, rows), strides_t{1, lda});
  b_tensor_t tB((device int8_t*)B, extents_t(K, cols), strides_t{1, ldb});

  auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
  auto b_chunk = [&](int k) { return tB.template slice<BK, TSN>(k, 0); };
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk(0));

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      TSM,
      TSN,
      BK,
      false,
      true,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

  auto Dacc = op.template get_destination_cooperative_tensor<
      a_chunk_t,
      b_chunk_t,
      float>();
  STEEL_PRAGMA_UNROLL
  for (uint16_t i = 0; i < Dacc.get_capacity(); i++) {
    Dacc[i] = 0.0f;
  }
  auto Areg = op.template get_left_input_cooperative_tensor<T, int8_t, float>();

  const int k_iters = params->gemm_k_iterations_aligned;
  for (int it = 0; it < k_iters; it++) {
    // Keep the simdgroups near each other in K, so that their shared loads
    // hit in cache.
    if ((it & 3) == 0) {
      threadgroup_barrier(mem_flags::mem_none);
    }
    const int k = (it * KS + k_group) * BK;
    if (!active || k >= K) {
      continue;
    }
    if (inside && k + BK <= K) {
      auto a = a_chunk(k);
      Areg.load(a);
      auto b = b_chunk(k);
      op.run(Areg, b, Dacc);
    } else if (inside) {
      auto a = tA.template slice<metal::dynamic_extent, TSM>(k, 0);
      auto b = tB.template slice<metal::dynamic_extent, TSN>(k, 0);
      op.run(a, b, Dacc);
    } else {
      // The views stop at the matrix edge and MPP masks the rest.
      const int kk = min(BK, K - k);
      a_tensor_t vA((device T*)A + k, extents_t(kk, rows), strides_t{1, lda});
      b_tensor_t vB(
          (device int8_t*)B + k, extents_t(kk, cols), strides_t{1, ldb});
      op.run(vA, vB, Dacc);
    }
  }

  if constexpr (KS > 1) {
    // K groups 1 to KS - 1 store their partial sums and group 0 adds them.
    using tg_tensor_t =
        metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;
    constexpr int kBlockSize = TSM * TSN;
    static_assert(
        (KS - 1) * SM * SN * kBlockSize <= 4096,
        "Partial sums need more than 16 KB of threadgroup memory");
    threadgroup float partials[(KS - 1) * SM * SN * kBlockSize];

    if (active && k_group > 0) {
      tg_tensor_t tP(
          partials + ((k_group - 1) * SM * SN + sg) * kBlockSize,
          extents_t(TSN, TSM),
          strides_t{1, TSN});
      Dacc.store(tP);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active && k_group == 0) {
      for (int g = 0; g < KS - 1; g++) {
        tg_tensor_t tP(
            partials + (g * SM * SN + sg) * kBlockSize,
            extents_t(TSN, TSM),
            strides_t{1, TSN});
        auto part = op.template get_destination_cooperative_tensor<
            a_chunk_t,
            b_chunk_t,
            float>();
        part.load(tP);
        STEEL_PRAGMA_UNROLL
        for (uint16_t i = 0; i < Dacc.get_capacity(); i++) {
          Dacc[i] += part[i];
        }
      }
    }
  }
  if (k_group != 0 || !active) {
    return;
  }

  auto Dout = op.template get_destination_cooperative_tensor<
      a_chunk_t,
      b_chunk_t,
      OutT>();
  STEEL_PRAGMA_UNROLL
  for (uint16_t i = 0; i < Dacc.get_capacity(); i++) {
    if constexpr (kPartial) {
      Dout[i] = Dacc[i];
    } else {
      const int c = min(int(Dacc.get_multidimensional_index(i)[0]), cols - 1);
      Dout[i] = static_cast<OutT>(Dacc[i] * S[n_off + c]);
    }
  }

  d_tensor_t tD(D, extents_t(cols, rows), strides_t{1, params->ldc});
  if (inside) {
    auto d = tD.template slice<TSN, TSM>(0, 0);
    Dout.store(d);
  } else {
    Dout.store(tD);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Partition sum (fixed order) and row scale of the weights.
///////////////////////////////////////////////////////////////////////////////

template <typename T>
[[kernel]] void convrot_w8a16_accum(
    const device float* C [[buffer(0)]],
    const device float* S [[buffer(1)]],
    device T* D [[buffer(2)]],
    const constant int& partitions [[buffer(3)]],
    const constant int& partition_stride [[buffer(4)]],
    const constant int& ldd [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const size_t idx = gid.x + gid.y * size_t(ldd);
  float acc = 0.0f;
  for (int p = 0; p < partitions; p++) {
    acc += C[idx + p * size_t(partition_stride)];
  }
  D[idx] = static_cast<T>(acc * S[gid.x]);
}
