// Copyright © 2026 ain-soph (MLX fork). Based on the Apple NAX GEMM kernels.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_convrot_nax.h"

// clang-format off
#define instantiate_convrot_tiled(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                           \
      "convrot_w8a16_nax_" #iname "_" #oname                                    \
      "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn,                        \
  convrot_w8a16_nax, itype, otype, bm, bn, bk, wm, wn)

#define instantiate_convrot_thin(iname, itype, oname, otype, um, un, sm, sn, ks) \
  instantiate_kernel(                                                           \
      "convrot_w8a16_thin_nax_" #iname "_" #oname                               \
      "_um" #um "_un" #un "_sm" #sm "_sn" #sn "_ks" #ks,                        \
  convrot_w8a16_thin_nax, itype, otype, um, un, sm, sn, ks)

#define instantiate_convrot_out(iname, itype, oname, otype)                   \
  instantiate_convrot_tiled(iname, itype, oname, otype,  64,  64, 256, 2, 2)    \
  instantiate_convrot_tiled(iname, itype, oname, otype,  64,  64, 512, 2, 2)    \
  instantiate_convrot_tiled(iname, itype, oname, otype, 128,  64, 256, 4, 2)    \
  instantiate_convrot_tiled(iname, itype, oname, otype, 128,  64, 512, 4, 2)    \
  instantiate_convrot_tiled(iname, itype, oname, otype,  64, 128, 512, 2, 4)    \
  instantiate_convrot_tiled(iname, itype, oname, otype, 128, 128, 512, 4, 4)    \
  instantiate_convrot_thin(iname, itype, oname, otype, 2, 2, 1, 1, 4)           \
  instantiate_convrot_thin(iname, itype, oname, otype, 2, 2, 1, 1, 2)           \
  instantiate_convrot_thin(iname, itype, oname, otype, 2, 2, 1, 2, 2)           \
  instantiate_convrot_thin(iname, itype, oname, otype, 2, 4, 1, 1, 2)

#define instantiate_convrot(iname, itype)                                     \
  instantiate_convrot_out(iname, itype, iname, itype)                         \
  instantiate_convrot_out(iname, itype, float32, float)                       \
  instantiate_kernel("convrot_w8a16_accum_" #iname, convrot_w8a16_accum, itype)

instantiate_convrot(float16, half);
instantiate_convrot(bfloat16, bfloat);
// clang-format on
