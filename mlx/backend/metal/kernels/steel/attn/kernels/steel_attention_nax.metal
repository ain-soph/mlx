// Copyright © 2024-26 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, bv, wm, wn, mname, mtype) \
  instantiate_kernel(                                                        \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd "_bv" #bv      \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                    \
  attention_nax, dtype, bq, bk, bd, bv, wm, wn, mtype, float)

// opts: see attention_nax_dsplit (D=256 uses all three: the bq32 tile with
// lane-minor exchange, Q reload and P in T measured 12-25% faster than bq64 on
// M5 for 1K-32K tokens).
#define instantiate_attn_dsplit(tname, dtype, bq, bk, bd, wm, wn, mname, mtype, opts) \
  instantiate_kernel(                                                           \
      "steel_attention_dsplit_" #tname "_bq" #bq "_bk" #bk "_bd" #bd "_bv" #bd  \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                       \
  attention_nax_dsplit, dtype, bq, bk, bd, wm, wn, mtype, float, opts)

#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)         \
    instantiate_attn_dsplit(iname, itype, 32, 32, 512, 2, 4, mname, mtype, 0) \
    instantiate_attn_dsplit(iname, itype, 32, 32, 256, 2, 2, mname, mtype, 7) \
    instantiate_attn(iname, itype, 64, 64, 128, 128, 4, 1, mname, mtype)   \
    instantiate_attn(iname, itype, 64, 64,  64,  64, 4, 1, mname, mtype)   \
    instantiate_attn(iname, itype, 64, 32, 256, 256, 4, 1, mname, mtype)   \
    instantiate_attn(iname, itype, 64, 32, 128, 128, 4, 1, mname, mtype)   \
    instantiate_attn(iname, itype, 64, 32,  96,  96, 4, 1, mname, mtype)   \
    instantiate_attn(iname, itype, 64, 32,  96,  64, 4, 1, mname, mtype)   \
    instantiate_attn(iname, itype, 64, 32,  64,  64, 4, 1, mname, mtype)


#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat);
instantiate_attn_mask_helper(float32, float);
// clang-format on
