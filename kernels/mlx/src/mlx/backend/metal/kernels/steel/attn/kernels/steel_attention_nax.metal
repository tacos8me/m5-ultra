// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention_nax, dtype, bq, bk, bd, wm, wn, mtype, float)

#define instantiate_attn_vdim(tname, dtype, bq, bk, bd, bdv, wm, wn, mname, mtype) \
  instantiate_kernel(                                                           \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd "_bdv" #bdv       \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                       \
  attention_nax, dtype, bq, bk, bd, wm, wn, mtype, float, bdv)

#define instantiate_attn_dsplit(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                           \
      "steel_attention_dsplit_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                       \
  attention_nax_dsplit, dtype, bq, bk, bd, wm, wn, mtype, float)

#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)         \
    instantiate_attn_dsplit(iname, itype, 64, 32, 256, 4, 2, mname, mtype) \
    instantiate_attn_vdim(iname, itype, 64, 32, 192, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 64, 32, 128, 4, 1, mname, mtype)        \
    instantiate_attn(iname, itype, 64, 32,  96, 4, 1, mname, mtype)        \
    instantiate_attn(iname, itype, 64, 32,  64, 4, 1, mname, mtype)        \
    instantiate_attn(iname, itype, 64, 64, 128, 4, 1, mname, mtype)        \
    instantiate_attn(iname, itype, 64, 64,  64, 4, 1, mname, mtype)

#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat);

instantiate_attn_mask_helper(float32, float);

#define instantiate_attn_decode(tname, dtype, bk, bd, bdv)            \
  instantiate_kernel(                                                 \
      "steel_attention_decode_" #tname "_bk" #bk "_bd" #bd "_bdv" #bdv, \
      attention_nax_decode, dtype, bk, bd, bdv, float)

instantiate_attn_decode(float16, half, 32, 192, 128)
instantiate_attn_decode(bfloat16, bfloat, 32, 192, 128)
    // clang-format on
