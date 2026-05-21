#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <ATen/record_function.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "common.h"
#include "riscv64/gemm.h"
#include "riscv64/vector_helpers.h"

#if defined(CPU_CAPABILITY_RVV)

namespace {

constexpr int64_t W4A8_DYNAMIC_MAX_K = 12288;
constexpr int64_t W4A8_DYNAMIC_MAX_GROUPS = W4A8_DYNAMIC_MAX_K / 128;

inline vint16m4_t decode_low_signed_nibble_to_i16m4(vuint8m2_t packed, size_t vl) {
  vuint8m2_t shifted = __riscv_vsll_vx_u8m2(packed, 4, vl);
  vint8m2_t signed_i8 = __riscv_vreinterpret_v_u8m2_i8m2(shifted);
  signed_i8 = __riscv_vsra_vx_i8m2(signed_i8, 4, vl);
  return __riscv_vsext_vf2_i16m4(signed_i8, vl);
}

inline vint16m4_t decode_high_signed_nibble_to_i16m4(vuint8m2_t packed, size_t vl) {
  vint8m2_t signed_i8 = __riscv_vreinterpret_v_u8m2_i8m2(packed);
  signed_i8 = __riscv_vsra_vx_i8m2(signed_i8, 4, vl);
  return __riscv_vsext_vf2_i16m4(signed_i8, vl);
}

inline vint16m2_t decode_low_signed_nibble_to_i16m2(vuint8m1_t packed, size_t vl) {
  vuint8m1_t shifted = __riscv_vsll_vx_u8m1(packed, 4, vl);
  vint8m1_t signed_i8 = __riscv_vreinterpret_v_u8m1_i8m1(shifted);
  signed_i8 = __riscv_vsra_vx_i8m1(signed_i8, 4, vl);
  return __riscv_vsext_vf2_i16m2(signed_i8, vl);
}

inline vint16m2_t decode_high_signed_nibble_to_i16m2(vuint8m1_t packed, size_t vl) {
  vint8m1_t signed_i8 = __riscv_vreinterpret_v_u8m1_i8m1(packed);
  signed_i8 = __riscv_vsra_vx_i8m1(signed_i8, 4, vl);
  return __riscv_vsext_vf2_i16m2(signed_i8, vl);
}

template <typename scalar_t>
inline void quantize_activation_w4a8_dynamic_groups(
    const scalar_t* __restrict__ A,
    int8_t* __restrict__ Aq,
    float* __restrict__ Aq_scales,
    const int64_t* __restrict__ input_permutation,
    int64_t K,
    int64_t group_size) {
  const int64_t num_groups = K / group_size;
  alignas(64) float load_scratch[rvv_constants::MAX_VL_ELEMENTS_M8];
  for (int64_t g = 0; g < num_groups; ++g) {
    const int64_t k_begin = g * group_size;
    float amax = 0.0f;

    int64_t k = 0;
    while (k < group_size) {
      const size_t vl = __riscv_vsetvl_e32m8(group_size - k);
      if (input_permutation == nullptr) {
        vfloat32m8_t v_x = load_as_float_m8(A + k_begin + k, vl, load_scratch);
        vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_x, vl);
        vfloat32m1_t v_init = __riscv_vfmv_s_f_f32m1(amax, 1);
        vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_init, vl);
        amax = __riscv_vfmv_f_s_f32m1_f32(v_max);
      } else {
        for (size_t i = 0; i < vl; ++i) {
          load_scratch[i] = static_cast<float>(A[input_permutation[k_begin + k + static_cast<int64_t>(i)]]);
        }
        vfloat32m8_t v_x = __riscv_vle32_v_f32m8(load_scratch, vl);
        vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_x, vl);
        vfloat32m1_t v_init = __riscv_vfmv_s_f_f32m1(amax, 1);
        vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_init, vl);
        amax = __riscv_vfmv_f_s_f32m1_f32(v_max);
      }
      k += vl;
    }

    const float scale = std::max(amax / 127.0f, 1.0e-8f);
    const float inv_scale = 1.0f / scale;
    Aq_scales[g] = scale;

    k = 0;
    while (k < group_size) {
      const size_t vl = __riscv_vsetvl_e32m8(group_size - k);
      vfloat32m8_t v_x;
      if (input_permutation == nullptr) {
        v_x = load_as_float_m8(A + k_begin + k, vl, load_scratch);
      } else {
        for (size_t i = 0; i < vl; ++i) {
          load_scratch[i] = static_cast<float>(A[input_permutation[k_begin + k + static_cast<int64_t>(i)]]);
        }
        v_x = __riscv_vle32_v_f32m8(load_scratch, vl);
      }
      vfloat32m8_t v_qf = __riscv_vfmul_vf_f32m8(v_x, inv_scale, vl);
      v_qf = __riscv_vfmax_vf_f32m8(v_qf, -127.0f, vl);
      v_qf = __riscv_vfmin_vf_f32m8(v_qf, 127.0f, vl);
      vint32m8_t v_i32 = __riscv_vfcvt_x_f_v_i32m8_rm(v_qf, __RISCV_FRM_RNE, vl);
      vint16m4_t v_i16 = __riscv_vnclip_wx_i16m4(v_i32, 0, __RISCV_VXRM_RNE, vl);
      vint8m2_t v_i8 = __riscv_vnclip_wx_i8m2(v_i16, 0, __RISCV_VXRM_RNE, vl);
      __riscv_vse8_v_i8m2(Aq + k_begin + k, v_i8, vl);
      k += vl;
    }
  }
}

template <typename scalar_t, int BLOCK_M, bool has_input_permutation>
inline void quantize_activation_w4a8_dynamic_groups_block(
    const scalar_t* __restrict__ A,
    int64_t lda,
    int8_t* __restrict__ Aq,
    float* __restrict__ Aq_scales,
    const int64_t* __restrict__ input_permutation,
    int64_t K,
    int64_t group_size) {
  static_assert(BLOCK_M >= 1 && BLOCK_M <= 4, "BLOCK_M must be 1-4");
  if constexpr (!has_input_permutation) {
    UNUSED(input_permutation);
  }
  const int64_t num_groups = K / group_size;
  alignas(64) float load_scratch[rvv_constants::MAX_VL_ELEMENTS_M8];

  for (int64_t g = 0; g < num_groups; ++g) {
    const int64_t k_begin = g * group_size;
    float amax[BLOCK_M] = {};

    int64_t k = 0;
    while (k < group_size) {
      const size_t vl = __riscv_vsetvl_e32m8(group_size - k);
      for (int m = 0; m < BLOCK_M; ++m) {
        const scalar_t* row = A + static_cast<int64_t>(m) * lda;
        vfloat32m8_t v_x;
        if constexpr (!has_input_permutation) {
          v_x = load_as_float_m8(row + k_begin + k, vl, load_scratch);
        } else {
          for (size_t i = 0; i < vl; ++i) {
            load_scratch[i] = static_cast<float>(row[input_permutation[k_begin + k + static_cast<int64_t>(i)]]);
          }
          v_x = __riscv_vle32_v_f32m8(load_scratch, vl);
        }
        vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_x, vl);
        vfloat32m1_t v_init = __riscv_vfmv_s_f_f32m1(amax[m], 1);
        vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_init, vl);
        amax[m] = __riscv_vfmv_f_s_f32m1_f32(v_max);
      }
      k += vl;
    }

    float inv_scale[BLOCK_M];
    for (int m = 0; m < BLOCK_M; ++m) {
      const float scale = std::max(amax[m] / 127.0f, 1.0e-8f);
      Aq_scales[static_cast<int64_t>(m) * W4A8_DYNAMIC_MAX_GROUPS + g] = scale;
      inv_scale[m] = 1.0f / scale;
    }

    k = 0;
    while (k < group_size) {
      const size_t vl = __riscv_vsetvl_e32m8(group_size - k);
      for (int m = 0; m < BLOCK_M; ++m) {
        const scalar_t* row = A + static_cast<int64_t>(m) * lda;
        vfloat32m8_t v_x;
        if constexpr (!has_input_permutation) {
          v_x = load_as_float_m8(row + k_begin + k, vl, load_scratch);
        } else {
          for (size_t i = 0; i < vl; ++i) {
            load_scratch[i] = static_cast<float>(row[input_permutation[k_begin + k + static_cast<int64_t>(i)]]);
          }
          v_x = __riscv_vle32_v_f32m8(load_scratch, vl);
        }
        vfloat32m8_t v_qf = __riscv_vfmul_vf_f32m8(v_x, inv_scale[m], vl);
        v_qf = __riscv_vfmax_vf_f32m8(v_qf, -127.0f, vl);
        v_qf = __riscv_vfmin_vf_f32m8(v_qf, 127.0f, vl);
        vint32m8_t v_i32 = __riscv_vfcvt_x_f_v_i32m8_rm(v_qf, __RISCV_FRM_RNE, vl);
        vint16m4_t v_i16 = __riscv_vnclip_wx_i16m4(v_i32, 0, __RISCV_VXRM_RNE, vl);
        vint8m2_t v_i8 = __riscv_vnclip_wx_i8m2(v_i16, 0, __RISCV_VXRM_RNE, vl);
        __riscv_vse8_v_i8m2(Aq + static_cast<int64_t>(m) * W4A8_DYNAMIC_MAX_K + k_begin + k, v_i8, vl);
      }
      k += vl;
    }
  }
}

template <typename scalar_t, bool has_bias>
void w4a8_dynamic_linear_kernel_impl_m1(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    const float* __restrict__ bias,
    const int64_t* __restrict__ input_permutation,
    int64_t N,
    int64_t K,
    int64_t group_size) {
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  const int64_t NB = div_up(N, BLOCK_N);
  const int64_t num_groups = K / group_size;
  const int64_t w4a8_dynamic_group_bytes = 1 + (group_size / 2) * BLOCK_N;
  constexpr int64_t parallel_nb_min = 1;
  constexpr int64_t parallel_grain_size = 0;

  TORCH_CHECK(K <= W4A8_DYNAMIC_MAX_K, "weight_w4a8_dynamic_linear: stack scratch supports K <= ", W4A8_DYNAMIC_MAX_K);
  TORCH_CHECK(
      num_groups <= W4A8_DYNAMIC_MAX_GROUPS,
      "weight_w4a8_dynamic_linear: stack scratch supports at most ",
      W4A8_DYNAMIC_MAX_GROUPS,
      " activation groups.");
  alignas(64) int8_t mat1_q8[W4A8_DYNAMIC_MAX_K];
  alignas(64) float mat1_scales[W4A8_DYNAMIC_MAX_GROUPS];
  quantize_activation_w4a8_dynamic_groups(mat1, mat1_q8, mat1_scales, input_permutation, K, group_size);

  auto compute_nb_range = [&](int64_t begin, int64_t end) {
    for (int64_t nb = begin; nb < end; ++nb) {
      const int64_t n_start = nb * BLOCK_N;
      const int64_t n_size = std::min(BLOCK_N, N - n_start);
      const size_t vl = __riscv_vsetvl_e32m8(n_size);

      vfloat32m8_t v_acc_f = __riscv_vfmv_v_f_f32m8(0.0f, vl);
      for (int64_t g = 0; g < num_groups; ++g) {
        vint32m8_t v_acc_i32 = __riscv_vmv_v_x_i32m8(0, vl);
        const uint8_t* group_block = packed_w + (nb * num_groups + g) * w4a8_dynamic_group_bytes;
        const uint8_t* q4_bytes = group_block + 1;
        constexpr int64_t PREFETCH_DIST = 4;
        const int64_t packed_group_elems = group_size / 2;

        for (int64_t kp = 0; kp < packed_group_elems; ++kp) {
          if (kp + PREFETCH_DIST < packed_group_elems) {
            __builtin_prefetch(q4_bytes + (kp + PREFETCH_DIST) * BLOCK_N, 0, 1);
          }
          const int64_t k0 = g * group_size + kp * 2;
          const int64_t k1 = k0 + 1;
          const vuint8m2_t v_packed = __riscv_vle8_v_u8m2(q4_bytes + kp * BLOCK_N, vl);
          const vint16m4_t v_low_i16 = decode_low_signed_nibble_to_i16m4(v_packed, vl);
          const vint16m4_t v_high_i16 = decode_high_signed_nibble_to_i16m4(v_packed, vl);
          v_acc_i32 = __riscv_vwmacc_vx_i32m8(v_acc_i32, static_cast<int16_t>(mat1_q8[k0]), v_low_i16, vl);
          v_acc_i32 = __riscv_vwmacc_vx_i32m8(v_acc_i32, static_cast<int16_t>(mat1_q8[k1]), v_high_i16, vl);
        }

        vfloat32m8_t v_group = __riscv_vfcvt_f_x_v_f32m8(v_acc_i32, vl);
        vfloat32m8_t v_w_scale = __riscv_vle32_v_f32m8(scales + (nb * num_groups + g) * BLOCK_N, vl);
        vfloat32m8_t v_scale = __riscv_vfmul_vf_f32m8(v_w_scale, mat1_scales[g], vl);
        v_acc_f = __riscv_vfmacc_vv_f32m8(v_acc_f, v_group, v_scale, vl);
      }

      if constexpr (has_bias) {
        v_acc_f = __riscv_vfadd_vv_f32m8(v_acc_f, __riscv_vle32_v_f32m8(bias + n_start, vl), vl);
      }

      alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M8];
      store_from_float_m8(out + n_start, v_acc_f, vl, scratch);
    }
  };

  if (parallel_nb_min > 0 && NB >= parallel_nb_min) {
    at::parallel_for(0, NB, parallel_grain_size, compute_nb_range);
  } else {
    compute_nb_range(0, NB);
  }
}

template <typename scalar_t, bool has_bias>
void w4a8_dynamic_gemm_kernel_impl_m2(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    const float* __restrict__ bias,
    const int64_t* __restrict__ input_permutation,
    int64_t N,
    int64_t K,
    int64_t mat1_stride0,
    int64_t out_stride0,
    int64_t group_size) {
  constexpr int64_t BLOCK_M = 2;
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  const int64_t NB = div_up(N, BLOCK_N);
  const int64_t num_groups = K / group_size;
  const int64_t w4a8_dynamic_group_bytes = 1 + (group_size / 2) * BLOCK_N;
  constexpr int64_t parallel_nb_min = 1;
  constexpr int64_t parallel_grain_size = 0;

  TORCH_CHECK(K <= W4A8_DYNAMIC_MAX_K, "weight_w4a8_dynamic_linear: stack scratch supports K <= ", W4A8_DYNAMIC_MAX_K);
  TORCH_CHECK(
      num_groups <= W4A8_DYNAMIC_MAX_GROUPS,
      "weight_w4a8_dynamic_linear: stack scratch supports at most ",
      W4A8_DYNAMIC_MAX_GROUPS,
      " activation groups.");

  alignas(64) int8_t mat1_q8[BLOCK_M][W4A8_DYNAMIC_MAX_K];
  alignas(64) float mat1_scales[BLOCK_M][W4A8_DYNAMIC_MAX_GROUPS];
  if (input_permutation == nullptr) {
    quantize_activation_w4a8_dynamic_groups_block<scalar_t, BLOCK_M, false>(
        mat1, mat1_stride0, &mat1_q8[0][0], &mat1_scales[0][0], nullptr, K, group_size);
  } else {
    quantize_activation_w4a8_dynamic_groups_block<scalar_t, BLOCK_M, true>(
        mat1, mat1_stride0, &mat1_q8[0][0], &mat1_scales[0][0], input_permutation, K, group_size);
  }

  auto compute_nb_range = [&](int64_t begin, int64_t end) {
    for (int64_t nb = begin; nb < end; ++nb) {
      const int64_t n_start = nb * BLOCK_N;
      const int64_t n_size = std::min(BLOCK_N, N - n_start);
      const size_t vl = __riscv_vsetvl_e32m8(n_size);

      vfloat32m8_t v_acc_f0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
      vfloat32m8_t v_acc_f1 = __riscv_vfmv_v_f_f32m8(0.0f, vl);

      for (int64_t g = 0; g < num_groups; ++g) {
        vint32m8_t v_acc_i0 = __riscv_vmv_v_x_i32m8(0, vl);
        vint32m8_t v_acc_i1 = __riscv_vmv_v_x_i32m8(0, vl);

        const uint8_t* group_block = packed_w + (nb * num_groups + g) * w4a8_dynamic_group_bytes;
        const uint8_t* q4_bytes = group_block + 1;
        constexpr int64_t PREFETCH_DIST = 4;
        const int64_t packed_group_elems = group_size / 2;

        for (int64_t kp = 0; kp < packed_group_elems; ++kp) {
          if (kp + PREFETCH_DIST < packed_group_elems) {
            __builtin_prefetch(q4_bytes + (kp + PREFETCH_DIST) * BLOCK_N, 0, 1);
          }
          const int64_t k0 = g * group_size + kp * 2;
          const int64_t k1 = k0 + 1;
          const vuint8m2_t v_packed = __riscv_vle8_v_u8m2(q4_bytes + kp * BLOCK_N, vl);
          const vint16m4_t v_low_i16 = decode_low_signed_nibble_to_i16m4(v_packed, vl);
          const vint16m4_t v_high_i16 = decode_high_signed_nibble_to_i16m4(v_packed, vl);
          v_acc_i0 = __riscv_vwmacc_vx_i32m8(v_acc_i0, static_cast<int16_t>(mat1_q8[0][k0]), v_low_i16, vl);
          v_acc_i0 = __riscv_vwmacc_vx_i32m8(v_acc_i0, static_cast<int16_t>(mat1_q8[0][k1]), v_high_i16, vl);
          v_acc_i1 = __riscv_vwmacc_vx_i32m8(v_acc_i1, static_cast<int16_t>(mat1_q8[1][k0]), v_low_i16, vl);
          v_acc_i1 = __riscv_vwmacc_vx_i32m8(v_acc_i1, static_cast<int16_t>(mat1_q8[1][k1]), v_high_i16, vl);
        }

        vfloat32m8_t v_w_scale = __riscv_vle32_v_f32m8(scales + (nb * num_groups + g) * BLOCK_N, vl);
        vfloat32m8_t v_group0 = __riscv_vfcvt_f_x_v_f32m8(v_acc_i0, vl);
        vfloat32m8_t v_scale0 = __riscv_vfmul_vf_f32m8(v_w_scale, mat1_scales[0][g], vl);
        v_acc_f0 = __riscv_vfmacc_vv_f32m8(v_acc_f0, v_group0, v_scale0, vl);
        vfloat32m8_t v_group1 = __riscv_vfcvt_f_x_v_f32m8(v_acc_i1, vl);
        vfloat32m8_t v_scale1 = __riscv_vfmul_vf_f32m8(v_w_scale, mat1_scales[1][g], vl);
        v_acc_f1 = __riscv_vfmacc_vv_f32m8(v_acc_f1, v_group1, v_scale1, vl);
      }

      if constexpr (has_bias) {
        vfloat32m8_t v_bias = __riscv_vle32_v_f32m8(bias + n_start, vl);
        v_acc_f0 = __riscv_vfadd_vv_f32m8(v_acc_f0, v_bias, vl);
        v_acc_f1 = __riscv_vfadd_vv_f32m8(v_acc_f1, v_bias, vl);
      }

      alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M8];
      store_from_float_m8(out + n_start, v_acc_f0, vl, scratch);
      store_from_float_m8(out + out_stride0 + n_start, v_acc_f1, vl, scratch);
    }
  };

  if (parallel_nb_min > 0 && NB >= parallel_nb_min) {
    at::parallel_for(0, NB, parallel_grain_size, compute_nb_range);
  } else {
    compute_nb_range(0, NB);
  }
}

template <typename scalar_t, bool has_bias>
void w4a8_dynamic_gemm_kernel_impl_m2_batched(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    const float* __restrict__ bias,
    const int64_t* __restrict__ input_permutation,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_stride0,
    int64_t out_stride0,
    int64_t group_size) {
  constexpr int64_t BLOCK_M = 2;
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  const int64_t num_m_blocks = M / BLOCK_M;
  const int64_t NB = div_up(N, BLOCK_N);
  const int64_t num_groups = K / group_size;
  const int64_t w4a8_dynamic_group_bytes = 1 + (group_size / 2) * BLOCK_N;
  constexpr int64_t parallel_mb_grain_size = 0;

  TORCH_CHECK(K <= W4A8_DYNAMIC_MAX_K, "weight_w4a8_dynamic_linear: stack scratch supports K <= ", W4A8_DYNAMIC_MAX_K);
  TORCH_CHECK(
      num_groups <= W4A8_DYNAMIC_MAX_GROUPS,
      "weight_w4a8_dynamic_linear: stack scratch supports at most ",
      W4A8_DYNAMIC_MAX_GROUPS,
      " activation groups.");

  auto compute_mb_range = [&](int64_t begin, int64_t end) {
    for (int64_t mb = begin; mb < end; ++mb) {
      const scalar_t* mat1_block = mat1 + mb * BLOCK_M * mat1_stride0;
      scalar_t* out_block = out + mb * BLOCK_M * out_stride0;

      alignas(64) int8_t mat1_q8[BLOCK_M][W4A8_DYNAMIC_MAX_K];
      alignas(64) float mat1_scales[BLOCK_M][W4A8_DYNAMIC_MAX_GROUPS];
      if (input_permutation == nullptr) {
        quantize_activation_w4a8_dynamic_groups_block<scalar_t, BLOCK_M, false>(
            mat1_block, mat1_stride0, &mat1_q8[0][0], &mat1_scales[0][0], nullptr, K, group_size);
      } else {
        quantize_activation_w4a8_dynamic_groups_block<scalar_t, BLOCK_M, true>(
            mat1_block, mat1_stride0, &mat1_q8[0][0], &mat1_scales[0][0], input_permutation, K, group_size);
      }

      for (int64_t nb = 0; nb < NB; ++nb) {
        const int64_t n_start = nb * BLOCK_N;
        const int64_t n_size = std::min(BLOCK_N, N - n_start);
        const size_t vl = __riscv_vsetvl_e32m8(n_size);

        vfloat32m8_t v_acc_f0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        vfloat32m8_t v_acc_f1 = __riscv_vfmv_v_f_f32m8(0.0f, vl);

        for (int64_t g = 0; g < num_groups; ++g) {
          vint32m8_t v_acc_i0 = __riscv_vmv_v_x_i32m8(0, vl);
          vint32m8_t v_acc_i1 = __riscv_vmv_v_x_i32m8(0, vl);

          const uint8_t* group_block = packed_w + (nb * num_groups + g) * w4a8_dynamic_group_bytes;
          const uint8_t* q4_bytes = group_block + 1;
          constexpr int64_t PREFETCH_DIST = 4;
          const int64_t packed_group_elems = group_size / 2;

          for (int64_t kp = 0; kp < packed_group_elems; ++kp) {
            if (kp + PREFETCH_DIST < packed_group_elems) {
              __builtin_prefetch(q4_bytes + (kp + PREFETCH_DIST) * BLOCK_N, 0, 1);
            }
            const int64_t k0 = g * group_size + kp * 2;
            const int64_t k1 = k0 + 1;
            const vuint8m2_t v_packed = __riscv_vle8_v_u8m2(q4_bytes + kp * BLOCK_N, vl);
            const vint16m4_t v_low_i16 = decode_low_signed_nibble_to_i16m4(v_packed, vl);
            const vint16m4_t v_high_i16 = decode_high_signed_nibble_to_i16m4(v_packed, vl);
            v_acc_i0 = __riscv_vwmacc_vx_i32m8(v_acc_i0, static_cast<int16_t>(mat1_q8[0][k0]), v_low_i16, vl);
            v_acc_i0 = __riscv_vwmacc_vx_i32m8(v_acc_i0, static_cast<int16_t>(mat1_q8[0][k1]), v_high_i16, vl);
            v_acc_i1 = __riscv_vwmacc_vx_i32m8(v_acc_i1, static_cast<int16_t>(mat1_q8[1][k0]), v_low_i16, vl);
            v_acc_i1 = __riscv_vwmacc_vx_i32m8(v_acc_i1, static_cast<int16_t>(mat1_q8[1][k1]), v_high_i16, vl);
          }

          vfloat32m8_t v_w_scale = __riscv_vle32_v_f32m8(scales + (nb * num_groups + g) * BLOCK_N, vl);
          vfloat32m8_t v_group0 = __riscv_vfcvt_f_x_v_f32m8(v_acc_i0, vl);
          vfloat32m8_t v_scale0 = __riscv_vfmul_vf_f32m8(v_w_scale, mat1_scales[0][g], vl);
          v_acc_f0 = __riscv_vfmacc_vv_f32m8(v_acc_f0, v_group0, v_scale0, vl);
          vfloat32m8_t v_group1 = __riscv_vfcvt_f_x_v_f32m8(v_acc_i1, vl);
          vfloat32m8_t v_scale1 = __riscv_vfmul_vf_f32m8(v_w_scale, mat1_scales[1][g], vl);
          v_acc_f1 = __riscv_vfmacc_vv_f32m8(v_acc_f1, v_group1, v_scale1, vl);
        }

        if constexpr (has_bias) {
          vfloat32m8_t v_bias = __riscv_vle32_v_f32m8(bias + n_start, vl);
          v_acc_f0 = __riscv_vfadd_vv_f32m8(v_acc_f0, v_bias, vl);
          v_acc_f1 = __riscv_vfadd_vv_f32m8(v_acc_f1, v_bias, vl);
        }

        alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M8];
        store_from_float_m8(out_block + n_start, v_acc_f0, vl, scratch);
        store_from_float_m8(out_block + out_stride0 + n_start, v_acc_f1, vl, scratch);
      }
    }
  };

  at::parallel_for(0, num_m_blocks, parallel_mb_grain_size, compute_mb_range);
}

template <typename scalar_t, bool has_bias>
void w4a8_dynamic_gemm_kernel_impl_m4(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    const float* __restrict__ bias,
    const int64_t* __restrict__ input_permutation,
    int64_t N,
    int64_t K,
    int64_t mat1_stride0,
    int64_t out_stride0,
    int64_t group_size) {
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  const int64_t NB = div_up(N, BLOCK_N);
  const int64_t num_groups = K / group_size;
  const int64_t w4a8_dynamic_group_bytes = 1 + (group_size / 2) * BLOCK_N;
  constexpr int64_t parallel_nb_min = 1;
  constexpr int64_t parallel_grain_size = 0;

  TORCH_CHECK(K <= W4A8_DYNAMIC_MAX_K, "weight_w4a8_dynamic_linear: stack scratch supports K <= ", W4A8_DYNAMIC_MAX_K);
  TORCH_CHECK(
      num_groups <= W4A8_DYNAMIC_MAX_GROUPS,
      "weight_w4a8_dynamic_linear: stack scratch supports at most ",
      W4A8_DYNAMIC_MAX_GROUPS,
      " activation groups.");

  alignas(64) int8_t mat1_q8[BLOCK_M][W4A8_DYNAMIC_MAX_K];
  alignas(64) float mat1_scales[BLOCK_M][W4A8_DYNAMIC_MAX_GROUPS];
  if (input_permutation == nullptr) {
    quantize_activation_w4a8_dynamic_groups_block<scalar_t, BLOCK_M, false>(
        mat1, mat1_stride0, &mat1_q8[0][0], &mat1_scales[0][0], nullptr, K, group_size);
  } else {
    quantize_activation_w4a8_dynamic_groups_block<scalar_t, BLOCK_M, true>(
        mat1, mat1_stride0, &mat1_q8[0][0], &mat1_scales[0][0], input_permutation, K, group_size);
  }

  auto compute_nb_range = [&](int64_t begin, int64_t end) {
    for (int64_t nb = begin; nb < end; ++nb) {
      const int64_t n_start = nb * BLOCK_N;
      const int64_t n_size = std::min(BLOCK_N, N - n_start);
      const size_t vl_max = __riscv_vsetvlmax_e32m4();

      for (int64_t j = 0; j < n_size; j += static_cast<int64_t>(vl_max)) {
        const size_t vl = (j + static_cast<int64_t>(vl_max) <= n_size) ? vl_max : __riscv_vsetvl_e32m4(n_size - j);

        vfloat32m4_t v_acc_f0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t v_acc_f1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t v_acc_f2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t v_acc_f3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        for (int64_t g = 0; g < num_groups; ++g) {
          vint32m4_t v_acc_i0 = __riscv_vmv_v_x_i32m4(0, vl);
          vint32m4_t v_acc_i1 = __riscv_vmv_v_x_i32m4(0, vl);
          vint32m4_t v_acc_i2 = __riscv_vmv_v_x_i32m4(0, vl);
          vint32m4_t v_acc_i3 = __riscv_vmv_v_x_i32m4(0, vl);

          const uint8_t* group_block = packed_w + (nb * num_groups + g) * w4a8_dynamic_group_bytes;
          const uint8_t* q4_bytes = group_block + 1 + j;
          constexpr int64_t PREFETCH_DIST = 4;
          const int64_t packed_group_elems = group_size / 2;

          for (int64_t kp = 0; kp < packed_group_elems; ++kp) {
            if (kp + PREFETCH_DIST < packed_group_elems) {
              __builtin_prefetch(q4_bytes + (kp + PREFETCH_DIST) * BLOCK_N, 0, 1);
            }
            const int64_t k0 = g * group_size + kp * 2;
            const int64_t k1 = k0 + 1;
            const vuint8m1_t v_packed = __riscv_vle8_v_u8m1(q4_bytes + kp * BLOCK_N, vl);
            const vint16m2_t v_low_i16 = decode_low_signed_nibble_to_i16m2(v_packed, vl);
            const vint16m2_t v_high_i16 = decode_high_signed_nibble_to_i16m2(v_packed, vl);
            v_acc_i0 = __riscv_vwmacc_vx_i32m4(v_acc_i0, static_cast<int16_t>(mat1_q8[0][k0]), v_low_i16, vl);
            v_acc_i0 = __riscv_vwmacc_vx_i32m4(v_acc_i0, static_cast<int16_t>(mat1_q8[0][k1]), v_high_i16, vl);
            v_acc_i1 = __riscv_vwmacc_vx_i32m4(v_acc_i1, static_cast<int16_t>(mat1_q8[1][k0]), v_low_i16, vl);
            v_acc_i1 = __riscv_vwmacc_vx_i32m4(v_acc_i1, static_cast<int16_t>(mat1_q8[1][k1]), v_high_i16, vl);
            v_acc_i2 = __riscv_vwmacc_vx_i32m4(v_acc_i2, static_cast<int16_t>(mat1_q8[2][k0]), v_low_i16, vl);
            v_acc_i2 = __riscv_vwmacc_vx_i32m4(v_acc_i2, static_cast<int16_t>(mat1_q8[2][k1]), v_high_i16, vl);
            v_acc_i3 = __riscv_vwmacc_vx_i32m4(v_acc_i3, static_cast<int16_t>(mat1_q8[3][k0]), v_low_i16, vl);
            v_acc_i3 = __riscv_vwmacc_vx_i32m4(v_acc_i3, static_cast<int16_t>(mat1_q8[3][k1]), v_high_i16, vl);
          }

          vfloat32m4_t v_w_scale = __riscv_vle32_v_f32m4(scales + (nb * num_groups + g) * BLOCK_N + j, vl);
          v_acc_f0 = __riscv_vfmacc_vv_f32m4(
              v_acc_f0,
              __riscv_vfcvt_f_x_v_f32m4(v_acc_i0, vl),
              __riscv_vfmul_vf_f32m4(v_w_scale, mat1_scales[0][g], vl),
              vl);
          v_acc_f1 = __riscv_vfmacc_vv_f32m4(
              v_acc_f1,
              __riscv_vfcvt_f_x_v_f32m4(v_acc_i1, vl),
              __riscv_vfmul_vf_f32m4(v_w_scale, mat1_scales[1][g], vl),
              vl);
          v_acc_f2 = __riscv_vfmacc_vv_f32m4(
              v_acc_f2,
              __riscv_vfcvt_f_x_v_f32m4(v_acc_i2, vl),
              __riscv_vfmul_vf_f32m4(v_w_scale, mat1_scales[2][g], vl),
              vl);
          v_acc_f3 = __riscv_vfmacc_vv_f32m4(
              v_acc_f3,
              __riscv_vfcvt_f_x_v_f32m4(v_acc_i3, vl),
              __riscv_vfmul_vf_f32m4(v_w_scale, mat1_scales[3][g], vl),
              vl);
        }

        if constexpr (has_bias) {
          vfloat32m4_t v_bias = __riscv_vle32_v_f32m4(bias + n_start + j, vl);
          v_acc_f0 = __riscv_vfadd_vv_f32m4(v_acc_f0, v_bias, vl);
          v_acc_f1 = __riscv_vfadd_vv_f32m4(v_acc_f1, v_bias, vl);
          v_acc_f2 = __riscv_vfadd_vv_f32m4(v_acc_f2, v_bias, vl);
          v_acc_f3 = __riscv_vfadd_vv_f32m4(v_acc_f3, v_bias, vl);
        }

        alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
        store_from_float_m4(out + n_start + j, v_acc_f0, vl, scratch);
        store_from_float_m4(out + out_stride0 + n_start + j, v_acc_f1, vl, scratch);
        store_from_float_m4(out + 2 * out_stride0 + n_start + j, v_acc_f2, vl, scratch);
        store_from_float_m4(out + 3 * out_stride0 + n_start + j, v_acc_f3, vl, scratch);
      }
    }
  };

  if (parallel_nb_min > 0 && NB >= parallel_nb_min) {
    at::parallel_for(0, NB, parallel_grain_size, compute_nb_range);
  } else {
    compute_nb_range(0, NB);
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor>
convert_weight_w4a8_dynamic_packed(at::Tensor& weight_q, at::Tensor& scales, int64_t group_size) {
  RECORD_FUNCTION(
      "sgl-kernel::convert_weight_w4a8_dynamic_packed", std::vector<c10::IValue>({weight_q, scales, group_size}));

  CHECK_INPUT(weight_q);
  CHECK_INPUT(scales);
  CHECK_DIM(2, weight_q);
  CHECK_DIM(2, scales);
  TORCH_CHECK(weight_q.scalar_type() == at::kByte, "convert_weight_w4a8_dynamic_packed: weight_q must be uint8.");
  TORCH_CHECK(scales.scalar_type() == at::kFloat, "convert_weight_w4a8_dynamic_packed: scales must be float32.");
  TORCH_CHECK(
      group_size > 0 && group_size % 2 == 0,
      "convert_weight_w4a8_dynamic_packed: group_size must be a positive even number.");

  const int64_t N = weight_q.size(0);
  const int64_t packed_k = weight_q.size(1);
  const int64_t K = packed_k * 2;
  const int64_t num_groups = K / group_size;
  TORCH_CHECK(K % group_size == 0, "convert_weight_w4a8_dynamic_packed: K must be divisible by group_size.");
  TORCH_CHECK(
      scales.size(0) == N && scales.size(1) == num_groups,
      "convert_weight_w4a8_dynamic_packed: scales must have shape [N, K / group_size].");

  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  const int64_t NB = div_up(N, BLOCK_N);
  const int64_t w4a8_dynamic_group_bytes = 1 + (group_size / 2) * BLOCK_N;

  auto packed_weight = at::zeros({NB, num_groups, w4a8_dynamic_group_bytes}, weight_q.options());
  auto packed_scales = at::zeros({NB, num_groups, BLOCK_N}, scales.options());

  const auto weight_q_contig = weight_q.contiguous();
  const auto scales_contig = scales.contiguous();
  const uint8_t* src_w = weight_q_contig.data_ptr<uint8_t>();
  const float* src_s = scales_contig.data_ptr<float>();
  uint8_t* dst_w = packed_weight.data_ptr<uint8_t>();
  float* dst_s = packed_scales.data_ptr<float>();

  at::parallel_for(0, NB * num_groups, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t nb = idx / num_groups;
      const int64_t g = idx % num_groups;
      const int64_t n_start = nb * BLOCK_N;
      const int64_t n_size = std::min<int64_t>(BLOCK_N, N - n_start);
      uint8_t* dst_group_w = dst_w + (nb * num_groups + g) * w4a8_dynamic_group_bytes;
      float* dst_group_s = dst_s + (nb * num_groups + g) * BLOCK_N;
      dst_group_w[0] = static_cast<uint8_t>(n_size);

      const int64_t kp_begin = (g * group_size) / 2;
      for (int64_t kp = 0; kp < group_size / 2; ++kp) {
        uint8_t* dst_row = dst_group_w + 1 + kp * BLOCK_N;
        for (int64_t n = 0; n < n_size; ++n) {
          dst_row[n] = src_w[(n_start + n) * packed_k + kp_begin + kp];
        }
      }

      for (int64_t n = 0; n < n_size; ++n) {
        dst_group_s[n] = src_s[(n_start + n) * num_groups + g];
      }
    }
  });

  return std::make_tuple(packed_weight, packed_scales);
}

at::Tensor weight_w4a8_dynamic_linear(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& input_permutation,
    int64_t group_size,
    bool is_packed) {
  RECORD_FUNCTION(
      "sgl-kernel::weight_w4a8_dynamic_linear",
      std::vector<c10::IValue>({mat1, mat2, scales, bias, input_permutation, group_size, is_packed}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales);
  CHECK_DIM(2, mat1);
  TORCH_CHECK(mat2.dim() == 2 || mat2.dim() == 3, "weight_w4a8_dynamic_linear: mat2 must be 2D unpacked or 3D packed.");
  TORCH_CHECK(
      scales.dim() == 2 || scales.dim() == 3, "weight_w4a8_dynamic_linear: scales must be 2D unpacked or 3D packed.");
  TORCH_CHECK(
      mat1.scalar_type() == at::kBFloat16 || mat1.scalar_type() == at::kHalf,
      "weight_w4a8_dynamic_linear: mat1 must be bfloat16 or float16.");
  TORCH_CHECK(mat2.scalar_type() == at::kByte, "weight_w4a8_dynamic_linear: mat2 must be uint8 packed INT4.");
  TORCH_CHECK(scales.scalar_type() == at::kFloat, "weight_w4a8_dynamic_linear: scales must be float32.");
  TORCH_CHECK(
      group_size > 0 && group_size % 2 == 0, "weight_w4a8_dynamic_linear: group_size must be a positive even number.");

  at::Tensor packed_w;
  at::Tensor packed_scales;
  if (is_packed) {
    CHECK_DIM(3, mat2);
    CHECK_DIM(3, scales);
    packed_w = mat2;
    packed_scales = scales;
  } else {
    CHECK_DIM(2, mat2);
    CHECK_DIM(2, scales);
    std::tie(packed_w, packed_scales) = convert_weight_w4a8_dynamic_packed(mat2, scales, group_size);
  }

  const int64_t M = mat1.size(0);
  const int64_t K = mat1.size(1);
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  const int64_t NB = packed_w.size(0);
  const int64_t num_groups = K / group_size;
  const int64_t w4a8_dynamic_group_bytes = 1 + (group_size / 2) * BLOCK_N;
  const uint8_t* packed_w_data = packed_w.data_ptr<uint8_t>();
  const int64_t last_n_size = static_cast<int64_t>(packed_w_data[((NB - 1) * num_groups) * w4a8_dynamic_group_bytes]);
  TORCH_CHECK(last_n_size > 0 && last_n_size <= BLOCK_N, "weight_w4a8_dynamic_linear: invalid packed tail block size.");
  const int64_t N = (NB - 1) * BLOCK_N + last_n_size;

  TORCH_CHECK(K % 2 == 0, "weight_w4a8_dynamic_linear: K must be even.");
  TORCH_CHECK(K % group_size == 0, "weight_w4a8_dynamic_linear: K must be divisible by group_size.");
  TORCH_CHECK(
      packed_w.size(1) == num_groups && packed_w.size(2) == w4a8_dynamic_group_bytes,
      "weight_w4a8_dynamic_linear: packed weight must have shape [NB, K / group_size, 1 + (group_size/2) * BLOCK_N].");
  TORCH_CHECK(
      packed_scales.size(0) == NB && packed_scales.size(1) == num_groups && packed_scales.size(2) == BLOCK_N,
      "weight_w4a8_dynamic_linear: packed scales must have shape [NB, K / group_size, BLOCK_N].");

  const float* bias_data = nullptr;
  if (bias.has_value()) {
    CHECK_INPUT(bias.value());
    CHECK_DIM(1, bias.value());
    CHECK_EQ(bias.value().size(0), N);
    TORCH_CHECK(bias.value().scalar_type() == at::kFloat, "weight_w4a8_dynamic_linear: bias must be float32.");
    bias_data = bias.value().data_ptr<float>();
  }

  const int64_t* input_permutation_data = nullptr;
  if (input_permutation.has_value()) {
    CHECK_INPUT(input_permutation.value());
    CHECK_DIM(1, input_permutation.value());
    CHECK_EQ(input_permutation.value().size(0), K);
    TORCH_CHECK(
        input_permutation.value().scalar_type() == at::kLong,
        "weight_w4a8_dynamic_linear: input_permutation must be int64.");
    input_permutation_data = input_permutation.value().data_ptr<int64_t>();
  }

  auto out = at::empty({M, N}, mat1.options());
  AT_DISPATCH_RVV_TYPES(mat1.scalar_type(), "weight_w4a8_dynamic_linear_kernel_impl", [&]() {
    AT_DISPATCH_BOOL(bias_data != nullptr, has_bias, [&] {
      const scalar_t* mat1_data = mat1.data_ptr<scalar_t>();
      scalar_t* out_data = out.data_ptr<scalar_t>();
      int64_t m = 0;
      constexpr int64_t batched_m2_min_blocks = 8;
      for (; m + 4 <= M && M < 2 * batched_m2_min_blocks; m += 4) {
        w4a8_dynamic_gemm_kernel_impl_m4<scalar_t, has_bias>(
            out_data + m * out.stride(0),
            mat1_data + m * mat1.stride(0),
            packed_w.data_ptr<uint8_t>(),
            packed_scales.data_ptr<float>(),
            bias_data,
            input_permutation_data,
            N,
            K,
            mat1.stride(0),
            out.stride(0),
            group_size);
      }
      const int64_t even_m = m + ((M - m) / 2) * 2;
      if (even_m / 2 >= batched_m2_min_blocks) {
        w4a8_dynamic_gemm_kernel_impl_m2_batched<scalar_t, has_bias>(
            out_data,
            mat1_data,
            packed_w.data_ptr<uint8_t>(),
            packed_scales.data_ptr<float>(),
            bias_data,
            input_permutation_data,
            even_m - m,
            N,
            K,
            mat1.stride(0),
            out.stride(0),
            group_size);
        m = even_m;
      } else {
        for (; m + 2 <= M; m += 2) {
          w4a8_dynamic_gemm_kernel_impl_m2<scalar_t, has_bias>(
              out_data + m * out.stride(0),
              mat1_data + m * mat1.stride(0),
              packed_w.data_ptr<uint8_t>(),
              packed_scales.data_ptr<float>(),
              bias_data,
              input_permutation_data,
              N,
              K,
              mat1.stride(0),
              out.stride(0),
              group_size);
        }
      }
      const int64_t tail = M - m;
      if (tail == 2) {
        w4a8_dynamic_gemm_kernel_impl_m2<scalar_t, has_bias>(
            out_data + m * out.stride(0),
            mat1_data + m * mat1.stride(0),
            packed_w.data_ptr<uint8_t>(),
            packed_scales.data_ptr<float>(),
            bias_data,
            input_permutation_data,
            N,
            K,
            mat1.stride(0),
            out.stride(0),
            group_size);
      } else if (tail == 1) {
        w4a8_dynamic_linear_kernel_impl_m1<scalar_t, has_bias>(
            out_data + m * out.stride(0),
            mat1_data + m * mat1.stride(0),
            packed_w.data_ptr<uint8_t>(),
            packed_scales.data_ptr<float>(),
            bias_data,
            input_permutation_data,
            N,
            K,
            group_size);
      }
    });
  });

  return out;
}

#endif  // CPU_CAPABILITY_RVV
