#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif
#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include <cstdio>
#include <vector>

#include "common.h"
#include "vector_helpers.h"
#include "vector_math.h"

#if defined(CPU_CAPABILITY_RVV)
// Clang 19+ version check
#if !defined(__clang__) || !defined(__clang_major__) || __clang_major__ < 19
#error "RVV backend requires Clang 19 or later. Please use Clang 19 or later to compile this file."
#endif

#include <riscv_vector.h>

// Check for Zvfh (FP16 vector) support

#if defined(__riscv_zvfh) || defined(__riscv_v)
#define HAS_ZVFH_DECODE 1
#else
#define HAS_ZVFH_DECODE 0
#endif

#endif

namespace {

#if defined(CPU_CAPABILITY_RVV)

template <typename scalar_t>
constexpr int64_t get_optimal_block_n_m1() {
  return 8;
}

template <typename scalar_t>
constexpr int64_t get_optimal_block_n_general() {
  if constexpr (std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>) {
    return 32;
  } else {
    return 64;
  }
}

#define DISPATCH_BLOCK_SIZE(FUNC, ...)                        \
  switch (nb_size) {                                          \
    case 1:                                                   \
      FUNC<scalar_t, index_t, 1>(__VA_ARGS__);                \
      break;                                                  \
    case 2:                                                   \
      FUNC<scalar_t, index_t, 2>(__VA_ARGS__);                \
      break;                                                  \
    case 3:                                                   \
      FUNC<scalar_t, index_t, 3>(__VA_ARGS__);                \
      break;                                                  \
    case 4:                                                   \
      FUNC<scalar_t, index_t, 4>(__VA_ARGS__);                \
      break;                                                  \
    case 5:                                                   \
      FUNC<scalar_t, index_t, 5>(__VA_ARGS__);                \
      break;                                                  \
    case 6:                                                   \
      FUNC<scalar_t, index_t, 6>(__VA_ARGS__);                \
      break;                                                  \
    case 7:                                                   \
      FUNC<scalar_t, index_t, 7>(__VA_ARGS__);                \
      break;                                                  \
    case 8:                                                   \
      FUNC<scalar_t, index_t, 8>(__VA_ARGS__);                \
      break;                                                  \
    default:                                                  \
      TORCH_CHECK(false, "Unexpected block size: ", nb_size); \
  }

template <typename scalar_t, typename index_t, int BLOCK_N_SIZE>
inline void index_gemm_kernel_nt_rvv_m1_block(
    const scalar_t* __restrict__ q_ptr,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    float scale,
    int64_t K,
    int64_t ldb,
    int64_t max_tokens) {
  int64_t b_idx[BLOCK_N_SIZE];
  const scalar_t* k_ptr[BLOCK_N_SIZE];
  for (int64_t i = 0; i < BLOCK_N_SIZE; ++i) {
    b_idx[i] = indices[i];
    TORCH_CHECK(b_idx[i] < max_tokens, "token index out of scope!");
    k_ptr[i] = B + b_idx[i] * ldb;
  }

  if constexpr (std::is_same_v<scalar_t, float>) {
    size_t vl_max = __riscv_vsetvlmax_e32m8();

    if constexpr (BLOCK_N_SIZE == 1) {
      vfloat32m8_t v_acc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);
      for (int64_t k = 0; k < K; k += vl_max) {
        size_t vl = __riscv_vsetvl_e32m8(K - k);
        vfloat32m8_t vq = __riscv_vle32_v_f32m8(q_ptr + k, vl);
        vfloat32m8_t vk0 = __riscv_vle32_v_f32m8(k_ptr[0] + k, vl);
        v_acc0 = __riscv_vfmacc_vv_f32m8_tu(v_acc0, vq, vk0, vl);
      }
      C[0] = reduce_sum_f32m8(v_acc0, vl_max) * scale;
    } else if constexpr (BLOCK_N_SIZE == 2) {
      vfloat32m8_t v_acc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);
      vfloat32m8_t v_acc1 = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);
      for (int64_t k = 0; k < K; k += vl_max) {
        size_t vl = __riscv_vsetvl_e32m8(K - k);
        vfloat32m8_t vq = __riscv_vle32_v_f32m8(q_ptr + k, vl);
        vfloat32m8_t vk0 = __riscv_vle32_v_f32m8(k_ptr[0] + k, vl);
        vfloat32m8_t vk1 = __riscv_vle32_v_f32m8(k_ptr[1] + k, vl);
        v_acc0 = __riscv_vfmacc_vv_f32m8_tu(v_acc0, vq, vk0, vl);
        v_acc1 = __riscv_vfmacc_vv_f32m8_tu(v_acc1, vq, vk1, vl);
      }
      C[0] = reduce_sum_f32m8(v_acc0, vl_max) * scale;
      C[1] = reduce_sum_f32m8(v_acc1, vl_max) * scale;
    } else if constexpr (BLOCK_N_SIZE == 4) {
      vfloat32m8_t v_acc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);
      vfloat32m8_t v_acc1 = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);
      vfloat32m8_t v_acc2 = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);
      vfloat32m8_t v_acc3 = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);
      for (int64_t k = 0; k < K; k += vl_max) {
        size_t vl = __riscv_vsetvl_e32m8(K - k);
        vfloat32m8_t vq = __riscv_vle32_v_f32m8(q_ptr + k, vl);
        vfloat32m8_t vk0 = __riscv_vle32_v_f32m8(k_ptr[0] + k, vl);
        vfloat32m8_t vk1 = __riscv_vle32_v_f32m8(k_ptr[1] + k, vl);
        vfloat32m8_t vk2 = __riscv_vle32_v_f32m8(k_ptr[2] + k, vl);
        vfloat32m8_t vk3 = __riscv_vle32_v_f32m8(k_ptr[3] + k, vl);
        v_acc0 = __riscv_vfmacc_vv_f32m8_tu(v_acc0, vq, vk0, vl);
        v_acc1 = __riscv_vfmacc_vv_f32m8_tu(v_acc1, vq, vk1, vl);
        v_acc2 = __riscv_vfmacc_vv_f32m8_tu(v_acc2, vq, vk2, vl);
        v_acc3 = __riscv_vfmacc_vv_f32m8_tu(v_acc3, vq, vk3, vl);
      }
      C[0] = reduce_sum_f32m8(v_acc0, vl_max) * scale;
      C[1] = reduce_sum_f32m8(v_acc1, vl_max) * scale;
      C[2] = reduce_sum_f32m8(v_acc2, vl_max) * scale;
      C[3] = reduce_sum_f32m8(v_acc3, vl_max) * scale;
    } else {
      float acc[BLOCK_N_SIZE];
      for (int64_t i = 0; i < BLOCK_N_SIZE; ++i) {
        vfloat32m8_t v_acc = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);
        for (int64_t k = 0; k < K; k += vl_max) {
          size_t vl = __riscv_vsetvl_e32m8(K - k);
          vfloat32m8_t vq = __riscv_vle32_v_f32m8(q_ptr + k, vl);
          vfloat32m8_t vk = __riscv_vle32_v_f32m8(k_ptr[i] + k, vl);
          v_acc = __riscv_vfmacc_vv_f32m8_tu(v_acc, vq, vk, vl);
        }
        acc[i] = reduce_sum_f32m8(v_acc, vl_max);
        C[i] = acc[i] * scale;
      }
    }

  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH_DECODE

    size_t vl_max = __riscv_vsetvlmax_e16m2();

    if constexpr (BLOCK_N_SIZE == 1) {
      vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      for (int64_t k = 0; k < K; k += vl_max) {
        size_t vl = __riscv_vsetvl_e16m2(K - k);
        vfloat16m2_t vq = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(q_ptr + k), vl);
        vfloat16m2_t vk0 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr[0] + k), vl);
        v_acc0 = __riscv_vfwmacc_vv_f32m4_tu(v_acc0, vq, vk0, vl);
      }
      size_t vl_max_f32 = __riscv_vsetvlmax_e32m4();
      C[0] = reduce_sum_f32m4(v_acc0, vl_max_f32) * scale;
    } else if constexpr (BLOCK_N_SIZE == 2) {
      vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      vfloat32m4_t v_acc1 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      for (int64_t k = 0; k < K; k += vl_max) {
        size_t vl = __riscv_vsetvl_e16m2(K - k);
        vfloat16m2_t vq = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(q_ptr + k), vl);
        vfloat16m2_t vk0 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr[0] + k), vl);
        vfloat16m2_t vk1 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr[1] + k), vl);
        v_acc0 = __riscv_vfwmacc_vv_f32m4_tu(v_acc0, vq, vk0, vl);
        v_acc1 = __riscv_vfwmacc_vv_f32m4_tu(v_acc1, vq, vk1, vl);
      }
      size_t vl_max_f32 = __riscv_vsetvlmax_e32m4();
      C[0] = reduce_sum_f32m4(v_acc0, vl_max_f32) * scale;
      C[1] = reduce_sum_f32m4(v_acc1, vl_max_f32) * scale;
    } else if constexpr (BLOCK_N_SIZE == 4) {
      vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      vfloat32m4_t v_acc1 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      vfloat32m4_t v_acc2 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      vfloat32m4_t v_acc3 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      for (int64_t k = 0; k < K; k += vl_max) {
        size_t vl = __riscv_vsetvl_e16m2(K - k);
        vfloat16m2_t vq = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(q_ptr + k), vl);
        vfloat16m2_t vk0 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr[0] + k), vl);
        vfloat16m2_t vk1 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr[1] + k), vl);
        vfloat16m2_t vk2 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr[2] + k), vl);
        vfloat16m2_t vk3 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr[3] + k), vl);
        v_acc0 = __riscv_vfwmacc_vv_f32m4_tu(v_acc0, vq, vk0, vl);
        v_acc1 = __riscv_vfwmacc_vv_f32m4_tu(v_acc1, vq, vk1, vl);
        v_acc2 = __riscv_vfwmacc_vv_f32m4_tu(v_acc2, vq, vk2, vl);
        v_acc3 = __riscv_vfwmacc_vv_f32m4_tu(v_acc3, vq, vk3, vl);
      }
      size_t vl_max_f32 = __riscv_vsetvlmax_e32m4();
      C[0] = reduce_sum_f32m4(v_acc0, vl_max_f32) * scale;
      C[1] = reduce_sum_f32m4(v_acc1, vl_max_f32) * scale;
      C[2] = reduce_sum_f32m4(v_acc2, vl_max_f32) * scale;
      C[3] = reduce_sum_f32m4(v_acc3, vl_max_f32) * scale;
    } else {
      // Fallback for other sizes (3, 5, 6, 7, 8)
      for (int64_t i = 0; i < BLOCK_N_SIZE; ++i) {
        size_t vl_max_f32 = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t v_acc = __riscv_vfmv_v_f_f32m4(0.0f, vl_max_f32);
        for (int64_t k = 0; k < K; k += vl_max) {
          size_t vl = __riscv_vsetvl_e16m2(K - k);
          vfloat16m2_t vq = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(q_ptr + k), vl);
          vfloat16m2_t vk = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr[i] + k), vl);
          // Widening MAC: v_acc (f32m4) += vq (f16m2) * vk (f16m2)
          v_acc = __riscv_vfwmacc_vv_f32m4_tu(v_acc, vq, vk, vl);
        }
        C[i] = reduce_sum_f32m4(v_acc, vl_max_f32) * scale;
      }
    }
#else

    for (int64_t i = 0; i < BLOCK_N_SIZE; ++i) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += static_cast<float>(q_ptr[k]) * static_cast<float>(k_ptr[i][k]);
      }
      C[i] = sum * scale;
    }
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    size_t vl_max = __riscv_vsetvlmax_e32m4();

    if constexpr (BLOCK_N_SIZE == 1) {
      vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
      for (int64_t k = 0; k < K; k += vl_max) {
        size_t vl = __riscv_vsetvl_e32m4(K - k);
        vfloat32m4_t vq = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(q_ptr + k), vl);
        vfloat32m4_t vk0 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr[0] + k), vl);
        v_acc0 = __riscv_vfmacc_vv_f32m4_tu(v_acc0, vq, vk0, vl);
      }
      C[0] = reduce_sum_f32m4(v_acc0, vl_max) * scale;
    } else if constexpr (BLOCK_N_SIZE == 2) {
      vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
      vfloat32m4_t v_acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
      for (int64_t k = 0; k < K; k += vl_max) {
        size_t vl = __riscv_vsetvl_e32m4(K - k);
        vfloat32m4_t vq = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(q_ptr + k), vl);
        vfloat32m4_t vk0 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr[0] + k), vl);
        vfloat32m4_t vk1 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr[1] + k), vl);
        v_acc0 = __riscv_vfmacc_vv_f32m4_tu(v_acc0, vq, vk0, vl);
        v_acc1 = __riscv_vfmacc_vv_f32m4_tu(v_acc1, vq, vk1, vl);
      }
      C[0] = reduce_sum_f32m4(v_acc0, vl_max) * scale;
      C[1] = reduce_sum_f32m4(v_acc1, vl_max) * scale;
    } else if constexpr (BLOCK_N_SIZE == 4) {
      vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
      vfloat32m4_t v_acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
      vfloat32m4_t v_acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
      vfloat32m4_t v_acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
      for (int64_t k = 0; k < K; k += vl_max) {
        size_t vl = __riscv_vsetvl_e32m4(K - k);
        vfloat32m4_t vq = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(q_ptr + k), vl);
        vfloat32m4_t vk0 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr[0] + k), vl);
        vfloat32m4_t vk1 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr[1] + k), vl);
        vfloat32m4_t vk2 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr[2] + k), vl);
        vfloat32m4_t vk3 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr[3] + k), vl);

        v_acc0 = __riscv_vfmacc_vv_f32m4_tu(v_acc0, vq, vk0, vl);
        v_acc1 = __riscv_vfmacc_vv_f32m4_tu(v_acc1, vq, vk1, vl);
        v_acc2 = __riscv_vfmacc_vv_f32m4_tu(v_acc2, vq, vk2, vl);
        v_acc3 = __riscv_vfmacc_vv_f32m4_tu(v_acc3, vq, vk3, vl);
      }
      C[0] = reduce_sum_f32m4(v_acc0, vl_max) * scale;
      C[1] = reduce_sum_f32m4(v_acc1, vl_max) * scale;
      C[2] = reduce_sum_f32m4(v_acc2, vl_max) * scale;
      C[3] = reduce_sum_f32m4(v_acc3, vl_max) * scale;
    } else {
      // Fallback for other sizes (3, 5, 6, 7, 8)
      for (int64_t i = 0; i < BLOCK_N_SIZE; ++i) {
        vfloat32m4_t v_acc = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
        for (int64_t k = 0; k < K; k += vl_max) {
          size_t vl = __riscv_vsetvl_e32m4(K - k);
          vfloat32m4_t vq = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(q_ptr + k), vl);

          vfloat32m4_t vk = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr[i] + k), vl);
          v_acc = __riscv_vfmacc_vv_f32m4_tu(v_acc, vq, vk, vl);
        }
        C[i] = reduce_sum_f32m4(v_acc, vl_max) * scale;
      }
    }
  } else {
    // Fallback
    for (int64_t i = 0; i < BLOCK_N_SIZE; ++i) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += static_cast<float>(q_ptr[k]) * static_cast<float>(k_ptr[i][k]);
      }
      C[i] = sum * scale;
    }
  }
}

template <typename scalar_t, typename index_t, int BLOCK_N_SIZE>
inline void index_gemm_kernel_nt_rvv_m1_block_int8(
    const scalar_t* __restrict__ q_ptr,
    const int8_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    float scale,
    int64_t K,
    int64_t ldb,
    int64_t max_tokens) {
  int64_t b_idx[BLOCK_N_SIZE];
  const int8_t* k_ptr[BLOCK_N_SIZE];
  for (int64_t i = 0; i < BLOCK_N_SIZE; ++i) {
    b_idx[i] = indices[i];
    TORCH_CHECK(b_idx[i] < max_tokens, "token index out of scope!");
    k_ptr[i] = B + b_idx[i] * ldb;
  }

#if HAS_ZVFH_DECODE
  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    // FP16 + INT8 path
    size_t vl_max = __riscv_vsetvlmax_e16m2();

    if constexpr (BLOCK_N_SIZE == 1) {
      vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      for (int64_t k = 0; k < K; k += vl_max) {
        size_t vl = __riscv_vsetvl_e16m2(K - k);
        vfloat16m2_t v_q = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(q_ptr + k), vl);
        vfloat16m2_t vk_f16 = int8_to_f16m2(k_ptr[0] + k, vl);
        v_acc0 = __riscv_vfwmacc_vv_f32m4_tu(v_acc0, v_q, vk_f16, vl);
      }
      size_t vl_max_f32 = __riscv_vsetvlmax_e32m4();
      C[0] = reduce_sum_f32m4(v_acc0, vl_max_f32) * scale;
    } else if constexpr (BLOCK_N_SIZE == 2) {
      vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      vfloat32m4_t v_acc1 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      for (int64_t k = 0; k < K; k += vl_max) {
        size_t vl = __riscv_vsetvl_e16m2(K - k);
        vfloat16m2_t v_q = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(q_ptr + k), vl);

        vfloat16m2_t vk0_f16 = int8_to_f16m2(k_ptr[0] + k, vl);
        v_acc0 = __riscv_vfwmacc_vv_f32m4_tu(v_acc0, v_q, vk0_f16, vl);

        vfloat16m2_t vk1_f16 = int8_to_f16m2(k_ptr[1] + k, vl);
        v_acc1 = __riscv_vfwmacc_vv_f32m4_tu(v_acc1, v_q, vk1_f16, vl);
      }
      size_t vl_max_f32 = __riscv_vsetvlmax_e32m4();
      C[0] = reduce_sum_f32m4(v_acc0, vl_max_f32) * scale;
      C[1] = reduce_sum_f32m4(v_acc1, vl_max_f32) * scale;
    } else if constexpr (BLOCK_N_SIZE == 4) {
      vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      vfloat32m4_t v_acc1 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      vfloat32m4_t v_acc2 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      vfloat32m4_t v_acc3 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
      for (int64_t k = 0; k < K; k += vl_max) {
        size_t vl = __riscv_vsetvl_e16m2(K - k);
        vfloat16m2_t v_q = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(q_ptr + k), vl);

        auto process_int8_k = [&](int idx, vfloat32m4_t& acc) {
          vfloat16m2_t vk_f16 = int8_to_f16m2(k_ptr[idx] + k, vl);
          acc = __riscv_vfwmacc_vv_f32m4_tu(acc, v_q, vk_f16, vl);
        };

        process_int8_k(0, v_acc0);
        process_int8_k(1, v_acc1);
        process_int8_k(2, v_acc2);
        process_int8_k(3, v_acc3);
      }
      size_t vl_max_f32 = __riscv_vsetvlmax_e32m4();
      C[0] = reduce_sum_f32m4(v_acc0, vl_max_f32) * scale;
      C[1] = reduce_sum_f32m4(v_acc1, vl_max_f32) * scale;
      C[2] = reduce_sum_f32m4(v_acc2, vl_max_f32) * scale;
      C[3] = reduce_sum_f32m4(v_acc3, vl_max_f32) * scale;
    } else {
      // Fallback for other sizes
      for (int64_t i = 0; i < BLOCK_N_SIZE; ++i) {
        vfloat32m4_t v_acc = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
        for (int64_t k = 0; k < K; k += vl_max) {
          size_t vl = __riscv_vsetvl_e16m2(K - k);
          vfloat16m2_t v_q = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(q_ptr + k), vl);
          vfloat16m2_t vk_f16 = int8_to_f16m2(k_ptr[i] + k, vl);
          v_acc = __riscv_vfwmacc_vv_f32m4_tu(v_acc, v_q, vk_f16, vl);
        }
        size_t vl_max_f32 = __riscv_vsetvlmax_e32m4();
        C[i] = reduce_sum_f32m4(v_acc, vl_max_f32) * scale;
      }
    }
  } else
#endif
  {
    // Fallback: scalar path for INT8
    for (int64_t i = 0; i < BLOCK_N_SIZE; ++i) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        float val_k = static_cast<float>(k_ptr[i][k]);
        sum += static_cast<float>(q_ptr[k]) * val_k;
      }
      C[i] = sum * scale;
    }
  }
}

template <typename scalar_t, typename index_t>
inline void index_gemm_kernel_nt_rvv(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    float scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t max_tokens) {
  // M=1 special path: Optimized for decode attention
  if (M == 1) {
    constexpr int64_t BLOCK_N = get_optimal_block_n_m1<scalar_t>();
    const int64_t NB = div_up(N, BLOCK_N);
    const scalar_t* q_ptr_base = A;

    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);
      DISPATCH_BLOCK_SIZE(
          index_gemm_kernel_nt_rvv_m1_block,
          q_ptr_base,
          B,
          C + nb_start,
          indices + nb_start,
          scale,
          K,
          ldb,
          max_tokens);
    }
    return;
  }

  // M > 1 path (rarely used in decode, mainly for prefill/extend)
  for (int64_t m = 0; m < M; ++m) {
    const scalar_t* q_ptr_base = A + m * lda;

    int64_t n = 0;

#if HAS_ZVFH_DECODE
    if constexpr (std::is_same_v<scalar_t, at::Half>) {
      size_t vl_max = __riscv_vsetvlmax_e16m2();

      for (; n + 3 < N; n += 4) {
        int64_t b_idx0 = indices[n];
        int64_t b_idx1 = indices[n + 1];
        int64_t b_idx2 = indices[n + 2];
        int64_t b_idx3 = indices[n + 3];

        TORCH_CHECK(b_idx0 >= 0 && b_idx0 < max_tokens, "token index out of scope!");
        TORCH_CHECK(b_idx1 >= 0 && b_idx1 < max_tokens, "token index out of scope!");
        TORCH_CHECK(b_idx2 >= 0 && b_idx2 < max_tokens, "token index out of scope!");
        TORCH_CHECK(b_idx3 >= 0 && b_idx3 < max_tokens, "token index out of scope!");

        const scalar_t* k_ptr0 = B + b_idx0 * ldb;
        const scalar_t* k_ptr1 = B + b_idx1 * ldb;
        const scalar_t* k_ptr2 = B + b_idx2 * ldb;
        const scalar_t* k_ptr3 = B + b_idx3 * ldb;

        vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
        vfloat32m4_t v_acc1 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
        vfloat32m4_t v_acc2 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
        vfloat32m4_t v_acc3 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());

        for (int64_t k = 0; k < K; k += vl_max) {
          size_t vl = __riscv_vsetvl_e16m2(K - k);

          vfloat16m2_t v_q = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(q_ptr_base + k), vl);
          vfloat16m2_t v_k0 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr0 + k), vl);
          vfloat16m2_t v_k1 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr1 + k), vl);
          vfloat16m2_t v_k2 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr2 + k), vl);
          vfloat16m2_t v_k3 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr3 + k), vl);
          v_acc0 = __riscv_vfwmacc_vv_f32m4_tu(v_acc0, v_q, v_k0, vl);
          v_acc1 = __riscv_vfwmacc_vv_f32m4_tu(v_acc1, v_q, v_k1, vl);
          v_acc2 = __riscv_vfwmacc_vv_f32m4_tu(v_acc2, v_q, v_k2, vl);
          v_acc3 = __riscv_vfwmacc_vv_f32m4_tu(v_acc3, v_q, v_k3, vl);
        }

        // Reductions
        size_t vl_max_f32 = __riscv_vsetvlmax_e32m4();
        C[m * ldc + n] = reduce_sum_f32m4(v_acc0, vl_max_f32) * scale;
        C[m * ldc + n + 1] = reduce_sum_f32m4(v_acc1, vl_max_f32) * scale;
        C[m * ldc + n + 2] = reduce_sum_f32m4(v_acc2, vl_max_f32) * scale;
        C[m * ldc + n + 3] = reduce_sum_f32m4(v_acc3, vl_max_f32) * scale;
      }
    }
#endif

    if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      size_t vl_max = __riscv_vsetvlmax_e32m4();
      bool is_contiguous = true;
      int64_t start_idx = indices[0];
      size_t vl_idx_max = __riscv_vsetvlmax_e64m4();

      for (int64_t ck = 0; ck < N; ck += vl_idx_max) {
        size_t vl = __riscv_vsetvl_e64m4(N - ck);
        vint64m4_t v_idx = __riscv_vle64_v_i64m4(reinterpret_cast<const int64_t*>(indices + ck), vl);
        vint64m4_t v_seq = __riscv_vid_v_i64m4(vl);
        v_seq = __riscv_vadd_vx_i64m4(v_seq, start_idx + ck, vl);
        vbool16_t v_mask = __riscv_vmseq_vv_i64m4_b16(v_idx, v_seq, vl);
        long matches = __riscv_vcpop_m_b16(v_mask, vl);
        if (matches != vl) {
          is_contiguous = false;
          break;
        }
      }

      for (; n + 1 < N; n += 2) {
        int64_t b_idx0, b_idx1;
        if (is_contiguous) {
          b_idx0 = start_idx + n;
          b_idx1 = start_idx + n + 1;
        } else {
          b_idx0 = indices[n];
          b_idx1 = indices[n + 1];
        }

        TORCH_CHECK(b_idx0 >= 0 && b_idx0 < max_tokens, "token index out of scope!");
        TORCH_CHECK(b_idx1 >= 0 && b_idx1 < max_tokens, "token index out of scope!");

        const scalar_t* k_ptr0 = B + b_idx0 * ldb;
        const scalar_t* k_ptr1 = B + b_idx1 * ldb;
        vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
        vfloat32m4_t v_acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);

        for (int64_t k = 0; k < K; k += vl_max) {
          size_t vl = __riscv_vsetvl_e32m4(K - k);
          vfloat32m4_t v_q = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(q_ptr_base + k), vl);

          auto process_token = [&](vfloat32m4_t& acc, const scalar_t* ptr) {
            vfloat32m4_t v_k = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(ptr + k), vl);
            acc = __riscv_vfmacc_vv_f32m4_tu(acc, v_q, v_k, vl);
          };

          process_token(v_acc0, k_ptr0);
          process_token(v_acc1, k_ptr1);
        }

        C[m * ldc + n] = reduce_sum_f32m4(v_acc0, vl_max) * scale;
        C[m * ldc + n + 1] = reduce_sum_f32m4(v_acc1, vl_max) * scale;
      }
    }

    for (; n < N; ++n) {
      int64_t b_idx = indices[n];
      TORCH_CHECK(b_idx >= 0 && b_idx < max_tokens, "token index out of scope!");
      const scalar_t* k_ptr = B + b_idx * ldb;

      float scalar_sum = 0.0f;

      if constexpr (std::is_same_v<scalar_t, float>) {
        size_t vl_max = __riscv_vsetvlmax_e32m8();
        vfloat32m8_t v_acc = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);

        for (int64_t k = 0; k < K; k += vl_max) {
          size_t vl = __riscv_vsetvl_e32m8(K - k);
          vfloat32m8_t vq = __riscv_vle32_v_f32m8(q_ptr_base + k, vl);
          vfloat32m8_t vk = __riscv_vle32_v_f32m8(k_ptr + k, vl);
          v_acc = __riscv_vfmacc_vv_f32m8_tu(v_acc, vq, vk, vl);
        }
        scalar_sum = reduce_sum_f32m8(v_acc, vl_max);

      } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH_DECODE
        size_t vl_max = __riscv_vsetvlmax_e16m4();
        vfloat32m8_t v_acc = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());

        for (int64_t k = 0; k < K; k += vl_max) {
          size_t vl = __riscv_vsetvl_e16m4(K - k);
          vfloat16m4_t vq = __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16*>(q_ptr_base + k), vl);
          vfloat16m4_t vk = __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16*>(k_ptr + k), vl);
          v_acc = __riscv_vfwmacc_vv_f32m8_tu(v_acc, vq, vk, vl);
        }
        scalar_sum = reduce_sum_f32m8(v_acc, __riscv_vsetvlmax_e32m8());
#else
        for (int64_t k = 0; k < K; ++k) {
          scalar_sum += static_cast<float>(q_ptr_base[k]) * static_cast<float>(k_ptr[k]);
        }
#endif
      } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        size_t vl_max = __riscv_vsetvlmax_e32m8();
        vfloat32m8_t v_acc = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);

        for (int64_t k = 0; k < K; k += vl_max) {
          size_t vl = __riscv_vsetvl_e32m8(K - k);
          vfloat32m8_t vq = bf16_to_f32m8(reinterpret_cast<const uint16_t*>(q_ptr_base + k), vl);
          vfloat32m8_t vk = bf16_to_f32m8(reinterpret_cast<const uint16_t*>(k_ptr + k), vl);
          v_acc = __riscv_vfmacc_vv_f32m8_tu(v_acc, vq, vk, vl);
        }
        scalar_sum = reduce_sum_f32m8(v_acc, vl_max);
      } else {
        for (int64_t k = 0; k < K; ++k) {
          scalar_sum += static_cast<float>(q_ptr_base[k]) * static_cast<float>(k_ptr[k]);
        }
      }

      C[m * ldc + n] = scalar_sum * scale;
    }
  }
}

template <typename scalar_t, typename index_t>
inline void index_gemm_kernel_nt_rvv(
    const scalar_t* __restrict__ A,
    const int8_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    float scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t max_tokens) {
  // M=1 path: Used by decode attention (M>1 is rare, handled by scalar fallback)
  if (M == 1) {
    constexpr int64_t BLOCK_N = 8;
    const int64_t NB = div_up(N, BLOCK_N);
    const scalar_t* q_ptr_base = A;

    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);
      DISPATCH_BLOCK_SIZE(
          index_gemm_kernel_nt_rvv_m1_block_int8,
          q_ptr_base,
          B,
          C + nb_start,
          indices + nb_start,
          scale,
          K,
          ldb,
          max_tokens);
    }
    return;
  }

  // M>1 fallback: Scalar path (rarely used in decode attention)
  for (int64_t m = 0; m < M; ++m) {
    const scalar_t* q_ptr_base = A + m * lda;
    for (int64_t n = 0; n < N; ++n) {
      int64_t b_idx = indices[n];
      const int8_t* k_ptr = B + b_idx * ldb;
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += static_cast<float>(q_ptr_base[k]) * static_cast<float>(k_ptr[k]);
      }
      C[m * ldc + n] = sum * scale;
    }
  }
}

// 2. Softmax
inline float exp_rvv(const float* __restrict__ scores, float* __restrict__ output, int64_t N, float m_i) {
  if (N <= 0) {
    return 0.0f;
  }
  size_t vl_max = __riscv_vsetvlmax_e32m8();
  vfloat32m8_t v_sum_acc = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);

  int64_t i = 0;
  for (; i < N; i += vl_max) {
    size_t vl = __riscv_vsetvl_e32m8(N - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(scores + i, vl);

    vx = __riscv_vfsub_vf_f32m8(vx, m_i, vl);

    vfloat32m8_t vex = vfexp_f32m8(vx, vl);

    __riscv_vse32_v_f32m8(output + i, vex, vl);

    v_sum_acc = __riscv_vfadd_vv_f32m8(v_sum_acc, vex, vl);
  }

  // Use helper function for reduction (v_sum_scalar is already initialized to 0.0f)
  return reduce_sum_f32m8(v_sum_acc, vl_max);
}

template <typename scalar_t, typename index_t>
inline void prob_value_aggregate_rvv(
    const float* __restrict__ probs,
    const scalar_t* __restrict__ values,
    float* __restrict__ output,
    const index_t* __restrict__ indices,
    int64_t N,
    int64_t head_dim,
    int64_t v_strideN,
    float scale,
    int64_t max_tokens) {
  if (N == 0 || head_dim == 0) return;

  size_t vl_max = __riscv_vsetvlmax_e32m8();
  for (int64_t d = 0; d < head_dim; d += vl_max) {
    size_t vl = __riscv_vsetvl_e32m8(head_dim - d);

    vfloat32m8_t vacc = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    int64_t n = 0;
    for (; n + 3 < N; n += 4) {
      float p0 = probs[n];
      float p1 = probs[n + 1];
      float p2 = probs[n + 2];
      float p3 = probs[n + 3];

      int64_t v_idx0 = indices[n];
      int64_t v_idx1 = indices[n + 1];
      int64_t v_idx2 = indices[n + 2];
      int64_t v_idx3 = indices[n + 3];

      const scalar_t* v_ptr0 = values + v_idx0 * v_strideN + d;
      const scalar_t* v_ptr1 = values + v_idx1 * v_strideN + d;
      const scalar_t* v_ptr2 = values + v_idx2 * v_strideN + d;
      const scalar_t* v_ptr3 = values + v_idx3 * v_strideN + d;

      if constexpr (std::is_same_v<scalar_t, float>) {
        vfloat32m8_t vv0 = __riscv_vle32_v_f32m8(v_ptr0, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p0, vv0, vl);

        vfloat32m8_t vv1 = __riscv_vle32_v_f32m8(v_ptr1, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p1, vv1, vl);

        vfloat32m8_t vv2 = __riscv_vle32_v_f32m8(v_ptr2, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p2, vv2, vl);

        vfloat32m8_t vv3 = __riscv_vle32_v_f32m8(v_ptr3, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p3, vv3, vl);
      } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH_DECODE
        // FP16 path
        vfloat16m4_t vv0_16 = __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16*>(v_ptr0), vl);
        vfloat32m8_t vv0 = __riscv_vfwcvt_f_f_v_f32m8(vv0_16, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p0, vv0, vl);

        vfloat16m4_t vv1_16 = __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16*>(v_ptr1), vl);
        vfloat32m8_t vv1 = __riscv_vfwcvt_f_f_v_f32m8(vv1_16, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p1, vv1, vl);

        vfloat16m4_t vv2_16 = __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16*>(v_ptr2), vl);
        vfloat32m8_t vv2 = __riscv_vfwcvt_f_f_v_f32m8(vv2_16, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p2, vv2, vl);

        vfloat16m4_t vv3_16 = __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16*>(v_ptr3), vl);
        vfloat32m8_t vv3 = __riscv_vfwcvt_f_f_v_f32m8(vv3_16, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p3, vv3, vl);
#else
        // Fallback
        float temp[256];

        // v0
        for (size_t j = 0; j < vl; ++j)
          temp[j] = static_cast<float>(v_ptr0[j]);
        vfloat32m8_t vv = __riscv_vle32_v_f32m8(temp, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p0, vv, vl);

        // v1
        for (size_t j = 0; j < vl; ++j)
          temp[j] = static_cast<float>(v_ptr1[j]);
        vv = __riscv_vle32_v_f32m8(temp, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p1, vv, vl);

        // v2
        for (size_t j = 0; j < vl; ++j)
          temp[j] = static_cast<float>(v_ptr2[j]);
        vv = __riscv_vle32_v_f32m8(temp, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p2, vv, vl);

        // v3
        for (size_t j = 0; j < vl; ++j)
          temp[j] = static_cast<float>(v_ptr3[j]);
        vv = __riscv_vle32_v_f32m8(temp, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p3, vv, vl);
#endif
      } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        vfloat32m8_t vv0 = bf16_to_f32m8(reinterpret_cast<const uint16_t*>(v_ptr0), vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p0, vv0, vl);
        vfloat32m8_t vv1 = bf16_to_f32m8(reinterpret_cast<const uint16_t*>(v_ptr1), vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p1, vv1, vl);
        vfloat32m8_t vv2 = bf16_to_f32m8(reinterpret_cast<const uint16_t*>(v_ptr2), vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p2, vv2, vl);
        vfloat32m8_t vv3 = bf16_to_f32m8(reinterpret_cast<const uint16_t*>(v_ptr3), vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p3, vv3, vl);

      } else {
        // Other types
        float temp[256];
        // v0
        for (size_t j = 0; j < vl; ++j)
          temp[j] = static_cast<float>(v_ptr0[j]);
        vfloat32m8_t vv = __riscv_vle32_v_f32m8(temp, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p0, vv, vl);
        // v1
        for (size_t j = 0; j < vl; ++j)
          temp[j] = static_cast<float>(v_ptr1[j]);
        vv = __riscv_vle32_v_f32m8(temp, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p1, vv, vl);
        // v2
        for (size_t j = 0; j < vl; ++j)
          temp[j] = static_cast<float>(v_ptr2[j]);
        vv = __riscv_vle32_v_f32m8(temp, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p2, vv, vl);
        // v3
        for (size_t j = 0; j < vl; ++j)
          temp[j] = static_cast<float>(v_ptr3[j]);
        vv = __riscv_vle32_v_f32m8(temp, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p3, vv, vl);
      }
    }

    for (; n < N; ++n) {
      float prob = probs[n];
      int64_t v_idx = indices[n];
      const scalar_t* v_ptr = values + v_idx * v_strideN + d;

      if constexpr (std::is_same_v<scalar_t, float>) {
        vfloat32m8_t vv = __riscv_vle32_v_f32m8(v_ptr, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, prob, vv, vl);
      } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH_DECODE
        vfloat16m4_t vv_16 = __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16*>(v_ptr), vl);

        // Widening MAC: vacc += prob * vv_16
        vfloat32m8_t vv = __riscv_vfwcvt_f_f_v_f32m8(vv_16, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, prob, vv, vl);
#else
        // Fallback
        float temp[256];
        for (size_t j = 0; j < vl; ++j)
          temp[j] = static_cast<float>(v_ptr[j]);
        vfloat32m8_t vv = __riscv_vle32_v_f32m8(temp, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, prob, vv, vl);
#endif
      } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        vfloat32m8_t vv = bf16_to_f32m8(reinterpret_cast<const uint16_t*>(v_ptr), vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, prob, vv, vl);
      } else {
        // Fallback
        float temp[256];
        for (size_t j = 0; j < vl; ++j)
          temp[j] = static_cast<float>(v_ptr[j]);
        vfloat32m8_t vv = __riscv_vle32_v_f32m8(temp, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, prob, vv, vl);
      }
    }

    vfloat32m8_t vout = __riscv_vle32_v_f32m8(output + d, vl);
    vout = __riscv_vfmul_vf_f32m8(vout, scale, vl);
    vout = __riscv_vfadd_vv_f32m8(vout, vacc, vl);
    __riscv_vse32_v_f32m8(output + d, vout, vl);
  }
}

template <typename scalar_t, typename index_t>
inline void prob_value_aggregate_rvv(
    const float* __restrict__ probs,
    const int8_t* __restrict__ values,
    float* __restrict__ output,
    const index_t* __restrict__ indices,
    int64_t N,
    int64_t head_dim,
    int64_t v_strideN,
    float scale,
    int64_t max_tokens,
    float v_scale) {
  if (N == 0 || head_dim == 0) return;

  size_t vl_max = __riscv_vsetvlmax_e32m8();

  for (int64_t d = 0; d < head_dim; d += vl_max) {
    size_t vl = __riscv_vsetvl_e32m8(head_dim - d);
    vfloat32m8_t vacc = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    int64_t n = 0;

    for (; n + 3 < N; n += 4) {
      float p0 = probs[n];
      float p1 = probs[n + 1];
      float p2 = probs[n + 2];
      float p3 = probs[n + 3];

      int64_t v_idx0 = indices[n];
      int64_t v_idx1 = indices[n + 1];
      int64_t v_idx2 = indices[n + 2];
      int64_t v_idx3 = indices[n + 3];

      const int8_t* v_ptr0 = values + v_idx0 * v_strideN + d;
      const int8_t* v_ptr1 = values + v_idx1 * v_strideN + d;
      const int8_t* v_ptr2 = values + v_idx2 * v_strideN + d;
      const int8_t* v_ptr3 = values + v_idx3 * v_strideN + d;

      auto process = [&](float p, const int8_t* ptr) {
        vfloat32m8_t vv_f32 = int8_to_f32m8(ptr, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, p, vv_f32, vl);
      };

      process(p0, v_ptr0);
      process(p1, v_ptr1);
      process(p2, v_ptr2);
      process(p3, v_ptr3);
    }

    for (; n < N; ++n) {
      float prob = probs[n];
      int64_t v_idx = indices[n];
      const int8_t* v_ptr = values + v_idx * v_strideN + d;

      vfloat32m8_t vv_f32 = int8_to_f32m8(v_ptr, vl);
      vacc = __riscv_vfmacc_vf_f32m8(vacc, prob, vv_f32, vl);
    }

    // Apply v_scale to vacc
    vacc = __riscv_vfmul_vf_f32m8(vacc, v_scale, vl);

    vfloat32m8_t vout = __riscv_vle32_v_f32m8(output + d, vl);
    vout = __riscv_vfmul_vf_f32m8(vout, scale, vl);
    vout = __riscv_vfadd_vv_f32m8(vout, vacc, vl);
    __riscv_vse32_v_f32m8(output + d, vout, vl);
  }
}

}  // namespace

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ src, int64_t size) {
  copy_stub_rvv<scalar_t>(out, src, size);
}

template <typename scalar_t>
void decode_set_kv_buffer(
    scalar_t* __restrict__ k_buffer,
    scalar_t* __restrict__ v_buffer,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ loc,
    int64_t batches,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    int64_t nk_strideN,
    int64_t nk_strideH,
    int64_t nv_strideN,
    int64_t nv_strideH,
    bool is_mla) {
  at::parallel_for(0, batches * num_heads_kv, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, head_kv_id{0};
    data_index_init(begin, bs, batches, head_kv_id, num_heads_kv);

    for (int64_t i = begin; i < end; i++) {
      int64_t loc_val = loc[bs];
      scalar_t* k_buffer_ptr = k_buffer + loc_val * k_strideN + head_kv_id * k_strideH;
      const scalar_t* new_key_ptr = key + bs * nk_strideN + head_kv_id * nk_strideH;
      copy_stub<scalar_t>(k_buffer_ptr, new_key_ptr, head_size);
      if (!is_mla) {
        scalar_t* v_buffer_ptr = v_buffer + loc_val * v_strideN + head_kv_id * v_strideH;
        const scalar_t* new_value_ptr = value + bs * nv_strideN + head_kv_id * nv_strideH;
        copy_stub<scalar_t>(v_buffer_ptr, new_value_ptr, head_size_v);
      }

      // move to the next index
      data_index_step(bs, batches, head_kv_id, num_heads_kv);
    }
  });
}

template <typename scalar_t>
inline void decode_accumulate_kv_splits_rvv(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t l_stride1,
    int64_t l_stride2) {
  // parallel on [batches, num_heads]
  at::parallel_for(0, batches * num_heads, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      float* __restrict__ acc = attn_logits + i * l_stride1;

      float s_prime = 0.f;
      float m_prime = -std::numeric_limits<float>::infinity();

      for (int64_t kv_id = 0; kv_id < num_kv_splits; ++kv_id) {
        float* __restrict__ tv = acc + kv_id * l_stride2;
        const float tlogic = tv[head_size_v];

        float m_i = std::max(tlogic, m_prime);
        float m_delta = std::exp(m_prime - m_i);
        float e_logic = std::exp(tlogic - m_i);

        if (kv_id != 0) {
          size_t vl_max = __riscv_vsetvlmax_e32m8();
          for (int64_t d = 0; d < head_size_v; d += vl_max) {
            size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(acc + d, vl);
            vfloat32m8_t vtv = __riscv_vle32_v_f32m8(tv + d, vl);

            vfloat32m8_t vm_delta = __riscv_vfmv_v_f_f32m8(m_delta, vl);
            vfloat32m8_t ve_logic = __riscv_vfmv_v_f_f32m8(e_logic, vl);

            // acc = acc * m_delta
            vacc = __riscv_vfmul_vv_f32m8(vacc, vm_delta, vl);
            // acc = acc + tv * e_logic
            vacc = __riscv_vfmacc_vv_f32m8(vacc, ve_logic, vtv, vl);

            __riscv_vse32_v_f32m8(acc + d, vacc, vl);
          }
        }

        s_prime = s_prime * m_delta + e_logic;
        m_prime = m_i;
      }
      float scale = (s_prime > 0.0f && std::isfinite(s_prime)) ? (1.0f / s_prime) : 0.0f;

      size_t vl_max = __riscv_vsetvlmax_e32m8();
      for (int64_t d = 0; d < head_size_v; d += vl_max) {
        size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
        vfloat32m8_t vacc = __riscv_vle32_v_f32m8(acc + d, vl);
        vfloat32m8_t vscale = __riscv_vfmv_v_f_f32m8(scale, vl);
        vacc = __riscv_vfmul_vv_f32m8(vacc, vscale, vl);

        if constexpr (std::is_same_v<scalar_t, float>) {
          __riscv_vse32_v_f32m8(output + i * head_size_v + d, vacc, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH_DECODE
          vfloat16m4_t vout = __riscv_vfncvt_f_f_w_f16m4(vacc, vl);
          __riscv_vse16_v_f16m4(reinterpret_cast<_Float16*>(output + i * head_size_v + d), vout, vl);
#else
              // Software conversion
              float temp[256];
              __riscv_vse32_v_f32m8(temp, vacc, vl);
              for(size_t j=0; j<vl; ++j) {
                  (output + i * head_size_v + d)[j] = static_cast<at::Half>(temp[j]);
              }
#endif
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          // BFloat16 Store
          vuint16m4_t vout = f32m8_to_bf16(vacc, vl);
          __riscv_vse16_v_u16m4(reinterpret_cast<uint16_t*>(output + i * head_size_v + d), vout, vl);

        } else {
          // Others
          float temp[256];
          __riscv_vse32_v_f32m8(temp, vacc, vl);
          for (size_t j = 0; j < vl; ++j) {
            (output + i * head_size_v + d)[j] = static_cast<scalar_t>(temp[j]);
          }
        }
      }
    }
  });
}

template <typename scalar_t, typename index_t, int64_t BLOCK_N>
void decode_attention_kernel_rvv_common(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,  // Added num_heads_kv
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale = 1.0f,
    float v_scale = 1.0f) {
  const bool has_logit_cap = logit_cap > 0;
  float rlogit_cap = has_logit_cap ? 1 / logit_cap : 0.f;
  bool is_int8 = (k_scale != 1.0f);

  // strides for accumulation
  const int64_t l_stride1 = num_kv_splits * (head_size_v + 1);
  const int64_t l_stride2 = head_size_v + 1;

  // GQA group size
  int64_t group_size = num_heads / num_heads_kv;

  // parallel on [batches, num_heads, num_kv_splits]
  at::parallel_for(0, batches * num_heads * num_kv_splits, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, head_id{0}, kv_id{0};
    data_index_init(begin, bs, batches, head_id, num_heads, kv_id, num_kv_splits);

    // s_prime and s_delta
    alignas(64) float s_i[BLOCK_N];
    float* __restrict__ s_delta = s_i;

    for (int64_t i = begin; i < end; ++i) {
      // get query
      const scalar_t* __restrict__ q_ptr = query + bs * q_strideM + head_id * q_strideH;

      // get key/value head index for GQA
      int64_t head_kv = head_id / group_size;

      // get key/value
      int64_t seq_len_kv = seq_lens[bs];
      int64_t req_pool_id = req_pool_indices[bs];

      const int64_t SPLIT_SIZE = div_up(seq_len_kv, num_kv_splits);
      const int64_t kv_start = kv_id * SPLIT_SIZE;
      const int64_t kv_end = std::min(kv_start + SPLIT_SIZE, seq_len_kv);

      float m_prime = -std::numeric_limits<float>::infinity();
      float s_prime = 0.f;

      float* __restrict__ v_prime = attn_logits + i * (head_size_v + 1);
      std::memset(v_prime, 0, head_size_v * sizeof(float));

      for (int64_t n = kv_start; n < kv_end; n += BLOCK_N) {
        int64_t n_size = std::min(BLOCK_N, kv_end - n);

        // calculate s_i <- scale * Q @ K
        if (is_int8) {
          index_gemm_kernel_nt_rvv<scalar_t, index_t>(
              /* A   */ q_ptr,
              /* B   */ static_cast<const int8_t*>(k_buffer) + head_kv * k_strideH,
              /* C   */ s_i,
              /* ind */ req_to_token + req_pool_id * max_context_len + n,
              /* scl */ scaling * k_scale,
              /* M   */ 1,
              /* N   */ n_size,
              /* K   */ head_size,
              /* lda */ 1,
              /* ldb */ k_strideN,
              /* ldc */ 1,
              /* mtt */ max_total_num_tokens);
        } else {
          index_gemm_kernel_nt_rvv<scalar_t, index_t>(
              /* A   */ q_ptr,
              /* B   */ static_cast<const scalar_t*>(k_buffer) + head_kv * k_strideH,
              /* C   */ s_i,
              /* ind */ req_to_token + req_pool_id * max_context_len + n,
              /* scl */ scaling,
              /* M   */ 1,
              /* N   */ n_size,
              /* K   */ head_size,
              /* lda */ 1,
              /* ldb */ k_strideN,
              /* ldc */ 1,
              /* mtt */ max_total_num_tokens);
        }

        if (has_logit_cap) {
          size_t vl_max = __riscv_vsetvlmax_e32m8();
          for (int64_t j = 0; j < n_size; j += vl_max) {
            size_t vl = __riscv_vsetvl_e32m8(n_size - j);
            vfloat32m8_t vx = __riscv_vle32_v_f32m8(s_i + j, vl);
            vx = __riscv_vfmul_vf_f32m8(vx, rlogit_cap, vl);
            vfloat32m8_t vtanh = vftanh_f32m8(vx, vl);
            vfloat32m8_t vres = __riscv_vfmul_vf_f32m8(vtanh, logit_cap, vl);
            __riscv_vse32_v_f32m8(s_i + j, vres, vl);
          }
        }

        // m_i: max value per row
        float m_i = (n_size > 0) ? std::max(rvv_reduce_max_f32(s_i, n_size), m_prime) : m_prime;

        // Ensure m_i is finite
        if (!std::isfinite(m_i)) {
          m_i = m_prime;
        }

        // m_delta <- exp(m' - m_i)
        float m_delta = std::isfinite(m_i) ? std::exp(m_prime - m_i) : 0.0f;

        // s_delta <- exp(s_i - m_i)
        // exp_rvv computes exp(s_i - m_i) and returns sum
        float local_sum = (n_size > 0 && std::isfinite(m_i)) ? exp_rvv(s_i, s_delta, n_size, m_i) : 0.0f;

        s_prime *= m_delta;
        s_prime += local_sum;

        m_prime = m_i;

        // calculate V' <- s_delta @ V + V' * m_delta
        // output = output * m_delta + sum(s_delta * values)
        if (is_int8) {
          prob_value_aggregate_rvv<scalar_t, index_t>(
              s_delta,
              static_cast<const int8_t*>(v_buffer) + head_kv * v_strideH,
              v_prime,
              req_to_token + req_pool_id * max_context_len + n,
              n_size,
              head_size_v,
              v_strideN,
              m_delta,
              max_total_num_tokens,
              v_scale);
        } else {
          prob_value_aggregate_rvv<scalar_t, index_t>(
              s_delta,
              static_cast<const scalar_t*>(v_buffer) + head_kv * v_strideH,
              v_prime,
              req_to_token + req_pool_id * max_context_len + n,
              n_size,
              head_size_v,
              v_strideN,
              m_delta,
              max_total_num_tokens);
        }
      }

      if (kv_end > kv_start && s_prime > 0.0f && std::isfinite(s_prime)) {
        float s = 1.0f / s_prime;
        size_t vl_v = __riscv_vsetvl_e32m8(head_size_v);
        for (int64_t d = 0; d < head_size_v; d += vl_v) {
          size_t current_vl = __riscv_vsetvl_e32m8(head_size_v - d);
          vfloat32m8_t vval = __riscv_vle32_v_f32m8(v_prime + d, current_vl);
          vfloat32m8_t vscale = __riscv_vfmv_v_f_f32m8(s, current_vl);
          vval = __riscv_vfmul_vv_f32m8(vval, vscale, current_vl);
          __riscv_vse32_v_f32m8(v_prime + d, vval, current_vl);
        }

        v_prime[head_size_v] = m_prime + std::log(s_prime);
      }

      // Check for next iteration
      data_index_step(bs, batches, head_id, num_heads, kv_id, num_kv_splits);
    }
  });
  decode_accumulate_kv_splits_rvv(
      output, attn_logits, batches, num_heads, head_size_v, num_kv_splits, l_stride1, l_stride2);
}

template <typename scalar_t, typename index_t>
void decode_attention_kernel_rvv_auto(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale = 1.0f,
    float v_scale = 1.0f) {
  // Select optimal BLOCK_N based on data type
  // SpacemiT K1: Vector-256bit
  // FP16: Use smaller block (32) for better register utilization
  // BF16/FP32: Use larger block (64) for better throughput
  constexpr int64_t BLOCK_N = get_optimal_block_n_general<scalar_t>();

  decode_attention_kernel_rvv_common<scalar_t, index_t, BLOCK_N>(
      output,
      attn_logits,
      query,
      k_buffer,
      v_buffer,
      req_to_token,
      req_pool_indices,
      seq_lens,
      batches,
      num_heads,
      num_heads,  // num_heads_kv = num_heads for MHA
      head_size,
      head_size_v,
      num_kv_splits,
      q_strideM,
      q_strideH,
      k_strideN,
      k_strideH,
      v_strideN,
      v_strideH,
      scaling,
      logit_cap,
      max_num_reqs,
      max_context_len,
      max_total_num_tokens,
      k_scale,
      v_scale);
}

// Wrapper for MHA (backward compatibility - keeps old signature for compatibility)
template <typename scalar_t, typename index_t, int64_t BLOCK_N>
void decode_attention_kernel_rvv(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale = 1.0f,
    float v_scale = 1.0f) {
  decode_attention_kernel_rvv_common<scalar_t, index_t, BLOCK_N>(
      output,
      attn_logits,
      query,
      k_buffer,
      v_buffer,
      req_to_token,
      req_pool_indices,
      seq_lens,
      batches,
      num_heads,
      num_heads,  // num_heads_kv = num_heads for MHA
      head_size,
      head_size_v,
      num_kv_splits,
      q_strideM,
      q_strideH,
      k_strideN,
      k_strideH,
      v_strideN,
      v_strideH,
      scaling,
      logit_cap,
      max_num_reqs,
      max_context_len,
      max_total_num_tokens,
      k_scale,
      v_scale);
}

template <typename scalar_t, typename index_t>
void decode_attention_grouped_kernel_rvv_auto(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,  // Explicit num_heads_kv
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale = 1.0f,
    float v_scale = 1.0f) {
  // Select optimal BLOCK_N based on data type
  constexpr int64_t BLOCK_N = get_optimal_block_n_general<scalar_t>();

  decode_attention_kernel_rvv_common<scalar_t, index_t, BLOCK_N>(
      output,
      attn_logits,
      query,
      k_buffer,
      v_buffer,
      req_to_token,
      req_pool_indices,
      seq_lens,
      batches,
      num_heads,
      num_heads_kv,
      head_size,
      head_size_v,
      num_kv_splits,
      q_strideM,
      q_strideH,
      k_strideN,
      k_strideH,
      v_strideN,
      v_strideH,
      scaling,
      logit_cap,
      max_num_reqs,
      max_context_len,
      max_total_num_tokens,
      k_scale,
      v_scale);
}

template <typename scalar_t, typename index_t>
void decode_attention_kernel_rvv_int8(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale) {
  // INT8 wrapper: calls the auto version which handles INT8 via k_scale/v_scale
  decode_attention_kernel_rvv_auto<scalar_t, index_t>(
      output,
      attn_logits,
      query,
      k_buffer,
      v_buffer,
      req_to_token,
      req_pool_indices,
      seq_lens,
      batches,
      num_heads,
      head_size,
      head_size_v,
      num_kv_splits,
      q_strideM,
      q_strideH,
      k_strideN,
      k_strideH,
      v_strideN,
      v_strideH,
      scaling,
      logit_cap,
      max_num_reqs,
      max_context_len,
      max_total_num_tokens,
      k_scale,
      v_scale);
}

template <typename scalar_t, typename index_t>
void decode_attention_grouped_kernel_rvv_int8(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale) {
  // INT8 wrapper: calls the auto version which handles INT8 via k_scale/v_scale
  decode_attention_grouped_kernel_rvv_auto<scalar_t, index_t>(
      output,
      attn_logits,
      query,
      k_buffer,
      v_buffer,
      req_to_token,
      req_pool_indices,
      seq_lens,
      batches,
      num_heads,
      num_heads_kv,
      head_size,
      head_size_v,
      num_kv_splits,
      q_strideM,
      q_strideH,
      k_strideN,
      k_strideH,
      v_strideN,
      v_strideH,
      scaling,
      logit_cap,
      max_num_reqs,
      max_context_len,
      max_total_num_tokens,
      k_scale,
      v_scale);
}

// Wrapper for GQA (backward compatibility - keeps old signature for compatibility)
template <typename scalar_t, typename index_t, int64_t BLOCK_N>
void decode_attention_grouped_kernel_rvv(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,  // Explicit num_heads_kv
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale = 1.0f,
    float v_scale = 1.0f) {
  decode_attention_kernel_rvv_common<scalar_t, index_t, BLOCK_N>(
      output,
      attn_logits,
      query,
      k_buffer,
      v_buffer,
      req_to_token,
      req_pool_indices,
      seq_lens,
      batches,
      num_heads,
      num_heads_kv,
      head_size,
      head_size_v,
      num_kv_splits,
      q_strideM,
      q_strideH,
      k_strideN,
      k_strideH,
      v_strideN,
      v_strideH,
      scaling,
      logit_cap,
      max_num_reqs,
      max_context_len,
      max_total_num_tokens,
      k_scale,
      v_scale);
}

// RVV-specific decode_attention_cpu entry point
void decode_attention_cpu(
    at::Tensor& query,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& output,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& loc,
    at::Tensor& attn_logits,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    double sm_scale,
    double logit_cap) {
  RECORD_FUNCTION(
      "sgl-kernel::decode_attention_cpu_rvv",
      std::vector<c10::IValue>(
          {query, output, k_buffer, v_buffer, attn_logits, req_to_token, req_pool_indices, seq_lens}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(value);
  CHECK_DIM(3, query);
  CHECK_DIM(3, k_buffer);
  CHECK_DIM(3, v_buffer);
  CHECK_DIM(3, key);
  CHECK_DIM(3, value);
  CHECK_DIM(1, loc);

  int64_t num_seqs = seq_lens.size(0);
  int64_t max_num_reqs = req_to_token.size(0);
  int64_t max_context_len = req_to_token.size(1);
  int64_t max_total_num_tokens = k_buffer.size(0);

  int64_t num_heads = query.size(1);
  int64_t num_heads_kv = k_buffer.size(1);
  int64_t head_size = query.size(2);
  int64_t head_size_v = v_buffer.size(2);

  int64_t num_kv_splits = attn_logits.size(2);

  CHECK_EQ(loc.numel(), num_seqs);
  CHECK_EQ(attn_logits.size(0), num_seqs);
  CHECK_EQ(attn_logits.size(1), num_heads);
  CHECK_EQ(attn_logits.size(3), head_size_v + 1);
  CHECK_EQ(attn_logits.scalar_type(), at::kFloat);

  // strides for query
  int64_t q_strideM = query.stride(0);
  int64_t q_strideH = query.stride(1);

  // strides for k_buffer and v_buffer
  int64_t k_strideN = k_buffer.stride(0);
  int64_t k_strideH = k_buffer.stride(1);
  int64_t v_strideN = v_buffer.stride(0);
  int64_t v_strideH = v_buffer.stride(1);
  // strides for new key and value
  int64_t nk_strideN = key.stride(0);
  int64_t nk_strideH = key.stride(1);
  int64_t nv_strideN = value.stride(0);
  int64_t nv_strideH = value.stride(1);

  // check index data types
  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "decode: expect req_to_token to be int32 or int64, got ",
      index_dtype);
  TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "decode: expect req_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kLong,
      "decode: expect req_pool_indices to be int64, got ",
      req_pool_indices.scalar_type());

  // check if we have MLA here
  void* k_buffer_data = k_buffer.data_ptr();
  void* v_buffer_data = v_buffer.data_ptr();
  const bool is_mla = (k_buffer_data == v_buffer_data) && (num_heads_kv == 1) && (head_size == head_size_v + 64);

  // buffer for packing k_cache and v_cache
  int num_threads = at::get_num_threads();
  // MLA is not yet implemented for RVV, so size_per_thread is 0
  int64_t size_per_thread = 0;
  auto buffer = at::empty({num_threads, size_per_thread}, k_buffer.options());

  // Use AT_DISPATCH_REDUCED_FLOATING_TYPES for Half and BFloat16 only
  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "decode_attention_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "decode_attention_indices", [&] {
      // update the kv buffer
      decode_set_kv_buffer(
          (scalar_t*)k_buffer_data,
          (scalar_t*)v_buffer_data,
          key.data_ptr<scalar_t>(),
          value.data_ptr<scalar_t>(),
          loc.data_ptr<int64_t>(),
          num_seqs,
          num_heads_kv,
          head_size,
          head_size_v,
          k_strideN,
          k_strideH,
          v_strideN,
          v_strideH,
          nk_strideN,
          nk_strideH,
          nv_strideN,
          nv_strideH,
          is_mla);

      if (num_heads == num_heads_kv) {
        // MHA
        // RVV path: Use auto block size selection (no BLOCK_N template parameter)
        decode_attention_kernel_rvv_auto<scalar_t, index_t>(
            output.data_ptr<scalar_t>(),
            attn_logits.data_ptr<float>(),
            query.data_ptr<scalar_t>(),
            (const void*)k_buffer_data,
            (const void*)v_buffer_data,
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            num_seqs,
            num_heads,
            head_size,
            head_size_v,
            num_kv_splits,
            q_strideM,
            q_strideH,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            sm_scale,
            logit_cap,
            max_num_reqs,
            max_context_len,
            max_total_num_tokens);
      } else if (is_mla) {
        // MLA - Currently, RVV MLA is not implemented. Fallback to GQA/MQA for now.
        // TODO: Implement MLA specific kernel for RVV
        decode_attention_grouped_kernel_rvv_auto<scalar_t, index_t>(
            output.data_ptr<scalar_t>(),
            attn_logits.data_ptr<float>(),
            query.data_ptr<scalar_t>(),
            (const void*)k_buffer_data,
            (const void*)v_buffer_data,
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            num_seqs,
            num_heads,
            num_heads_kv,
            head_size,
            head_size_v,
            num_kv_splits,
            q_strideM,
            q_strideH,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            sm_scale,
            logit_cap,
            max_num_reqs,
            max_context_len,
            max_total_num_tokens);
      } else {
        // GQA/MQA
        // RVV path: Use auto block size selection (no BLOCK_N template parameter)
        decode_attention_grouped_kernel_rvv_auto<scalar_t, index_t>(
            output.data_ptr<scalar_t>(),
            attn_logits.data_ptr<float>(),
            query.data_ptr<scalar_t>(),
            (const void*)k_buffer_data,
            (const void*)v_buffer_data,
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            num_seqs,
            num_heads,
            num_heads_kv,
            head_size,
            head_size_v,
            num_kv_splits,
            q_strideM,
            q_strideH,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            sm_scale,
            logit_cap,
            max_num_reqs,
            max_context_len,
            max_total_num_tokens);
      }
    });
  });
}

template void decode_attention_kernel_rvv_auto<at::Half, int32_t>(
    at::Half* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::Half* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_kernel_rvv_auto<at::Half, int64_t>(
    at::Half* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::Half* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int64_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_kernel_rvv_auto<at::BFloat16, int32_t>(
    at::BFloat16* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::BFloat16* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_kernel_rvv_auto<at::BFloat16, int64_t>(
    at::BFloat16* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::BFloat16* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int64_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_grouped_kernel_rvv_auto<at::Half, int32_t>(
    at::Half* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::Half* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_grouped_kernel_rvv_auto<at::Half, int64_t>(
    at::Half* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::Half* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int64_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_grouped_kernel_rvv_auto<at::BFloat16, int32_t>(
    at::BFloat16* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::BFloat16* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_grouped_kernel_rvv_auto<at::BFloat16, int64_t>(
    at::BFloat16* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::BFloat16* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int64_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_kernel_rvv_int8<at::Half, int32_t>(
    at::Half* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::Half* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_kernel_rvv_int8<at::Half, int64_t>(
    at::Half* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::Half* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int64_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_kernel_rvv_int8<at::BFloat16, int32_t>(
    at::BFloat16* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::BFloat16* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_kernel_rvv_int8<at::BFloat16, int64_t>(
    at::BFloat16* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::BFloat16* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int64_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_grouped_kernel_rvv_int8<at::Half, int32_t>(
    at::Half* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::Half* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_grouped_kernel_rvv_int8<at::Half, int64_t>(
    at::Half* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::Half* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int64_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_grouped_kernel_rvv_int8<at::BFloat16, int32_t>(
    at::BFloat16* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::BFloat16* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template void decode_attention_grouped_kernel_rvv_int8<at::BFloat16, int64_t>(
    at::BFloat16* __restrict__ output,
    float* __restrict__ attn_logits,
    const at::BFloat16* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const int64_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

#endif  // CPU_CAPABILITY_RVV
