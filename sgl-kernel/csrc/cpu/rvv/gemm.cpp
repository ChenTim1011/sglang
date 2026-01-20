/*
 * 1. FP16/FP32 GEMM: Standard floating-point matrix multiplication.
 * 2. Dispatching: Selecting the best kernel based on problem size.
 * 3. Native RVV Support: Implementing GEMM using RVV intrinsics for non-quantized types.
 */

#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "gemm.h"

#include <ATen/core/Tensor.h>

#include <cstdint>

#include "common.h"
#include "vector_helpers.h"
#include "vector_math.h"

#if defined(CPU_CAPABILITY_RVV)
// Clang 19+ version check
#if !defined(__clang__) || !defined(__clang_major__) || __clang_major__ < 19
#error "RVV backend requires Clang 19 or later. Please use Clang 19 or later to compile this file."
#endif

#include <omp.h>
#include <riscv_vector.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#if defined(__riscv_zvfh) || defined(__riscv_v)
#define HAS_ZVFH 1
#else
#define HAS_ZVFH 0
#endif

namespace {

using namespace rvv_constants;

#if HAS_ZVFH

// =============================================================================
// Kernel: Native GEMM (FP32 / FP16 / BF16)
// =============================================================================
template <typename scalar_t>
void gemm_native_rvv(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    size_t vl_max = __riscv_vsetvlmax_e32m4();

#pragma omp parallel for schedule(static) if (M > 4)
    for (int64_t m = 0; m < M; ++m) {
      const float* a_row = A + m * lda;
      float* c_row = C + m * ldc;

      for (int64_t n = 0; n < N; ++n) {
        const float* b_row = B + n * ldb;
        float bias_val = (bias != nullptr) ? bias[n] : 0.0f;

        vfloat32m4_t v_sum = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);

        int64_t k = 0;
        for (; k + static_cast<int64_t>(vl_max) <= K; k += vl_max) {
          vfloat32m4_t v_a = __riscv_vle32_v_f32m4(a_row + k, vl_max);
          vfloat32m4_t v_b = __riscv_vle32_v_f32m4(b_row + k, vl_max);
          __builtin_prefetch(b_row + k + 128, 0, 0);  // Prefetch weights (Read, Non-temporal)
          v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_a, v_b, vl_max);
        }

        float sum = reduce_sum_f32m4(v_sum, vl_max);
        if (k < K) {
          size_t tail_vl = __riscv_vsetvl_e32m4(K - k);
          vfloat32m4_t v_a = __riscv_vle32_v_f32m4(a_row + k, tail_vl);
          vfloat32m4_t v_b = __riscv_vle32_v_f32m4(b_row + k, tail_vl);
          vfloat32m4_t v_tail = __riscv_vfmul_vv_f32m4(v_a, v_b, tail_vl);
          sum += reduce_sum_f32m4(v_tail, tail_vl);
        }

        c_row[n] = sum + bias_val;
      }
    }
    return;
  }

  if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    size_t vl_max_fp32m4 = __riscv_vsetvlmax_e32m4();

    constexpr int64_t TILE_M = rvv_constants::TILE_M;

#pragma omp parallel for schedule(static) if (M > TILE_M)
    for (int64_t m0 = 0; m0 < M; m0 += TILE_M) {
      int64_t current_m_size = std::min(TILE_M, M - m0);

      for (int64_t n = 0; n < N; ++n) {
        const uint16_t* b_ptr_start = reinterpret_cast<const uint16_t*>(B + n * ldb);
        float bias_val = (bias != nullptr) ? bias[n] : 0.0f;

        vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max_fp32m4);
        vfloat32m4_t v_acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max_fp32m4);
        vfloat32m4_t v_acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max_fp32m4);
        vfloat32m4_t v_acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max_fp32m4);

        int64_t k = 0;
        for (; k + static_cast<int64_t>(vl_max_fp32m4) <= K; k += vl_max_fp32m4) {
          vfloat32m4_t v_b_f32 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(b_ptr_start + k), vl_max_fp32m4);
          if (current_m_size > 0) {
            vfloat32m4_t v_a_f32 =
                bf16_to_f32m4(reinterpret_cast<const uint16_t*>(A + (m0 + 0) * lda + k), vl_max_fp32m4);
            v_acc0 = __riscv_vfmacc_vv_f32m4(v_acc0, v_a_f32, v_b_f32, vl_max_fp32m4);
          }
          if (current_m_size > 1) {
            vfloat32m4_t v_a_f32 =
                bf16_to_f32m4(reinterpret_cast<const uint16_t*>(A + (m0 + 1) * lda + k), vl_max_fp32m4);
            v_acc1 = __riscv_vfmacc_vv_f32m4(v_acc1, v_a_f32, v_b_f32, vl_max_fp32m4);
          }
          if (current_m_size > 2) {
            vfloat32m4_t v_a_f32 =
                bf16_to_f32m4(reinterpret_cast<const uint16_t*>(A + (m0 + 2) * lda + k), vl_max_fp32m4);
            v_acc2 = __riscv_vfmacc_vv_f32m4(v_acc2, v_a_f32, v_b_f32, vl_max_fp32m4);
          }
          if (current_m_size > 3) {
            vfloat32m4_t v_a_f32 =
                bf16_to_f32m4(reinterpret_cast<const uint16_t*>(A + (m0 + 3) * lda + k), vl_max_fp32m4);
            v_acc3 = __riscv_vfmacc_vv_f32m4(v_acc3, v_a_f32, v_b_f32, vl_max_fp32m4);
          }
        }

        float tail_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (k < K) {
          size_t tail_vl = __riscv_vsetvl_e32m4(K - k);
          vfloat32m4_t v_b_f32 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(b_ptr_start + k), tail_vl);

          auto process_tail_element = [&](int idx) {
            if (current_m_size > idx) {
              vfloat32m4_t v_a_f32 =
                  bf16_to_f32m4(reinterpret_cast<const uint16_t*>(A + (m0 + idx) * lda + k), tail_vl);
              vfloat32m4_t v_tail = __riscv_vfmul_vv_f32m4(v_a_f32, v_b_f32, tail_vl);
              tail_sums[idx] = reduce_sum_f32m4(v_tail, tail_vl);
            }
          };

          process_tail_element(0);
          process_tail_element(1);
          process_tail_element(2);
          process_tail_element(3);
        }

        auto store_result = [&](vfloat32m4_t v_acc, int offset, float tail_sum) {
          float sum = reduce_sum_f32m4(v_acc, vl_max_fp32m4);
          sum += tail_sum;
          sum += bias_val;
          uint32_t sum_as_int;
          std::memcpy(&sum_as_int, &sum, sizeof(float));
          if (std::isnan(sum)) {
            sum_as_int = 0x7FC00000;
          } else {
            sum_as_int += 0x8000;
          }
          uint16_t res_bf16 = static_cast<uint16_t>(sum_as_int >> 16);
          at::BFloat16* c_val = C + (m0 + offset) * ldc + n;
          *reinterpret_cast<uint16_t*>(c_val) = res_bf16;
        };

        if (current_m_size > 0) store_result(v_acc0, 0, tail_sums[0]);
        if (current_m_size > 1) store_result(v_acc1, 1, tail_sums[1]);
        if (current_m_size > 2) store_result(v_acc2, 2, tail_sums[2]);
        if (current_m_size > 3) store_result(v_acc3, 3, tail_sums[3]);
      }
    }
    return;
  }

  if constexpr (!std::is_same_v<scalar_t, at::Half>) {
    return;
  }

  size_t vl_max_f16m4 = __riscv_vsetvlmax_e16m4();
  size_t vl_max_f16m1 = __riscv_vsetvlmax_e16m1();

#pragma omp parallel for schedule(static) if (M > 1)
  for (int64_t m = 0; m < M; ++m) {
    const _Float16* a_row = reinterpret_cast<const _Float16*>(A + m * lda);
    _Float16* c_row = reinterpret_cast<_Float16*>(C + m * ldc);

    for (int64_t n = 0; n < N; ++n) {
      const _Float16* b_row = reinterpret_cast<const _Float16*>(B + n * ldb);

      vfloat16m4_t v_sum = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vl_max_f16m4);

      int64_t k = 0;
      for (; k + static_cast<int64_t>(vl_max_f16m4) <= K; k += vl_max_f16m4) {
        vfloat16m4_t v_a = __riscv_vle16_v_f16m4(a_row + k, vl_max_f16m4);
        vfloat16m4_t v_b = __riscv_vle16_v_f16m4(b_row + k, vl_max_f16m4);
        v_sum = __riscv_vfmacc_vv_f16m4(v_sum, v_a, v_b, vl_max_f16m4);
      }

      _Float16 sum = reduce_sum_f16m4(v_sum, vl_max_f16m4);

      if (k < K) {
        size_t tail_vl = __riscv_vsetvl_e16m4(K - k);
        vfloat16m4_t v_a = __riscv_vle16_v_f16m4(a_row + k, tail_vl);
        vfloat16m4_t v_b = __riscv_vle16_v_f16m4(b_row + k, tail_vl);
        vfloat16m4_t v_tail = __riscv_vfmul_vv_f16m4(v_a, v_b, tail_vl);
        sum += reduce_sum_f16m4(v_tail, tail_vl);
      }

      float result = static_cast<float>(sum);
      if (bias != nullptr) {
        result += bias[n];
      }

      c_row[n] = static_cast<_Float16>(result);
    }
  }
}

// =============================================================================
// Kernel: Tiled FP32 GEMM
// =============================================================================
template <typename scalar_t>
void gemm_fp16_tiled_rvv(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  if constexpr (!std::is_same_v<scalar_t, at::Half>) {
    return;
  }

  constexpr int64_t TILE_M = rvv_constants::TILE_M;
  constexpr int64_t TILE_N = rvv_constants::TILE_N;

  size_t vl_max = __riscv_vsetvlmax_e16m4();

#pragma omp parallel for schedule(static) if (M > TILE_M)
  for (int64_t m0 = 0; m0 < M; m0 += TILE_M) {
    int64_t m_size = std::min(TILE_M, M - m0);

    for (int64_t n0 = 0; n0 < N; n0 += TILE_N) {
      int64_t n_size = std::min(TILE_N, N - n0);
      float acc[TILE_M][TILE_N] = {{0.0f}};
      if (bias != nullptr) {
        for (int64_t m = 0; m < m_size; ++m) {
          for (int64_t n = 0; n < n_size; ++n) {
            acc[m][n] = bias[n0 + n];
          }
        }
      }

      for (int64_t m = 0; m < m_size; ++m) {
        const _Float16* a_row = reinterpret_cast<const _Float16*>(A + (m0 + m) * lda);

        for (int64_t n = 0; n < n_size; ++n) {
          const _Float16* b_row = reinterpret_cast<const _Float16*>(B + (n0 + n) * ldb);

          vfloat16m4_t v_sum = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vl_max);

          int64_t k = 0;
          for (; k + static_cast<int64_t>(vl_max) <= K; k += vl_max) {
            vfloat16m4_t v_a = __riscv_vle16_v_f16m4(a_row + k, vl_max);
            vfloat16m4_t v_b = __riscv_vle16_v_f16m4(b_row + k, vl_max);
            v_sum = __riscv_vfmacc_vv_f16m4(v_sum, v_a, v_b, vl_max);
          }

          _Float16 partial = reduce_sum_f16m4(v_sum, vl_max);

          if (k < K) {
            size_t tail_vl = __riscv_vsetvl_e16m4(K - k);
            vfloat16m4_t v_a = __riscv_vle16_v_f16m4(a_row + k, tail_vl);
            vfloat16m4_t v_b = __riscv_vle16_v_f16m4(b_row + k, tail_vl);
            vfloat16m4_t v_tail = __riscv_vfmul_vv_f16m4(v_a, v_b, tail_vl);
            partial += reduce_sum_f16m4(v_tail, tail_vl);
          }

          acc[m][n] += static_cast<float>(partial);
        }
      }

      for (int64_t m = 0; m < m_size; ++m) {
        _Float16* c_row = reinterpret_cast<_Float16*>(C + (m0 + m) * ldc);
        for (int64_t n = 0; n < n_size; ++n) {
          c_row[n0 + n] = static_cast<_Float16>(acc[m][n]);
        }
      }
    }
  }
}

#endif  // HAS_ZVFH

template <typename scalar_t>
void gemm_rvv_impl(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    float* __restrict__ scratch_a,
    float* __restrict__ scratch_b,
    float* __restrict__ acc_buf) {
  size_t vl_max_m1 = __riscv_vsetvlmax_e32m1();
  size_t vl_max_m4 = __riscv_vsetvlmax_e32m4();

  // Ensure M loop handles all rows correctly
  for (int64_t m_start = 0; m_start < M; m_start += BLOCK_M_RVV) {
    int64_t m_size = std::min(static_cast<int64_t>(BLOCK_M_RVV), M - m_start);

    // Initialize accumulator buffer
    for (int64_t m = 0; m < m_size; ++m) {
      float* acc_row = acc_buf + m * N;
      if (bias != nullptr) {
        size_t vl;
        for (int64_t n = 0; n < N; n += vl) {
          vl = __riscv_vsetvl_e32m4(N - n);
          vfloat32m4_t v_bias = __riscv_vle32_v_f32m4(bias + n, vl);
          __riscv_vse32_v_f32m4(acc_row + n, v_bias, vl);
        }
      } else {
        std::memset(acc_row, 0, N * sizeof(float));
      }
    }

    for (int64_t k = 0; k < K; ++k) {
      for (int64_t m = 0; m < m_size; ++m) {
        float a_val = static_cast<float>(A[(m_start + m) * lda + k]);
        // Remove threshold check to ensure correctness for all values
        // if (std::abs(a_val) < rvv_constants::ZERO_THRESHOLD) continue;

        float* acc_row = acc_buf + m * N;
        // Correct implementation assuming A[M, K] * B[K, N]
        // This K-loop implementation assumes we take A[m, k] (scalar) and multiple it by B[k, :] (vector)
        // So b_row should point to the start of row k in B.
        const scalar_t* b_row = B + k * ldb;

        size_t vl;
        for (int64_t n = 0; n < N; n += vl) {
          vl = __riscv_vsetvl_e32m4(N - n);

          ptrdiff_t stride_bytes = ldb * sizeof(scalar_t);
          vfloat32m4_t v_b = load_strided_as_float_m4(B + n * ldb + k, stride_bytes, vl, scratch_b);
          vfloat32m4_t v_acc = __riscv_vle32_v_f32m4(acc_row + n, vl);
          v_acc = __riscv_vfmacc_vf_f32m4(v_acc, a_val, v_b, vl);
          __riscv_vse32_v_f32m4(acc_row + n, v_acc, vl);
        }
      }
    }

    for (int64_t m = 0; m < m_size; ++m) {
      float* acc_row = acc_buf + m * N;
      scalar_t* c_row = C + (m_start + m) * ldc;

      size_t vl;
      for (int64_t n = 0; n < N; n += vl) {
        vl = __riscv_vsetvl_e32m4(N - n);
        vfloat32m4_t v_acc = __riscv_vle32_v_f32m4(acc_row + n, vl);
        store_from_float_m4(c_row + n, v_acc, vl, scratch_a);
      }
    }
  }
}

template <typename scalar_t>
void gemm_nt_rvv_linear(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    float* __restrict__ scratch) {
  size_t vl_max = __riscv_vsetvlmax_e32m1();

#pragma omp parallel for if (M > 1)
  for (int64_t m = 0; m < M; ++m) {
    const scalar_t* a_row = A + m * lda;
    scalar_t* c_row = C + m * ldc;

    alignas(64) float local_scratch_a[MAX_VL_ELEMENTS];
    alignas(64) float local_scratch_b[MAX_VL_ELEMENTS];

    for (int64_t n = 0; n < N; ++n) {
      const scalar_t* b_row = B + n * ldb;

      vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, vl_max);

      int64_t k = 0;
      for (; k + static_cast<int64_t>(vl_max) <= K; k += vl_max) {
        vfloat32m1_t v_a = load_as_float_m1(a_row + k, vl_max, local_scratch_a);
        vfloat32m1_t v_b = load_as_float_m1(b_row + k, vl_max, local_scratch_b);
        v_sum = __riscv_vfmacc_vv_f32m1(v_sum, v_a, v_b, vl_max);
      }

      float sum = reduce_sum_f32m1(v_sum, vl_max);
      if (k < K) {
        size_t tail_vl = __riscv_vsetvl_e32m1(K - k);
        vfloat32m1_t v_a = load_as_float_m1(a_row + k, tail_vl, local_scratch_a);
        vfloat32m1_t v_b = load_as_float_m1(b_row + k, tail_vl, local_scratch_b);
        vfloat32m1_t v_tail = __riscv_vfmul_vv_f32m1(v_a, v_b, tail_vl);
        sum += reduce_sum_f32m1(v_tail, tail_vl);
      }

      if (bias != nullptr) {
        sum += bias[n];
      }

      c_row[n] = static_cast<scalar_t>(sum);
    }
  }
}

template <typename scalar_t>
void gemm_tiled_rvv(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  int64_t TILE_M = rvv_constants::TILE_M;
  int64_t TILE_K = rvv_constants::TILE_K;
  const char* env_tile_m = std::getenv("SGL_TUNE_TILE_M");
  const char* env_tile_k = std::getenv("SGL_TUNE_TILE_K");
  if (env_tile_m != nullptr) {
    int tile_m = std::atoi(env_tile_m);
    if (tile_m > 0) TILE_M = tile_m;
  }
  if (env_tile_k != nullptr) {
    int tile_k = std::atoi(env_tile_k);
    if (tile_k > 0) TILE_K = tile_k;
  }

  size_t vl_max = __riscv_vsetvlmax_e32m1();

#pragma omp parallel
  {
    alignas(64) float scratch_a[MAX_VL_ELEMENTS];
    alignas(64) float scratch_b[MAX_VL_ELEMENTS];
    alignas(64) float acc_rows[TILE_M * 4096];

#pragma omp for schedule(static)
    for (int64_t m0 = 0; m0 < M; m0 += TILE_M) {
      int64_t m_size = std::min(TILE_M, M - m0);

      for (int64_t n = 0; n < N; ++n) {
        const scalar_t* b_row = B + n * ldb;
        float acc[TILE_M];
        for (int64_t m = 0; m < m_size; ++m) {
          acc[m] = (bias != nullptr) ? bias[n] : 0.0f;
        }

        for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
          int64_t k_size = std::min(TILE_K, K - k0);

          for (int64_t m = 0; m < m_size; ++m) {
            const scalar_t* a_row = A + (m0 + m) * lda + k0;

            vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, vl_max);

            int64_t k = 0;
            for (; k + static_cast<int64_t>(vl_max) <= k_size; k += vl_max) {
              vfloat32m1_t v_a = load_as_float_m1(a_row + k, vl_max, scratch_a);
              vfloat32m1_t v_b = load_as_float_m1(b_row + k0 + k, vl_max, scratch_b);
              v_sum = __riscv_vfmacc_vv_f32m1(v_sum, v_a, v_b, vl_max);
            }

            float partial_sum = reduce_sum_f32m1(v_sum, vl_max);

            if (k < k_size) {
              size_t tail_vl = __riscv_vsetvl_e32m1(k_size - k);
              vfloat32m1_t v_a = load_as_float_m1(a_row + k, tail_vl, scratch_a);
              vfloat32m1_t v_b = load_as_float_m1(b_row + k0 + k, tail_vl, scratch_b);
              vfloat32m1_t v_tail = __riscv_vfmul_vv_f32m1(v_a, v_b, tail_vl);
              partial_sum += reduce_sum_f32m1(v_tail, tail_vl);
            }

            acc[m] += partial_sum;
          }
        }

        for (int64_t m = 0; m < m_size; ++m) {
          C[(m0 + m) * ldc + n] = static_cast<scalar_t>(acc[m]);
        }
      }
    }
  }
}

// =============================================================================
// Kernel: TinyGEMM
// =============================================================================
template <typename scalar_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_rvv {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ PackedB,
      scalar_t* __restrict__ C,
      const float* __restrict__ bias,
      int64_t K,
      int64_t lda,
      int64_t ldc,
      int64_t nb,
      int64_t n_size) {
    size_t vl_max = __riscv_vsetvlmax_e32m4();
    constexpr int64_t n_start = nb * BLOCK_N;

    // We want to keep accumulators in registers: vfloat32m4_t acc[BLOCK_M]
    // BLOCK_M is typically small (e.g. 4), so 4 * m4 registers = 16 registers.
    // This fits in the 32 vector registers of RISC-V.

    // However, if n_size > vl_max, we need to loop. In tinygemm, N is usually small (<=32).
    // vl_max for e32m4 on 256-bit VLEN is 32.
    // So for N<=32, we can do it in one shot without a loop over columns.
    // If N > 32, we should loop over columns chunks.

    // To support generic N, we loop over column chunks.

    for (int64_t j = 0; j < n_size; j += vl_max) {
      size_t vl = (j + vl_max <= n_size) ? vl_max : __riscv_vsetvl_e32m4(n_size - j);

      // 1. Initialize Accumulators in Registers
      static_assert(BLOCK_M <= 4, "tinygemm_kernel_rvv supports BLOCK_M <= 4 only");
      vfloat32m4_t v_acc0, v_acc1, v_acc2, v_acc3;

      auto init_one = [&](int m, vfloat32m4_t& v) {
        if constexpr (has_bias) {
          v = __riscv_vle32_v_f32m4(bias + n_start + j, vl);
        } else {
          v = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        }
      };

      if (BLOCK_M > 0) init_one(0, v_acc0);
      if (BLOCK_M > 1) init_one(1, v_acc1);
      if (BLOCK_M > 2) init_one(2, v_acc2);
      if (BLOCK_M > 3) init_one(3, v_acc3);

      // 2. K-Loop Accumulation
      const scalar_t* b_ptr_base = PackedB + nb * K * BLOCK_N + j;

      for (int64_t k = 0; k < K; ++k) {
        if (k + 1 < K) {
          __builtin_prefetch(PackedB + nb * K * BLOCK_N + (k + 1) * BLOCK_N + j, 0, 3);
        }
        const scalar_t* b_ptr = b_ptr_base + k * BLOCK_N;

        // Load B vector once
        vfloat32m4_t v_b;
        if constexpr (std::is_same_v<scalar_t, float>) {
          v_b = __riscv_vle32_v_f32m4(b_ptr, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_b16 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(b_ptr), vl);
          v_b = __riscv_vfwcvt_f_f_v_f32m4(v_b16, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          v_b = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(b_ptr), vl);
        }

        // Compute for each M row
        auto compute_one = [&](int m, vfloat32m4_t& v) {
          float a_val = static_cast<float>(A[m * lda + k]);
          if (std::abs(a_val) >= 1e-8f) {
            v = __riscv_vfmacc_vf_f32m4(v, a_val, v_b, vl);
          }
        };

        if (BLOCK_M > 0) compute_one(0, v_acc0);
        if (BLOCK_M > 1) compute_one(1, v_acc1);
        if (BLOCK_M > 2) compute_one(2, v_acc2);
        if (BLOCK_M > 3) compute_one(3, v_acc3);
      }

      // 3. Store Results
      auto store_one = [&](int m, vfloat32m4_t& v) {
        scalar_t* c_ptr = C + m * ldc + n_start + j;

        if constexpr (std::is_same_v<scalar_t, float>) {
          __riscv_vse32_v_f32m4(c_ptr, v, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_f16 = f32m4_to_f16(v, vl);
          __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(c_ptr), v_f16, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          vuint16m2_t v_bf16 = f32m4_to_bf16(v, vl);
          __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(c_ptr), v_bf16, vl);
        }
      };

      if (BLOCK_M > 0) store_one(0, v_acc0);
      if (BLOCK_M > 1) store_one(1, v_acc1);
      if (BLOCK_M > 2) store_one(2, v_acc2);
      if (BLOCK_M > 3) store_one(3, v_acc3);
    }
  }
};

#define LAUNCH_TINYGEMM_KERNEL_RVV(MB_SIZE, NB_SIZE)                \
  tinygemm_kernel_rvv<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda, PackedB, C + mb_start * ldc, has_bias ? bias : nullptr, K, lda, ldc, nb, n_size);

#include <cstdio>

// =============================================================================
// Kernel: Packed GEMM
// =============================================================================
template <typename scalar_t>
void gemm_packed_rvv_impl(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ PackedB,
    scalar_t* __restrict__ C,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldc) {
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  int64_t num_blocks_n = (N + BLOCK_N - 1) / BLOCK_N;

  // Optimized Schedule:
  // Parallelize over N (blocks of columns) instead of M.
  // 1. For Decode (M=1): Enables multi-core utilization (previously single-threaded).
  // 2. For Prefill (M>1): Reduces memory bandwidth by reading PackedB once per thread
  //    (while broadcasting smaller A) instead of reading PackedB M times.
#pragma omp parallel for schedule(static)
  for (int64_t nb = 0; nb < num_blocks_n; ++nb) {
    int64_t n_start = nb * BLOCK_N;
    int64_t n_size = std::min(BLOCK_N, N - n_start);

    // We process n_size elements. We can process them in chunks of vl_max.
    // For each M, we need to accumulate results.
    // Ideally we keep acc in registers for the whole K loop.
    // If n_size is large (e.g. 64), and we have M=1, we can do it.
    // If M > 1, we iterate M.

    size_t vl_max = __riscv_vsetvlmax_e32m4();

    for (int64_t m = 0; m < M; ++m) {
      const scalar_t* a_row = A + m * lda;
      scalar_t* c_row = C + m * ldc;

      // Use a loop over N chunks to keep accumulators in registers
      for (int64_t j = 0; j < n_size; j += vl_max) {
        size_t vl = (j + vl_max <= n_size) ? vl_max : __riscv_vsetvl_e32m4(n_size - j);

        // 1. Initialize Accumulators
        vfloat32m4_t v_acc;
        if (bias) {
          v_acc = __riscv_vle32_v_f32m4(bias + n_start + j, vl);
        } else {
          v_acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        }

        // 2. K-Loop Accumulation (Keep v_acc in registers)
        const scalar_t* b_ptr_base = PackedB + nb * K * BLOCK_N + j;
        // Note: PackedB layout is [NB, K, BLOCK_N]
        // b_ptr address: base + k * BLOCK_N

        for (int64_t k = 0; k < K; ++k) {
          float a_val = static_cast<float>(a_row[k]);
          // Skip zero values
          if (std::abs(a_val) < rvv_constants::ZERO_THRESHOLD) continue;

          const scalar_t* b_ptr = b_ptr_base + k * BLOCK_N;

          // Prefetch
          if (k + rvv_constants::PREFETCH_DISTANCE < K) {
            __builtin_prefetch(b_ptr + rvv_constants::PREFETCH_DISTANCE * BLOCK_N, 0, rvv_constants::PREFETCH_LOCALITY);
          }

          vfloat32m4_t v_b;
          if constexpr (std::is_same_v<scalar_t, float>) {
            v_b = __riscv_vle32_v_f32m4(b_ptr, vl);
          } else {
            if constexpr (std::is_same_v<scalar_t, at::Half>) {
              vfloat16m2_t v_b16 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(b_ptr), vl);
              v_b = __riscv_vfwcvt_f_f_v_f32m4(v_b16, vl);
            } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
              v_b = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(b_ptr), vl);
            }
          }
          v_acc = __riscv_vfmacc_vf_f32m4(v_acc, a_val, v_b, vl);
        }

        // 3. Store Results
        if constexpr (std::is_same_v<scalar_t, float>) {
          __riscv_vse32_v_f32m4(c_row + n_start + j, v_acc, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_f16 = f32m4_to_f16(v_acc, vl);
          __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(c_row + n_start + j), v_f16, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          vuint16m2_t v_bf16 = f32m4_to_bf16(v_acc, vl);
          __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(c_row + n_start + j), v_bf16, vl);
        }
      }
    }
  }
}

}  // anonymous namespace

template <typename scalar_t>
void pack_weight_rvv(scalar_t* packed_w, const scalar_t* orig_w, int64_t N, int64_t K) {
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  int64_t NB = (N + BLOCK_N - 1) / BLOCK_N;

#pragma omp parallel for collapse(2)
  for (int64_t nb = 0; nb < NB; ++nb) {
    for (int64_t k = 0; k < K; ++k) {
      int64_t n_start = nb * BLOCK_N;
      int64_t n_size = std::min(BLOCK_N, N - n_start);

      scalar_t* dst = packed_w + nb * K * BLOCK_N + k * BLOCK_N;

      if constexpr (std::is_same_v<scalar_t, float>) {
        size_t vl;
        for (int64_t j = 0; j < n_size; j += vl) {
          vl = __riscv_vsetvl_e32m4(n_size - j);

          vfloat32m4_t v_data = __riscv_vlse32_v_f32m4(orig_w + (n_start + j) * K + k, K * sizeof(float), vl);

          __riscv_vse32_v_f32m4(dst + j, v_data, vl);
        }

        if (n_size < BLOCK_N) {
          size_t vl_pad = __riscv_vsetvl_e32m4(BLOCK_N - n_size);
          vfloat32m4_t v_zero = __riscv_vfmv_v_f_f32m4(0.0f, vl_pad);
          __riscv_vse32_v_f32m4(dst + n_size, v_zero, vl_pad);
        }
      } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
        size_t vl;
        for (int64_t j = 0; j < n_size; j += vl) {
          vl = __riscv_vsetvl_e16m2(n_size - j);

          vfloat16m2_t v_data = __riscv_vlse16_v_f16m2(
              reinterpret_cast<const _Float16*>(orig_w + (n_start + j) * K + k), K * sizeof(_Float16), vl);

          __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(dst + j), v_data, vl);
        }

        if (n_size < BLOCK_N) {
          size_t vl_pad = __riscv_vsetvl_e16m2(BLOCK_N - n_size);
          vfloat16m2_t v_zero = __riscv_vfmv_v_f_f16m2((_Float16)0.0f, vl_pad);
          __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(dst + n_size), v_zero, vl_pad);
        }
      } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        size_t vl;
        for (int64_t j = 0; j < n_size; j += vl) {
          vl = __riscv_vsetvl_e16m2(n_size - j);

          vuint16m2_t v_data = __riscv_vlse16_v_u16m2(
              reinterpret_cast<const uint16_t*>(orig_w + (n_start + j) * K + k), K * sizeof(uint16_t), vl);

          __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(dst + j), v_data, vl);
        }

        if (n_size < BLOCK_N) {
          size_t vl_pad = __riscv_vsetvl_e16m2(BLOCK_N - n_size);
          vuint16m2_t v_zero = __riscv_vmv_v_x_u16m2(0, vl_pad);
          __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(dst + n_size), v_zero, vl_pad);
        }
      } else {
        for (int64_t j = 0; j < n_size; ++j) {
          dst[j] = orig_w[(n_start + j) * K + k];
        }
        for (int64_t j = n_size; j < BLOCK_N; ++j) {
          dst[j] = static_cast<scalar_t>(0);
        }
      }
    }
  }
}

template <typename scalar_t>
void weight_packed_linear_kernel_impl_rvv(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const scalar_t* __restrict__ mat2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed) {
  if (is_packed) {
    gemm_packed_rvv_impl(mat1, mat2, out, bias, M, N, K, mat1_strideM, out_strideM);
    return;
  }

#if HAS_ZVFH
  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if (M <= 4) {
      gemm_native_rvv(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM);
    } else {
      gemm_fp16_tiled_rvv(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM);
    }
    return;
  }
  if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    gemm_native_rvv(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM);
    return;
  }
#endif

  if (M == 1) {
    gemm_nt_rvv_linear(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM, nullptr);
  } else {
    gemm_tiled_rvv(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM);
  }
}

template void weight_packed_linear_kernel_impl_rvv<float>(
    float* out,
    const float* mat1,
    const float* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed);

template void weight_packed_linear_kernel_impl_rvv<at::BFloat16>(
    at::BFloat16* out,
    const at::BFloat16* mat1,
    const at::BFloat16* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed);

template void weight_packed_linear_kernel_impl_rvv<at::Half>(
    at::Half* out,
    const at::Half* mat1,
    const at::Half* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed);

template void pack_weight_rvv<float>(float* packed_w, const float* orig_w, int64_t N, int64_t K);
template void pack_weight_rvv<at::BFloat16>(at::BFloat16* packed_w, const at::BFloat16* orig_w, int64_t N, int64_t K);
template void pack_weight_rvv<at::Half>(at::Half* packed_w, const at::Half* orig_w, int64_t N, int64_t K);
template void pack_weight_rvv<int8_t>(int8_t* packed_w, const int8_t* orig_w, int64_t N, int64_t K);

#else  // !CPU_CAPABILITY_RVV

template <typename scalar_t>
void weight_packed_linear_kernel_impl_rvv(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const scalar_t* __restrict__ mat2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed) {
  TORCH_CHECK(false, "RVV GEMM not available - built without CPU_CAPABILITY_RVV");
}

template <typename scalar_t>
void pack_weight_rvv(scalar_t* packed_w, const scalar_t* orig_w, int64_t N, int64_t K) {
  TORCH_CHECK(false, "RVV GEMM not available - built without CPU_CAPABILITY_RVV");
}

template void weight_packed_linear_kernel_impl_rvv<float>(
    float* out,
    const float* mat1,
    const float* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed);

template void weight_packed_linear_kernel_impl_rvv<at::BFloat16>(
    at::BFloat16* out,
    const at::BFloat16* mat1,
    const at::BFloat16* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed);

template void weight_packed_linear_kernel_impl_rvv<at::Half>(
    at::Half* out,
    const at::Half* mat1,
    const at::Half* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed);

template void pack_weight_rvv<float>(float* packed_w, const float* orig_w, int64_t N, int64_t K);
template void pack_weight_rvv<at::BFloat16>(at::BFloat16* packed_w, const at::BFloat16* orig_w, int64_t N, int64_t K);
template void pack_weight_rvv<at::Half>(at::Half* packed_w, const at::Half* orig_w, int64_t N, int64_t K);
template void pack_weight_rvv<int8_t>(int8_t* packed_w, const int8_t* orig_w, int64_t N, int64_t K);

#endif  // CPU_CAPABILITY_RVV

// Local dispatch macro for RVV adding Float support (not in common.h)
#define CPU_DISPATCH_PACKED_TYPES_RVV(TYPE, ...)                 \
  [&] {                                                          \
    switch (TYPE) {                                              \
      case at::ScalarType::Float: {                              \
        using packed_t = float;                                  \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::BFloat16: {                           \
        using packed_t = at::BFloat16;                           \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Half: {                               \
        using packed_t = at::Half;                               \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Char: {                               \
        using packed_t = int8_t;                                 \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Float8_e4m3fn: {                      \
        using packed_t = at::Float8_e4m3fn;                      \
        return __VA_ARGS__();                                    \
      }                                                          \
      default:                                                   \
        TORCH_CHECK(false, "Unsupported floating data type.\n"); \
    }                                                            \
  }()

// Implementation of convert_weight_packed required by common_ops
at::Tensor convert_weight_packed(at::Tensor& weight) {
  TORCH_CHECK(weight.defined(), "weight is undefined");

  const int64_t ndim = weight.ndimension();
  TORCH_CHECK(ndim == 2 || ndim == 3, "expect weight to be 2d or 3d, got ", ndim, "d tensor.");

  const int64_t E = ndim == 3 ? weight.size(0) : 1;
  const int64_t OC = ndim == 3 ? weight.size(1) : weight.size(0);
  const int64_t IC = ndim == 3 ? weight.size(2) : weight.size(1);

  // For small OC or unaligned OC, we use the FMA path (no packing, just transpose)
  // to avoid padding overhead and to preserve 2D shape for standard linear layers (fixing N deduction).
  constexpr int64_t BLOCK_N_RVV = 64;
  if (ndim == 2 && (OC < BLOCK_N_RVV || OC % BLOCK_N_RVV != 0)) {
    return weight.to(at::kFloat).t().contiguous();
  }

  auto st = weight.scalar_type();
  auto packed_weight = at::empty({}, weight.options());

  CPU_DISPATCH_PACKED_TYPES_RVV(st, [&] {
    // NB = ceil(OC / BLOCK_N_RVV)
    const int64_t NB = (OC + BLOCK_N_RVV - 1) / BLOCK_N_RVV;
    const int64_t packed_size_per_expert = NB * IC * BLOCK_N_RVV;

    packed_weight.resize_({E * packed_size_per_expert});

    packed_t* packed_data = packed_weight.data_ptr<packed_t>();
    const packed_t* w_data = weight.data_ptr<packed_t>();

    int64_t stride_input = OC * IC;
    int64_t stride_packed = packed_size_per_expert;

    for (int64_t e = 0; e < E; ++e) {
      pack_weight_rvv<packed_t>(packed_data + e * stride_packed, w_data + e * stride_input, OC, IC);
    }
  });

  return packed_weight;
}

// ============================================================================
//  Wrappers and entry points ported from gemm.cpp
// ============================================================================

// Wrappers for weight_packed_linear_kernel_impl calling RVV implementation

template <typename scalar_t>
void weight_packed_linear_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const scalar_t* __restrict__ mat2,
    const float* __restrict__ bias,
    const scalar_t* __restrict__ post_mul_mat,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed) {
  if (post_mul_mat == nullptr) {
    weight_packed_linear_kernel_impl_rvv(out, mat1, mat2, bias, M, N, K, mat1_strideM, out_strideM, is_packed);
  } else {
    TORCH_CHECK(false, "RVV fused linear sigmoid mul not supported");
  }
}

template <typename scalar_t, std::enable_if_t<!std::is_same_v<scalar_t, float>, int> = 0>
void weight_packed_linear_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const float* __restrict__ mat2,
    const float* __restrict__ bias,
    const scalar_t* __restrict__ post_mul_mat,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed) {
  if (post_mul_mat != nullptr) {
    TORCH_CHECK(false, "RVV fused linear sigmoid mul not supported");
  }

  // Fallback FMA path: Convert float weights to scalar_t and use native GEMM
  int64_t weight_size = K * N;
  auto options = at::TensorOptions().dtype(c10::CppTypeToScalarType<scalar_t>::value).device(at::kCPU);
  at::Tensor temp_weight = at::empty({weight_size}, options);
  scalar_t* temp_w_ptr = temp_weight.data_ptr<scalar_t>();

  // Convert float -> scalar_t
  // mat2 is K x N (transposed), we want NxK (standard linear weight) for gemm_native_rvv
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      temp_w_ptr[n * K + k] = static_cast<scalar_t>(mat2[k * N + n]);
    }
  }

  // Now temp_weight is NxK (standard linear weight).
  // Call gemm_native_rvv with ldb = K.
  gemm_native_rvv(
      mat1,
      temp_w_ptr,
      out,
      bias,
      M,
      N,
      K,
      mat1_strideM,
      K,  // ldb (stride of B)
      out_strideM);
}

// weight_packed_linear Implementation

at::Tensor
weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2, const std::optional<at::Tensor>& bias, bool is_vnni) {
  RECORD_FUNCTION("sgl-kernel::weight_packed_linear", std::vector<c10::IValue>({mat1, mat2, bias}));

  bool is_packed_call = is_vnni;
  at::Tensor packed_w;

  int64_t M = mat1.size(0);

  if (!is_vnni && M == 1) {
    // Fast path for decode (M=1) with unpacked weights:
    // Skip convert_weight_packed and use the implementation that handles unpacked weights directly (gemm_nt_rvv_linear)
    packed_w = mat2;
    is_packed_call = false;
  } else {
    packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);
    is_packed_call = true;
  }

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_DIM(2, mat1);
  // Relax check for packed weights as they might be 1D
  if (!is_vnni) {
    CHECK_DIM(2, mat2);
  }

  int64_t K = mat1.size(1);
  bool use_fma_gemm = (packed_w.scalar_type() == at::kFloat);

  int64_t N = 0;
  if (use_fma_gemm && is_vnni) {
    // FMA (Float) path: packed_w is usually [K, N] (transposed)
    // So N is size(1)
    N = packed_w.size(1);
  } else if (is_vnni) {
    // Packed (scalar_t) path: packed_w might be 1D flat tensor
    if (bias.has_value()) {
      N = bias.value().size(0);
    } else if (packed_w.dim() == 1) {
      // Assume exact packing (no padding) because convert_weight_packed
      // now enforces FMA for unaligned cases.
      // PackedSize = N * K
      N = packed_w.numel() / K;
    }
  } else {
    // Unpacked path (M=1 fast path): mat2 is [N, K] (standard Linear weight)
    N = mat2.size(0);
  }

  if (N == 0 && bias.has_value()) N = bias.value().size(0);

  if (!is_packed_call) {
    TORCH_CHECK(mat2.size(1) == K, "K dimension mismatch: input ", K, " vs weight ", mat2.size(1));
  }

  auto dispatch_type = mat1.scalar_type();
  auto out = at::empty({M, N}, mat1.options());
  // strides
  int64_t out_strideM = out.stride(0);
  int64_t mat1_strideM = mat1.stride(0);

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  auto kernel_func = [&](auto dummy_t) {
    using scalar_t = decltype(dummy_t);
    if (use_fma_gemm) {
      weight_packed_linear_kernel_impl<scalar_t>(
          out.data_ptr<scalar_t>(),
          mat1.data_ptr<scalar_t>(),
          packed_w.data_ptr<float>(),
          bias_data,
          nullptr,
          M,
          N,
          K,
          mat1_strideM,
          out_strideM,
          is_packed_call);
    } else {
      weight_packed_linear_kernel_impl<scalar_t>(
          out.data_ptr<scalar_t>(),
          mat1.data_ptr<scalar_t>(),
          packed_w.data_ptr<scalar_t>(),
          bias_data,
          nullptr,
          M,
          N,
          K,
          mat1_strideM,
          out_strideM,
          is_packed_call);
    }
  };

  AT_DISPATCH_SWITCH(dispatch_type, "weight_packed_linear_kernel_impl", AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
                       kernel_func(float{});
                     }) AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
                       kernel_func(at::Half{});
                     }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] { kernel_func(at::BFloat16{}); }));

  return out;
}

at::Tensor fused_linear_sigmoid_mul(
    at::Tensor& mat1,
    at::Tensor& mat2,
    const std::optional<at::Tensor>& bias,
    bool is_vnni,
    const at::Tensor& post_mul_mat) {
  RECORD_FUNCTION("sgl-kernel::fused_linear_sigmoid_mul", std::vector<c10::IValue>({mat1, mat2, bias, post_mul_mat}));

  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);
  TORCH_CHECK(packed_w.scalar_type() == at::kFloat, "fused_linear_sigmoid_mul requires packed float weight")

  int64_t M = mat1.size(0);
  int64_t K = mat1.size(1);
  int64_t N = mat2.size(1);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  int64_t out_strideM = post_mul_mat.size(1);
  int64_t mat1_strideM = mat1.stride(0);
  auto dispatch_type = mat1.scalar_type();
  auto out = at::empty({M, out_strideM}, mat1.options());

  TORCH_CHECK(
      N == 1 && out_strideM % 32 == 0,
      "post_mul_mat tensor size(1) should be 32 dividable, and the mat2 OC=1 (Mx1 as linear output shape)")

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  auto fused_kernel_func = [&](auto dummy_t) {
    using scalar_t = decltype(dummy_t);
    weight_packed_linear_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        mat1.data_ptr<scalar_t>(),
        packed_w.data_ptr<float>(),
        bias_data,
        post_mul_mat.data_ptr<scalar_t>(),
        M,
        N,
        K,
        mat1_strideM,
        out_strideM,
        true);
  };

  AT_DISPATCH_SWITCH(dispatch_type, "fused_linear_sigmoid_mul", AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
                       fused_kernel_func(float{});
                     }) AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
                       fused_kernel_func(at::Half{});
                     }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] { fused_kernel_func(at::BFloat16{}); }));

  return out;
}

// Implementation of tinygemm_kernel for RISC-V
// This is required by bmm.cpp and other generic components.
// We map it to the RVV native implementation.

template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    float* __restrict__ Ctmp,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {
  // On RISC-V, we ignore 'brg' (BRGEMM) and 'Ctmp' (used for BRGEMM accumulation).
  // We use the simpler gemm_native_rvv which accumulates in registers.
  gemm_native_rvv(A, B, C, nullptr, M, N, K, lda, ldb, ldc);
}

// Explicit instantiations
template void tinygemm_kernel<at::BFloat16>(
    const at::BFloat16* __restrict__ A,
    const at::BFloat16* __restrict__ B,
    at::BFloat16* __restrict__ C,
    float* __restrict__ Ctmp,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);

template void tinygemm_kernel<at::Half>(
    const at::Half* __restrict__ A,
    const at::Half* __restrict__ B,
    at::Half* __restrict__ C,
    float* __restrict__ Ctmp,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);

template void tinygemm_kernel<float>(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    float* __restrict__ Ctmp,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);
