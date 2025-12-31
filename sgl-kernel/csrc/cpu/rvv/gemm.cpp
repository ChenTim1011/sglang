#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include <ATen/core/Tensor.h>

#include <cstdint>

#include "common.h"
#include "vector_helpers.h"

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

template <typename scalar_t>
void gemm_fp16_native_rvv(
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

  for (int64_t m_start = 0; m_start < M; m_start += BLOCK_M_RVV) {
    int64_t m_size = std::min(static_cast<int64_t>(BLOCK_M_RVV), M - m_start);
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
        if (std::abs(a_val) < rvv_constants::ZERO_THRESHOLD) continue;
        float* acc_row = acc_buf + m * N;
        const scalar_t* b_row = B + k;
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
    alignas(64) float acc[BLOCK_M][BLOCK_N];
    auto init_acc = [&](auto i) {
      constexpr int m = i.value;
      if constexpr (has_bias) {
        for (int64_t j = 0; j < n_size; ++j) {
          acc[m][j] = bias[n_start + j];
        }
        for (int64_t j = n_size; j < BLOCK_N; ++j) {
          acc[m][j] = 0.0f;
        }
      } else {
        std::memset(acc[m], 0, BLOCK_N * sizeof(float));
      }
    };
    Unroll<BLOCK_M>{}(init_acc);
    for (int64_t k = 0; k < K; ++k) {
      if (k + 1 < K) {
        __builtin_prefetch(PackedB + nb * K * BLOCK_N + (k + 1) * BLOCK_N, 0, 3);
      }
      const scalar_t* b_ptr = PackedB + nb * K * BLOCK_N + k * BLOCK_N;
      auto compute = [&](auto i) {
        constexpr int m = i.value;
        float a_val = static_cast<float>(A[m * lda + k]);
        if (std::abs(a_val) < 1e-8f) return;
        size_t vl;
        for (int64_t j = 0; j < n_size; j += vl) {
          vl = __riscv_vsetvl_e32m4(n_size - j);
          vfloat32m4_t v_b;
          if constexpr (std::is_same_v<scalar_t, float>) {
            v_b = __riscv_vle32_v_f32m4(b_ptr + j, vl);
          } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
            vfloat16m2_t v_b16 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(b_ptr + j), vl);
            v_b = __riscv_vfwcvt_f_f_v_f32m4(v_b16, vl);
          } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
            v_b = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(b_ptr + j), vl);
          }
          vfloat32m4_t v_acc = __riscv_vle32_v_f32m4(acc[m] + j, vl);
          v_acc = __riscv_vfmacc_vf_f32m4(v_acc, a_val, v_b, vl);
          __riscv_vse32_v_f32m4(acc[m] + j, v_acc, vl);
        }
      };
      Unroll<BLOCK_M>{}(compute);
    }
    auto store = [&](auto i) {
      constexpr int m = i.value;
      scalar_t* c_ptr = C + m * ldc + n_start;
      size_t vl;
      for (int64_t j = 0; j < n_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(n_size - j);
        vfloat32m4_t v_val = __riscv_vle32_v_f32m4(acc[m] + j, vl);

        if constexpr (std::is_same_v<scalar_t, float>) {
          __riscv_vse32_v_f32m4(c_ptr + j, v_val, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_f16 = f32m4_to_f16(v_val, vl);
          __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(c_ptr + j), v_f16, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          vuint16m2_t v_bf16 = f32m4_to_bf16(v_val, vl);
          __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(c_ptr + j), v_bf16, vl);
        }
      }
    };
    Unroll<BLOCK_M>{}(store);
  }
};

#define LAUNCH_TINYGEMM_KERNEL_RVV(MB_SIZE, NB_SIZE)                \
  tinygemm_kernel_rvv<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda, PackedB, C + mb_start * ldc, has_bias ? bias : nullptr, K, lda, ldc, nb, n_size);

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

  size_t vl_max = __riscv_vsetvlmax_e32m4();

#pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const scalar_t* a_row = A + m * lda;
    scalar_t* c_row = C + m * ldc;

    for (int64_t nb = 0; nb < (N + BLOCK_N - 1) / BLOCK_N; ++nb) {
      int64_t n_start = nb * BLOCK_N;
      int64_t n_size = std::min(BLOCK_N, N - n_start);

      float acc[BLOCK_N];
      if (bias) {
        for (int j = 0; j < n_size; ++j)
          acc[j] = bias[n_start + j];
      } else {
        std::memset(acc, 0, n_size * sizeof(float));
      }

      for (int64_t k = 0; k < K; ++k) {
        float a_val = static_cast<float>(a_row[k]);
        if (std::abs(a_val) < rvv_constants::ZERO_THRESHOLD) continue;
        const scalar_t* b_ptr = PackedB + nb * K * BLOCK_N + k * BLOCK_N;

        if (k + rvv_constants::PREFETCH_DISTANCE < K) {
          __builtin_prefetch(
              PackedB + nb * K * BLOCK_N + (k + rvv_constants::PREFETCH_DISTANCE) * BLOCK_N,
              0,
              rvv_constants::PREFETCH_LOCALITY);
        }

        size_t vl_max = __riscv_vsetvlmax_e32m4();
        size_t vl;
        for (int64_t j = 0; j < n_size; j += vl_max) {
          vl = (j + vl_max <= n_size) ? vl_max : __riscv_vsetvl_e32m4(n_size - j);

          vfloat32m4_t v_b;
          if constexpr (std::is_same_v<scalar_t, float>) {
            v_b = __riscv_vle32_v_f32m4(b_ptr + j, vl);
          } else {
            if constexpr (std::is_same_v<scalar_t, at::Half>) {
              vfloat16m2_t v_b16 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(b_ptr + j), vl);
              v_b = __riscv_vfwcvt_f_f_v_f32m4(v_b16, vl);
            } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
              v_b = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(b_ptr + j), vl);
            }
          }
          vfloat32m4_t v_acc = __riscv_vle32_v_f32m4(acc + j, vl);
          v_acc = __riscv_vfmacc_vf_f32m4(v_acc, a_val, v_b, vl);
          __riscv_vse32_v_f32m4(acc + j, v_acc, vl);
        }
      }

      for (int64_t j = 0; j < n_size; ++j) {
        c_row[n_start + j] = static_cast<scalar_t>(acc[j]);
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
      gemm_fp16_native_rvv(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM);
    } else {
      gemm_fp16_tiled_rvv(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM);
    }
    return;
  }
  if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    gemm_fp16_native_rvv(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM);
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
template void
pack_weight_rvv<at::Float8_e4m3fn>(at::Float8_e4m3fn* packed_w, const at::Float8_e4m3fn* orig_w, int64_t N, int64_t K);

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
template void
pack_weight_rvv<at::Float8_e4m3fn>(at::Float8_e4m3fn* packed_w, const at::Float8_e4m3fn* orig_w, int64_t N, int64_t K);

#endif  // CPU_CAPABILITY_RVV
