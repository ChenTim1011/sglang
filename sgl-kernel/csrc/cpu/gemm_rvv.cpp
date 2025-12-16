// RISC-V Vector Extension (RVV) optimized GEMM kernels
// This file contains RVV specific implementations for Linear layers (QKV projection, FFN)

#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include <ATen/core/Tensor.h>

#include <cstdint>

#include "common.h"

#if defined(CPU_CAPABILITY_RVV)

#include <omp.h>
#include <riscv_vector.h>

#include <algorithm>
#include <cstring>

// Check for Zvfh (FP16 vector) support
#if defined(__riscv_zvfh) || defined(__riscv_v)
#define HAS_ZVFH 1
#else
#define HAS_ZVFH 0
#endif

namespace {

constexpr size_t MAX_VL_ELEMENTS = 512;
constexpr int BLOCK_M_RVV = 4;
constexpr int BLOCK_K_RVV = 64;

#if HAS_ZVFH

// C = A @ B^T + bias
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

    constexpr int64_t TILE_M = 4;

// Parallelize over M tiles
#pragma omp parallel for schedule(static) if (M > TILE_M)
    for (int64_t m0 = 0; m0 < M; m0 += TILE_M) {
      int64_t current_m_size = std::min(TILE_M, M - m0);

      for (int64_t n = 0; n < N; ++n) {
        // Pointers for B (weight row - logically col in matrix algebra terms)
        const uint16_t* b_ptr_start = reinterpret_cast<const uint16_t*>(B + n * ldb);
        float bias_val = (bias != nullptr) ? bias[n] : 0.0f;

        // Initialize accumulators
        vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max_fp32m4);
        vfloat32m4_t v_acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max_fp32m4);
        vfloat32m4_t v_acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max_fp32m4);
        vfloat32m4_t v_acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max_fp32m4);

        // Inner loop over K
        int64_t k = 0;
        for (; k + static_cast<int64_t>(vl_max_fp32m4) <= K; k += vl_max_fp32m4) {
          // 1. Load B vector (shared across M rows)
          vuint16m2_t v_b_u16 = __riscv_vle16_v_u16m2(b_ptr_start + k, vl_max_fp32m4);
          vuint32m4_t v_b_u32 = __riscv_vzext_vf2_u32m4(v_b_u16, vl_max_fp32m4);
          v_b_u32 = __riscv_vsll_vx_u32m4(v_b_u32, 16, vl_max_fp32m4);
          vfloat32m4_t v_b_f32 = __riscv_vreinterpret_v_u32m4_f32m4(v_b_u32);

          // 2. Process each M row (Unrolled)
          if (current_m_size > 0) {
            const uint16_t* a_ptr = reinterpret_cast<const uint16_t*>(A + (m0 + 0) * lda + k);
            vuint16m2_t v_a_u16 = __riscv_vle16_v_u16m2(a_ptr, vl_max_fp32m4);
            vuint32m4_t v_a_u32 = __riscv_vzext_vf2_u32m4(v_a_u16, vl_max_fp32m4);
            v_a_u32 = __riscv_vsll_vx_u32m4(v_a_u32, 16, vl_max_fp32m4);
            vfloat32m4_t v_a_f32 = __riscv_vreinterpret_v_u32m4_f32m4(v_a_u32);
            v_acc0 = __riscv_vfmacc_vv_f32m4(v_acc0, v_a_f32, v_b_f32, vl_max_fp32m4);
          }
          if (current_m_size > 1) {
            const uint16_t* a_ptr = reinterpret_cast<const uint16_t*>(A + (m0 + 1) * lda + k);
            vuint16m2_t v_a_u16 = __riscv_vle16_v_u16m2(a_ptr, vl_max_fp32m4);
            vuint32m4_t v_a_u32 = __riscv_vzext_vf2_u32m4(v_a_u16, vl_max_fp32m4);
            v_a_u32 = __riscv_vsll_vx_u32m4(v_a_u32, 16, vl_max_fp32m4);
            vfloat32m4_t v_a_f32 = __riscv_vreinterpret_v_u32m4_f32m4(v_a_u32);
            v_acc1 = __riscv_vfmacc_vv_f32m4(v_acc1, v_a_f32, v_b_f32, vl_max_fp32m4);
          }
          if (current_m_size > 2) {
            const uint16_t* a_ptr = reinterpret_cast<const uint16_t*>(A + (m0 + 2) * lda + k);
            vuint16m2_t v_a_u16 = __riscv_vle16_v_u16m2(a_ptr, vl_max_fp32m4);
            vuint32m4_t v_a_u32 = __riscv_vzext_vf2_u32m4(v_a_u16, vl_max_fp32m4);
            v_a_u32 = __riscv_vsll_vx_u32m4(v_a_u32, 16, vl_max_fp32m4);
            vfloat32m4_t v_a_f32 = __riscv_vreinterpret_v_u32m4_f32m4(v_a_u32);
            v_acc2 = __riscv_vfmacc_vv_f32m4(v_acc2, v_a_f32, v_b_f32, vl_max_fp32m4);
          }
          if (current_m_size > 3) {
            const uint16_t* a_ptr = reinterpret_cast<const uint16_t*>(A + (m0 + 3) * lda + k);
            vuint16m2_t v_a_u16 = __riscv_vle16_v_u16m2(a_ptr, vl_max_fp32m4);
            vuint32m4_t v_a_u32 = __riscv_vzext_vf2_u32m4(v_a_u16, vl_max_fp32m4);
            v_a_u32 = __riscv_vsll_vx_u32m4(v_a_u32, 16, vl_max_fp32m4);
            vfloat32m4_t v_a_f32 = __riscv_vreinterpret_v_u32m4_f32m4(v_a_u32);
            v_acc3 = __riscv_vfmacc_vv_f32m4(v_acc3, v_a_f32, v_b_f32, vl_max_fp32m4);
          }
        }

        // Handle tail loop
        if (k < K) {
          size_t tail_vl = __riscv_vsetvl_e32m4(K - k);

          // Initialize zero vectors
          vuint16m2_t v_b_u16 = __riscv_vmv_v_x_u16m2(0, vl_max_fp32m4);
          v_b_u16 = __riscv_vle16_v_u16m2_tu(v_b_u16, b_ptr_start + k, tail_vl);

          vuint32m4_t v_b_u32 = __riscv_vzext_vf2_u32m4(v_b_u16, vl_max_fp32m4);
          v_b_u32 = __riscv_vsll_vx_u32m4(v_b_u32, 16, vl_max_fp32m4);
          vfloat32m4_t v_b_f32 = __riscv_vreinterpret_v_u32m4_f32m4(v_b_u32);

          auto load_a_and_acc = [&](const uint16_t* ptr, vfloat32m4_t& acc) {
            vuint16m2_t v_a_u16 = __riscv_vmv_v_x_u16m2(0, vl_max_fp32m4);
            v_a_u16 = __riscv_vle16_v_u16m2_tu(v_a_u16, ptr, tail_vl);

            vuint32m4_t v_a_u32 = __riscv_vzext_vf2_u32m4(v_a_u16, vl_max_fp32m4);
            v_a_u32 = __riscv_vsll_vx_u32m4(v_a_u32, 16, vl_max_fp32m4);
            vfloat32m4_t v_a_f32 = __riscv_vreinterpret_v_u32m4_f32m4(v_a_u32);

            // Accumulate using FULL VL to preserve the non-tail partial sums
            acc = __riscv_vfmacc_vv_f32m4(acc, v_a_f32, v_b_f32, vl_max_fp32m4);
          };

          if (current_m_size > 0) load_a_and_acc(reinterpret_cast<const uint16_t*>(A + (m0 + 0) * lda + k), v_acc0);
          if (current_m_size > 1) load_a_and_acc(reinterpret_cast<const uint16_t*>(A + (m0 + 1) * lda + k), v_acc1);
          if (current_m_size > 2) load_a_and_acc(reinterpret_cast<const uint16_t*>(A + (m0 + 2) * lda + k), v_acc2);
          if (current_m_size > 3) load_a_and_acc(reinterpret_cast<const uint16_t*>(A + (m0 + 3) * lda + k), v_acc3);
        }

        // 3. Reduction and Store
        vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);

        // Helper lambda for store
        auto store_result = [&](vfloat32m4_t v_acc, int offset) {
          vfloat32m1_t v_red = __riscv_vfredusum_vs_f32m4_f32m1(v_acc, v_zero, vl_max_fp32m4);
          float sum = __riscv_vfmv_f_s_f32m1_f32(v_red);
          sum += bias_val;
          uint32_t sum_as_int;
          std::memcpy(&sum_as_int, &sum, sizeof(float));
          // Use Round-to-Nearest (RN) instead of Truncation for better precision
          if (std::isnan(sum)) {
            sum_as_int = 0x7FC00000;  // Canonical NaN
          } else {
            sum_as_int += 0x8000;
          }
          uint16_t res_bf16 = static_cast<uint16_t>(sum_as_int >> 16);
          at::BFloat16* c_val = C + (m0 + offset) * ldc + n;
          *reinterpret_cast<uint16_t*>(c_val) = res_bf16;
        };

        if (current_m_size > 0) store_result(v_acc0, 0);
        if (current_m_size > 1) store_result(v_acc1, 1);
        if (current_m_size > 2) store_result(v_acc2, 2);
        if (current_m_size > 3) store_result(v_acc3, 3);
      }
    }
    return;
  }

  // Only use FP16 path for Half type
  if constexpr (!std::is_same_v<scalar_t, at::Half>) {
    // Fall back to FP32 path for other types
    return;
  }

  // Get max vector lengths for different LMUL
  size_t vl_max_f16m4 = __riscv_vsetvlmax_e16m4();
  size_t vl_max_f16m1 = __riscv_vsetvlmax_e16m1();

// Parallel over M (each row of output is independent)
#pragma omp parallel for schedule(static) if (M > 1)
  for (int64_t m = 0; m < M; ++m) {
    const _Float16* a_row = reinterpret_cast<const _Float16*>(A + m * lda);
    _Float16* c_row = reinterpret_cast<_Float16*>(C + m * ldc);

    // Process each output column
    for (int64_t n = 0; n < N; ++n) {
      const _Float16* b_row = reinterpret_cast<const _Float16*>(B + n * ldb);

      // Use FP16 accumulation with LMUL=4 for high throughput
      vfloat16m4_t v_sum = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vl_max_f16m4);

      int64_t k = 0;
      // Main vectorized loop with LMUL=4
      for (; k + static_cast<int64_t>(vl_max_f16m4) <= K; k += vl_max_f16m4) {
        vfloat16m4_t v_a = __riscv_vle16_v_f16m4(a_row + k, vl_max_f16m4);
        vfloat16m4_t v_b = __riscv_vle16_v_f16m4(b_row + k, vl_max_f16m4);
        v_sum = __riscv_vfmacc_vv_f16m4(v_sum, v_a, v_b, vl_max_f16m4);
      }

      // Reduce FP16 vector to scalar
      vfloat16m1_t v_zero = __riscv_vfmv_v_f_f16m1((_Float16)0.0f, 1);
      vfloat16m1_t v_red = __riscv_vfredusum_vs_f16m4_f16m1(v_sum, v_zero, vl_max_f16m4);
      _Float16 sum = __riscv_vfmv_f_s_f16m1_f16(v_red);

      // Handle tail elements
      if (k < K) {
        size_t tail_vl = __riscv_vsetvl_e16m4(K - k);
        vfloat16m4_t v_a = __riscv_vle16_v_f16m4(a_row + k, tail_vl);
        vfloat16m4_t v_b = __riscv_vle16_v_f16m4(b_row + k, tail_vl);
        vfloat16m4_t v_tail = __riscv_vfmul_vv_f16m4(v_a, v_b, tail_vl);
        vfloat16m1_t v_tail_red = __riscv_vfredusum_vs_f16m4_f16m1(v_tail, v_zero, tail_vl);
        sum += __riscv_vfmv_f_s_f16m1_f16(v_tail_red);
      }

      // Add bias (convert to FP32 for accuracy, then back to FP16)
      float result = static_cast<float>(sum);
      if (bias != nullptr) {
        result += bias[n];
      }

      c_row[n] = static_cast<_Float16>(result);
    }
  }
}

// Tiled FP16 GEMM for better cache utilization with larger batches
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

  constexpr int64_t TILE_M = 4;
  constexpr int64_t TILE_N = 4;

  size_t vl_max = __riscv_vsetvlmax_e16m4();

#pragma omp parallel for schedule(static) if (M > TILE_M)
  for (int64_t m0 = 0; m0 < M; m0 += TILE_M) {
    int64_t m_size = std::min(TILE_M, M - m0);

    for (int64_t n0 = 0; n0 < N; n0 += TILE_N) {
      int64_t n_size = std::min(TILE_N, N - n0);

      // Accumulators for the tile (in FP32 for accuracy)
      float acc[TILE_M][TILE_N] = {{0.0f}};

      // Initialize with bias
      if (bias != nullptr) {
        for (int64_t m = 0; m < m_size; ++m) {
          for (int64_t n = 0; n < n_size; ++n) {
            acc[m][n] = bias[n0 + n];
          }
        }
      }

      // Main GEMM loop
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

          vfloat16m1_t v_zero = __riscv_vfmv_v_f_f16m1((_Float16)0.0f, 1);
          vfloat16m1_t v_red = __riscv_vfredusum_vs_f16m4_f16m1(v_sum, v_zero, vl_max);
          _Float16 partial = __riscv_vfmv_f_s_f16m1_f16(v_red);

          // Tail
          if (k < K) {
            size_t tail_vl = __riscv_vsetvl_e16m4(K - k);
            vfloat16m4_t v_a = __riscv_vle16_v_f16m4(a_row + k, tail_vl);
            vfloat16m4_t v_b = __riscv_vle16_v_f16m4(b_row + k, tail_vl);
            vfloat16m4_t v_tail = __riscv_vfmul_vv_f16m4(v_a, v_b, tail_vl);
            vfloat16m1_t v_tail_red = __riscv_vfredusum_vs_f16m4_f16m1(v_tail, v_zero, tail_vl);
            partial += __riscv_vfmv_f_s_f16m1_f16(v_tail_red);
          }

          acc[m][n] += static_cast<float>(partial);
        }
      }

      // Store results
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

// Load scalar_t data into float vector with type conversion
template <typename scalar_t>
inline vfloat32m1_t load_as_float_m1(const scalar_t* ptr, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return __riscv_vle32_v_f32m1(ptr, vl);
  } else {
    for (size_t i = 0; i < vl; ++i) {
      scratch[i] = static_cast<float>(ptr[i]);
    }
    return __riscv_vle32_v_f32m1(scratch, vl);
  }
}

// Load with m4
template <typename scalar_t>
inline vfloat32m4_t load_as_float_m4(const scalar_t* ptr, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return __riscv_vle32_v_f32m4(ptr, vl);
  } else {
    for (size_t i = 0; i < vl; ++i) {
      scratch[i] = static_cast<float>(ptr[i]);
    }
    return __riscv_vle32_v_f32m4(scratch, vl);
  }
}

// Store float vector to scalar_t with type conversion
template <typename scalar_t>
inline void store_from_float_m1(scalar_t* ptr, vfloat32m1_t v, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vse32_v_f32m1(ptr, v, vl);
  } else {
    __riscv_vse32_v_f32m1(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i) {
      ptr[i] = static_cast<scalar_t>(scratch[i]);
    }
  }
}

template <typename scalar_t>
inline void store_from_float_m4(scalar_t* ptr, vfloat32m4_t v, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vse32_v_f32m4(ptr, v, vl);
  } else {
    __riscv_vse32_v_f32m4(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i) {
      ptr[i] = static_cast<scalar_t>(scratch[i]);
    }
  }
}

// Load strided scalar_t data into float vector
template <typename scalar_t>
inline vfloat32m4_t load_strided_as_float_m4(const scalar_t* ptr, ptrdiff_t stride_byte, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    // Native FP32 strided load
    return __riscv_vlse32_v_f32m4(ptr, stride_byte, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
    // FP16 strided load -> extend to FP32
    // Load as FP16 (vlse16)
    vfloat16m2_t v_f16 = __riscv_vlse16_v_f16m2(reinterpret_cast<const _Float16*>(ptr), stride_byte, vl);
    // Convert to FP32 (vfwcvt)
    return __riscv_vfwcvt_f_f_v_f32m4(v_f16, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    // BF16 strided load -> convert to FP32
    // Load as u16 bits
    vuint16m2_t v_bf16 = __riscv_vlse16_v_u16m2(reinterpret_cast<const uint16_t*>(ptr), stride_byte, vl);
    // Zero extend to u32
    vuint32m4_t v_u32 = __riscv_vzext_vf2_u32m4(v_bf16, vl);
    // Shift left 16
    v_u32 = __riscv_vsll_vx_u32m4(v_u32, 16, vl);
    // Reinterpret as float
    return __riscv_vreinterpret_v_u32m4_f32m4(v_u32);
  } else {
    // Generic fallback
    for (size_t i = 0; i < vl; ++i) {
      const scalar_t* elem_ptr =
          reinterpret_cast<const scalar_t*>(reinterpret_cast<const char*>(ptr) + i * stride_byte);
      scratch[i] = static_cast<float>(*elem_ptr);
    }
    return __riscv_vle32_v_f32m4(scratch, vl);
  }
}

// GEMM kernel: C = A @ B
// A: [M, K], B: [K, N]
// C: [M, N]
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
  // Use m4 for higher throughput when possible
  size_t vl_max_m1 = __riscv_vsetvlmax_e32m1();
  size_t vl_max_m4 = __riscv_vsetvlmax_e32m4();

  // Process M in blocks
  for (int64_t m_start = 0; m_start < M; m_start += BLOCK_M_RVV) {
    int64_t m_size = std::min(static_cast<int64_t>(BLOCK_M_RVV), M - m_start);

    // Initialize accumulator with bias or zero
    for (int64_t m = 0; m < m_size; ++m) {
      float* acc_row = acc_buf + m * N;
      if (bias != nullptr) {
        // Copy bias to accumulator
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

    // Main GEMM computation: C[m, n] += A[m, k] * B[n, k]
    // Iterate over K
    for (int64_t k = 0; k < K; ++k) {
      // For each row in the block
      for (int64_t m = 0; m < m_size; ++m) {
        float a_val = static_cast<float>(A[(m_start + m) * lda + k]);
        if (std::abs(a_val) < 1e-8f) continue;  // Skip near-zero

        float* acc_row = acc_buf + m * N;
        const scalar_t* b_row = B + k;  // B[n, k] -> access B[0, k], B[1, k], ...

        // Vectorized accumulation over N
        size_t vl;
        for (int64_t n = 0; n < N; n += vl) {
          vl = __riscv_vsetvl_e32m4(N - n);

          // ldb is in elements, so stride in bytes is ldb * sizeof(scalar_t)
          ptrdiff_t stride_bytes = ldb * sizeof(scalar_t);
          vfloat32m4_t v_b = load_strided_as_float_m4(B + n * ldb + k, stride_bytes, vl, scratch_b);

          // Load accumulator
          vfloat32m4_t v_acc = __riscv_vle32_v_f32m4(acc_row + n, vl);

          // acc += a_val * b
          v_acc = __riscv_vfmacc_vf_f32m4(v_acc, a_val, v_b, vl);

          // Store back
          __riscv_vse32_v_f32m4(acc_row + n, v_acc, vl);
        }
      }
    }

    // Convert accumulator to output
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

// Optimized version for contiguous B (when weight is pre-arranged)
// C = A @ B^T where B is [N, K] stored contiguously
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

// Parallel over M (each row of output is independent)
#pragma omp parallel for if (M > 1)
  for (int64_t m = 0; m < M; ++m) {
    const scalar_t* a_row = A + m * lda;
    scalar_t* c_row = C + m * ldc;

    // Thread-local scratch buffer
    alignas(64) float local_scratch_a[MAX_VL_ELEMENTS];
    alignas(64) float local_scratch_b[MAX_VL_ELEMENTS];

    // For each output column
    for (int64_t n = 0; n < N; ++n) {
      const scalar_t* b_row = B + n * ldb;

      // Dot product: sum_k(A[m, k] * B[n, k])
      vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, vl_max);

      int64_t k = 0;
      // Main vectorized loop
      for (; k + static_cast<int64_t>(vl_max) <= K; k += vl_max) {
        vfloat32m1_t v_a = load_as_float_m1(a_row + k, vl_max, local_scratch_a);
        vfloat32m1_t v_b = load_as_float_m1(b_row + k, vl_max, local_scratch_b);
        v_sum = __riscv_vfmacc_vv_f32m1(v_sum, v_a, v_b, vl_max);
      }

      // Reduce
      vfloat32m1_t v_red = __riscv_vfredusum_vs_f32m1_f32m1(v_sum, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl_max);
      float sum = __riscv_vfmv_f_s_f32m1_f32(v_red);

      // Tail
      if (k < K) {
        size_t tail_vl = __riscv_vsetvl_e32m1(K - k);
        vfloat32m1_t v_a = load_as_float_m1(a_row + k, tail_vl, local_scratch_a);
        vfloat32m1_t v_b = load_as_float_m1(b_row + k, tail_vl, local_scratch_b);
        vfloat32m1_t v_tail = __riscv_vfmul_vv_f32m1(v_a, v_b, tail_vl);
        vfloat32m1_t v_tail_red = __riscv_vfredusum_vs_f32m1_f32m1(v_tail, __riscv_vfmv_v_f_f32m1(0.0f, 1), tail_vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(v_tail_red);
      }

      // Add bias if present
      if (bias != nullptr) {
        sum += bias[n];
      }

      c_row[n] = static_cast<scalar_t>(sum);
    }
  }
}

// Tiled GEMM: C = A @ B^T + bias
// Uses blocking to improve cache efficiency
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
  constexpr int64_t TILE_M = 4;
  constexpr int64_t TILE_K = 128;

  size_t vl_max = __riscv_vsetvlmax_e32m1();

#pragma omp parallel
  {
    // Thread-local scratch buffers
    alignas(64) float scratch_a[MAX_VL_ELEMENTS];
    alignas(64) float scratch_b[MAX_VL_ELEMENTS];
    alignas(64) float acc_rows[TILE_M * 4096];

#pragma omp for schedule(static)
    for (int64_t m0 = 0; m0 < M; m0 += TILE_M) {
      int64_t m_size = std::min(TILE_M, M - m0);

      // Process each output column
      for (int64_t n = 0; n < N; ++n) {
        const scalar_t* b_row = B + n * ldb;

        // Initialize accumulators with bias
        float acc[TILE_M];
        for (int64_t m = 0; m < m_size; ++m) {
          acc[m] = (bias != nullptr) ? bias[n] : 0.0f;
        }

        // Process K in tiles for better cache utilization
        for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
          int64_t k_size = std::min(TILE_K, K - k0);

          // Compute dot products for each row in the tile
          for (int64_t m = 0; m < m_size; ++m) {
            const scalar_t* a_row = A + (m0 + m) * lda + k0;

            // Vectorized dot product
            vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, vl_max);

            int64_t k = 0;
            for (; k + static_cast<int64_t>(vl_max) <= k_size; k += vl_max) {
              vfloat32m1_t v_a = load_as_float_m1(a_row + k, vl_max, scratch_a);
              vfloat32m1_t v_b = load_as_float_m1(b_row + k0 + k, vl_max, scratch_b);
              v_sum = __riscv_vfmacc_vv_f32m1(v_sum, v_a, v_b, vl_max);
            }

            // Reduce vector to scalar
            vfloat32m1_t v_red = __riscv_vfredusum_vs_f32m1_f32m1(v_sum, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl_max);
            float partial_sum = __riscv_vfmv_f_s_f32m1_f32(v_red);

            // Handle tail elements
            if (k < k_size) {
              size_t tail_vl = __riscv_vsetvl_e32m1(k_size - k);
              vfloat32m1_t v_a = load_as_float_m1(a_row + k, tail_vl, scratch_a);
              vfloat32m1_t v_b = load_as_float_m1(b_row + k0 + k, tail_vl, scratch_b);
              vfloat32m1_t v_tail = __riscv_vfmul_vv_f32m1(v_a, v_b, tail_vl);
              vfloat32m1_t v_tail_red =
                  __riscv_vfredusum_vs_f32m1_f32m1(v_tail, __riscv_vfmv_v_f_f32m1(0.0f, 1), tail_vl);
              partial_sum += __riscv_vfmv_f_s_f32m1_f32(v_tail_red);
            }

            acc[m] += partial_sum;
          }
        }

        // Store results
        for (int64_t m = 0; m < m_size; ++m) {
          C[(m0 + m) * ldc + n] = static_cast<scalar_t>(acc[m]);
        }
      }
    }
  }
}

}  // anonymous namespace

// Main entry point for RVV GEMM - matches the signature in gemm.h
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
    int64_t out_strideM) {
#if HAS_ZVFH
  // Use native FP16 kernel for Half type (fastest path)
  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if (M <= 4) {
      // Small batch: use simpler kernel
      gemm_fp16_native_rvv(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM);
    } else {
      // Larger batch: use tiled version
      gemm_fp16_tiled_rvv(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM);
    }
    return;
  }
  // Use Optimized BF16 Emulation kernel
  if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    gemm_fp16_native_rvv(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM);
    return;
  }
#endif

  // Fallback to FP32-based kernels for non-Half types or when Zvfh is unavailable
  if (M == 1) {
    // Single row: use simpler gemv-like kernel
    gemm_nt_rvv_linear(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM, nullptr);
  } else {
    // Multiple rows: use tiled GEMM
    gemm_tiled_rvv(mat1, mat2, out, bias, M, N, K, mat1_strideM, K, out_strideM);
  }
}

// Explicit instantiations
template void weight_packed_linear_kernel_impl_rvv<float>(
    float* out,
    const float* mat1,
    const float* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM);

template void weight_packed_linear_kernel_impl_rvv<at::BFloat16>(
    at::BFloat16* out,
    const at::BFloat16* mat1,
    const at::BFloat16* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM);

template void weight_packed_linear_kernel_impl_rvv<at::Half>(
    at::Half* out,
    const at::Half* mat1,
    const at::Half* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM);

#else  // !CPU_CAPABILITY_RVV

// Fallback: provide stub implementations when RVV is not available
// These should never be called - the dispatcher in gemm.cpp handles this
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
    int64_t out_strideM) {
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
    int64_t out_strideM);

template void weight_packed_linear_kernel_impl_rvv<at::BFloat16>(
    at::BFloat16* out,
    const at::BFloat16* mat1,
    const at::BFloat16* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM);

template void weight_packed_linear_kernel_impl_rvv<at::Half>(
    at::Half* out,
    const at::Half* mat1,
    const at::Half* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM);

#endif  // CPU_CAPABILITY_RVV
