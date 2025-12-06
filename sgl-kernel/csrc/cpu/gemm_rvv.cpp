// RISC-V Vector Extension (RVV) optimized GEMM kernels
// This file contains RVV specific implementations for Linear layers (QKV projection, FFN)
// Note: This provides RVV-accelerated alternatives to the AVX512 implementations in gemm.cpp
//
// Key optimizations:
// 1. Uses RVV intrinsics for vectorized computation
// 2. Native FP16 computation with Zvfh extension (no type conversion overhead)
// 3. Tiled GEMM for better cache utilization
// 4. Multi-row processing for better instruction-level parallelism
// 5. Thread-local scratch buffers to avoid allocation overhead
// 6. OpenMP parallelization across M dimension

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
// Most modern RISC-V implementations with RVV 1.0 include Zvfh
#if defined(__riscv_zvfh) || defined(__riscv_v)
#define HAS_ZVFH 1
#else
#define HAS_ZVFH 0
#endif

namespace {

// ============================================================================
// Constants and Configuration
// ============================================================================

// Maximum vector length elements (conservative for VLEN up to 16384 bits)
constexpr size_t MAX_VL_ELEMENTS = 512;

// Block sizes for tiled GEMM
// These are tuned for typical RISC-V implementations
constexpr int BLOCK_M_RVV = 4;   // Process 4 rows at a time
constexpr int BLOCK_K_RVV = 64;  // K dimension block size

// ============================================================================
// FP16 Native GEMM Kernels (Zvfh extension)
// ============================================================================

#if HAS_ZVFH

// Optimized FP16 GEMM for Linear layers using native Zvfh instructions
// C = A @ B^T + bias where:
//   A: [M, K] input activation (float16)
//   B: [N, K] weight (float16, stored as [out_features, in_features])
//   C: [M, N] output (float16)
//   bias: [N] (float32)
//
// This version uses native FP16 computation throughout, avoiding costly
// FP16->FP32 conversions. The bias addition is done with widening to FP32.
template <typename scalar_t>
void gemm_fp16_native_rvv(
    const scalar_t* __restrict__ A,  // [M, K]
    const scalar_t* __restrict__ B,  // [N, K]
    scalar_t* __restrict__ C,        // [M, N]
    const float* __restrict__ bias,  // [N] or nullptr
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  // Only use FP16 path for Half type
  if constexpr (!std::is_same_v<scalar_t, at::Half>) {
    // Fall back to FP32 path for other types
    return;
  }

  // Get max vector lengths for different LMUL
  size_t vl_max_f16m4 = __riscv_vsetvlmax_e16m4();  // FP16 with LMUL=4
  size_t vl_max_f16m1 = __riscv_vsetvlmax_e16m1();  // FP16 with LMUL=1

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
    const scalar_t* __restrict__ A,  // [M, K]
    const scalar_t* __restrict__ B,  // [N, K]
    scalar_t* __restrict__ C,        // [M, N]
    const float* __restrict__ bias,  // [N] or nullptr
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  if constexpr (!std::is_same_v<scalar_t, at::Half>) {
    return;
  }

  // Tile sizes
  constexpr int64_t TILE_M = 4;
  constexpr int64_t TILE_N = 4;  // Process multiple N at once to reduce reduce operations

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

// ============================================================================
// Helper Functions
// ============================================================================

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

// Load with m4 (4x register group for higher throughput)
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

// ============================================================================
// Core GEMM Kernels
// ============================================================================

// GEMM kernel: C = A @ B
// A: [M, K], B: [K, N] (B is in packed/transposed format for Linear layers)
// C: [M, N]
// This is the main kernel for Linear layers (QKV projection, FFN)
//
// For Linear layers, the weight is typically stored as [out_features, in_features]
// which means B^T in our notation. We compute C = A @ B^T
template <typename scalar_t>
void gemm_rvv_impl(
    const scalar_t* __restrict__ A,  // [M, K]
    const scalar_t* __restrict__ B,  // [N, K] (weight, stored as [out_features, in_features])
    scalar_t* __restrict__ C,        // [M, N]
    const float* __restrict__ bias,  // [N] or nullptr
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,                    // stride of A (typically K)
    int64_t ldb,                    // stride of B (typically K)
    int64_t ldc,                    // stride of C (typically N)
    float* __restrict__ scratch_a,  // scratch buffer for A conversion
    float* __restrict__ scratch_b,  // scratch buffer for B conversion
    float* __restrict__ acc_buf) {  // accumulator buffer [BLOCK_M_RVV, N]

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

    // Main GEMM computation: C[m, n] += A[m, k] * B[n, k] (B is transposed)
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

          // Load B values (strided access: B[n, k], B[n+1, k], ...)
          // Since B is stored as [N, K], B[n, k] = B[n * ldb + k]
          // We need to gather or use scalar loop for non-unit stride

          // For now, use scalar gather (can be optimized with segment load if available)
          for (size_t i = 0; i < vl; ++i) {
            scratch_b[i] = static_cast<float>(B[(n + i) * ldb + k]);
          }
          vfloat32m4_t v_b = __riscv_vle32_v_f32m4(scratch_b, vl);

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
    const scalar_t* __restrict__ A,  // [M, K]
    const scalar_t* __restrict__ B,  // [N, K]
    scalar_t* __restrict__ C,        // [M, N]
    const float* __restrict__ bias,  // [N] or nullptr
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

// ============================================================================
// Tiled GEMM for better cache utilization
// ============================================================================

// Tiled GEMM: C = A @ B^T + bias
// Uses blocking to improve cache efficiency
//
// Key optimization: Process multiple M rows per iteration to amortize
// the overhead of loading B weights
template <typename scalar_t>
void gemm_tiled_rvv(
    const scalar_t* __restrict__ A,  // [M, K]
    const scalar_t* __restrict__ B,  // [N, K]
    scalar_t* __restrict__ C,        // [M, N]
    const float* __restrict__ bias,  // [N] or nullptr
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  // Tile sizes tuned for typical RISC-V cache configurations
  constexpr int64_t TILE_M = 4;    // Process 4 rows at a time
  constexpr int64_t TILE_K = 128;  // K tile size for cache efficiency

  // Get max vector length
  size_t vl_max = __riscv_vsetvlmax_e32m1();

// Parallel over M dimension
#pragma omp parallel
  {
    // Thread-local scratch buffers (avoid allocation per call)
    alignas(64) float scratch_a[MAX_VL_ELEMENTS];
    alignas(64) float scratch_b[MAX_VL_ELEMENTS];
    alignas(64) float acc_rows[TILE_M * 4096];  // Accumulator for tile rows

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

// ============================================================================
// Public API - Called from gemm.cpp when RVV is available
// ============================================================================

// Main entry point for RVV GEMM - matches the signature in gemm.h
// This is called from weight_packed_linear_kernel_impl when running on RISC-V
//
// Parameters:
//   out: [M, N] output tensor
//   mat1: [M, K] input activation
//   mat2: [N, K] weight (transposed, so B^T layout)
//   bias: [N] bias vector or nullptr
//   M, N, K: matrix dimensions
//   mat1_strideM: stride of mat1 in M dimension
//   out_strideM: stride of out in M dimension
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
