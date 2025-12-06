// RISC-V Vector Extension (RVV) optimized decode attention kernels
// This file contains RVV specific implementations for decode attention
// Note: This file is included directly in decode.cpp, not compiled separately

// For RISC-V with Clang, we need pragma BEFORE any includes
#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include <vector>

#include "common.h"

#if defined(CPU_CAPABILITY_RVV)
// Clang 19 version check
#if !defined(__clang__)
#error "RVV backend in decode_rvv.cpp requires Clang compiler. Please use Clang 19.1.0 or later to compile this file."
#endif

#if !defined(__clang_major__) || __clang_major__ < 19
#error "RVV backend requires Clang 19 or later. Current version: " __clang_version__ ". Please use Clang 19.1.0."
#endif

#include <riscv_vector.h>

// Check for Zvfh (FP16 vector) support
// Spacemit X60 (Banana Pi) supports zvfh_zvfhmin
#if defined(__riscv_zvfh) || defined(__riscv_v)
#define HAS_ZVFH_DECODE 1
#else
#define HAS_ZVFH_DECODE 0
#endif

#endif

namespace {

#if defined(CPU_CAPABILITY_RVV)

// Helper: Load scalar_t data to float32 buffer using Zvfh when available
// For FP16 with Zvfh: Uses hardware vle16 + vfwcvt (widening convert)
// For FP16 without Zvfh: Software conversion loop
// For FP32: Direct copy
//
// This is used for Q/K/V loading before dot product computation
template <typename scalar_t>
inline void load_to_float32_buffer_rvv(const scalar_t* src, float* dst, int64_t len) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    // FP32: Direct copy
    for (int64_t i = 0; i < len; ++i) {
      dst[i] = src[i];
    }
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH_DECODE
    // FP16 with Zvfh: Hardware widening conversion (fast path)
    int64_t offset = 0;
    while (offset < len) {
      size_t vl = __riscv_vsetvl_e16mf2(len - offset);
      vfloat16mf2_t v_f16 = __riscv_vle16_v_f16mf2(reinterpret_cast<const _Float16*>(src + offset), vl);
      vfloat32m1_t v_f32 = __riscv_vfwcvt_f_f_v_f32m1(v_f16, vl);
      __riscv_vse32_v_f32m1(dst + offset, v_f32, vl);
      offset += vl;
    }
#else
    // FP16 without Zvfh: Software conversion (slow path)
    for (int64_t i = 0; i < len; ++i) {
      dst[i] = static_cast<float>(src[i]);
    }
#endif
  } else {
    // Other types (BFloat16, etc.): Software conversion
    for (int64_t i = 0; i < len; ++i) {
      dst[i] = static_cast<float>(src[i]);
    }
  }
}

inline float rvv_reduce_sum_f32(const float* data, int64_t len) {
  size_t vl = __riscv_vsetvl_e32m1(len);
  if (vl == static_cast<size_t>(len)) {
    vfloat32m1_t vdata = __riscv_vle32_v_f32m1(data, vl);
    std::vector<float> temp(vl);
    __riscv_vse32_v_f32m1(temp.data(), vdata, vl);
    float sum = 0.f;
    for (size_t i = 0; i < vl; ++i) {
      sum += temp[i];
    }
    return sum;
  }
  float sum = 0.f;
  for (int64_t i = 0; i < len; ++i) {
    sum += data[i];
  }
  return sum;
}

inline float rvv_reduce_max_f32(const float* data, int64_t len) {
  size_t vl = __riscv_vsetvl_e32m1(len);
  if (vl == static_cast<size_t>(len)) {
    vfloat32m1_t vdata = __riscv_vle32_v_f32m1(data, vl);
    std::vector<float> temp(vl);
    __riscv_vse32_v_f32m1(temp.data(), vdata, vl);
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vl; ++i) {
      max_val = std::max(max_val, temp[i]);
    }
    return max_val;
  }
  float max_val = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < len; ++i) {
    max_val = std::max(max_val, data[i]);
  }
  return max_val;
}

// 1. Query-Key dot product
// Efficient query-key matrix multiplication using RVV intrinsics
// with proper float16 to float32 conversion to ensure numerical accuracy
// Uses Zvfh hardware acceleration when available for FP16 loading
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
  // Set vector length (based on K size)
  size_t vl = __riscv_vsetvl_e32m1(K);

  // Check head_dim size to prevent buffer overflow
  TORCH_CHECK(K <= 256, "head_dim (", K, ") exceeds maximum supported size (256)");

  // Temporary buffer for converting scalar_t to float32
  // This is needed when scalar_t is float16 (2 bytes) but we need float32 (4 bytes) for RVV
  alignas(64) float q_float32[256];  // Max head_dim typically 256
  alignas(64) float k_float32[256];

  for (int64_t m = 0; m < M; ++m) {
    // Convert query from scalar_t to float32 using Zvfh when available
    load_to_float32_buffer_rvv(A + m * lda, q_float32, K);

    for (int64_t n = 0; n < N; ++n) {
      int64_t b_idx = indices[n];
      TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");

      // Convert key from scalar_t to float32 using Zvfh when available
      load_to_float32_buffer_rvv(B + b_idx * ldb, k_float32, K);

      // Initialize dot product accumulator
      vfloat32m1_t vdot = __riscv_vfmv_v_f_f32m1(0.0f, vl);

      // Vectorized dot product calculation
      int64_t k = 0;
      for (; k + vl <= K; k += vl) {
        // Load vectors (now correctly as float32)
        vfloat32m1_t vq = __riscv_vle32_v_f32m1(q_float32 + k, vl);
        vfloat32m1_t vk = __riscv_vle32_v_f32m1(k_float32 + k, vl);

        // Multiply-add: dot += q * k
        vdot = __riscv_vfmacc_vv_f32m1(vdot, vq, vk, vl);
      }

      // Handle tail elements
      if (k < K) {
        size_t tail_vl = __riscv_vsetvl_e32m1(K - k);
        vfloat32m1_t vq = __riscv_vle32_v_f32m1(q_float32 + k, tail_vl);
        vfloat32m1_t vk = __riscv_vle32_v_f32m1(k_float32 + k, tail_vl);
        vdot = __riscv_vfmacc_vv_f32m1(vdot, vq, vk, tail_vl);
      }

      // Reduce sum
      // Use dynamic buffer size based on actual vl to avoid overflow
      // RVV supports VLEN up to 65536, so we need to handle variable sizes
      float sum = 0.0f;
      size_t temp_size = (vl + 31) / 32;  // Round up to handle any VLEN
      if (temp_size > 32) {
        // For very large VLEN, use dynamic allocation
        std::vector<float> temp_vec(temp_size);
        __riscv_vse32_v_f32m1(temp_vec.data(), vdot, vl);
        for (size_t i = 0; i < vl; i++) {
          sum += temp_vec[i];
        }
      } else {
        // For normal VLEN (<= 1024), use stack buffer
        float temp[32];  // Sufficient for VLEN up to 1024 (32 * 32 = 1024)
        __riscv_vse32_v_f32m1(temp, vdot, vl);
        for (size_t i = 0; i < vl; i++) {
          sum += temp[i];
        }
      }

      // Apply scaling and store result
      C[m * ldc + n] = sum * scale;
    }
  }
}

// 2. Numerically stable softmax using log-sum-exp trick
// Calculate exp(scores - m_i) and normalize
// Handles edge cases (zero/infinite sum by falling back to uniform distribution)
inline void softmax_rvv(const float* __restrict__ scores, float* __restrict__ probs, int64_t N, float m_i) {
  if (N == 0) {
    return;
  }

  std::vector<float> exp_scores(N);

  int64_t offset = 0;
  while (offset < N) {
    size_t vl = __riscv_vsetvl_e32m1(N - offset);
    vfloat32m1_t vmax_broadcast = __riscv_vfmv_v_f_f32m1(m_i, vl);
    vfloat32m1_t vscores = __riscv_vle32_v_f32m1(scores + offset, vl);
    vfloat32m1_t vshifted = __riscv_vfsub_vv_f32m1(vscores, vmax_broadcast, vl);
    __riscv_vse32_v_f32m1(exp_scores.data() + offset, vshifted, vl);
    offset += vl;
  }

  for (int64_t i = 0; i < N; ++i) {
    exp_scores[i] = std::exp(exp_scores[i]);
  }

  float sum_val = 0.0f;
  for (int64_t i = 0; i < N; ++i) {
    sum_val += exp_scores[i];
  }

  // Handle edge cases: sum_val is 0, Inf, or NaN
  if (!std::isfinite(sum_val) || sum_val == 0.0f) {
    // If sum is invalid, set uniform distribution (1/N for each element)
    float uniform_prob = 1.0f / static_cast<float>(N);
    offset = 0;
    while (offset < N) {
      size_t vl = __riscv_vsetvl_e32m1(N - offset);
      vfloat32m1_t vuniform = __riscv_vfmv_v_f_f32m1(uniform_prob, vl);
      __riscv_vse32_v_f32m1(probs + offset, vuniform, vl);
      offset += vl;
    }
    return;
  }

  offset = 0;
  while (offset < N) {
    size_t vl = __riscv_vsetvl_e32m1(N - offset);
    vfloat32m1_t vexp_chunk = __riscv_vle32_v_f32m1(exp_scores.data() + offset, vl);
    vfloat32m1_t vsum_broadcast = __riscv_vfmv_v_f_f32m1(sum_val, vl);
    vfloat32m1_t vprobs_chunk = __riscv_vfdiv_vv_f32m1(vexp_chunk, vsum_broadcast, vl);
    __riscv_vse32_v_f32m1(probs + offset, vprobs_chunk, vl);
    offset += vl;
  }
}

// 3. Probabilities * Value aggregation
// Calculate output = output * scale + sum(probs * values)
// Weighted value aggregation using computed attention probabilities
// Uses Zvfh hardware acceleration when available for FP16 loading
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

  // Set vector length (based on head_dim)
  size_t vl = __riscv_vsetvl_e32m1(head_dim);

  std::vector<float> value_buffer(head_dim);
  // For each head_dim element, calculate output = output * scale + sum(probs * values)
  for (int64_t d = 0; d < head_dim; d += vl) {
    size_t current_vl = __riscv_vsetvl_e32m1(head_dim - d);

    // 1. Calculate sum(probs * values)
    vfloat32m1_t vacuum = __riscv_vfmv_v_f_f32m1(0.0f, current_vl);
    for (int64_t n = 0; n < N; ++n) {
      float prob = probs[n];
      int64_t v_idx = indices[n];
      TORCH_CHECK(v_idx < max_tokens, "value index out of scope!");

      // Convert value from scalar_t to float32 using Zvfh when available
      const scalar_t* v_scalar_ptr = values + v_idx * v_strideN + d;
      load_to_float32_buffer_rvv(v_scalar_ptr, value_buffer.data(), current_vl);

      // Load value vector (now correctly as float32)
      vfloat32m1_t vv = __riscv_vle32_v_f32m1(value_buffer.data(), current_vl);
      // Accumulate: accum += prob * value
      vacuum = __riscv_vfmacc_vf_f32m1(vacuum, prob, vv, current_vl);
    }

    // 2. Load corresponding part of output, multiply by scale
    vfloat32m1_t vout_part = __riscv_vle32_v_f32m1(output + d, current_vl);
    vfloat32m1_t vscale = __riscv_vfmv_v_f_f32m1(scale, current_vl);
    vout_part = __riscv_vfmul_vv_f32m1(vout_part, vscale, current_vl);

    // 3. Add accumulated result: output = output * scale + sum(probs * values)
    vout_part = __riscv_vfadd_vv_f32m1(vout_part, vacuum, current_vl);

    // 4. Store result
    __riscv_vse32_v_f32m1(output + d, vout_part, current_vl);
  }
}

#endif  // CPU_CAPABILITY_RVV

}  // namespace

// NOTE: decode_attention_kernel_rvv is now defined in decode.cpp directly
// inside the anonymous namespace, after all *_impl functions are defined.
// This allows it to call decode_attention_kernel_impl and other helper functions.
