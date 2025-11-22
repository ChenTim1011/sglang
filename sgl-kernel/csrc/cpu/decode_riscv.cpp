// RISC-V Vector Extension (RVV) optimized decode attention kernels
// This file contains RISC-V specific implementations for decode attention
// Note: This file is included directly in decode.cpp, not compiled separately

// For RISC-V with Clang, we need pragma BEFORE any includes
#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "common.h"

#if defined(CPU_CAPABILITY_RVV)
// Clang 19 version check
#if !defined(__clang__)
#error "RVV backend in decode_riscv.cpp requires Clang compiler. Please use Clang 19.1.0 or later to compile this file."
#endif

#if !defined(__clang_major__) || __clang_major__ < 19
#error "RVV backend requires Clang 19 or later. Current version: " __clang_version__ ". Please use Clang 19.1.0."
#endif

#include <riscv_vector.h>

#endif

namespace {

#if defined(CPU_CAPABILITY_RVV)

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

  // Temporary buffer for converting scalar_t to float32
  // This is needed when scalar_t is float16 (2 bytes) but we need float32 (4 bytes) for RVV
  alignas(64) float q_float32[256];  // Max head_dim typically 256
  alignas(64) float k_float32[256];

  for (int64_t m = 0; m < M; ++m) {
    // Convert query from scalar_t to float32
    // This handles both float16 and float32 cases correctly
    for (int64_t k = 0; k < K; ++k) {
      q_float32[k] = static_cast<float>(A[m * lda + k]);
    }

    for (int64_t n = 0; n < N; ++n) {
      int64_t b_idx = indices[n];
      TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");

      // Convert key from scalar_t to float32
      for (int64_t k = 0; k < K; ++k) {
        k_float32[k] = static_cast<float>(B[b_idx * ldb + k]);
      }

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
      float temp[32];  // Max VLEN/32 for RVV (support up to VLEN=1024)
      __riscv_vse32_v_f32m1(temp, vdot, vl);
      float sum = 0.0f;
      for (size_t i = 0; i < vl; i++) {
        sum += temp[i];
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

      // Convert value from scalar_t to float32
      // This handles both float16 and float32 cases correctly
      const scalar_t* v_scalar_ptr = values + v_idx * v_strideN;
      for (int64_t i = 0; i < current_vl; ++i) {
        value_buffer[i] = static_cast<float>(v_scalar_ptr[d + i]);
      }

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
