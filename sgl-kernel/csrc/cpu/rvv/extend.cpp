/*
 * 1. Prefill Kernel: Handling the initial prompt processing (extend phase).
 * 2. KV Cache Management: Computing and storing K/V values into the cache (including quantization/transposition).
 * 3. Attention Computation: Softmax and reduction logic for the extend phase.
 */

#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "common.h"
#include "vec.h"
#include "vector_math.h"

#if defined(CPU_CAPABILITY_RVV)
// Clang 19+ version check
#if !defined(__clang__) || !defined(__clang_major__) || __clang_major__ < 19
#error "RVV backend requires Clang 19 or later. Please use Clang 19 or later to compile this file."
#endif

#include <riscv_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "gemm.h"
#include "vector_helpers.h"

// Check for Zvfh (FP16 vector) support
#if defined(__riscv_zvfh) || defined(__riscv_v)
#define HAS_ZVFH 1
#else
#define HAS_ZVFH 0
#endif

namespace {

// =============================================================================
// Helper: Safe Math
// =============================================================================
// Helper for safe exp on RISC-V to avoid denormal traps
inline float safe_exp(float x) {
  // Avoid denormal traps on some RISC-V implementations
  // e^-88 is approx 5.9e-39, close to FLT_MIN (1.17e-38)
  if (x < -88.0f) return 0.0f;
  return std::exp(x);
}

// =============================================================================
// Helper: KV Cache Management
// =============================================================================
template <typename scalar_t, typename index_t>
void extend_set_kv_buffer_int8_quantize(
    int8_t* __restrict__ k_buffer,
    int8_t* __restrict__ v_buffer,
    const scalar_t* __restrict__ k_extend,
    const scalar_t* __restrict__ v_extend,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    const index_t* __restrict__ extend_seq_lens,
    const index_t* __restrict__ extend_start_loc,
    int num_seqs,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    int ke_strideN,
    int ke_strideH,
    int ve_strideN,
    int ve_strideH,
    int max_context_len,
    float k_scale,
    float v_scale) {
  at::parallel_for(0, num_seqs * num_heads_kv, 0, [&](int64_t begin, int64_t end) {
    int64_t seq_idx{0}, head_kv_id{0};
    int64_t num_seqs_i64 = static_cast<int64_t>(num_seqs);
    int64_t num_heads_kv_i64 = static_cast<int64_t>(num_heads_kv);
    data_index_init(begin, seq_idx, num_seqs_i64, head_kv_id, num_heads_kv_i64);

    for (int64_t i = begin; i < end; i++) {
      int64_t req_idx = req_pool_indices[seq_idx];
      int64_t extend_len = static_cast<int64_t>(extend_seq_lens[seq_idx]);
      int64_t start_loc = static_cast<int64_t>(extend_start_loc[seq_idx]);
      int64_t seq_len = seq_lens[seq_idx];

      // For each token in the extend part, append to buffer
      for (int64_t t = 0; t < extend_len; ++t) {
        // Get the token index in buffer (prefix_len + t)
        int64_t prefix_len = seq_len - extend_len;
        int64_t token_idx_in_seq = prefix_len + t;
        int64_t token_idx = req_to_token[req_idx * max_context_len + token_idx_in_seq];

        // Get pointers
        int8_t* k_buffer_ptr = k_buffer + token_idx * k_strideN + head_kv_id * k_strideH;
        int8_t* v_buffer_ptr = v_buffer + token_idx * v_strideN + head_kv_id * v_strideH;
        const scalar_t* k_extend_ptr = k_extend + t * ke_strideN + head_kv_id * ke_strideH;
        const scalar_t* v_extend_ptr = v_extend + t * ve_strideN + head_kv_id * ve_strideH;

        // On-the-fly quantization for float input
        // Always use SYMMETRIC quantization for KV cache (signed int8)
        quantize_and_copy_rvv<scalar_t>(
            k_buffer_ptr,  // int8_t* directly
            k_extend_ptr,
            head_size,
            k_scale);

        quantize_and_copy_rvv<scalar_t>(
            v_buffer_ptr,  // int8_t* directly
            v_extend_ptr,
            head_size_v,
            v_scale);
      }

      data_index_step(seq_idx, num_seqs_i64, head_kv_id, num_heads_kv_i64);
    }
  });
}

template <typename scalar_t>
inline void transpose_block(const scalar_t* src, scalar_t* dst, int rows, int cols, int src_stride) {
  transpose_block_rvv<scalar_t>(src, dst, rows, cols, src_stride);
}

// =============================================================================
// Kernel: Matrix Multiplication
// =============================================================================
template <typename scalar_t>
void gemm_nt_rvv_tiny_m1(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K_trans,
    float* __restrict__ C,
    int N,
    int head_size,
    int q_strideM,
    int block_n,
    int ldc,
    float scale) {
  size_t vl;
  for (int n = 0; n < N; n += vl) {
    vl = __riscv_vsetvl_e16m2(N - n);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    for (int k = 0; k < head_size; ++k) {
      const scalar_t* k_ptr = K_trans + k * block_n + n;
      float q_val = static_cast<float>(Q[k]);

      if constexpr (std::is_same_v<scalar_t, at::Half>) {
        vfloat16m2_t v_k = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr), vl);
        acc = vfwmacc_f16_scalar_to_f32m4(acc, (_Float16)q_val, v_k, vl);
      } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        vfloat32m4_t v_k = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr), vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, q_val, v_k, vl);
      } else {
        float k_tmp[64];
        for (size_t i = 0; i < vl; ++i)
          k_tmp[i] = static_cast<float>(k_ptr[i]);
        vfloat32m4_t v_k = __riscv_vle32_v_f32m4(k_tmp, vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, q_val, v_k, vl);
      }
    }
    __riscv_vse32_v_f32m4(C + n, __riscv_vfmul_vf_f32m4(acc, scale, vl), vl);
  }
}

template <typename scalar_t>
void gemm_nt_rvv_tiled_transposed(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K_trans,
    float* __restrict__ C,
    int M,
    int N,
    int head_size,
    int q_strideM,
    int block_n,
    int ldc,
    float scale) {
  if (M == 1) {
    gemm_nt_rvv_tiny_m1<scalar_t>(Q, K_trans, C, N, head_size, q_strideM, block_n, ldc, scale);
    return;
  }

  size_t vl;
  for (int n = 0; n < N; n += vl) {
    vl = __riscv_vsetvl_e16m2(N - n);

    int m = 0;
    for (; m < M - 3; m += 4) {
      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      int k = 0;
      for (; k < head_size - 1; k += 2) {
        const scalar_t* k_ptr0 = K_trans + (k + 0) * block_n + n;
        const scalar_t* k_ptr1 = K_trans + (k + 1) * block_n + n;

        if (k + 2 < head_size) {
          __builtin_prefetch(K_trans + (k + 2) * block_n + n, 0, 3);
        }

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_k0 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr0), vl);
          vfloat16m2_t v_k1 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr1), vl);

          float q0_0 = static_cast<float>(Q[(m + 0) * q_strideM + k + 0]);
          float q0_1 = static_cast<float>(Q[(m + 0) * q_strideM + k + 1]);
          float q1_0 = static_cast<float>(Q[(m + 1) * q_strideM + k + 0]);
          float q1_1 = static_cast<float>(Q[(m + 1) * q_strideM + k + 1]);
          float q2_0 = static_cast<float>(Q[(m + 2) * q_strideM + k + 0]);
          float q2_1 = static_cast<float>(Q[(m + 2) * q_strideM + k + 1]);
          float q3_0 = static_cast<float>(Q[(m + 3) * q_strideM + k + 0]);
          float q3_1 = static_cast<float>(Q[(m + 3) * q_strideM + k + 1]);

          acc0 = vfwmacc_f16_scalar_to_f32m4(acc0, (_Float16)(q0_0), v_k0, vl);
          acc0 = vfwmacc_f16_scalar_to_f32m4(acc0, (_Float16)(q0_1), v_k1, vl);
          acc1 = vfwmacc_f16_scalar_to_f32m4(acc1, (_Float16)(q1_0), v_k0, vl);
          acc1 = vfwmacc_f16_scalar_to_f32m4(acc1, (_Float16)(q1_1), v_k1, vl);
          acc2 = vfwmacc_f16_scalar_to_f32m4(acc2, (_Float16)(q2_0), v_k0, vl);
          acc2 = vfwmacc_f16_scalar_to_f32m4(acc2, (_Float16)(q2_1), v_k1, vl);
          acc3 = vfwmacc_f16_scalar_to_f32m4(acc3, (_Float16)(q3_0), v_k0, vl);
          acc3 = vfwmacc_f16_scalar_to_f32m4(acc3, (_Float16)(q3_1), v_k1, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          vfloat32m4_t v_k0 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr0), vl);
          vfloat32m4_t v_k1 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr1), vl);

          float q0_0 = static_cast<float>(Q[(m + 0) * q_strideM + k + 0]);
          float q0_1 = static_cast<float>(Q[(m + 0) * q_strideM + k + 1]);
          float q1_0 = static_cast<float>(Q[(m + 1) * q_strideM + k + 0]);
          float q1_1 = static_cast<float>(Q[(m + 1) * q_strideM + k + 1]);
          float q2_0 = static_cast<float>(Q[(m + 2) * q_strideM + k + 0]);
          float q2_1 = static_cast<float>(Q[(m + 2) * q_strideM + k + 1]);
          float q3_0 = static_cast<float>(Q[(m + 3) * q_strideM + k + 0]);
          float q3_1 = static_cast<float>(Q[(m + 3) * q_strideM + k + 1]);

          acc0 = __riscv_vfmacc_vf_f32m4(acc0, q0_0, v_k0, vl);
          acc0 = __riscv_vfmacc_vf_f32m4(acc0, q0_1, v_k1, vl);
          acc1 = __riscv_vfmacc_vf_f32m4(acc1, q1_0, v_k0, vl);
          acc1 = __riscv_vfmacc_vf_f32m4(acc1, q1_1, v_k1, vl);
          acc2 = __riscv_vfmacc_vf_f32m4(acc2, q2_0, v_k0, vl);
          acc2 = __riscv_vfmacc_vf_f32m4(acc2, q2_1, v_k1, vl);
          acc3 = __riscv_vfmacc_vf_f32m4(acc3, q3_0, v_k0, vl);
          acc3 = __riscv_vfmacc_vf_f32m4(acc3, q3_1, v_k1, vl);
        } else {
          float k_tmp0[64], k_tmp1[64];
          for (size_t i = 0; i < vl; ++i) {
            k_tmp0[i] = static_cast<float>(k_ptr0[i]);
            k_tmp1[i] = static_cast<float>(k_ptr1[i]);
          }
          vfloat32m4_t v_k0 = __riscv_vle32_v_f32m4(k_tmp0, vl);
          vfloat32m4_t v_k1 = __riscv_vle32_v_f32m4(k_tmp1, vl);

          float q0_0 = static_cast<float>(Q[(m + 0) * q_strideM + k + 0]);
          float q0_1 = static_cast<float>(Q[(m + 0) * q_strideM + k + 1]);
          float q1_0 = static_cast<float>(Q[(m + 1) * q_strideM + k + 0]);
          float q1_1 = static_cast<float>(Q[(m + 1) * q_strideM + k + 1]);
          float q2_0 = static_cast<float>(Q[(m + 2) * q_strideM + k + 0]);
          float q2_1 = static_cast<float>(Q[(m + 2) * q_strideM + k + 1]);
          float q3_0 = static_cast<float>(Q[(m + 3) * q_strideM + k + 0]);
          float q3_1 = static_cast<float>(Q[(m + 3) * q_strideM + k + 1]);

          acc0 = __riscv_vfmacc_vf_f32m4(acc0, q0_0, v_k0, vl);
          acc0 = __riscv_vfmacc_vf_f32m4(acc0, q0_1, v_k1, vl);
          acc1 = __riscv_vfmacc_vf_f32m4(acc1, q1_0, v_k0, vl);
          acc1 = __riscv_vfmacc_vf_f32m4(acc1, q1_1, v_k1, vl);
          acc2 = __riscv_vfmacc_vf_f32m4(acc2, q2_0, v_k0, vl);
          acc2 = __riscv_vfmacc_vf_f32m4(acc2, q2_1, v_k1, vl);
          acc3 = __riscv_vfmacc_vf_f32m4(acc3, q3_0, v_k0, vl);
          acc3 = __riscv_vfmacc_vf_f32m4(acc3, q3_1, v_k1, vl);
        }
      }
      for (; k < head_size; ++k) {
        const scalar_t* k_ptr = K_trans + k * block_n + n;

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_k = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr), vl);
          float q0 = static_cast<float>(Q[(m + 0) * q_strideM + k]);
          float q1 = static_cast<float>(Q[(m + 1) * q_strideM + k]);
          float q2 = static_cast<float>(Q[(m + 2) * q_strideM + k]);
          float q3 = static_cast<float>(Q[(m + 3) * q_strideM + k]);
          acc0 = vfwmacc_f16_scalar_to_f32m4(acc0, (_Float16)(q0), v_k, vl);
          acc1 = vfwmacc_f16_scalar_to_f32m4(acc1, (_Float16)(q1), v_k, vl);
          acc2 = vfwmacc_f16_scalar_to_f32m4(acc2, (_Float16)(q2), v_k, vl);
          acc3 = vfwmacc_f16_scalar_to_f32m4(acc3, (_Float16)(q3), v_k, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          vfloat32m4_t v_k = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr), vl);
          float q0 = static_cast<float>(Q[(m + 0) * q_strideM + k]);
          float q1 = static_cast<float>(Q[(m + 1) * q_strideM + k]);
          float q2 = static_cast<float>(Q[(m + 2) * q_strideM + k]);
          float q3 = static_cast<float>(Q[(m + 3) * q_strideM + k]);
          acc0 = __riscv_vfmacc_vf_f32m4(acc0, q0, v_k, vl);
          acc1 = __riscv_vfmacc_vf_f32m4(acc1, q1, v_k, vl);
          acc2 = __riscv_vfmacc_vf_f32m4(acc2, q2, v_k, vl);
          acc3 = __riscv_vfmacc_vf_f32m4(acc3, q3, v_k, vl);
        } else {
          float k_tmp[64];
          for (size_t i = 0; i < vl; ++i)
            k_tmp[i] = static_cast<float>(k_ptr[i]);
          vfloat32m4_t v_k = __riscv_vle32_v_f32m4(k_tmp, vl);
          float q0 = static_cast<float>(Q[(m + 0) * q_strideM + k]);
          float q1 = static_cast<float>(Q[(m + 1) * q_strideM + k]);
          float q2 = static_cast<float>(Q[(m + 2) * q_strideM + k]);
          float q3 = static_cast<float>(Q[(m + 3) * q_strideM + k]);
          acc0 = __riscv_vfmacc_vf_f32m4(acc0, q0, v_k, vl);
          acc1 = __riscv_vfmacc_vf_f32m4(acc1, q1, v_k, vl);
          acc2 = __riscv_vfmacc_vf_f32m4(acc2, q2, v_k, vl);
          acc3 = __riscv_vfmacc_vf_f32m4(acc3, q3, v_k, vl);
        }
      }

      __riscv_vse32_v_f32m4(C + (m + 0) * ldc + n, __riscv_vfmul_vf_f32m4(acc0, scale, vl), vl);
      __riscv_vse32_v_f32m4(C + (m + 1) * ldc + n, __riscv_vfmul_vf_f32m4(acc1, scale, vl), vl);
      __riscv_vse32_v_f32m4(C + (m + 2) * ldc + n, __riscv_vfmul_vf_f32m4(acc2, scale, vl), vl);
      __riscv_vse32_v_f32m4(C + (m + 3) * ldc + n, __riscv_vfmul_vf_f32m4(acc3, scale, vl), vl);
    }

    for (; m < M; ++m) {
      vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      for (int k = 0; k < head_size; ++k) {
        const scalar_t* k_ptr = K_trans + k * block_n + n;
        float q_val = static_cast<float>(Q[m * q_strideM + k]);

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_k = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr), vl);
          acc = vfwmacc_f16_scalar_to_f32m4(acc, (_Float16)q_val, v_k, vl);
        } else {
          float k_tmp[64];
          for (size_t i = 0; i < vl; ++i)
            k_tmp[i] = static_cast<float>(k_ptr[i]);
          vfloat32m4_t v_k = __riscv_vle32_v_f32m4(k_tmp, vl);
          acc = __riscv_vfmacc_vf_f32m4(acc, q_val, v_k, vl);
        }
      }
      __riscv_vse32_v_f32m4(C + m * ldc + n, __riscv_vfmul_vf_f32m4(acc, scale, vl), vl);
    }
  }
}

template <typename scalar_t>
void gemm_nn_rvv_tiled(
    const float* __restrict__ P,
    const scalar_t* __restrict__ V,
    float* __restrict__ O,
    int M,
    int N,
    int head_size_v,
    int p_strideN,
    int v_strideH) {
  for (int m_base = 0; m_base < M; m_base += 4) {
    int m_count = std::min(4, M - m_base);
    size_t vl;
    for (int d = 0; d < head_size_v; d += vl) {
      vl = __riscv_vsetvl_e16m2(head_size_v - d);

      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      for (int n = 0; n < N; ++n) {
        const scalar_t* v_ptr = V + n * v_strideH + d;

        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        if (m_count > 0) p0 = P[(m_base + 0) * p_strideN + n];
        if (m_count > 1) p1 = P[(m_base + 1) * p_strideN + n];
        if (m_count > 2) p2 = P[(m_base + 2) * p_strideN + n];
        if (m_count > 3) p3 = P[(m_base + 3) * p_strideN + n];

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_v = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(v_ptr), vl);

          acc0 = vfwmacc_f16_scalar_to_f32m4(acc0, (_Float16)p0, v_v, vl);
          if (m_count > 1) acc1 = vfwmacc_f16_scalar_to_f32m4(acc1, (_Float16)p1, v_v, vl);
          if (m_count > 2) acc2 = vfwmacc_f16_scalar_to_f32m4(acc2, (_Float16)p2, v_v, vl);
          if (m_count > 3) acc3 = vfwmacc_f16_scalar_to_f32m4(acc3, (_Float16)p3, v_v, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          vfloat32m4_t v_v = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(v_ptr), vl);

          acc0 = __riscv_vfmacc_vf_f32m4(acc0, p0, v_v, vl);
          if (m_count > 1) acc1 = __riscv_vfmacc_vf_f32m4(acc1, p1, v_v, vl);
          if (m_count > 2) acc2 = __riscv_vfmacc_vf_f32m4(acc2, p2, v_v, vl);
          if (m_count > 3) acc3 = __riscv_vfmacc_vf_f32m4(acc3, p3, v_v, vl);
        } else {
          float v_tmp[64];
          for (size_t i = 0; i < vl; ++i)
            v_tmp[i] = static_cast<float>(v_ptr[i]);
          vfloat32m4_t v_v = __riscv_vle32_v_f32m4(v_tmp, vl);

          acc0 = __riscv_vfmacc_vf_f32m4(acc0, p0, v_v, vl);
          if (m_count > 1) acc1 = __riscv_vfmacc_vf_f32m4(acc1, p1, v_v, vl);
          if (m_count > 2) acc2 = __riscv_vfmacc_vf_f32m4(acc2, p2, v_v, vl);
          if (m_count > 3) acc3 = __riscv_vfmacc_vf_f32m4(acc3, p3, v_v, vl);
        }
      }

      auto store_o = [&](int idx, vfloat32m4_t acc) {
        if (idx < m_count) {
          float* o_ptr = O + (m_base + idx) * head_size_v + d;
          vfloat32m4_t old_o = __riscv_vle32_v_f32m4(o_ptr, vl);
          acc = __riscv_vfadd_vv_f32m4(old_o, acc, vl);
          __riscv_vse32_v_f32m4(o_ptr, acc, vl);
        }
      };

      store_o(0, acc0);
      store_o(1, acc1);
      store_o(2, acc2);
      store_o(3, acc3);
    }
  }
}

template <typename scalar_t>
void gemm_nt_rvv_tiled_transposed_int8(
    const scalar_t* __restrict__ Q,
    const int8_t* __restrict__ K_trans,
    float* __restrict__ C,
    int M,
    int N,
    int head_size,
    int q_strideM,
    int block_n,  // Dynamic block size for K_trans layout
    int ldc,
    float scale,
    float k_scale) {
  size_t vl;
  constexpr int LOCAL_MAX_HEAD = 256;
  int8_t q_int8[4][LOCAL_MAX_HEAD];
  float q_scales[4];

  for (int n = 0; n < N; n += vl) {
    vl = __riscv_vsetvl_e16m2(N - n);

    int m = 0;
    for (; m < M - 3; m += 4) {
      // W8A8 Optimization: Quantize Q tile on-the-fly
      for (int i = 0; i < 4; ++i) {
        float max_abs = 0.0f;
        int m_idx = m + i;
        for (int k = 0; k < head_size; ++k)
          max_abs = std::max(max_abs, std::abs((float)Q[m_idx * q_strideM + k]));
        q_scales[i] = max_abs / 127.0f;
        float inv_scale = (q_scales[i] > 1e-9f) ? 1.0f / q_scales[i] : 0.0f;
        for (int k = 0; k < head_size; ++k) {
          float val = (float)Q[m_idx * q_strideM + k] * inv_scale;
          q_int8[i][k] = (int8_t)std::clamp(std::round(val), -128.0f, 127.0f);
        }
      }

      // Use Integer Accumulators
      vint32m4_t acc0 = __riscv_vmv_v_x_i32m4(0, vl);
      vint32m4_t acc1 = __riscv_vmv_v_x_i32m4(0, vl);
      vint32m4_t acc2 = __riscv_vmv_v_x_i32m4(0, vl);
      vint32m4_t acc3 = __riscv_vmv_v_x_i32m4(0, vl);

      // K reduction loop with unroll factor 2
      int k = 0;
      for (; k < head_size - 1; k += 2) {
        const int8_t* k_ptr0 = K_trans + (k + 0) * block_n + n;
        const int8_t* k_ptr1 = K_trans + (k + 1) * block_n + n;
        if (k + 2 < head_size) {
          __builtin_prefetch(K_trans + (k + 2) * block_n + n, 0, 3);
        }

        vint8m1_t v_k0_8 = __riscv_vle8_v_i8m1(k_ptr0, vl);
        vint8m1_t v_k1_8 = __riscv_vle8_v_i8m1(k_ptr1, vl);
        vint16m2_t v_k0_16 = __riscv_vsext_vf2_i16m2(v_k0_8, vl);
        vint16m2_t v_k1_16 = __riscv_vsext_vf2_i16m2(v_k1_8, vl);

        int16_t q0_0 = (int16_t)q_int8[0][k];
        int16_t q0_1 = (int16_t)q_int8[0][k + 1];
        acc0 = __riscv_vwmacc_vx_i32m4(acc0, q0_0, v_k0_16, vl);
        acc0 = __riscv_vwmacc_vx_i32m4(acc0, q0_1, v_k1_16, vl);

        int16_t q1_0 = (int16_t)q_int8[1][k];
        int16_t q1_1 = (int16_t)q_int8[1][k + 1];
        acc1 = __riscv_vwmacc_vx_i32m4(acc1, q1_0, v_k0_16, vl);
        acc1 = __riscv_vwmacc_vx_i32m4(acc1, q1_1, v_k1_16, vl);

        int16_t q2_0 = (int16_t)q_int8[2][k];
        int16_t q2_1 = (int16_t)q_int8[2][k + 1];
        acc2 = __riscv_vwmacc_vx_i32m4(acc2, q2_0, v_k0_16, vl);
        acc2 = __riscv_vwmacc_vx_i32m4(acc2, q2_1, v_k1_16, vl);

        int16_t q3_0 = (int16_t)q_int8[3][k];
        int16_t q3_1 = (int16_t)q_int8[3][k + 1];
        acc3 = __riscv_vwmacc_vx_i32m4(acc3, q3_0, v_k0_16, vl);
        acc3 = __riscv_vwmacc_vx_i32m4(acc3, q3_1, v_k1_16, vl);
      }

      // Handle remaining K iterations
      for (; k < head_size; ++k) {
        const int8_t* k_ptr = K_trans + k * block_n + n;
        vint8m1_t v_k_8 = __riscv_vle8_v_i8m1(k_ptr, vl);
        vint16m2_t v_k_16 = __riscv_vsext_vf2_i16m2(v_k_8, vl);

        acc0 = __riscv_vwmacc_vx_i32m4(acc0, (int16_t)q_int8[0][k], v_k_16, vl);
        acc1 = __riscv_vwmacc_vx_i32m4(acc1, (int16_t)q_int8[1][k], v_k_16, vl);
        acc2 = __riscv_vwmacc_vx_i32m4(acc2, (int16_t)q_int8[2][k], v_k_16, vl);
        acc3 = __riscv_vwmacc_vx_i32m4(acc3, (int16_t)q_int8[3][k], v_k_16, vl);
      }

      // Dequantize
      auto dequant = [&](vint32m4_t vac, float qs) {
        vfloat32m4_t vf = __riscv_vfcvt_f_x_v_f32m4(vac, vl);
        float combined = qs * k_scale * scale;
        return __riscv_vfmul_vf_f32m4(vf, combined, vl);
      };

      __riscv_vse32_v_f32m4(C + (m + 0) * ldc + n, dequant(acc0, q_scales[0]), vl);
      __riscv_vse32_v_f32m4(C + (m + 1) * ldc + n, dequant(acc1, q_scales[1]), vl);
      __riscv_vse32_v_f32m4(C + (m + 2) * ldc + n, dequant(acc2, q_scales[2]), vl);
      __riscv_vse32_v_f32m4(C + (m + 3) * ldc + n, dequant(acc3, q_scales[3]), vl);
    }

    for (; m < M; ++m) {
      // Single row quantization
      float max_abs = 0.0f;
      for (int k = 0; k < head_size; ++k)
        max_abs = std::max(max_abs, std::abs((float)Q[m * q_strideM + k]));
      float q_s = max_abs / 127.0f;
      float inv_s = (q_s > 1e-9f) ? 1.0f / q_s : 0.0f;

      vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);

      for (int k = 0; k < head_size; ++k) {
        int8_t q_val = (int8_t)std::clamp(std::round((float)Q[m * q_strideM + k] * inv_s), -128.0f, 127.0f);
        const int8_t* k_ptr = K_trans + k * block_n + n;
        vint8m1_t v_k_8 = __riscv_vle8_v_i8m1(k_ptr, vl);
        vint16m2_t v_k_16 = __riscv_vsext_vf2_i16m2(v_k_8, vl);
        acc = __riscv_vwmacc_vx_i32m4(acc, (int16_t)q_val, v_k_16, vl);
      }

      vfloat32m4_t vf = __riscv_vfcvt_f_x_v_f32m4(acc, vl);
      __riscv_vse32_v_f32m4(C + m * ldc + n, __riscv_vfmul_vf_f32m4(vf, q_s * k_scale * scale, vl), vl);
    }
  }
}

void gemm_nn_rvv_tiled_int8(
    const float* __restrict__ P,
    const int8_t* __restrict__ V,
    float* __restrict__ O,
    int M,
    int N,
    int head_size_v,
    int p_strideN,
    int v_strideH,
    float v_scale) {
  for (int m_base = 0; m_base < M; m_base += 4) {
    int m_count = std::min(4, M - m_base);
    size_t vl;
    for (int d = 0; d < head_size_v; d += vl) {
      vl = __riscv_vsetvl_e16m2(head_size_v - d);

      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      for (int n = 0; n < N; ++n) {
        const int8_t* v_ptr = V + n * v_strideH + d;

        vfloat32m4_t v_v_f32 = int8_to_f32m4(v_ptr, vl);

        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        if (m_count > 0) p0 = P[(m_base + 0) * p_strideN + n] * v_scale;
        if (m_count > 1) p1 = P[(m_base + 1) * p_strideN + n] * v_scale;
        if (m_count > 2) p2 = P[(m_base + 2) * p_strideN + n] * v_scale;
        if (m_count > 3) p3 = P[(m_base + 3) * p_strideN + n] * v_scale;

        acc0 = __riscv_vfmacc_vf_f32m4(acc0, p0, v_v_f32, vl);
        if (m_count > 1) acc1 = __riscv_vfmacc_vf_f32m4(acc1, p1, v_v_f32, vl);
        if (m_count > 2) acc2 = __riscv_vfmacc_vf_f32m4(acc2, p2, v_v_f32, vl);
        if (m_count > 3) acc3 = __riscv_vfmacc_vf_f32m4(acc3, p3, v_v_f32, vl);
      }

      auto store_o = [&](int idx, vfloat32m4_t acc) {
        if (idx < m_count) {
          float* o_ptr = O + (m_base + idx) * head_size_v + d;
          vfloat32m4_t old_o = __riscv_vle32_v_f32m4(o_ptr, vl);
          acc = __riscv_vfadd_vv_f32m4(old_o, acc, vl);
          __riscv_vse32_v_f32m4(o_ptr, acc, vl);
        }
      };

      store_o(0, acc0);
      store_o(1, acc1);
      store_o(2, acc2);
      store_o(3, acc3);
    }
  }
}

template <typename scalar_t, typename index_t, bool IS_INT8>
void extend_attention_kernel_rvv_impl_template(
    scalar_t* __restrict__ o_extend,
    const scalar_t* __restrict__ q_extend,
    const scalar_t* __restrict__ k_extend,
    const scalar_t* __restrict__ v_extend,
    const void* __restrict__ k_buffer,  // void* to support scalar_t* or int8_t*
    const void* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    const index_t* __restrict__ extend_seq_lens,
    const index_t* __restrict__ extend_start_loc,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int o_strideM,
    int o_strideH,
    int q_strideM,
    int q_strideH,
    int ke_strideN,
    int ke_strideH,
    int ve_strideN,
    int ve_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float scaling,
    float logit_cap,
    int max_num_reqs,
    int max_context_len,
    int max_total_num_tokens,
    int max_len_extend,
    bool is_prefix_skipped,
    float k_scale = 1.0f,
    float v_scale = 1.0f) {
  if (head_size > MAX_HEAD_SIZE || head_size_v > MAX_HEAD_SIZE) {
    return;
  }

  // Cast void* buffers to correct types based on IS_INT8
  using buffer_t = std::conditional_t<IS_INT8, int8_t, scalar_t>;
  const buffer_t* k_buffer_typed = static_cast<const buffer_t*>(k_buffer);
  const buffer_t* v_buffer_typed = static_cast<const buffer_t*>(v_buffer);

  // Dynamic block size selection
  const int BLOCK_M = get_optimal_block_m<scalar_t>(head_size, head_size_v);
  const int BLOCK_N = get_optimal_block_n<scalar_t>(head_size, head_size_v);

#pragma omp parallel
  {
    // Use max possible size for dynamic allocation
    // Use std::vector to avoid stack overflow
    std::vector<float> scores_block_vec(32 * 64);
    float* scores_block = scores_block_vec.data();

    std::vector<buffer_t> k_trans_buf_vec(MAX_HEAD_SIZE * 64);
    buffer_t* k_trans_buf = k_trans_buf_vec.data();

    std::vector<buffer_t> v_buf_vec(64 * MAX_HEAD_SIZE);
    buffer_t* v_buf = v_buf_vec.data();

    std::vector<scalar_t> k_trans_buf_fp_vec(MAX_HEAD_SIZE * 64);
    scalar_t* k_trans_buf_fp = k_trans_buf_fp_vec.data();

    std::vector<scalar_t> v_buf_fp_vec(64 * MAX_HEAD_SIZE);
    scalar_t* v_buf_fp = v_buf_fp_vec.data();

    std::vector<float> o_acc_vec(32 * MAX_HEAD_SIZE);
    float* o_acc = o_acc_vec.data();

    alignas(64) float l_acc[32];
    alignas(64) float m_acc[32];

#pragma omp for collapse(2)
    for (int b = 0; b < batches; ++b) {
      for (int h = 0; h < num_heads; ++h) {
        int head_kv = h / (num_heads / num_heads_kv);
        int seq_len = seq_lens[b];
        int extend_len = extend_seq_lens[b];
        int prefix_len = seq_len - extend_len;
        int start_loc = extend_start_loc[b];
        int req_idx = req_pool_indices[b];

        // using the batch's slice offset (start_loc).
        const scalar_t* q_ptr = q_extend + h * q_strideH + start_loc * q_strideM;
        scalar_t* o_ptr = o_extend + h * o_strideH + start_loc * o_strideM;

        for (int m_start = 0; m_start < extend_len; m_start += BLOCK_M) {
          int m_size = std::min(BLOCK_M, extend_len - m_start);

          std::fill(l_acc, l_acc + m_size, 0.0f);
          std::fill(m_acc, m_acc + m_size, -std::numeric_limits<float>::infinity());
          std::memset(o_acc, 0, m_size * head_size_v * sizeof(float));
          std::memset(scores_block, 0, BLOCK_M * BLOCK_N * sizeof(float));

          // ============================================
          // PREFIX PART (KV Cache)
          // ============================================
          if (prefix_len > 0) {
            bool is_contiguous = false;
            int start_token_idx = -1;
            if (prefix_len > 0) {
              start_token_idx = req_to_token[req_idx * max_context_len + 0];
              is_contiguous = true;
              size_t vl_idx_max = __riscv_vsetvlmax_e64m4();
              for (int ck = 0; ck < prefix_len && ck < BLOCK_N; ck += vl_idx_max) {
                size_t vl = __riscv_vsetvl_e64m4(std::min((int)vl_idx_max, prefix_len - ck));
                vint64m4_t v_idx = __riscv_vle64_v_i64m4(
                    reinterpret_cast<const int64_t*>(req_to_token + req_idx * max_context_len + ck), vl);
                vint64m4_t v_seq = __riscv_vid_v_i64m4(vl);
                v_seq = __riscv_vadd_vx_i64m4(v_seq, start_token_idx + ck, vl);
                vbool16_t v_mask = __riscv_vmseq_vv_i64m4_b16(v_idx, v_seq, vl);
                long matches = __riscv_vcpop_m_b16(v_mask, vl);
                if (matches != vl) {
                  is_contiguous = false;
                  break;
                }
              }
            }

            for (int n_start = 0; n_start < prefix_len; n_start += BLOCK_N) {
              int n_size = std::min(BLOCK_N, prefix_len - n_start);

              // Helper to copy/transpose K buffer (works for both buffer types conceptually if strides match)
              if constexpr (IS_INT8) {
                // INT8 Specific Copy Logic
                if (is_contiguous && n_start == 0) {
                  int base_token_idx = start_token_idx + n_start;
                  for (int j = 0; j < n_size; ++j) {
                    int token_idx = base_token_idx + j;
                    const int8_t* k_src = k_buffer_typed + token_idx * k_strideN + head_kv * k_strideH;
                    for (int d = 0; d < head_size; ++d) {
                      k_trans_buf[d * BLOCK_N + j] = k_src[d];
                    }
                  }
                } else {
                  for (int j = 0; j < n_size; ++j) {
                    int token_idx = req_to_token[req_idx * max_context_len + n_start + j];
                    const int8_t* k_src = k_buffer_typed + token_idx * k_strideN + head_kv * k_strideH;
                    for (int d = 0; d < head_size; ++d) {
                      k_trans_buf[d * BLOCK_N + j] = k_src[d];
                    }
                  }
                }
              } else {
                // FP16/BF16/FP32 Logic
                if (is_contiguous && n_start == 0) {
                  int base_token_idx = start_token_idx + n_start;
                  for (int j = 0; j < n_size; ++j) {
                    int token_idx = base_token_idx + j;
                    const scalar_t* src_tok = k_buffer_typed + token_idx * k_strideN + head_kv * k_strideH;
                    // Transpose: dst[d * BLOCK_N + j] = src[d]
                    // NOTE: Original FP code had vectorization for Half, let's preserve it
                    if constexpr (std::is_same_v<scalar_t, at::Half>) {
                      size_t vl_max = __riscv_vsetvlmax_e16m2();
                      for (int d = 0; d < head_size; d += vl_max) {
                        size_t vl = __riscv_vsetvl_e16m2(head_size - d);
                        vfloat16m2_t v_k = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(src_tok + d), vl);
                        ptrdiff_t stride = BLOCK_N * sizeof(_Float16);
                        __riscv_vsse16_v_f16m2(
                            reinterpret_cast<_Float16*>(k_trans_buf + d * BLOCK_N + j), stride, v_k, vl);
                      }
                    } else {
                      for (int d = 0; d < head_size; ++d) {
                        k_trans_buf[d * BLOCK_N + j] = src_tok[d];
                      }
                    }
                  }
                } else {
                  for (int j = 0; j < n_size; ++j) {
                    int token_idx = req_to_token[req_idx * max_context_len + n_start + j];
                    const scalar_t* src_tok = k_buffer_typed + token_idx * k_strideN + head_kv * k_strideH;
                    if constexpr (std::is_same_v<scalar_t, at::Half>) {
                      size_t vl_max = __riscv_vsetvlmax_e16m2();
                      for (int d = 0; d < head_size; d += vl_max) {
                        size_t vl = __riscv_vsetvl_e16m2(head_size - d);
                        vfloat16m2_t v_k = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(src_tok + d), vl);
                        ptrdiff_t stride = BLOCK_N * sizeof(_Float16);
                        __riscv_vsse16_v_f16m2(
                            reinterpret_cast<_Float16*>(k_trans_buf + d * BLOCK_N + j), stride, v_k, vl);
                      }
                    } else {
                      for (int d = 0; d < head_size; ++d) {
                        k_trans_buf[d * BLOCK_N + j] = src_tok[d];
                      }
                    }
                  }
                }
              }

              // GEMM Call - Branch on INT8
              if constexpr (IS_INT8) {
                gemm_nt_rvv_tiled_transposed_int8(
                    q_ptr + m_start * q_strideM,
                    k_trans_buf,
                    scores_block,
                    m_size,
                    n_size,
                    head_size,
                    q_strideM,
                    BLOCK_N,
                    BLOCK_N,
                    scaling,
                    k_scale);
              } else {
                gemm_nt_rvv_tiled_transposed(
                    q_ptr + m_start * q_strideM,
                    k_trans_buf,
                    scores_block,
                    m_size,
                    n_size,
                    head_size,
                    q_strideM,
                    BLOCK_N,
                    BLOCK_N,
                    scaling);
              }

              // Softma Logic (Common)
              // NOTE: Padding logic is identical
              constexpr int VLEN_BYTES = 32;
              constexpr int VLEN_FLOATS = VLEN_BYTES / sizeof(float);
              int padded_n_size = ((n_size + VLEN_FLOATS - 1) / VLEN_FLOATS) * VLEN_FLOATS;
              padded_n_size = std::min(padded_n_size, BLOCK_N);

              if (padded_n_size > n_size) {
                size_t vl_max = __riscv_vsetvlmax_e32m4();
                for (int i = 0; i < m_size; ++i) {
                  float* row = scores_block + i * BLOCK_N;
                  for (int j = n_size; j < padded_n_size; j += vl_max) {
                    size_t vl = __riscv_vsetvl_e32m4(padded_n_size - j);
                    __riscv_vse32_v_f32m4(row + j, __riscv_vfmv_v_f_f32m4(0.0f, vl), vl);
                  }
                }
              }

              // V Buffer Copy
              if constexpr (IS_INT8) {
                // Direct copy for INT8
                for (int j = 0; j < n_size; ++j) {
                  int token_idx = req_to_token[req_idx * max_context_len + n_start + j];
                  const int8_t* v_src = v_buffer_typed + token_idx * v_strideN + head_kv * v_strideH;
                  int8_t* v_dst = v_buf + j * head_size_v;
                  std::memcpy(v_dst, v_src, head_size_v * sizeof(int8_t));
                }
              } else {
                // Direct copy for FP
                for (int j = 0; j < n_size; ++j) {
                  int token_idx = req_to_token[req_idx * max_context_len + n_start + j];
                  const buffer_t* v_src = v_buffer_typed + token_idx * v_strideN + head_kv * v_strideH;
                  buffer_t* v_dst = v_buf + j * head_size_v;
                  std::memcpy(v_dst, v_src, head_size_v * sizeof(buffer_t));
                }
              }

              // Update Acc Accumulators (Common Logic)
              for (int i = 0; i < m_size; ++i) {
                float* row_scores = scores_block + i * BLOCK_N;
                if (logit_cap > 0.0f) {
                  float rlogit_cap = 1.0f / logit_cap;
                  size_t vl_max = __riscv_vsetvlmax_e32m4();
                  for (int j = 0; j < n_size; j += vl_max) {
                    size_t vl = __riscv_vsetvl_e32m4(n_size - j);
                    vfloat32m4_t vx = __riscv_vle32_v_f32m4(row_scores + j, vl);
                    vx = __riscv_vfmul_vf_f32m4(vx, rlogit_cap, vl);
                    vfloat32m4_t vtanh = vftanh_f32m4(vx, vl);
                    vfloat32m4_t vres = __riscv_vfmul_vf_f32m4(vtanh, logit_cap, vl);
                    __riscv_vse32_v_f32m4(row_scores + j, vres, vl);
                  }
                }

                // Max reduction
                float m_curr = -std::numeric_limits<float>::infinity();
                if (n_size > 0) {
                  size_t vl_max = __riscv_vsetvlmax_e32m4();
                  vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(m_curr, 1);
                  for (int j = 0; j < n_size; j += vl_max) {
                    size_t vl = __riscv_vsetvl_e32m4(n_size - j);
                    vfloat32m4_t v_val = __riscv_vle32_v_f32m4(row_scores + j, vl);
                    v_max = __riscv_vfredmax_vs_f32m4_f32m1(v_val, v_max, vl);
                  }
                  m_curr = __riscv_vfmv_f_s_f32m1_f32(v_max);
                  if (!std::isfinite(m_curr)) m_curr = -std::numeric_limits<float>::infinity();
                }

                float m_prev = m_acc[i];
                if (!std::isfinite(m_prev)) m_prev = -std::numeric_limits<float>::infinity();

                float m_new = std::max(m_prev, m_curr);
                if (!std::isfinite(m_new)) m_new = -std::numeric_limits<float>::infinity();

                float alpha = (m_prev == -std::numeric_limits<float>::infinity()) ? 0.0f : safe_exp(m_prev - m_new);
                if (!std::isfinite(alpha)) alpha = 0.0f;

                l_acc[i] = l_acc[i] * alpha;
                m_acc[i] = m_new;

                if (alpha != 1.0f) {
                  float* acc_row = o_acc + i * head_size_v;
                  size_t vl_max = __riscv_vsetvlmax_e32m4();
                  for (int d = 0; d < head_size_v; d += vl_max) {
                    size_t vl = __riscv_vsetvl_e32m4(head_size_v - d);
                    vfloat32m4_t v_o = __riscv_vle32_v_f32m4(acc_row + d, vl);
                    v_o = __riscv_vfmul_vf_f32m4(v_o, alpha, vl);
                    __riscv_vse32_v_f32m4(acc_row + d, v_o, vl);
                  }
                }
                float l_curr = exp_and_sum_rvv(row_scores, n_size, m_new);
                if (!std::isfinite(l_curr)) l_curr = 0.0f;
                l_acc[i] += l_curr;
                if (!std::isfinite(l_acc[i])) l_acc[i] = 0.0f;
              }

              // Value Accumulation GEMM
              if constexpr (IS_INT8) {
                gemm_nn_rvv_tiled_int8(
                    scores_block, v_buf, o_acc, m_size, n_size, head_size_v, BLOCK_N, head_size_v, v_scale);
              } else {
                gemm_nn_rvv_tiled(scores_block, v_buf, o_acc, m_size, n_size, head_size_v, BLOCK_N, head_size_v);
              }
            }  // end loop over prefix blocks
          }  // end if prefix_len > 0

          // ============================================
          // EXTEND PART (Self-Attention)
          // ============================================
          for (int n_start = 0; n_start < extend_len; n_start += BLOCK_N) {
            int n_size = std::min(BLOCK_N, extend_len - n_start);
            // printf("Extend Loop: m_start=%d, n_start=%d, m_size=%d, n_size=%d\n", m_start, n_start, m_size, n_size);

            const scalar_t* k_ext_ptr = k_extend + n_start * ke_strideN + head_kv * ke_strideH + start_loc * ke_strideN;

            // Prepare K Transpose (always FP for extend part)
            for (int j = 0; j < n_size; ++j) {
              const scalar_t* src_tok = k_ext_ptr + j * ke_strideN;
              if constexpr (std::is_same_v<scalar_t, at::Half>) {
                size_t vl_max = __riscv_vsetvlmax_e16m2();
                for (int d = 0; d < head_size; d += vl_max) {
                  size_t vl = __riscv_vsetvl_e16m2(head_size - d);
                  vfloat16m2_t v_k = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(src_tok + d), vl);
                  ptrdiff_t stride = BLOCK_N * sizeof(_Float16);
                  __riscv_vsse16_v_f16m2(
                      reinterpret_cast<_Float16*>(k_trans_buf_fp + d * BLOCK_N + j), stride, v_k, vl);
                }
              } else {
                for (int d = 0; d < head_size; ++d) {
                  k_trans_buf_fp[d * BLOCK_N + j] = src_tok[d];
                }
              }
            }

            gemm_nt_rvv_tiled_transposed(
                q_ptr + m_start * q_strideM,
                k_trans_buf_fp,  // Always use FP buffer for extend part
                scores_block,
                m_size,
                n_size,
                head_size,
                q_strideM,
                BLOCK_N,
                BLOCK_N,
                scaling);

            // Causal Masking
            for (int i = 0; i < m_size; ++i) {
              int m_idx = m_start + i;
              float* row = scores_block + i * BLOCK_N;
              if (n_start > m_idx) {
                // Entire block masked
                size_t vl_max = __riscv_vsetvlmax_e32m4();
                for (int j = 0; j < n_size; j += vl_max) {
                  size_t vl = __riscv_vsetvl_e32m4(n_size - j);
                  __riscv_vse32_v_f32m4(
                      row + j, __riscv_vfmv_v_f_f32m4(-std::numeric_limits<float>::infinity(), vl), vl);
                }
              } else if (n_start + n_size > m_idx + 1) {
                // Partial block masked
                int valid = m_idx - n_start + 1;
                size_t vl_max = __riscv_vsetvlmax_e32m4();
                for (int j = valid; j < n_size; j += vl_max) {
                  size_t vl = __riscv_vsetvl_e32m4(n_size - j);
                  __riscv_vse32_v_f32m4(
                      row + j, __riscv_vfmv_v_f_f32m4(-std::numeric_limits<float>::infinity(), vl), vl);
                }
              }
            }

            // Copy V Extend (always FP)
            const scalar_t* v_ext_ptr = v_extend + n_start * ve_strideN + head_kv * ve_strideH + start_loc * ve_strideN;
            for (int j = 0; j < n_size; ++j) {
              const scalar_t* src_tok = v_ext_ptr + j * ve_strideN;
              scalar_t* dst_tok = v_buf_fp + j * head_size_v;
              std::memcpy(dst_tok, src_tok, head_size_v * sizeof(scalar_t));
            }

            // Update Accumulators (Softmax Step)
            for (int i = 0; i < m_size; ++i) {
              // Same logic as Prefix Part
              float* row_scores = scores_block + i * BLOCK_N;
              if (logit_cap > 0.0f) {
                float rlogit_cap = 1.0f / logit_cap;
                size_t vl_max = __riscv_vsetvlmax_e32m4();
                for (int j = 0; j < n_size; j += vl_max) {
                  size_t vl = __riscv_vsetvl_e32m4(n_size - j);
                  vfloat32m4_t vx = __riscv_vle32_v_f32m4(row_scores + j, vl);
                  vx = __riscv_vfmul_vf_f32m4(vx, rlogit_cap, vl);
                  vfloat32m4_t vtanh = vftanh_f32m4(vx, vl);
                  vfloat32m4_t vres = __riscv_vfmul_vf_f32m4(vtanh, logit_cap, vl);
                  __riscv_vse32_v_f32m4(row_scores + j, vres, vl);
                }
              }

              float m_curr = -std::numeric_limits<float>::infinity();
              if (n_size > 0) {
                size_t vl_max = __riscv_vsetvlmax_e32m4();
                vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(m_curr, 1);
                for (int j = 0; j < n_size; j += vl_max) {
                  size_t vl = __riscv_vsetvl_e32m4(n_size - j);
                  vfloat32m4_t v_val = __riscv_vle32_v_f32m4(row_scores + j, vl);
                  v_max = __riscv_vfredmax_vs_f32m4_f32m1(v_val, v_max, vl);
                }
                m_curr = __riscv_vfmv_f_s_f32m1_f32(v_max);
                if (!std::isfinite(m_curr)) m_curr = -std::numeric_limits<float>::infinity();
              }

              // printf("Debug: i=%d, m_curr=%f, m_prev=%f\n", i, m_curr, m_acc[i]);

              float m_prev = m_acc[i];
              if (!std::isfinite(m_prev)) m_prev = -std::numeric_limits<float>::infinity();
              float m_new = std::max(m_prev, m_curr);
              if (!std::isfinite(m_new)) m_new = -std::numeric_limits<float>::infinity();

              float alpha = (m_prev == -std::numeric_limits<float>::infinity()) ? 0.0f : safe_exp(m_prev - m_new);
              if (!std::isfinite(alpha)) alpha = 0.0f;

              l_acc[i] = l_acc[i] * alpha;
              m_acc[i] = m_new;

              if (alpha != 1.0f) {
                float* acc_row = o_acc + i * head_size_v;
                size_t vl_max = __riscv_vsetvlmax_e32m4();
                for (int d = 0; d < head_size_v; d += vl_max) {
                  size_t vl = __riscv_vsetvl_e32m4(head_size_v - d);
                  vfloat32m4_t v_o = __riscv_vle32_v_f32m4(acc_row + d, vl);
                  v_o = __riscv_vfmul_vf_f32m4(v_o, alpha, vl);
                  __riscv_vse32_v_f32m4(acc_row + d, v_o, vl);
                }
              }

              float l_curr = exp_and_sum_rvv(row_scores, n_size, m_new);
              if (!std::isfinite(l_curr)) l_curr = 0.0f;
              l_acc[i] += l_curr;
              if (!std::isfinite(l_acc[i])) l_acc[i] = 0.0f;
            }

            // Value Accumulation GEMM for Extend (Using v_buf_fp)
            gemm_nn_rvv_tiled(scores_block, v_buf_fp, o_acc, m_size, n_size, head_size_v, BLOCK_N, head_size_v);

          }  // end loop with seq_len_extend

          // Final Normalization and Write-back
          for (int i = 0; i < m_size; ++i) {
            scalar_t* out_row = o_ptr + (m_start + i) * o_strideM;
            float* acc_row = o_acc + i * head_size_v;
            float l_val = l_acc[i];

            float inv_l = (l_val > 1e-6f) ? 1.0f / l_val : 0.0f;
            if (!std::isfinite(inv_l)) inv_l = 0.0f;

            size_t vl;
            for (int d = 0; d < head_size_v; d += vl) {
              vl = __riscv_vsetvl_e32m4(head_size_v - d);
              vfloat32m4_t v_val = __riscv_vle32_v_f32m4(acc_row + d, vl);
              v_val = __riscv_vfmul_vf_f32m4(v_val, inv_l, vl);

              if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH
                vfloat16m2_t v_f16 = f32m4_to_f16(v_val, vl);
                __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(out_row + d), v_f16, vl);
#else
                float temp[128];
                __riscv_vse32_v_f32m4(temp, v_val, vl);
                for (int k = 0; k < vl; ++k)
                  out_row[d + k] = static_cast<scalar_t>(temp[k]);
#endif
              } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
                vuint16m2_t v_bf16 = f32m4_to_bf16(v_val, vl);
                __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(out_row + d), v_bf16, vl);
              } else {
                __riscv_vse32_v_f32m4((float*)out_row + d, v_val, vl);
              }
            }
          }
        }
      }
    }
  }
}
template <typename scalar_t, typename index_t>
// =============================================================================
// Kernel: Attention Computation
// =============================================================================
void extend_attention_kernel_rvv_impl(
    scalar_t* __restrict__ o_extend,
    const scalar_t* __restrict__ q_extend,
    const scalar_t* __restrict__ k_extend,
    const scalar_t* __restrict__ v_extend,
    const scalar_t* __restrict__ k_buffer,
    const scalar_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    const index_t* __restrict__ extend_seq_lens,
    const index_t* __restrict__ extend_start_loc,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int o_strideM,
    int o_strideH,
    int q_strideM,
    int q_strideH,
    int ke_strideN,
    int ke_strideH,
    int ve_strideN,
    int ve_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float scaling,
    float logit_cap,
    int max_num_reqs,
    int max_context_len,
    int max_total_num_tokens,
    int max_len_extend,
    bool is_prefix_skipped) {
  extend_attention_kernel_rvv_impl_template<scalar_t, index_t, false>(
      o_extend,
      q_extend,
      k_extend,
      v_extend,
      k_buffer,
      v_buffer,
      req_to_token,
      req_pool_indices,
      seq_lens,
      extend_seq_lens,
      extend_start_loc,
      batches,
      num_heads,
      num_heads_kv,
      head_size,
      head_size_v,
      o_strideM,
      o_strideH,
      q_strideM,
      q_strideH,
      ke_strideN,
      ke_strideH,
      ve_strideN,
      ve_strideH,
      k_strideN,
      k_strideH,
      v_strideN,
      v_strideH,
      scaling,
      logit_cap,
      max_num_reqs,
      max_context_len,
      max_total_num_tokens,
      max_len_extend,
      is_prefix_skipped);
}

template <typename scalar_t, typename index_t>
void extend_attention_kernel_rvv_int8_impl(
    scalar_t* __restrict__ o_extend,
    const scalar_t* __restrict__ q_extend,
    const scalar_t* __restrict__ k_extend,
    const scalar_t* __restrict__ v_extend,
    const int8_t* __restrict__ k_buffer,
    const int8_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    const index_t* __restrict__ extend_seq_lens,
    const index_t* __restrict__ extend_start_loc,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int o_strideM,
    int o_strideH,
    int q_strideM,
    int q_strideH,
    int ke_strideN,
    int ke_strideH,
    int ve_strideN,
    int ve_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float scaling,
    float logit_cap,
    int max_num_reqs,
    int max_context_len,
    int max_total_num_tokens,
    int max_len_extend,
    bool is_prefix_skipped,
    float k_scale,
    float v_scale) {
  extend_attention_kernel_rvv_impl_template<scalar_t, index_t, true>(
      o_extend,
      q_extend,
      k_extend,
      v_extend,
      k_buffer,
      v_buffer,
      req_to_token,
      req_pool_indices,
      seq_lens,
      extend_seq_lens,
      extend_start_loc,
      batches,
      num_heads,
      num_heads_kv,
      head_size,
      head_size_v,
      o_strideM,
      o_strideH,
      q_strideM,
      q_strideH,
      ke_strideN,
      ke_strideH,
      ve_strideN,
      ve_strideH,
      k_strideN,
      k_strideH,
      v_strideN,
      v_strideH,
      scaling,
      logit_cap,
      max_num_reqs,
      max_context_len,
      max_total_num_tokens,
      max_len_extend,
      is_prefix_skipped,
      k_scale,
      v_scale);
}

// =============================================================================
// Entry Point: CPU Interface
// =============================================================================
void extend_attention_kernel_rvv(
    at::Tensor& q_extend,
    at::Tensor& k_extend,
    at::Tensor& v_extend,
    at::Tensor& o_extend,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_seq_lens,
    at::Tensor& extend_start_loc,
    int64_t max_len_extend,
    double sm_scale,
    double logit_cap) {
  int num_seqs = seq_lens.size(0);
  int num_heads = q_extend.size(1);
  int num_heads_kv = k_extend.size(1);
  int head_size = q_extend.size(2);
  int head_size_v = v_extend.size(2);

  int q_strideM = q_extend.stride(0);
  int q_strideH = q_extend.stride(1);
  int o_strideM = o_extend.stride(0);
  int o_strideH = o_extend.stride(1);
  int ke_strideN = k_extend.stride(0);
  int ke_strideH = k_extend.stride(1);
  int ve_strideN = v_extend.stride(0);
  int ve_strideH = v_extend.stride(1);

  int k_strideN = k_buffer.stride(0);
  int k_strideH = k_buffer.stride(1);
  int v_strideN = v_buffer.stride(0);
  int v_strideH = v_buffer.stride(1);

  int64_t max_num_reqs = req_to_token.size(0);
  int64_t max_context_len = req_to_token.size(1);
  int64_t max_total_num_tokens = k_buffer.size(0);
  bool is_prefix_skipped = k_buffer.size(1) != num_heads_kv;

  const auto index_dtype = req_to_token.scalar_type();

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q_extend.scalar_type(), "extend_attention_kernel_rvv", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "extend_attention_kernel_rvv_indices", [&] {
      extend_attention_kernel_rvv_impl<scalar_t, index_t>(
          o_extend.data_ptr<scalar_t>(),
          q_extend.data_ptr<scalar_t>(),
          k_extend.data_ptr<scalar_t>(),
          v_extend.data_ptr<scalar_t>(),
          k_buffer.data_ptr<scalar_t>(),
          v_buffer.data_ptr<scalar_t>(),
          req_to_token.data_ptr<index_t>(),
          req_pool_indices.data_ptr<int64_t>(),
          seq_lens.data_ptr<int64_t>(),
          extend_seq_lens.data_ptr<index_t>(),
          extend_start_loc.data_ptr<index_t>(),
          num_seqs,
          num_heads,
          num_heads_kv,
          head_size,
          head_size_v,
          o_strideM,
          o_strideH,
          q_strideM,
          q_strideH,
          ke_strideN,
          ke_strideH,
          ve_strideN,
          ve_strideH,
          k_strideN,
          k_strideH,
          v_strideN,
          v_strideH,
          (float)sm_scale,
          (float)logit_cap,
          (int)max_num_reqs,
          (int)max_context_len,
          (int)max_total_num_tokens,
          (int)max_len_extend,
          is_prefix_skipped);
    });
  });
}

}  // namespace

// RVV-specific extend_attention_cpu entry point
void extend_attention_cpu(
    at::Tensor& q_extend,
    at::Tensor& k_extend,
    at::Tensor& v_extend,
    at::Tensor& o_extend,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_seq_lens,
    at::Tensor& extend_start_loc,
    int64_t max_len_extend,
    double sm_scale,
    double logit_cap) {
  RECORD_FUNCTION(
      "sgl-kernel::extend_attention_cpu_rvv",
      std::vector<c10::IValue>(
          {q_extend,
           k_extend,
           v_extend,
           o_extend,
           k_buffer,
           v_buffer,
           req_to_token,
           req_pool_indices,
           seq_lens,
           extend_seq_lens,
           extend_start_loc}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_extend);
  CHECK_INPUT(o_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);

  int num_seqs = seq_lens.size(0);
  int max_num_reqs = req_to_token.size(0);
  int max_context_len = req_to_token.size(1);
  int max_total_num_tokens = k_buffer.size(0);

  int num_heads = q_extend.size(1);
  int num_heads_kv = k_extend.size(1);
  int head_size = q_extend.size(2);
  int head_size_v = v_extend.size(2);

  // strides for q_extend, k_extend and v_extend
  int q_strideM = q_extend.stride(0);
  int q_strideH = q_extend.stride(1);
  int o_strideM = o_extend.stride(0);
  int o_strideH = o_extend.stride(1);
  int ke_strideN = k_extend.stride(0);
  int ke_strideH = k_extend.stride(1);
  int ve_strideN = v_extend.stride(0);
  int ve_strideH = v_extend.stride(1);

  // strides for k_buffer and v_buffer
  int k_strideN = k_buffer.stride(0);
  int k_strideH = k_buffer.stride(1);
  int v_strideN = v_buffer.stride(0);
  int v_strideH = v_buffer.stride(1);

  // check sizes
  CHECK_EQ(req_pool_indices.size(0), num_seqs);
  CHECK_EQ(extend_seq_lens.size(0), num_seqs);
  CHECK_EQ(extend_start_loc.size(0), num_seqs);
  CHECK_EQ(v_extend.size(1), num_heads_kv);
  CHECK_EQ(k_buffer.size(1), v_buffer.size(1));

  // MLA will skip prefix part
  const bool is_prefix_skipped = k_buffer.size(1) != num_heads_kv;

  // check index data types
  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "extend: expect req_to_token to be int32 or int64, got ",
      index_dtype);
  TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "extend: expect req_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kLong,
      "extend: expect req_pool_indices to be int64, got ",
      req_pool_indices.scalar_type());
  TORCH_CHECK(
      extend_seq_lens.scalar_type() == index_dtype && extend_start_loc.scalar_type() == index_dtype,
      "extend: expect extend_seq_lens and extend_start_loc to have same dtype as req_to_token.");

  // Use AT_DISPATCH_RVV_TYPES for Float, Half and BFloat16
  AT_DISPATCH_RVV_TYPES(q_extend.scalar_type(), "extend_attention_kernel_rvv", [&] {
    {
      AT_DISPATCH_INDEX_TYPES(index_dtype, "extend_attention_kernel_rvv_indices", [&] {
        extend_attention_kernel_rvv_impl<scalar_t, index_t>(
            o_extend.data_ptr<scalar_t>(),
            q_extend.data_ptr<scalar_t>(),
            k_extend.data_ptr<scalar_t>(),
            v_extend.data_ptr<scalar_t>(),
            k_buffer.data_ptr<scalar_t>(),
            v_buffer.data_ptr<scalar_t>(),
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            extend_seq_lens.data_ptr<index_t>(),
            extend_start_loc.data_ptr<index_t>(),
            num_seqs,
            num_heads,
            num_heads_kv,
            head_size,
            head_size_v,
            o_strideM,
            o_strideH,
            q_strideM,
            q_strideH,
            ke_strideN,
            ke_strideH,
            ve_strideN,
            ve_strideH,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            (float)sm_scale,
            (float)logit_cap,
            (int)max_num_reqs,
            (int)max_context_len,
            (int)max_total_num_tokens,
            (int)max_len_extend,
            is_prefix_skipped);
      });
    }  // end block
  });
}

void extend_attention_int8_cpu(
    at::Tensor& q_extend,
    at::Tensor& k_extend,
    at::Tensor& v_extend,
    at::Tensor& o_extend,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_seq_lens,
    at::Tensor& extend_start_loc,
    int64_t max_len_extend,
    double sm_scale,
    double logit_cap,
    double k_scale,
    double v_scale) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_extend);
  CHECK_INPUT(o_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);

  int num_seqs = seq_lens.size(0);
  int max_num_reqs = req_to_token.size(0);
  int max_context_len = req_to_token.size(1);
  int max_total_num_tokens = k_buffer.size(0);

  int num_heads = q_extend.size(1);
  int num_heads_kv = k_extend.size(1);
  int head_size = q_extend.size(2);
  int head_size_v = v_extend.size(2);

  int q_strideM = q_extend.stride(0);
  int q_strideH = q_extend.stride(1);
  int o_strideM = o_extend.stride(0);
  int o_strideH = o_extend.stride(1);
  int ke_strideN = k_extend.stride(0);
  int ke_strideH = k_extend.stride(1);
  int ve_strideN = v_extend.stride(0);
  int ve_strideH = v_extend.stride(1);

  int k_strideN = k_buffer.stride(0);
  int k_strideH = k_buffer.stride(1);
  int v_strideN = v_buffer.stride(0);
  int v_strideH = v_buffer.stride(1);

  CHECK_EQ(req_pool_indices.size(0), num_seqs);
  CHECK_EQ(extend_seq_lens.size(0), num_seqs);
  CHECK_EQ(extend_start_loc.size(0), num_seqs);
  CHECK_EQ(v_extend.size(1), num_heads_kv);
  CHECK_EQ(k_buffer.size(1), v_buffer.size(1));

  bool is_prefix_skipped = k_buffer.size(1) != num_heads_kv;

  // check index data types
  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "extend_int8: expect req_to_token to be int32 or int64, got ",
      index_dtype);
  TORCH_CHECK(
      seq_lens.scalar_type() == at::kLong, "extend_int8: expect seq_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kLong,
      "extend_int8: expect req_pool_indices to be int64, got ",
      req_pool_indices.scalar_type());
  TORCH_CHECK(
      extend_seq_lens.scalar_type() == index_dtype && extend_start_loc.scalar_type() == index_dtype,
      "extend_int8: expect extend_seq_lens and extend_start_loc to have same dtype as req_to_token.");

  // Validate extend_len <= seq_len for each sequence
  for (int i = 0; i < num_seqs; ++i) {
    int64_t seq_len = seq_lens[i].item<int64_t>();
    int64_t extend_len = extend_seq_lens[i].item<int64_t>();
    TORCH_CHECK(
        extend_len <= seq_len,
        "extend_int8: extend_len (",
        extend_len,
        ") must be <= seq_len (",
        seq_len,
        ") for sequence ",
        i);
    TORCH_CHECK(extend_len >= 0, "extend_int8: extend_len (", extend_len, ") must be >= 0 for sequence ", i);
    TORCH_CHECK(seq_len >= 0, "extend_int8: seq_len (", seq_len, ") must be >= 0 for sequence ", i);
  }

  // Validate types for INT8 mode
  TORCH_CHECK(
      k_buffer.scalar_type() == at::kByte || k_buffer.scalar_type() == at::kChar,
      "extend_int8: k_buffer must be int8/uint8, got ",
      k_buffer.scalar_type());
  TORCH_CHECK(
      v_buffer.scalar_type() == at::kByte || v_buffer.scalar_type() == at::kChar,
      "extend_int8: v_buffer must be int8/uint8, got ",
      v_buffer.scalar_type());

  // Use AT_DISPATCH_RVV_TYPES for Float, Half and BFloat16
  AT_DISPATCH_RVV_TYPES(q_extend.scalar_type(), "extend_attention_int8_cpu", [&] {
    {
      AT_DISPATCH_INDEX_TYPES(index_dtype, "extend_attention_int8_cpu_indices", [&] {
        if (k_extend.scalar_type() == at::kByte || k_extend.scalar_type() == at::kChar) {
        } else {
          extend_set_kv_buffer_int8_quantize<scalar_t, index_t>(
              (int8_t*)k_buffer.data_ptr(),
              (int8_t*)v_buffer.data_ptr(),
              k_extend.data_ptr<scalar_t>(),
              v_extend.data_ptr<scalar_t>(),
              req_to_token.data_ptr<index_t>(),
              req_pool_indices.data_ptr<int64_t>(),
              seq_lens.data_ptr<int64_t>(),
              extend_seq_lens.data_ptr<index_t>(),
              extend_start_loc.data_ptr<index_t>(),
              num_seqs,
              num_heads_kv,
              head_size,
              head_size_v,
              k_strideN,
              k_strideH,
              v_strideN,
              v_strideH,
              ke_strideN,
              ke_strideH,
              ve_strideN,
              ve_strideH,
              max_context_len,
              (float)k_scale,
              (float)v_scale);
        }
        extend_attention_kernel_rvv_int8_impl<scalar_t, index_t>(
            o_extend.data_ptr<scalar_t>(),
            q_extend.data_ptr<scalar_t>(),
            k_extend.data_ptr<scalar_t>(),
            v_extend.data_ptr<scalar_t>(),
            (int8_t*)k_buffer.data_ptr(),
            (int8_t*)v_buffer.data_ptr(),
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            extend_seq_lens.data_ptr<index_t>(),
            extend_start_loc.data_ptr<index_t>(),
            num_seqs,
            num_heads,
            num_heads_kv,
            head_size,
            head_size_v,
            o_strideM,
            o_strideH,
            q_strideM,
            q_strideH,
            ke_strideN,
            ke_strideH,
            ve_strideN,
            ve_strideH,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            (float)sm_scale,
            (float)logit_cap,
            max_num_reqs,
            max_context_len,
            max_total_num_tokens,
            max_len_extend,
            is_prefix_skipped,
            (float)k_scale,
            (float)v_scale);
      });
    }
  });
}

#endif  // CPU_CAPABILITY_RVV
