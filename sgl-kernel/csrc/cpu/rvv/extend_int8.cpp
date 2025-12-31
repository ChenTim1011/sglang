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

#include "vector_helpers.h"

// Check for Zvfh (FP16 vector) support
#if defined(__riscv_zvfh) || defined(__riscv_v)
#define HAS_ZVFH 1
#else
#define HAS_ZVFH 0
#endif

namespace {

template <typename scalar_t>
inline void
quantize_row_int8_with_scale(uint8_t* __restrict__ Aq, const scalar_t* __restrict__ A, int64_t K, float scale) {
  quantize_row_int8_symmetric_rvv<scalar_t>(reinterpret_cast<int8_t*>(Aq), A, K, scale);
}

template <typename scalar_t>
inline void quantize_row_int8_symmetric_auto(
    int8_t* __restrict__ Aq, float& scale_out, const scalar_t* __restrict__ A, int64_t K, float eps = 1e-7) {
  float amax = 0.f;
  for (int64_t k = 0; k < K; ++k) {
    const float val = static_cast<float>(A[k]);
    amax = std::max(amax, std::abs(val));
  }

  amax = std::max(amax, eps);
  const float scale = amax / 127.0f;
  const float inv_scale = 127.0f / amax;

  for (int64_t k = 0; k < K; ++k) {
    const float val = static_cast<float>(A[k]) * inv_scale;
    int32_t quantized = (int32_t)(std::round(val));
    quantized = std::max(-128, std::min(127, quantized));
    Aq[k] = (int8_t)(quantized);
  }

  scale_out = scale;
}

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
        const scalar_t* k_extend_ptr = k_extend + (start_loc + t) * ke_strideN + head_kv_id * ke_strideH;
        const scalar_t* v_extend_ptr = v_extend + (start_loc + t) * ve_strideN + head_kv_id * ve_strideH;

        // On-the-fly quantization for float input
        // Always use SYMMETRIC quantization for KV cache (signed int8)
        if (k_scale != 1.0f && k_scale > 0.0f) {
          // Use provided scale with symmetric quantization
          quantize_row_int8_symmetric_rvv<scalar_t>(
              k_buffer_ptr,  // int8_t* directly, no type conversion needed
              k_extend_ptr,
              head_size,
              k_scale);
        } else {
          // Auto-compute scale with symmetric quantization
          float computed_k_scale;
          quantize_row_int8_symmetric_auto<scalar_t>(
              k_buffer_ptr,  // int8_t* directly
              computed_k_scale,
              k_extend_ptr,
              head_size);
        }

        if (v_scale != 1.0f && v_scale > 0.0f) {
          // Use provided scale with symmetric quantization
          quantize_row_int8_symmetric_rvv<scalar_t>(
              v_buffer_ptr,  // int8_t* directly
              v_extend_ptr,
              head_size_v,
              v_scale);
        } else {
          // Auto-compute scale with symmetric quantization
          float computed_v_scale;
          quantize_row_int8_symmetric_auto<scalar_t>(
              v_buffer_ptr,  // int8_t* directly
              computed_v_scale,
              v_extend_ptr,
              head_size_v);
        }
      }

      data_index_step(seq_idx, num_seqs_i64, head_kv_id, num_heads_kv_i64);
    }
  });
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
  size_t vl;
  for (int n = 0; n < N; n += vl) {
    vl = __riscv_vsetvl_e16m2(N - n);

    int m = 0;
    for (; m < M - 3; m += 4) {
      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      // K reduction loop with unroll factor 2 for better ILP
      int k = 0;
      for (; k < head_size - 1; k += 2) {
        const scalar_t* k_ptr0 = K_trans + (k + 0) * block_n + n;
        const scalar_t* k_ptr1 = K_trans + (k + 1) * block_n + n;

        // Prefetch next iteration's K data
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
          acc0 = __riscv_vfwmacc_vf_f32m4(acc0, (_Float16)(q0_0), v_k0, vl);
          acc0 = __riscv_vfwmacc_vf_f32m4(acc0, (_Float16)(q0_1), v_k1, vl);
          acc1 = __riscv_vfwmacc_vf_f32m4(acc1, (_Float16)(q1_0), v_k0, vl);
          acc1 = __riscv_vfwmacc_vf_f32m4(acc1, (_Float16)(q1_1), v_k1, vl);
          acc2 = __riscv_vfwmacc_vf_f32m4(acc2, (_Float16)(q2_0), v_k0, vl);
          acc2 = __riscv_vfwmacc_vf_f32m4(acc2, (_Float16)(q2_1), v_k1, vl);
          acc3 = __riscv_vfwmacc_vf_f32m4(acc3, (_Float16)(q3_0), v_k0, vl);
          acc3 = __riscv_vfwmacc_vf_f32m4(acc3, (_Float16)(q3_1), v_k1, vl);
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
          float k_tmp0[32], k_tmp1[32];
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

      // Handle remaining K iterations
      for (; k < head_size; ++k) {
        const scalar_t* k_ptr = K_trans + k * block_n + n;
        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_k = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr), vl);
          float q0 = static_cast<float>(Q[(m + 0) * q_strideM + k]);
          float q1 = static_cast<float>(Q[(m + 1) * q_strideM + k]);
          float q2 = static_cast<float>(Q[(m + 2) * q_strideM + k]);
          float q3 = static_cast<float>(Q[(m + 3) * q_strideM + k]);
          acc0 = __riscv_vfwmacc_vf_f32m4(acc0, (_Float16)(q0), v_k, vl);
          acc1 = __riscv_vfwmacc_vf_f32m4(acc1, (_Float16)(q1), v_k, vl);
          acc2 = __riscv_vfwmacc_vf_f32m4(acc2, (_Float16)(q2), v_k, vl);
          acc3 = __riscv_vfwmacc_vf_f32m4(acc3, (_Float16)(q3), v_k, vl);
        } else {
          float k_tmp[32];
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
          acc = __riscv_vfwmacc_vf_f32m4(acc, (_Float16)q_val, v_k, vl);
        } else {
          float k_tmp[32];
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
  for (int n = 0; n < N; n += vl) {
    vl = __riscv_vsetvl_e16m2(N - n);

    int m = 0;
    for (; m < M - 3; m += 4) {
      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      // K reduction loop with unroll factor 2 for better ILP
      int k = 0;
      for (; k < head_size - 1; k += 2) {
        const int8_t* k_ptr0 = K_trans + (k + 0) * block_n + n;
        const int8_t* k_ptr1 = K_trans + (k + 1) * block_n + n;
        if (k + 2 < head_size) {
          __builtin_prefetch(K_trans + (k + 2) * block_n + n, 0, 3);
        }

        float q0_0 = static_cast<float>(Q[(m + 0) * q_strideM + k + 0]);
        float q0_1 = static_cast<float>(Q[(m + 0) * q_strideM + k + 1]);
        float q1_0 = static_cast<float>(Q[(m + 1) * q_strideM + k + 0]);
        float q1_1 = static_cast<float>(Q[(m + 1) * q_strideM + k + 1]);
        float q2_0 = static_cast<float>(Q[(m + 2) * q_strideM + k + 0]);
        float q2_1 = static_cast<float>(Q[(m + 2) * q_strideM + k + 1]);
        float q3_0 = static_cast<float>(Q[(m + 3) * q_strideM + k + 0]);
        float q3_1 = static_cast<float>(Q[(m + 3) * q_strideM + k + 1]);

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_k0_f16 = int8_to_f16m2(k_ptr0, vl);
          vfloat16m2_t v_k1_f16 = int8_to_f16m2(k_ptr1, vl);
          acc0 = __riscv_vfwmacc_vf_f32m4(acc0, (_Float16)(q0_0), v_k0_f16, vl);
          acc0 = __riscv_vfwmacc_vf_f32m4(acc0, (_Float16)(q0_1), v_k1_f16, vl);
          acc1 = __riscv_vfwmacc_vf_f32m4(acc1, (_Float16)(q1_0), v_k0_f16, vl);
          acc1 = __riscv_vfwmacc_vf_f32m4(acc1, (_Float16)(q1_1), v_k1_f16, vl);
          acc2 = __riscv_vfwmacc_vf_f32m4(acc2, (_Float16)(q2_0), v_k0_f16, vl);
          acc2 = __riscv_vfwmacc_vf_f32m4(acc2, (_Float16)(q2_1), v_k1_f16, vl);
          acc3 = __riscv_vfwmacc_vf_f32m4(acc3, (_Float16)(q3_0), v_k0_f16, vl);
          acc3 = __riscv_vfwmacc_vf_f32m4(acc3, (_Float16)(q3_1), v_k1_f16, vl);
        } else {
          vfloat32m4_t v_k0_f32 = int8_to_f32m4(k_ptr0, vl);
          vfloat32m4_t v_k1_f32 = int8_to_f32m4(k_ptr1, vl);
          acc0 = __riscv_vfmacc_vf_f32m4(acc0, q0_0, v_k0_f32, vl);
          acc0 = __riscv_vfmacc_vf_f32m4(acc0, q0_1, v_k1_f32, vl);
          acc1 = __riscv_vfmacc_vf_f32m4(acc1, q1_0, v_k0_f32, vl);
          acc1 = __riscv_vfmacc_vf_f32m4(acc1, q1_1, v_k1_f32, vl);
          acc2 = __riscv_vfmacc_vf_f32m4(acc2, q2_0, v_k0_f32, vl);
          acc2 = __riscv_vfmacc_vf_f32m4(acc2, q2_1, v_k1_f32, vl);
          acc3 = __riscv_vfmacc_vf_f32m4(acc3, q3_0, v_k0_f32, vl);
          acc3 = __riscv_vfmacc_vf_f32m4(acc3, q3_1, v_k1_f32, vl);
        }
      }

      // Handle remaining K iterations
      for (; k < head_size; ++k) {
        const int8_t* k_ptr = K_trans + k * block_n + n;
        float q_val0 = static_cast<float>(Q[(m + 0) * q_strideM + k]);
        float q_val1 = static_cast<float>(Q[(m + 1) * q_strideM + k]);
        float q_val2 = static_cast<float>(Q[(m + 2) * q_strideM + k]);
        float q_val3 = static_cast<float>(Q[(m + 3) * q_strideM + k]);

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_k_f16 = int8_to_f16m2(k_ptr, vl);
          acc0 = __riscv_vfwmacc_vf_f32m4(acc0, (_Float16)(q_val0), v_k_f16, vl);
          acc1 = __riscv_vfwmacc_vf_f32m4(acc1, (_Float16)(q_val1), v_k_f16, vl);
          acc2 = __riscv_vfwmacc_vf_f32m4(acc2, (_Float16)(q_val2), v_k_f16, vl);
          acc3 = __riscv_vfwmacc_vf_f32m4(acc3, (_Float16)(q_val3), v_k_f16, vl);
        } else {
          vfloat32m4_t v_k_f32 = int8_to_f32m4(k_ptr, vl);
          acc0 = __riscv_vfmacc_vf_f32m4(acc0, q_val0, v_k_f32, vl);
          acc1 = __riscv_vfmacc_vf_f32m4(acc1, q_val1, v_k_f32, vl);
          acc2 = __riscv_vfmacc_vf_f32m4(acc2, q_val2, v_k_f32, vl);
          acc3 = __riscv_vfmacc_vf_f32m4(acc3, q_val3, v_k_f32, vl);
        }
      }

      float combined_scale = scale * k_scale;
      __riscv_vse32_v_f32m4(C + (m + 0) * ldc + n, __riscv_vfmul_vf_f32m4(acc0, combined_scale, vl), vl);
      __riscv_vse32_v_f32m4(C + (m + 1) * ldc + n, __riscv_vfmul_vf_f32m4(acc1, combined_scale, vl), vl);
      __riscv_vse32_v_f32m4(C + (m + 2) * ldc + n, __riscv_vfmul_vf_f32m4(acc2, combined_scale, vl), vl);
      __riscv_vse32_v_f32m4(C + (m + 3) * ldc + n, __riscv_vfmul_vf_f32m4(acc3, combined_scale, vl), vl);
    }

    for (; m < M; ++m) {
      vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      for (int k = 0; k < head_size; ++k) {
        const int8_t* k_ptr = K_trans + k * block_n + n;
        float q_val = static_cast<float>(Q[m * q_strideM + k]);

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_k_f16 = int8_to_f16m2(k_ptr, vl);
          acc = __riscv_vfwmacc_vf_f32m4(acc, (_Float16)q_val, v_k_f16, vl);
        } else {
          vfloat32m4_t v_k_f32 = int8_to_f32m4(k_ptr, vl);
          acc = __riscv_vfmacc_vf_f32m4(acc, q_val, v_k_f32, vl);
        }
      }
      float combined_scale = scale * k_scale;
      __riscv_vse32_v_f32m4(C + m * ldc + n, __riscv_vfmul_vf_f32m4(acc, combined_scale, vl), vl);
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
          acc0 = __riscv_vfwmacc_vf_f32m4(acc0, (_Float16)p0, v_v, vl);
          if (m_count > 1) acc1 = __riscv_vfwmacc_vf_f32m4(acc1, (_Float16)p1, v_v, vl);
          if (m_count > 2) acc2 = __riscv_vfwmacc_vf_f32m4(acc2, (_Float16)p2, v_v, vl);
          if (m_count > 3) acc3 = __riscv_vfwmacc_vf_f32m4(acc3, (_Float16)p3, v_v, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          vfloat32m4_t v_v = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(v_ptr), vl);
          acc0 = __riscv_vfmacc_vf_f32m4(acc0, p0, v_v, vl);
          if (m_count > 1) acc1 = __riscv_vfmacc_vf_f32m4(acc1, p1, v_v, vl);
          if (m_count > 2) acc2 = __riscv_vfmacc_vf_f32m4(acc2, p2, v_v, vl);
          if (m_count > 3) acc3 = __riscv_vfmacc_vf_f32m4(acc3, p3, v_v, vl);
        } else {
          float v_tmp[32];
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
  if (head_size > MAX_HEAD_SIZE || head_size_v > MAX_HEAD_SIZE) {
    return;
  }

  // Dynamic block size selection
  const int BLOCK_M = get_optimal_block_m<scalar_t>(head_size, head_size_v);
  const int BLOCK_N = get_optimal_block_n<scalar_t>(head_size, head_size_v);

#pragma omp parallel
  {
    // Use max possible size for dynamic allocation
    alignas(64) float scores_block[32 * 64];  // Max size
    alignas(64) int8_t k_trans_buf[MAX_HEAD_SIZE * 64];
    alignas(64) int8_t v_buf[64 * MAX_HEAD_SIZE];
    alignas(64) scalar_t k_trans_buf_fp[MAX_HEAD_SIZE * 64];
    alignas(64) scalar_t v_buf_fp[64 * MAX_HEAD_SIZE];

    alignas(64) float o_acc[32 * MAX_HEAD_SIZE];
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

        const scalar_t* q_ptr = q_extend + start_loc * q_strideM + h * q_strideH;
        scalar_t* o_ptr = o_extend + start_loc * q_strideM + h * q_strideH;

        for (int m_start = 0; m_start < extend_len; m_start += BLOCK_M) {
          int m_size = std::min(BLOCK_M, extend_len - m_start);

          std::fill(l_acc, l_acc + m_size, 0.0f);
          std::fill(m_acc, m_acc + m_size, -std::numeric_limits<float>::infinity());
          std::memset(o_acc, 0, m_size * head_size_v * sizeof(float));
          std::memset(scores_block, 0, BLOCK_M * BLOCK_N * sizeof(float));

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

              // Padding to VLEN multiple for better vectorization
              constexpr int VLEN_BYTES = 32;
              constexpr int VLEN_FLOATS = VLEN_BYTES / sizeof(float);
              int padded_n_size = ((n_size + VLEN_FLOATS - 1) / VLEN_FLOATS) * VLEN_FLOATS;
              padded_n_size = std::min(padded_n_size, BLOCK_N);
              if (is_contiguous && n_start == 0) {
                int base_token_idx = start_token_idx + n_start;
                for (int j = 0; j < n_size; ++j) {
                  int token_idx = base_token_idx + j;
                  const int8_t* k_src = k_buffer + token_idx * k_strideN + head_kv * k_strideH;
                  for (int d = 0; d < head_size; ++d) {
                    k_trans_buf[d * BLOCK_N + j] = k_src[d];
                  }
                }
              } else {
                for (int j = 0; j < n_size; ++j) {
                  int token_idx = req_to_token[req_idx * max_context_len + n_start + j];
                  const int8_t* k_src = k_buffer + token_idx * k_strideN + head_kv * k_strideH;
                  for (int d = 0; d < head_size; ++d) {
                    k_trans_buf[d * BLOCK_N + j] = k_src[d];
                  }
                }
              }

              gemm_nt_rvv_tiled_transposed_int8(
                  q_ptr + m_start * q_strideM,
                  k_trans_buf,
                  scores_block,
                  m_size,
                  n_size,
                  head_size,
                  q_strideM,
                  BLOCK_N,  // block_n for K_trans layout
                  BLOCK_N,  // ldc
                  scaling,
                  k_scale);

              for (int i = 0; i < m_size; ++i) {
                float* row = scores_block + i * BLOCK_N;
                for (int j = 0; j < n_size; ++j) {
                  if (!std::isfinite(row[j])) {
                    row[j] = 0.0f;
                  }
                }
              }
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

              for (int j = 0; j < n_size; ++j) {
                int token_idx = req_to_token[req_idx * max_context_len + n_start + j];
                const int8_t* v_src = v_buffer + token_idx * v_strideN + head_kv * v_strideH;
                int8_t* v_dst = v_buf + j * head_size_v;
                std::memcpy(v_dst, v_src, head_size_v * sizeof(int8_t));
              }

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

                // Vectorized max reduction
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
                  if (!std::isfinite(m_curr)) {
                    m_curr = -std::numeric_limits<float>::infinity();
                  }
                }
                float m_prev = m_acc[i];
                if (!std::isfinite(m_prev)) {
                  m_prev = -std::numeric_limits<float>::infinity();
                }
                float m_new = std::max(m_prev, m_curr);
                if (!std::isfinite(m_new)) {
                  m_new = -std::numeric_limits<float>::infinity();
                }
                float alpha = (m_prev == -std::numeric_limits<float>::infinity()) ? 0.0f : std::exp(m_prev - m_new);
                if (!std::isfinite(alpha)) {
                  alpha = 0.0f;
                }

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
                if (!std::isfinite(l_curr)) {
                  l_curr = 0.0f;
                }
                l_acc[i] += l_curr;
                if (!std::isfinite(l_acc[i])) {
                  l_acc[i] = 0.0f;
                }
              }

              gemm_nn_rvv_tiled_int8(
                  scores_block, v_buf, o_acc, m_size, n_size, head_size_v, BLOCK_N, head_size_v, v_scale);
            }
          }

          for (int n_start = 0; n_start < extend_len; n_start += BLOCK_N) {
            int n_size = std::min(BLOCK_N, extend_len - n_start);
            const scalar_t* k_ext_ptr = k_extend + start_loc * ke_strideN + head_kv * ke_strideH + n_start * ke_strideN;
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
                k_trans_buf_fp,
                scores_block,
                m_size,
                n_size,
                head_size,
                q_strideM,
                BLOCK_N,  // block_n for K_trans layout
                BLOCK_N,  // ldc
                scaling);

            for (int i = 0; i < m_size; ++i) {
              float* row = scores_block + i * BLOCK_N;
              for (int j = 0; j < n_size; ++j) {
                if (!std::isfinite(row[j])) {
                  row[j] = 0.0f;
                }
              }
            }
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

            for (int i = 0; i < m_size; ++i) {
              int m_idx = m_start + i;
              float* row = scores_block + i * BLOCK_N;
              if (n_start > m_idx) {
                size_t vl;
                for (int j = 0; j < n_size; j += vl) {
                  vl = __riscv_vsetvl_e32m4(n_size - j);
                  __riscv_vse32_v_f32m4(
                      row + j, __riscv_vfmv_v_f_f32m4(-std::numeric_limits<float>::infinity(), vl), vl);
                }
              } else if (n_start + n_size > m_idx + 1) {
                int valid = m_idx - n_start + 1;
                size_t vl;
                for (int j = valid; j < n_size; j += vl) {
                  vl = __riscv_vsetvl_e32m4(n_size - j);
                  __riscv_vse32_v_f32m4(
                      row + j, __riscv_vfmv_v_f_f32m4(-std::numeric_limits<float>::infinity(), vl), vl);
                }
              }
            }

            const scalar_t* v_ext_ptr = v_extend + start_loc * ve_strideN + head_kv * ve_strideH + n_start * ve_strideN;
            for (int j = 0; j < n_size; ++j) {
              const scalar_t* src_tok = v_ext_ptr + j * ve_strideN;
              scalar_t* dst_tok = v_buf_fp + j * head_size_v;
              std::memcpy(dst_tok, src_tok, head_size_v * sizeof(scalar_t));
            }

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
                if (!std::isfinite(m_curr)) {
                  m_curr = -std::numeric_limits<float>::infinity();
                }
              }
              float m_prev = m_acc[i];
              if (!std::isfinite(m_prev)) {
                m_prev = -std::numeric_limits<float>::infinity();
              }
              float m_new = std::max(m_prev, m_curr);
              if (!std::isfinite(m_new)) {
                m_new = -std::numeric_limits<float>::infinity();
              }
              float alpha = (m_prev == -std::numeric_limits<float>::infinity()) ? 0.0f : std::exp(m_prev - m_new);
              if (!std::isfinite(alpha)) {
                alpha = 0.0f;
              }

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
              if (!std::isfinite(l_curr)) {
                l_curr = 0.0f;
              }
              l_acc[i] += l_curr;
              if (!std::isfinite(l_acc[i])) {
                l_acc[i] = 0.0f;
              }
            }

            gemm_nn_rvv_tiled(scores_block, v_buf_fp, o_acc, m_size, n_size, head_size_v, BLOCK_N, head_size_v);
          }

          for (int i = 0; i < m_size; ++i) {
            scalar_t* out_row = o_ptr + (m_start + i) * q_strideM;
            float* acc_row = o_acc + i * head_size_v;
            // Ensure l_acc is finite and positive
            float l_val = l_acc[i];
            if (!std::isfinite(l_val) || l_val <= 0.0f) {
              l_val = 1e-6f;
            }
            float inv_l = 1.0f / (l_val + 1e-6f);
            if (!std::isfinite(inv_l)) {
              inv_l = 0.0f;
            }

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

}  // namespace

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
  int ke_strideN = k_extend.stride(0);
  int ke_strideH = k_extend.stride(1);
  int ve_strideN = v_extend.stride(0);
  int ve_strideH = v_extend.stride(1);

  int k_strideN = k_buffer.stride(0);
  int k_strideH = k_buffer.stride(1);
  int v_strideN = v_buffer.stride(0);
  int v_strideH = v_buffer.stride(1);

  bool is_prefix_skipped = k_buffer.size(1) != num_heads_kv;

  // We assume k_buffer is INT8 if this function is called
  const auto index_dtype = req_to_token.scalar_type();

  // Validate types for INT8 mode
  TORCH_CHECK(
      k_buffer.scalar_type() == at::kByte || k_buffer.scalar_type() == at::kChar,
      "extend_int8: k_buffer must be int8/uint8, got ",
      k_buffer.scalar_type());
  TORCH_CHECK(
      v_buffer.scalar_type() == at::kByte || v_buffer.scalar_type() == at::kChar,
      "extend_int8: v_buffer must be int8/uint8, got ",
      v_buffer.scalar_type());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q_extend.scalar_type(), "extend_attention_int8_cpu", [&] {
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
  });
}

#endif  // CPU_CAPABILITY_RVV
