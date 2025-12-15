// RISC-V Vector Extension (RVV) optimized decode attention kernels
// This file contains RVV specific implementations for decode attention

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

inline float rvv_reduce_sum_f32(const float* data, int64_t len) {
  size_t vl = __riscv_vsetvl_e32m8(len);
  if (vl == static_cast<size_t>(len)) {
    vfloat32m8_t vdata = __riscv_vle32_v_f32m8(data, vl);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
    vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m8_f32m1(vdata, vzero, vl);
    return __riscv_vfmv_f_s_f32m1_f32(vsum);
  }
  float sum = 0.f;
  for (int64_t i = 0; i < len; ++i) {
    sum += data[i];
  }
  return sum;
}

inline float rvv_reduce_max_f32(const float* data, int64_t len) {
  size_t vl = __riscv_vsetvl_e32m8(len);
  if (vl == static_cast<size_t>(len)) {
    vfloat32m8_t vdata = __riscv_vle32_v_f32m8(data, vl);
    vfloat32m1_t vmin = __riscv_vfmv_s_f_f32m1(-std::numeric_limits<float>::infinity(), 1);
    vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m8_f32m1(vdata, vmin, vl);
    return __riscv_vfmv_f_s_f32m1_f32(vmax);
  }
  float max_val = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < len; ++i) {
    max_val = std::max(max_val, data[i]);
  }
  return max_val;
}

// 1. Query-Key dot product
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
  for (int64_t m = 0; m < M; ++m) {
    const scalar_t* q_ptr_base = A + m * lda;

    int64_t n = 0;

#if HAS_ZVFH_DECODE
    if constexpr (std::is_same_v<scalar_t, at::Half>) {
      // FP16 Optimized Path
      size_t vl_max = __riscv_vsetvlmax_e16m2();

      for (; n + 3 < N; n += 4) {
        int64_t b_idx0 = indices[n];
        int64_t b_idx1 = indices[n + 1];
        int64_t b_idx2 = indices[n + 2];
        int64_t b_idx3 = indices[n + 3];

        const scalar_t* k_ptr0 = B + b_idx0 * ldb;
        const scalar_t* k_ptr1 = B + b_idx1 * ldb;
        const scalar_t* k_ptr2 = B + b_idx2 * ldb;
        const scalar_t* k_ptr3 = B + b_idx3 * ldb;

        // Initialize accumulators
        vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
        vfloat32m4_t v_acc1 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
        vfloat32m4_t v_acc2 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
        vfloat32m4_t v_acc3 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());

        for (int64_t k = 0; k < K; k += vl_max) {
          size_t vl = __riscv_vsetvl_e16m2(K - k);

          // Persistent Query: Loaded once for the block of 4 tokens
          vfloat16m2_t v_q = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(q_ptr_base + k), vl);

          // Token 0
          vfloat16m2_t v_k0 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr0 + k), vl);
          v_acc0 = __riscv_vfwmacc_vv_f32m4_tu(v_acc0, v_q, v_k0, vl);

          // Token 1
          vfloat16m2_t v_k1 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr1 + k), vl);
          v_acc1 = __riscv_vfwmacc_vv_f32m4_tu(v_acc1, v_q, v_k1, vl);

          // Token 2
          vfloat16m2_t v_k2 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr2 + k), vl);
          v_acc2 = __riscv_vfwmacc_vv_f32m4_tu(v_acc2, v_q, v_k2, vl);

          // Token 3
          vfloat16m2_t v_k3 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr3 + k), vl);
          v_acc3 = __riscv_vfwmacc_vv_f32m4_tu(v_acc3, v_q, v_k3, vl);
        }

        // Reductions
        vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vfloat32m1_t v_sum0 = __riscv_vfredusum_vs_f32m4_f32m1(v_acc0, v_zero, __riscv_vsetvlmax_e32m4());
        vfloat32m1_t v_sum1 = __riscv_vfredusum_vs_f32m4_f32m1(v_acc1, v_zero, __riscv_vsetvlmax_e32m4());
        vfloat32m1_t v_sum2 = __riscv_vfredusum_vs_f32m4_f32m1(v_acc2, v_zero, __riscv_vsetvlmax_e32m4());
        vfloat32m1_t v_sum3 = __riscv_vfredusum_vs_f32m4_f32m1(v_acc3, v_zero, __riscv_vsetvlmax_e32m4());

        C[m * ldc + n] = __riscv_vfmv_f_s_f32m1_f32(v_sum0) * scale;
        C[m * ldc + n + 1] = __riscv_vfmv_f_s_f32m1_f32(v_sum1) * scale;
        C[m * ldc + n + 2] = __riscv_vfmv_f_s_f32m1_f32(v_sum2) * scale;
        C[m * ldc + n + 3] = __riscv_vfmv_f_s_f32m1_f32(v_sum3) * scale;
      }
    }
#endif

    // Remainder loop / Non-FP16 loop
    for (; n < N; ++n) {
      int64_t b_idx = indices[n];
      const scalar_t* k_ptr = B + b_idx * ldb;

      float scalar_sum = 0.0f;

      if constexpr (std::is_same_v<scalar_t, float>) {
        // Pure FP32 path
        size_t vl_max = __riscv_vsetvlmax_e32m8();
        vfloat32m8_t v_acc = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);

        for (int64_t k = 0; k < K; k += vl_max) {
          size_t vl = __riscv_vsetvl_e32m8(K - k);
          vfloat32m8_t vq = __riscv_vle32_v_f32m8(q_ptr_base + k, vl);
          vfloat32m8_t vk = __riscv_vle32_v_f32m8(k_ptr + k, vl);
          v_acc = __riscv_vfmacc_vv_f32m8_tu(v_acc, vq, vk, vl);
        }
        vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vfloat32m1_t vred = __riscv_vfredusum_vs_f32m8_f32m1(v_acc, vzero, vl_max);
        scalar_sum = __riscv_vfmv_f_s_f32m1_f32(vred);

      } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH_DECODE
        // FP16 Widening path (Remainder)
        size_t vl_max = __riscv_vsetvlmax_e16m4();
        vfloat32m8_t v_acc = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());

        for (int64_t k = 0; k < K; k += vl_max) {
          size_t vl = __riscv_vsetvl_e16m4(K - k);
          vfloat16m4_t vq = __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16*>(q_ptr_base + k), vl);
          vfloat16m4_t vk = __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16*>(k_ptr + k), vl);
          v_acc = __riscv_vfwmacc_vv_f32m8_tu(v_acc, vq, vk, vl);
        }
        vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vfloat32m1_t vred = __riscv_vfredusum_vs_f32m8_f32m1(v_acc, vzero, __riscv_vsetvlmax_e32m8());
        scalar_sum = __riscv_vfmv_f_s_f32m1_f32(vred);
#else
        // Fallback Software Convert
        for (int64_t k = 0; k < K; ++k) {
          scalar_sum += static_cast<float>(q_ptr_base[k]) * static_cast<float>(k_ptr[k]);
        }
#endif
      } else {
        // BFloat16 etc
        for (int64_t k = 0; k < K; ++k) {
          scalar_sum += static_cast<float>(q_ptr_base[k]) * static_cast<float>(k_ptr[k]);
        }
      }

      C[m * ldc + n] = scalar_sum * scale;
    }
  }
}

// 2. Softmax
inline float exp_rvv(const float* __restrict__ scores, float* __restrict__ output, int64_t N, float m_i) {
  float sum = 0.0f;
  for (int64_t i = 0; i < N; ++i) {
    float x = scores[i] - m_i;
    float val = std::exp(x);
    output[i] = val;
    sum += val;
  }
  return sum;
}

// 3. Probabilities * Value aggregation
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

  // Block over head_dim using vector length
  for (int64_t d = 0; d < head_dim; d += vl_max) {
    size_t vl = __riscv_vsetvl_e32m8(head_dim - d);

    // Initialize accumulator for this chunk of head dimensions
    vfloat32m8_t vacc = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    int64_t n = 0;

    // Unroll N (Tokens) by 4
    for (; n + 3 < N; n += 4) {
      float p0 = probs[n];
      float p1 = probs[n + 1];
      float p2 = probs[n + 2];
      float p3 = probs[n + 3];

      int64_t v_idx0 = indices[n];
      int64_t v_idx1 = indices[n + 1];
      int64_t v_idx2 = indices[n + 2];
      int64_t v_idx3 = indices[n + 3];

      // Process 4 tokens
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
      } else {
        // BFloat16/Other
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
        // Load FP16
        vfloat16m4_t vv_16 = __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16*>(v_ptr), vl);

        // Widening MAC: vacc += prob * vv_16
        // vfwmacc here uses float16 * scalar_float? No such intrinsic.
        // We must convert vv_16 to float32 first.
        // There is vfwmacc_vf but it might expect scalar to be fp16 too?
        // RISC-V V spec: vfwmacc.vf (wide accumulate vector-scalar)
        // vd[i] += f[rs1] * vs2[i]
        // If vd is f32, f[rs1] is f16, vs2 is f16.
        // But prob is float(32).
        // So we need to convert vv_16 to vv_32, then mac.
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
      } else {
        // BFloat16
        float temp[256];
        for (size_t j = 0; j < vl; ++j)
          temp[j] = static_cast<float>(v_ptr[j]);
        vfloat32m8_t vv = __riscv_vle32_v_f32m8(temp, vl);
        vacc = __riscv_vfmacc_vf_f32m8(vacc, prob, vv, vl);
      }
    }

    // Load current output, scale it, and add accumulator
    vfloat32m8_t vout = __riscv_vle32_v_f32m8(output + d, vl);
    vout = __riscv_vfmul_vf_f32m8(vout, scale, vl);
    vout = __riscv_vfadd_vv_f32m8(vout, vacc, vl);
    __riscv_vse32_v_f32m8(output + d, vout, vl);
  }
}

#endif  // CPU_CAPABILITY_RVV

}  // namespace

// 4. Accumulate KV splits and write to output
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
          // Vectorized update: acc = acc * m_delta + tv * e_logic
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

      // Final copy to output with scaling: output = acc * (1/s_prime)
      float scale = 1.0f / s_prime;

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
        } else {
          // BFloat16 or others
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

// 5. FlashDecoding Kernel

template <typename scalar_t, typename index_t, int64_t BLOCK_N>
void decode_attention_kernel_rvv(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ k_buffer,
    const scalar_t* __restrict__ v_buffer,
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
    int64_t max_total_num_tokens) {
  const bool has_logit_cap = logit_cap > 0;
  float rlogit_cap = has_logit_cap ? 1 / logit_cap : 0.f;

  // strides for accumulation
  const int64_t l_stride1 = num_kv_splits * (head_size_v + 1);
  const int64_t l_stride2 = head_size_v + 1;

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

      // get key/value
      int64_t seq_len_kv = seq_lens[bs];
      int64_t req_pool_id = req_pool_indices[bs];

      const int64_t SPLIT_SIZE = div_up(seq_len_kv, num_kv_splits);
      const int64_t kv_start = kv_id * SPLIT_SIZE;
      const int64_t kv_end = std::min(kv_start + SPLIT_SIZE, seq_len_kv);

      float m_prime = -std::numeric_limits<float>::infinity();
      float s_prime = 0.f;

      // get v_prime, and init to zero
      float* __restrict__ v_prime = attn_logits + i * (head_size_v + 1);
      // fill_stub is generic, but we can use memset for 0
      std::memset(v_prime, 0, head_size_v * sizeof(float));

      // loop over K and V sequence with BLOCK_N
      for (int64_t n = kv_start; n < kv_end; n += BLOCK_N) {
        int64_t n_size = std::min(BLOCK_N, kv_end - n);

        // calculate s_i <- scale * Q @ K
        index_gemm_kernel_nt_rvv<scalar_t, index_t>(
            /* A   */ q_ptr,
            /* B   */ k_buffer + head_id * k_strideH,
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

        if (has_logit_cap) {
          // TODO: Vectorize tanh if needed, for now scalar loop or use sleef if available
          // But since we are in RVV file, we could implement tanh_rvv later
          for (int j = 0; j < n_size; ++j) {
            s_i[j] = logit_cap * std::tanh(s_i[j] * rlogit_cap);
          }
        }

        // m_i: max value per row
        float m_i = std::max(rvv_reduce_max_f32(s_i, n_size), m_prime);

        // m_delta <- exp(m' - m_i)
        float m_delta = std::exp(m_prime - m_i);

        // s_delta <- exp(s_i - m_i)
        // exp_rvv computes exp(s_i - m_i) and returns sum
        float local_sum = exp_rvv(s_i, s_delta, n_size, m_i);

        s_prime *= m_delta;
        s_prime += local_sum;

        m_prime = m_i;

        // calculate V' <- s_delta @ V + V' * m_delta
        // output = output * m_delta + sum(s_delta * values)
        prob_value_aggregate_rvv<scalar_t, index_t>(
            s_delta,
            v_buffer + head_id * v_strideH,
            v_prime,
            req_to_token + req_pool_id * max_context_len + n,
            n_size,
            head_size_v,
            v_strideN,
            m_delta,
            max_total_num_tokens);
      }

      // update m_prime and s_prime
      // Normalize v_prime by s_prime
      if (kv_end > kv_start) {
        float s = 1.0f / s_prime;

        // Vectorized scaling: v_prime = v_prime * s
        // We reuse head_size_v for loop limit
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

  // Accumulate splits and write to output
  decode_accumulate_kv_splits_rvv(
      output, attn_logits, batches, num_heads, head_size_v, num_kv_splits, l_stride1, l_stride2);
}
