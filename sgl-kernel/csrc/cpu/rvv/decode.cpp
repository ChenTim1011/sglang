/*
 * 1. Decoding Kernel: Primary entry point for decode phase attention.
 * 2. Index GEMM: Matrix multiplication where indices are used for lookups (e.g., gathering KV cache).
 * 3. Precision Support: Handing FP32, FP16, and Int8 data types via specialized kernels.
 */

#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

// Clang 19+ version check
#if !defined(__clang__) || !defined(__clang_major__) || __clang_major__ < 19
#error "RVV backend requires Clang 19 or later. Please use Clang 19 or later to compile this file."
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#include "common.h"
#include "gemm.h"
#include "vec.h"
#include "vector_helpers.h"
#include "vector_math.h"

namespace sgl_kernel {
namespace rvv {

#if defined(CPU_CAPABILITY_RVV)

// =============================================================================
// Kernel: Index GEMM
// =============================================================================

template <typename scalar_t, typename index_t, size_t BLOCK_N, typename B_TYPE = scalar_t>
struct IndexGemmKernel {
  static void
  run(const scalar_t* __restrict__ q_ptr,
      const B_TYPE* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      float scale,
      int64_t K,
      int64_t ldb) {
#pragma unroll
    for (size_t i = 0; i < BLOCK_N; ++i) {
      IndexGemmKernel<scalar_t, index_t, 1, B_TYPE>::run(q_ptr, B, C + i, indices + i, scale, K, ldb);
    }
  }
};

// N=1 Specialization
template <typename scalar_t, typename index_t, typename B_TYPE>
struct IndexGemmKernel<scalar_t, index_t, 1, B_TYPE> {
  static void
  run(const scalar_t* __restrict__ q_ptr,
      const B_TYPE* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      float scale,
      int64_t K,
      int64_t ldb) {
    const B_TYPE* k_ptr0 = B + indices[0] * ldb;
    size_t vl_max = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);

    size_t vl;
    float scratch[128];
    for (int64_t k = 0; k < K; k += vl) {
      vl = __riscv_vsetvl_e32m4(K - k);
      vfloat32m4_t v_q = load_as_float_m4(q_ptr + k, vl, scratch);
      vfloat32m4_t v_k0 = load_as_float_m4(k_ptr0 + k, vl, scratch);
      acc0 = __riscv_vfmacc_vv_f32m4_tu(acc0, v_q, v_k0, vl);
    }
    C[0] = reduce_sum_f32m4(acc0, vl_max) * scale;
  }
};

// N=2 Specialization
template <typename scalar_t, typename index_t, typename B_TYPE>
struct IndexGemmKernel<scalar_t, index_t, 2, B_TYPE> {
  static void
  run(const scalar_t* __restrict__ q_ptr,
      const B_TYPE* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      float scale,
      int64_t K,
      int64_t ldb) {
    const B_TYPE* k_ptr0 = B + indices[0] * ldb;
    const B_TYPE* k_ptr1 = B + indices[1] * ldb;
    size_t vl_max = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
    vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);

    size_t vl;
    float scratch[128];
    for (int64_t k = 0; k < K; k += vl) {
      vl = __riscv_vsetvl_e32m4(K - k);
      vfloat32m4_t v_q = load_as_float_m4(q_ptr + k, vl, scratch);

      vfloat32m4_t v_k0 = load_as_float_m4(k_ptr0 + k, vl, scratch);
      acc0 = __riscv_vfmacc_vv_f32m4_tu(acc0, v_q, v_k0, vl);

      vfloat32m4_t v_k1 = load_as_float_m4(k_ptr1 + k, vl, scratch);
      acc1 = __riscv_vfmacc_vv_f32m4_tu(acc1, v_q, v_k1, vl);
    }
    C[0] = reduce_sum_f32m4(acc0, vl_max) * scale;
    C[1] = reduce_sum_f32m4(acc1, vl_max) * scale;
  }
};

// N=4 Specialization
template <typename scalar_t, typename index_t, typename B_TYPE>
struct IndexGemmKernel<scalar_t, index_t, 4, B_TYPE> {
  static void
  run(const scalar_t* __restrict__ q_ptr,
      const B_TYPE* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      float scale,
      int64_t K,
      int64_t ldb) {
    const B_TYPE* k_ptr0 = B + indices[0] * ldb;
    const B_TYPE* k_ptr1 = B + indices[1] * ldb;
    const B_TYPE* k_ptr2 = B + indices[2] * ldb;
    const B_TYPE* k_ptr3 = B + indices[3] * ldb;

    size_t vl_max = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
    vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
    vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
    vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);

    size_t vl;
    float scratch[128];
    for (int64_t k = 0; k < K; k += vl) {
      vl = __riscv_vsetvl_e32m4(K - k);
      vfloat32m4_t v_q = load_as_float_m4(q_ptr + k, vl, scratch);

      vfloat32m4_t v_k;
      v_k = load_as_float_m4(k_ptr0 + k, vl, scratch);
      acc0 = __riscv_vfmacc_vv_f32m4_tu(acc0, v_q, v_k, vl);
      v_k = load_as_float_m4(k_ptr1 + k, vl, scratch);
      acc1 = __riscv_vfmacc_vv_f32m4_tu(acc1, v_q, v_k, vl);
      v_k = load_as_float_m4(k_ptr2 + k, vl, scratch);
      acc2 = __riscv_vfmacc_vv_f32m4_tu(acc2, v_q, v_k, vl);
      v_k = load_as_float_m4(k_ptr3 + k, vl, scratch);
      acc3 = __riscv_vfmacc_vv_f32m4_tu(acc3, v_q, v_k, vl);
    }
    C[0] = reduce_sum_f32m4(acc0, vl_max) * scale;
    C[1] = reduce_sum_f32m4(acc1, vl_max) * scale;
    C[2] = reduce_sum_f32m4(acc2, vl_max) * scale;
    C[3] = reduce_sum_f32m4(acc3, vl_max) * scale;
  }
};

// Dispatch Templates
template <typename scalar_t, typename index_t, size_t BLOCK_N_SIZE>
inline void index_gemm_kernel_nt_rvv_m1_template(
    const scalar_t* __restrict__ q_ptr,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    float scale,
    int64_t K,
    int64_t ldb,
    int64_t max_tokens) {
  IndexGemmKernel<scalar_t, index_t, BLOCK_N_SIZE>::run(q_ptr, B, C, indices, scale, K, ldb);
}

template <typename scalar_t, typename index_t, size_t BLOCK_N_SIZE>
inline void index_gemm_kernel_nt_rvv_m1_template_int8(
    const scalar_t* __restrict__ q_ptr,
    const int8_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    float scale,
    int64_t K,
    int64_t ldb,
    int64_t max_tokens) {
  IndexGemmKernel<scalar_t, index_t, BLOCK_N_SIZE, int8_t>::run(q_ptr, B, C, indices, scale, K, ldb);
}

// =============================================================================
// Kernel: Value Aggregate
// =============================================================================

template <typename scalar_t, typename index_t, typename V_TYPE>
inline void prob_value_aggregate_rvv_impl(
    const float* __restrict__ probs,
    const V_TYPE* __restrict__ values,
    float* __restrict__ output,
    const index_t* __restrict__ indices,
    int64_t N,
    int64_t head_dim,
    int64_t v_strideN,
    float scale,
    float v_scale_val = 1.0f) {
  if (N == 0 || head_dim == 0) return;
  size_t vl_max = __riscv_vsetvlmax_e32m8();

  for (int64_t d = 0; d < head_dim; d += vl_max) {
    size_t vl = __riscv_vsetvl_e32m8(head_dim - d);
    vfloat32m8_t vacc = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    for (int64_t n = 0; n < N; ++n) {
      float prob = probs[n];
      int64_t v_idx = indices[n];
      const V_TYPE* v_ptr = values + v_idx * v_strideN + d;

      vfloat32m8_t v_val_f32;

      if constexpr (std::is_same_v<V_TYPE, int8_t>) {
        vint8m2_t v_i8 = __riscv_vle8_v_i8m2(v_ptr, vl);
        vint16m4_t v_i16 = __riscv_vsext_vf2_i16m4(v_i8, vl);
        vint32m8_t v_i32 = __riscv_vsext_vf2_i32m8(v_i16, vl);
        v_val_f32 = __riscv_vfcvt_f_x_v_f32m8(v_i32, vl);
      } else {
        if constexpr (std::is_same_v<V_TYPE, float>) {
          v_val_f32 = __riscv_vle32_v_f32m8(reinterpret_cast<const float*>(v_ptr), vl);
        } else if constexpr (std::is_same_v<V_TYPE, at::Half>) {
          vfloat16m4_t v_f16 = __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16*>(v_ptr), vl);
          v_val_f32 = __riscv_vfwcvt_f_f_v_f32m8(v_f16, vl);
        } else if constexpr (std::is_same_v<V_TYPE, at::BFloat16>) {
          v_val_f32 = bf16_to_f32m8(reinterpret_cast<const uint16_t*>(v_ptr), vl);
        } else {
          float temp[256];
          for (size_t j = 0; j < vl; ++j)
            temp[j] = static_cast<float>(v_ptr[j]);
          v_val_f32 = __riscv_vle32_v_f32m8(temp, vl);
        }
      }

      vacc = __riscv_vfmacc_vf_f32m8(vacc, prob, v_val_f32, vl);
    }

    if constexpr (std::is_same_v<V_TYPE, int8_t>) {
      vacc = __riscv_vfmul_vf_f32m8(vacc, v_scale_val, vl);
    }

    vfloat32m8_t vout = __riscv_vle32_v_f32m8(output + d, vl);
    vout = __riscv_vfmul_vf_f32m8(vout, scale, vl);
    vout = __riscv_vfadd_vv_f32m8(vout, vacc, vl);
    float scratch[256];
    store_from_float_m8(output + d, vout, vl, scratch);
  }
}

// =============================================================================
// Helper: Reduce Splits
// =============================================================================

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

            vacc = __riscv_vfmul_vv_f32m8(vacc, vm_delta, vl);
            vacc = __riscv_vfmacc_vv_f32m8(vacc, ve_logic, vtv, vl);

            __riscv_vse32_v_f32m8(acc + d, vacc, vl);
          }
        }

        s_prime = s_prime * m_delta + e_logic;
        m_prime = m_i;
      }

      float scale = (s_prime > 0.0f && std::isfinite(s_prime)) ? (1.0f / s_prime) : 0.0f;

      size_t vl_max = __riscv_vsetvlmax_e32m8();
      float scratch[256];
      for (int64_t d = 0; d < head_size_v; d += vl_max) {
        size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
        vfloat32m8_t vacc = __riscv_vle32_v_f32m8(acc + d, vl);
        vfloat32m8_t vscale = __riscv_vfmv_v_f_f32m8(scale, vl);
        vacc = __riscv_vfmul_vv_f32m8(vacc, vscale, vl);

        store_from_float_m8(output + i * head_size_v + d, vacc, vl, scratch);
      }
    }
  });
}

// =============================================================================
// Adapter: Decode Attention (Generic & Grouped)
// =============================================================================

template <typename scalar_t, typename index_t, int64_t BLOCK_N_IGNORED>
void decode_attention_grouped_kernel(
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
  // Implementation of the main loop
  bool is_int8 = (k_scale != 1.0f);
  int64_t group_size = num_heads / num_heads_kv;
  bool has_logit_cap = (logit_cap > 0.0f);
  float rlogit_cap = has_logit_cap ? (1.0f / logit_cap) : 0.0f;

  // Strides for accumulation
  const int64_t l_stride1 = num_kv_splits * (head_size_v + 1);
  const int64_t l_stride2 = head_size_v + 1;

  constexpr int64_t TARGET_BLOCK_N = 4;  // Reduced to 4 for debugging/correctness

  at::parallel_for(0, batches * num_heads * num_kv_splits, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, head_id{0}, kv_id{0};
    data_index_init(begin, bs, batches, head_id, num_heads, kv_id, num_kv_splits);

    // Scratch buffer for s_prime/s_delta
    alignas(64) float s_i[TARGET_BLOCK_N];

    for (int64_t i = begin; i < end; ++i) {
      const scalar_t* q_ptr = query + bs * q_strideM + head_id * q_strideH;

      int64_t head_kv = head_id / group_size;
      int64_t seq_len_kv = seq_lens[bs];
      int64_t req_pool_id = req_pool_indices[bs];

      const int64_t SPLIT_SIZE = (seq_len_kv + num_kv_splits - 1) / num_kv_splits;
      const int64_t kv_start = kv_id * SPLIT_SIZE;
      const int64_t kv_end = std::min(kv_start + SPLIT_SIZE, seq_len_kv);

      float m_prime = -std::numeric_limits<float>::infinity();
      float s_prime = 0.f;

      float* v_prime = attn_logits + i * (head_size_v + 1);
      std::memset(v_prime, 0, head_size_v * sizeof(float));

      // Loop over tokens in blocks
      for (int64_t n = kv_start; n < kv_end; n += TARGET_BLOCK_N) {
        int64_t n_size = std::min(TARGET_BLOCK_N, kv_end - n);

// 1. GEMM: Q * K -> s_i
#define CASE_GEMM(N)                                                               \
  case N: {                                                                        \
    constexpr size_t N_SIZE = N;                                                   \
    const index_t* cur_indices = req_to_token + req_pool_id * max_context_len + n; \
    if (is_int8) {                                                                 \
      index_gemm_kernel_nt_rvv_m1_template_int8<scalar_t, index_t, N_SIZE>(        \
          q_ptr,                                                                   \
          reinterpret_cast<const int8_t*>(k_buffer) + head_kv * k_strideH,         \
          s_i,                                                                     \
          cur_indices,                                                             \
          scaling * k_scale,                                                       \
          head_size,                                                               \
          k_strideN,                                                               \
          max_total_num_tokens);                                                   \
    } else {                                                                       \
      index_gemm_kernel_nt_rvv_m1_template<scalar_t, index_t, N_SIZE>(             \
          q_ptr,                                                                   \
          reinterpret_cast<const scalar_t*>(k_buffer) + head_kv * k_strideH,       \
          s_i,                                                                     \
          cur_indices,                                                             \
          scaling,                                                                 \
          head_size,                                                               \
          k_strideN,                                                               \
          max_total_num_tokens);                                                   \
    }                                                                              \
    break;                                                                         \
  }

        switch (n_size) {
          CASE_GEMM(1)
          CASE_GEMM(2)
          CASE_GEMM(3)
          CASE_GEMM(4)
        }
#undef CASE_GEMM

        // 2. Logit Cap & Softmax parts
        if (has_logit_cap) {
          for (int k = 0; k < n_size; ++k)
            s_i[k] = std::tanh(s_i[k] * rlogit_cap) * logit_cap;
        }

        // Max reduction
        float m_i = (n_size > 0) ? std::max(rvv_reduce_max_f32(s_i, n_size), m_prime) : m_prime;
        if (!std::isfinite(m_i)) m_i = m_prime;

        float m_delta = std::isfinite(m_i) ? std::exp(m_prime - m_i) : 0.0f;

        // exp(s_i - m_i) and sum
        float local_sum = 0.f;
        for (int k = 0; k < n_size; ++k) {
          s_i[k] = std::exp(s_i[k] - m_i);  // s_delta
          local_sum += s_i[k];
        }

        s_prime = s_prime * m_delta + local_sum;
        m_prime = m_i;

        // 3. Aggregate: V' = s_delta * V + V' * m_delta

        if (m_delta != 1.0f) {
          size_t vl_max = __riscv_vsetvlmax_e32m8();
          for (int64_t d = 0; d < head_size_v; d += vl_max) {
            size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
            vfloat32m8_t vv = __riscv_vle32_v_f32m8(v_prime + d, vl);
            vv = __riscv_vfmul_vf_f32m8(vv, m_delta, vl);
            __riscv_vse32_v_f32m8(v_prime + d, vv, vl);
          }
        }

        // Accumulate new values
        const index_t* cur_indices = req_to_token + req_pool_id * max_context_len + n;

        if (is_int8) {
          prob_value_aggregate_rvv_impl<scalar_t, index_t, int8_t>(
              s_i,
              reinterpret_cast<const int8_t*>(v_buffer) + head_kv * v_strideH,
              v_prime,
              cur_indices,
              n_size,
              head_size_v,
              v_strideN,
              1.0f,
              v_scale);
        } else {
          prob_value_aggregate_rvv_impl<scalar_t, index_t, scalar_t>(
              s_i,
              reinterpret_cast<const scalar_t*>(v_buffer) + head_kv * v_strideH,
              v_prime,
              cur_indices,
              n_size,
              head_size_v,
              v_strideN,
              1.0f);
        }
      }  // end loop over tokens

      // Final normalization
      if (kv_end > kv_start && s_prime > 0.0f && std::isfinite(s_prime)) {
        float s = 1.0f / s_prime;
        size_t vl_max = __riscv_vsetvlmax_e32m8();
        for (int64_t d = 0; d < head_size_v; d += vl_max) {
          size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
          vfloat32m8_t vval = __riscv_vle32_v_f32m8(v_prime + d, vl);
          vval = __riscv_vfmul_vf_f32m8(vval, s, vl);
          __riscv_vse32_v_f32m8(v_prime + d, vval, vl);
        }
        v_prime[head_size_v] = m_prime + std::log(s_prime);
      }

      data_index_step(bs, batches, head_id, num_heads, kv_id, num_kv_splits);
    }
  });

  // Final merge of splits
  decode_accumulate_kv_splits_rvv(
      output, attn_logits, batches, num_heads, head_size_v, num_kv_splits, l_stride1, l_stride2);
}

// =============================================================================
// Wrappers
// =============================================================================

template <typename scalar_t, typename index_t, int64_t BLOCK_N_IGNORED>
void decode_attention_kernel(
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
  // Check if we can use RVV
  if (get_rvv_vlen() < 128) {
    // Fallback? or just error
    // For now assume dispatched here means capability checked
  }
  // MHA delegating to Grouped
  decode_attention_grouped_kernel<scalar_t, index_t, BLOCK_N_IGNORED>(
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
      num_heads,  // kv_heads = num_heads
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
  decode_attention_grouped_kernel<scalar_t, index_t, 256L>(
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
  decode_attention_grouped_kernel<scalar_t, index_t, 256L>(
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
      num_heads,  // MHA
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
// =============================================================================
// Helper: MLA Kernel
// =============================================================================

template <typename scalar_t, typename index_t, int64_t BLOCK_N>
void decode_attention_mla_kernel(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    scalar_t* __restrict__ buffer,
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
    int64_t buffer_size_per_thread) {
  // Dynamic BLOCK_H: Process up to 8 heads at once.
  const int64_t TARGET_BLOCK_H = 8;
  const int64_t BLOCK_H = std::min(num_heads, TARGET_BLOCK_H);

  // strides
  const int64_t l_stride0 = num_heads * num_kv_splits * (head_size_v + 1);
  const int64_t l_stride1 = num_kv_splits * (head_size_v + 1);
  const int64_t l_stride2 = head_size_v + 1;

  TORCH_CHECK(logit_cap == 0.f, "RVV decode MLA: expect no logit_cap.");

  // partition the heads into blocks for parallel
  const int64_t num_blocks = div_up(num_heads, BLOCK_H);

  // Reuse existing parallel infrastructure
  at::parallel_for(0, batches * num_blocks * num_kv_splits, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, block_id{0}, kv_id{0};
    data_index_init(begin, bs, batches, block_id, num_blocks, kv_id, num_kv_splits);

    // Scratch buffers
    alignas(64) float s_prime[TARGET_BLOCK_H];
    alignas(64) float m_prime[TARGET_BLOCK_H];
    alignas(64) float m_delta[TARGET_BLOCK_H];

    // Todo: tune MAX_BLOCK_SIZE
    constexpr int64_t MAX_BLOCK_SIZE = 8 * 256;
    alignas(64) float s_i[MAX_BLOCK_SIZE];

    for (int64_t i = begin; i < end; ++i) {
      const int64_t h_start = block_id * BLOCK_H;
      const int64_t h_end = std::min(block_id * BLOCK_H + BLOCK_H, num_heads);
      const int64_t h_size = h_end - h_start;

      const scalar_t* __restrict__ q_ptr = query + bs * q_strideM + h_start * q_strideH;

      int64_t seq_len_kv = seq_lens[bs];
      int64_t req_pool_id = req_pool_indices[bs];

      const int64_t SPLIT_SIZE = div_up(seq_len_kv, num_kv_splits);
      const int64_t kv_start = kv_id * SPLIT_SIZE;
      const int64_t kv_end = std::min(kv_start + SPLIT_SIZE, seq_len_kv);

      // Init accumulators
      for (int h = 0; h < h_size; ++h) {
        s_prime[h] = 0.f;
        m_prime[h] = -std::numeric_limits<float>::infinity();
      }

      // v_prime inited to zero
      float* __restrict__ v_prime_base = attn_logits + bs * l_stride0 + h_start * l_stride1 + kv_id * l_stride2;
      for (int64_t h = 0; h < h_size; ++h) {
        float* v_ptr = v_prime_base + h * l_stride1;
        std::memset(v_ptr, 0, head_size_v * sizeof(float));
      }

      // Loop over K/V
      for (int64_t n = kv_start; n < kv_end; n += BLOCK_N) {
        int64_t n_size = std::min(BLOCK_N, kv_end - n);

        // 1. Compute s_i[h, n] for h in [0..h_size), n in [0..n_size)
        // Optim: Parallelize over h_size
        alignas(64) float scratch[256];
        size_t vl_max = __riscv_vsetvlmax_e32m4();

        // Clear s_i
        std::memset(s_i, 0, h_size * n_size * sizeof(float));

        for (int64_t j = 0; j < n_size; ++j) {
          index_t idx = req_to_token[req_pool_id * max_context_len + n + j];
          const scalar_t* k_ptr_j =
              reinterpret_cast<const scalar_t*>(k_buffer) + /*head=0*/ 0 * k_strideH + idx * k_strideN;

          // For each head h, compute dot(Q[h], K[j])
          for (int64_t h = 0; h < h_size; ++h) {
            const scalar_t* q_ptr_h = q_ptr + h * q_strideH;

            // Vector dot product: q_ptr_h vs k_ptr_j, length=head_size
            vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vl_max);
            for (int64_t d = 0; d < head_size;) {
              size_t vl = __riscv_vsetvl_e32m4(head_size - d);
              // Load Q (contiguous)
              vfloat32m4_t vq;
              if constexpr (std::is_same_v<scalar_t, at::Half>) {
                vfloat16m2_t vq_f16 = __riscv_vle16_v_f16m2((const _Float16*)q_ptr_h + d, vl);
                vq = __riscv_vfwcvt_f_f_v_f32m4(vq_f16, vl);
              } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
                vq = bf16_to_f32m4((const uint16_t*)q_ptr_h + d, vl);
              } else {
                // Fallback float
                vq = __riscv_vle32_v_f32m4((const float*)q_ptr_h + d, vl);
              }

              // Load K (contiguous)
              vfloat32m4_t vk;
              if constexpr (std::is_same_v<scalar_t, at::Half>) {
                vfloat16m2_t vk_f16 = __riscv_vle16_v_f16m2((const _Float16*)k_ptr_j + d, vl);
                vk = __riscv_vfwcvt_f_f_v_f32m4(vk_f16, vl);
              } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
                vk = bf16_to_f32m4((const uint16_t*)k_ptr_j + d, vl);
              } else {
                vk = __riscv_vle32_v_f32m4((const float*)k_ptr_j + d, vl);
              }

              vsum = __riscv_vfmacc_vv_f32m4(vsum, vq, vk, vl);
              d += vl;
            }
            float dot = reduce_sum_f32m4(vsum, vl_max);
            s_i[h * BLOCK_N + j] = dot * scaling;
          }
        }

        // 2. Softmax Logic
        for (int64_t h = 0; h < h_size; ++h) {
          float* s_row = s_i + h * BLOCK_N;

          // max
          float local_max = -std::numeric_limits<float>::infinity();
          for (int j = 0; j < n_size; ++j)
            local_max = std::max(local_max, s_row[j]);

          float m_i = std::max(local_max, m_prime[h]);
          float m_d = std::exp(m_prime[h] - m_i);
          m_delta[h] = m_d;

          // exp & sum
          float row_sum = 0.f;
          for (int j = 0; j < n_size; ++j) {
            s_row[j] = std::exp(s_row[j] - m_i);
            row_sum += s_row[j];
          }

          s_prime[h] = s_prime[h] * m_d + row_sum;
          m_prime[h] = m_i;

          // Update V'
          // V' = V' * m_delta + s_delta @ V
          // Rescale current V'
          float* v_ptr = v_prime_base + h * l_stride1;
          size_t vl_max_v = __riscv_vsetvlmax_e32m8();
          for (int d = 0; d < head_size_v; d += vl_max_v) {
            size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
            vfloat32m8_t vv = __riscv_vle32_v_f32m8(v_ptr + d, vl);
            vv = __riscv_vfmul_vf_f32m8(vv, m_d, vl);
            __riscv_vse32_v_f32m8(v_ptr + d, vv, vl);
          }

          // Accumulate s_delta @ V
          // s_row: [n_size], V: [n_size, head_size_v] gathered
          for (int j = 0; j < n_size; ++j) {
            float prob = s_row[j];
            index_t idx = req_to_token[req_pool_id * max_context_len + n + j];
            const scalar_t* val_ptr = reinterpret_cast<const scalar_t*>(v_buffer) + /*head=0*/ 0 * v_strideH +
                                      idx * v_strideN;  // MLA shares KV

            for (int d = 0; d < head_size_v; d += vl_max_v) {
              size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
              vfloat32m8_t vout = __riscv_vle32_v_f32m8(v_ptr + d, vl);

              // Load Value
              vfloat32m8_t vval;
              if constexpr (std::is_same_v<scalar_t, at::Half>) {
                vfloat16m4_t vf16 = __riscv_vle16_v_f16m4((const _Float16*)val_ptr + d, vl);
                vval = __riscv_vfwcvt_f_f_v_f32m8(vf16, vl);
              } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
                vval = bf16_to_f32m8((const uint16_t*)val_ptr + d, vl);
              } else {
                vval = __riscv_vle32_v_f32m8((const float*)val_ptr + d, vl);
              }

              vout = __riscv_vfmacc_vf_f32m8(vout, prob, vval, vl);
              __riscv_vse32_v_f32m8(v_ptr + d, vout, vl);
            }
          }
        }
      }  // end loop over tokens (N)

      // Final Cleanup loop
      if (kv_end > kv_start) {
        for (int h = 0; h < h_size; ++h) {
          float sp = s_prime[h];
          float mp = m_prime[h];
          float* v_ptr = v_prime_base + h * l_stride1;

          if (sp > 0.0f && std::isfinite(sp)) {
            float s = 1.0f / sp;
            size_t vl_max_v = __riscv_vsetvlmax_e32m8();
            for (int d = 0; d < head_size_v; d += vl_max_v) {
              size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
              vfloat32m8_t vv = __riscv_vle32_v_f32m8(v_ptr + d, vl);
              vv = __riscv_vfmul_vf_f32m8(vv, s, vl);
              __riscv_vse32_v_f32m8(v_ptr + d, vv, vl);
            }
            v_ptr[head_size_v] = mp + std::log(sp);
          }
        }
      }

      data_index_step(bs, batches, block_id, num_blocks, kv_id, num_kv_splits);
    }
  });

  // Final merge of splits
  decode_accumulate_kv_splits_rvv(
      output, attn_logits, batches, num_heads, head_size_v, num_kv_splits, l_stride1, l_stride2);
}

// =============================================================================
// Helper: Decode Set KV Buffer (RVV Implementation)
// =============================================================================

namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ src, int64_t size) {
  copy_stub_rvv<scalar_t>(out, src, size);
}

template <typename buffer_t>
void decode_set_kv_buffer_int8_copy(
    buffer_t* __restrict__ k_buffer,
    buffer_t* __restrict__ v_buffer,
    const buffer_t* __restrict__ key,
    const buffer_t* __restrict__ value,
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
    int64_t nv_strideH) {
  at::parallel_for(0, batches * num_heads_kv, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, head_kv_id{0};
    data_index_init(begin, bs, batches, head_kv_id, num_heads_kv);

    for (int64_t i = begin; i < end; i++) {
      int64_t loc_val = loc[bs];
      buffer_t* k_buffer_ptr = k_buffer + loc_val * k_strideN + head_kv_id * k_strideH;
      const buffer_t* new_key_ptr = key + bs * nk_strideN + head_kv_id * nk_strideH;
      copy_stub_rvv<buffer_t>(k_buffer_ptr, new_key_ptr, head_size);

      buffer_t* v_buffer_ptr = v_buffer + loc_val * v_strideN + head_kv_id * v_strideH;
      const buffer_t* new_value_ptr = value + bs * nv_strideN + head_kv_id * nv_strideH;
      copy_stub_rvv<buffer_t>(v_buffer_ptr, new_value_ptr, head_size_v);
      data_index_step(bs, batches, head_kv_id, num_heads_kv);
    }
  });
}

template <typename scalar_t, typename buffer_t>
void decode_set_kv_buffer_int8_quantize(
    buffer_t* __restrict__ k_buffer,
    buffer_t* __restrict__ v_buffer,
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
    float k_scale,
    float v_scale) {
  at::parallel_for(0, batches * num_heads_kv, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, head_kv_id{0};
    data_index_init(begin, bs, batches, head_kv_id, num_heads_kv);

    for (int64_t i = begin; i < end; i++) {
      int64_t loc_val = loc[bs];
      int8_t* k_buffer_ptr = reinterpret_cast<int8_t*>(k_buffer + loc_val * k_strideN + head_kv_id * k_strideH);
      const scalar_t* new_key_ptr = key + bs * nk_strideN + head_kv_id * nk_strideH;

      int8_t* v_buffer_ptr = reinterpret_cast<int8_t*>(v_buffer + loc_val * v_strideN + head_kv_id * v_strideH);
      const scalar_t* new_value_ptr = value + bs * nv_strideN + head_kv_id * nv_strideH;

      quantize_and_copy_rvv(k_buffer_ptr, new_key_ptr, head_size, k_scale);
      quantize_and_copy_rvv(v_buffer_ptr, new_value_ptr, head_size_v, v_scale);

      data_index_step(bs, batches, head_kv_id, num_heads_kv);
    }
  });
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
      std::memcpy(k_buffer_ptr, new_key_ptr, head_size * sizeof(scalar_t));

      if (!is_mla) {
        scalar_t* v_buffer_ptr = v_buffer + loc_val * v_strideN + head_kv_id * v_strideH;
        const scalar_t* new_value_ptr = value + bs * nv_strideN + head_kv_id * nv_strideH;
        std::memcpy(v_buffer_ptr, new_value_ptr, head_size_v * sizeof(scalar_t));
      }
      data_index_step(bs, batches, head_kv_id, num_heads_kv);
    }
  });
}

}  // namespace

// =============================================================================
// Explicit Instantiations
// =============================================================================

#define INSTANTIATE_FLOAT_AND_INT8(scalar_t, index_t)                        \
  template void decode_attention_kernel<scalar_t, index_t, 256L>(            \
      scalar_t*,                                                             \
      float*,                                                                \
      const scalar_t*,                                                       \
      const void*,                                                           \
      const void*,                                                           \
      const index_t*,                                                        \
      const int64_t*,                                                        \
      const int64_t*,                                                        \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      float,                                                                 \
      float,                                                                 \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      float,                                                                 \
      float);                                                                \
  template void decode_attention_grouped_kernel<scalar_t, index_t, 256L>(    \
      scalar_t*,                                                             \
      float*,                                                                \
      const scalar_t*,                                                       \
      const void*,                                                           \
      const void*,                                                           \
      const index_t*,                                                        \
      const int64_t*,                                                        \
      const int64_t*,                                                        \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      float,                                                                 \
      float,                                                                 \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      float,                                                                 \
      float);                                                                \
  template void decode_attention_kernel_rvv_int8<scalar_t, index_t>(         \
      scalar_t*,                                                             \
      float*,                                                                \
      const scalar_t*,                                                       \
      const void*,                                                           \
      const void*,                                                           \
      const index_t*,                                                        \
      const int64_t*,                                                        \
      const int64_t*,                                                        \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      float,                                                                 \
      float,                                                                 \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      float,                                                                 \
      float);                                                                \
  template void decode_attention_mla_kernel<scalar_t, index_t, 256L>(        \
      scalar_t*,                                                             \
      float*,                                                                \
      const scalar_t*,                                                       \
      const void*,                                                           \
      const void*,                                                           \
      const index_t*,                                                        \
      const int64_t*,                                                        \
      const int64_t*,                                                        \
      scalar_t*,                                                             \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      float,                                                                 \
      float,                                                                 \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t);                                                              \
  template void decode_attention_grouped_kernel_rvv_int8<scalar_t, index_t>( \
      scalar_t*,                                                             \
      float*,                                                                \
      const scalar_t*,                                                       \
      const void*,                                                           \
      const void*,                                                           \
      const index_t*,                                                        \
      const int64_t*,                                                        \
      const int64_t*,                                                        \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      float,                                                                 \
      float,                                                                 \
      int64_t,                                                               \
      int64_t,                                                               \
      int64_t,                                                               \
      float,                                                                 \
      float);

INSTANTIATE_FLOAT_AND_INT8(float, int32_t)
INSTANTIATE_FLOAT_AND_INT8(float, int64_t)
INSTANTIATE_FLOAT_AND_INT8(c10::Half, int32_t)
INSTANTIATE_FLOAT_AND_INT8(c10::Half, int64_t)
INSTANTIATE_FLOAT_AND_INT8(c10::BFloat16, int32_t)
INSTANTIATE_FLOAT_AND_INT8(c10::BFloat16, int64_t)

#undef INSTANTIATE_FLOAT_AND_INT8

#endif  // CPU_CAPABILITY_RVV

}  // namespace rvv
}  // namespace sgl_kernel

// =============================================================================
// Top-level Entry Point (Replacing decode.cpp for RISC-V)
// =============================================================================

#if defined(CPU_CAPABILITY_RVV)

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
    double logit_cap,
    double k_scale,
    double v_scale) {
  RECORD_FUNCTION(
      "sgl-kernel::decode_attention_cpu",
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

  int64_t q_strideM = query.stride(0);
  int64_t q_strideH = query.stride(1);
  int64_t k_strideN = k_buffer.stride(0);
  int64_t k_strideH = k_buffer.stride(1);
  int64_t v_strideN = v_buffer.stride(0);
  int64_t v_strideH = v_buffer.stride(1);
  int64_t nk_strideN = key.stride(0);
  int64_t nk_strideH = key.stride(1);
  int64_t nv_strideN = value.stride(0);
  int64_t nv_strideH = value.stride(1);

  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "decode: expect req_to_token to be int32 or int64, got ",
      index_dtype);

  void* k_buffer_data = k_buffer.data_ptr();
  void* v_buffer_data = v_buffer.data_ptr();
  const bool is_mla = (k_buffer_data == v_buffer_data) && (num_heads_kv == 1) && (head_size == head_size_v + 64);

  constexpr int BLOCK_N = 256;

  // Buffer for MLA (if needed)
  int num_threads = at::get_num_threads();
  int64_t size_per_thread = is_mla ? BLOCK_N * head_size + BLOCK_N * head_size_v : 0;
  auto buffer = at::empty({num_threads, size_per_thread}, k_buffer.options());

  // Use AT_DISPATCH_RVV_TYPES for Float, Half and BFloat16
  AT_DISPATCH_RVV_TYPES(query.scalar_type(), "decode_attention_kernel", [&] {
    {
      AT_DISPATCH_INDEX_TYPES(index_dtype, "decode_attention_indices", [&] {
        // update the kv buffer
        sgl_kernel::rvv::decode_set_kv_buffer(
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
          sgl_kernel::rvv::decode_attention_kernel<scalar_t, index_t, BLOCK_N>(
              output.data_ptr<scalar_t>(),
              attn_logits.data_ptr<float>(),
              query.data_ptr<scalar_t>(),
              (const scalar_t*)k_buffer_data,
              (const scalar_t*)v_buffer_data,
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
              max_total_num_tokens,
              (float)k_scale,
              (float)v_scale);
        } else if (is_mla) {
          // MLA
          sgl_kernel::rvv::decode_attention_mla_kernel<scalar_t, index_t, BLOCK_N>(
              output.data_ptr<scalar_t>(),
              attn_logits.data_ptr<float>(),
              query.data_ptr<scalar_t>(),
              (const scalar_t*)k_buffer_data,
              (const scalar_t*)v_buffer_data,
              req_to_token.data_ptr<index_t>(),
              req_pool_indices.data_ptr<int64_t>(),
              seq_lens.data_ptr<int64_t>(),
              buffer.data_ptr<scalar_t>(),
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
              max_total_num_tokens,
              size_per_thread);
        } else {
          // GQA/MQA
          sgl_kernel::rvv::decode_attention_grouped_kernel<scalar_t, index_t, BLOCK_N>(
              output.data_ptr<scalar_t>(),
              attn_logits.data_ptr<float>(),
              query.data_ptr<scalar_t>(),
              (const scalar_t*)k_buffer_data,
              (const scalar_t*)v_buffer_data,
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
              max_total_num_tokens,
              (float)k_scale,
              (float)v_scale);
        }
      });
    }
  });
}

void decode_attention_int8_cpu(
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
    double logit_cap,
    double k_scale,
    double v_scale) {
  RECORD_FUNCTION(
      "sgl-kernel::decode_attention_int8_cpu",
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
  TORCH_CHECK(
      k_buffer.scalar_type() == at::kByte || k_buffer.scalar_type() == at::kChar,
      "decode_int8: k_buffer must be int8/uint8, got ",
      k_buffer.scalar_type());
  TORCH_CHECK(
      v_buffer.scalar_type() == at::kByte || v_buffer.scalar_type() == at::kChar,
      "decode_int8: v_buffer must be int8/uint8, got ",
      v_buffer.scalar_type());

  int64_t num_seqs = seq_lens.size(0);
  int64_t max_num_reqs = req_to_token.size(0);
  int64_t max_context_len = req_to_token.size(1);
  int64_t max_total_num_tokens = k_buffer.size(0);

  int64_t num_heads = query.size(1);
  int64_t num_heads_kv = k_buffer.size(1);
  int64_t head_size = query.size(2);
  int64_t head_size_v = v_buffer.size(2);

  int64_t num_kv_splits = attn_logits.size(2);

  int64_t q_strideM = query.stride(0);
  int64_t q_strideH = query.stride(1);
  int64_t k_strideN = k_buffer.stride(0);
  int64_t k_strideH = k_buffer.stride(1);
  int64_t v_strideN = v_buffer.stride(0);
  int64_t v_strideH = v_buffer.stride(1);
  int64_t nk_strideN = key.stride(0);
  int64_t nk_strideH = key.stride(1);
  int64_t nv_strideN = value.stride(0);
  int64_t nv_strideH = value.stride(1);

  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "decode: expect req_to_token to be int32 or int64, got ",
      index_dtype);

  void* k_buffer_data = k_buffer.data_ptr();
  void* v_buffer_data = v_buffer.data_ptr();
  // Use AT_DISPATCH_RVV_TYPES for Float, Half and BFloat16
  AT_DISPATCH_RVV_TYPES(query.scalar_type(), "decode_attention_int8_kernel", [&] {
    {
      AT_DISPATCH_INDEX_TYPES(index_dtype, "decode_attention_indices", [&] {
        // Update KV buffer with INT8 quantization
        if (key.scalar_type() == at::kByte || key.scalar_type() == at::kChar) {
          // Input already INT8: just copy
          sgl_kernel::rvv::decode_set_kv_buffer_int8_copy<int8_t>(
              (int8_t*)k_buffer_data,
              (int8_t*)v_buffer_data,
              (int8_t*)key.data_ptr(),
              (int8_t*)value.data_ptr(),
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
              nv_strideH);
        } else {
          // Use AT_DISPATCH_RVV_TYPES for Float, Half and BFloat16
          AT_DISPATCH_RVV_TYPES(key.scalar_type(), "decode_set_kv_buffer_quantize", [&] {
            {
              // Quantize FP input to INT8 (symmetric)
              sgl_kernel::rvv::decode_set_kv_buffer_int8_quantize<scalar_t, int8_t>(
                  (int8_t*)k_buffer_data,
                  (int8_t*)v_buffer_data,
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
                  (float)k_scale,
                  (float)v_scale);
            }  // end block
          });
        }

        // Dispatch to INT8-aware RVV kernel
        if (num_heads == num_heads_kv) {
          // MHA : num_heads == num_heads_kv
          sgl_kernel::rvv::decode_attention_kernel_rvv_int8<scalar_t, index_t>(
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
              max_total_num_tokens,
              (float)k_scale,
              (float)v_scale);
        } else {
          // GQA : num_heads != num_heads_kv
          sgl_kernel::rvv::decode_attention_grouped_kernel_rvv_int8<scalar_t, index_t>(
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
              max_total_num_tokens,
              (float)k_scale,
              (float)v_scale);
        }
      });
    }
  });
}

#endif
