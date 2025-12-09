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

      // Initialize dot product accumulators (unrolled 4x)
      vfloat32m1_t vdot0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      vfloat32m1_t vdot1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      vfloat32m1_t vdot2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      vfloat32m1_t vdot3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

      // Vectorized dot product calculation with 4x unrolling
      int64_t k = 0;
      for (; k + 4 * vl <= K; k += 4 * vl) {
        vfloat32m1_t vq0 = __riscv_vle32_v_f32m1(q_float32 + k, vl);
        vfloat32m1_t vk0 = __riscv_vle32_v_f32m1(k_float32 + k, vl);
        vdot0 = __riscv_vfmacc_vv_f32m1(vdot0, vq0, vk0, vl);

        vfloat32m1_t vq1 = __riscv_vle32_v_f32m1(q_float32 + k + vl, vl);
        vfloat32m1_t vk1 = __riscv_vle32_v_f32m1(k_float32 + k + vl, vl);
        vdot1 = __riscv_vfmacc_vv_f32m1(vdot1, vq1, vk1, vl);

        vfloat32m1_t vq2 = __riscv_vle32_v_f32m1(q_float32 + k + 2 * vl, vl);
        vfloat32m1_t vk2 = __riscv_vle32_v_f32m1(k_float32 + k + 2 * vl, vl);
        vdot2 = __riscv_vfmacc_vv_f32m1(vdot2, vq2, vk2, vl);

        vfloat32m1_t vq3 = __riscv_vle32_v_f32m1(q_float32 + k + 3 * vl, vl);
        vfloat32m1_t vk3 = __riscv_vle32_v_f32m1(k_float32 + k + 3 * vl, vl);
        vdot3 = __riscv_vfmacc_vv_f32m1(vdot3, vq3, vk3, vl);
      }

      // Handle remaining chunks
      for (; k + vl <= K; k += vl) {
        vfloat32m1_t vq = __riscv_vle32_v_f32m1(q_float32 + k, vl);
        vfloat32m1_t vk = __riscv_vle32_v_f32m1(k_float32 + k, vl);
        vdot0 = __riscv_vfmacc_vv_f32m1(vdot0, vq, vk, vl);
      }

      // Combine accumulators
      vfloat32m1_t vdot = __riscv_vfadd_vv_f32m1(
          __riscv_vfadd_vv_f32m1(vdot0, vdot1, vl), __riscv_vfadd_vv_f32m1(vdot2, vdot3, vl), vl);

      // Handle tail elements
      if (k < K) {
        size_t tail_vl = __riscv_vsetvl_e32m1(K - k);
        vfloat32m1_t vq = __riscv_vle32_v_f32m1(q_float32 + k, tail_vl);
        vfloat32m1_t vk = __riscv_vle32_v_f32m1(k_float32 + k, tail_vl);

        // Use a separate accumulator for tail to avoid corrupting vdot elements past tail_vl
        vfloat32m1_t vtail_acc = __riscv_vfmv_v_f_f32m1(0.0f, tail_vl);
        vtail_acc = __riscv_vfmacc_vv_f32m1(vtail_acc, vq, vk, tail_vl);

        // Reduce tail accumulator to scalar sum immediately.
        vfloat32m1_t vtail_sum =
            __riscv_vfredusum_vs_f32m1_f32m1(vtail_acc, __riscv_vfmv_v_f_f32m1(0.0f, tail_vl), tail_vl);
        float tail_sum_val = __riscv_vfmv_f_s_f32m1_f32(vtail_sum);

        // Add this scalar to the first element of vdot
        // Use vfmv.s.f with tail undisturbed policy to update only index 0
        float vdot0_scalar = __riscv_vfmv_f_s_f32m1_f32(vdot);
        vdot0_scalar += tail_sum_val;
        vdot = __riscv_vfmv_s_f_f32m1_tu(vdot, vdot0_scalar, vl);  // Update index 0, preserve rest
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

// 2. Fast approximate exponential and sum
// Calculate exp(scores - m_i) and return sum
inline float exp_rvv(const float* __restrict__ scores, float* __restrict__ output, int64_t N, float m_i) {
  if (N == 0) return 0.0f;

  // 1. Calculate shifted scores: x = scores - m_i
  int64_t offset = 0;
  while (offset < N) {
    size_t vl = __riscv_vsetvl_e32m1(N - offset);
    vfloat32m1_t vscores = __riscv_vle32_v_f32m1(scores + offset, vl);
    vfloat32m1_t vshifted = __riscv_vfsub_vf_f32m1(vscores, m_i, vl);
    __riscv_vse32_v_f32m1(output + offset, vshifted, vl);
    offset += vl;
  }

  // 2. Calculate exp using scalar std::exp and sum
  float sum_val = 0.0f;
  for (int64_t i = 0; i < N; ++i) {
    float val = std::exp(output[i]);
    output[i] = val;
    sum_val += val;
  }

  return sum_val;
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

// 4. Accumulate KV splits and write to output
// This function merges partial results from different KV splits and writes the final result to output
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
          size_t vl_max = __riscv_vsetvlmax_e32m1();
          for (int64_t d = 0; d < head_size_v; d += vl_max) {
            size_t vl = __riscv_vsetvl_e32m1(head_size_v - d);
            vfloat32m1_t vacc = __riscv_vle32_v_f32m1(acc + d, vl);
            vfloat32m1_t vtv = __riscv_vle32_v_f32m1(tv + d, vl);

            vfloat32m1_t vm_delta = __riscv_vfmv_v_f_f32m1(m_delta, vl);
            vfloat32m1_t ve_logic = __riscv_vfmv_v_f_f32m1(e_logic, vl);

            // acc = acc * m_delta
            vacc = __riscv_vfmul_vv_f32m1(vacc, vm_delta, vl);
            // acc = acc + tv * e_logic
            vacc = __riscv_vfmacc_vv_f32m1(vacc, ve_logic, vtv, vl);

            __riscv_vse32_v_f32m1(acc + d, vacc, vl);
          }
        }

        s_prime = s_prime * m_delta + e_logic;
        m_prime = m_i;
      }

      // Final copy to output with scaling: output = acc * (1/s_prime)
      float scale = 1.0f / s_prime;

      size_t vl_max = __riscv_vsetvlmax_e32m1();
      for (int64_t d = 0; d < head_size_v; d += vl_max) {
        size_t vl = __riscv_vsetvl_e32m1(head_size_v - d);
        vfloat32m1_t vacc = __riscv_vle32_v_f32m1(acc + d, vl);
        vfloat32m1_t vscale = __riscv_vfmv_v_f_f32m1(scale, vl);
        vacc = __riscv_vfmul_vv_f32m1(vacc, vscale, vl);

        if constexpr (std::is_same_v<scalar_t, float>) {
          __riscv_vse32_v_f32m1(output + i * head_size_v + d, vacc, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH_DECODE
          vfloat16mf2_t vout = __riscv_vfncvt_f_f_w_f16mf2(vacc, vl);
          __riscv_vse16_v_f16mf2(reinterpret_cast<_Float16*>(output + i * head_size_v + d), vout, vl);
#else
              // Software conversion
              float temp[256];
              __riscv_vse32_v_f32m1(temp, vacc, vl);
              for(size_t j=0; j<vl; ++j) {
                  (output + i * head_size_v + d)[j] = static_cast<at::Half>(temp[j]);
              }
#endif
        } else {
          // BFloat16 or others
          float temp[256];
          __riscv_vse32_v_f32m1(temp, vacc, vl);
          for (size_t j = 0; j < vl; ++j) {
            (output + i * head_size_v + d)[j] = static_cast<scalar_t>(temp[j]);
          }
        }
      }
    }
  });
}

// 5. Full Decode Attention Kernel (RVV Optimized)
// This function implements the full FlashDecoding loop using RVV intrinsics
// It is called by decode_attention_cpu when RVV is available
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
        size_t vl_v = __riscv_vsetvl_e32m1(head_size_v);
        for (int64_t d = 0; d < head_size_v; d += vl_v) {
          size_t current_vl = __riscv_vsetvl_e32m1(head_size_v - d);
          vfloat32m1_t vval = __riscv_vle32_v_f32m1(v_prime + d, current_vl);
          vfloat32m1_t vscale = __riscv_vfmv_v_f_f32m1(s, current_vl);
          vval = __riscv_vfmul_vv_f32m1(vval, vscale, current_vl);
          __riscv_vse32_v_f32m1(v_prime + d, vval, current_vl);
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
