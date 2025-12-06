// RISC-V Vector Extension (RVV) optimized extend attention kernels
// This file contains RVV specific implementations for extend attention
// Note: This file is included directly in extend.cpp

#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "common.h"

#if defined(CPU_CAPABILITY_RVV)

#include <riscv_vector.h>

#include <cstring>

// Check for Zvfh (FP16 vector) support
// Spacemit X60 (Banana Pi) supports zvfh_zvfhmin
#if defined(__riscv_zvfh) || defined(__riscv_v)
#define HAS_ZVFH 1
#else
#define HAS_ZVFH 0
#endif

namespace {

// Thread-local scratch buffer for type conversion
// Avoids repeated allocation in hot loops
// Size is conservative upper bound for typical RVV implementations (VLEN up to 16384 bits)
constexpr size_t MAX_VL_ELEMENTS = 512;

// Helper: Load scalar_t data into float vector, handling type conversion
// Uses provided scratch buffer to avoid allocation
//
// For FP16 with Zvfh support: Uses hardware vle16 + vfwcvt (widening convert)
// For FP16 without Zvfh: Falls back to software conversion loop
// For FP32: Direct vector load
template <typename scalar_t>
inline vfloat32m1_t load_as_float_rvv(const scalar_t* ptr, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    // FP32: Direct load
    return __riscv_vle32_v_f32m1(ptr, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH
    // FP16 with Zvfh: Hardware widening conversion (fast path)
    // vle16 loads FP16 vector, vfwcvt_f_f widens to FP32
    // LMUL doubles: f16m1 -> f32m2, so we use f16mf2 -> f32m1
    vfloat16mf2_t v_f16 = __riscv_vle16_v_f16mf2(reinterpret_cast<const _Float16*>(ptr), vl);
    return __riscv_vfwcvt_f_f_v_f32m1(v_f16, vl);
#else
    // FP16 without Zvfh: Software conversion (slow path)
    for (size_t i = 0; i < vl; ++i) {
      scratch[i] = static_cast<float>(ptr[i]);
    }
    return __riscv_vle32_v_f32m1(scratch, vl);
#endif
  } else {
    // Other types (BFloat16, etc.): Software conversion
    for (size_t i = 0; i < vl; ++i) {
      scratch[i] = static_cast<float>(ptr[i]);
    }
    return __riscv_vle32_v_f32m1(scratch, vl);
  }
}

// Helper: Compute C = A @ B^T
// A: [M, K], B: [N, K], C: [M, N]
// lda, ldb, ldc are strides in elements
// scratch_a, scratch_b: pre-allocated buffers of size >= MAX_VL_ELEMENTS for type conversion
template <typename scalar_t>
void gemm_nt_rvv(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float scale,
    float* __restrict__ scratch_a,
    float* __restrict__ scratch_b) {
  size_t vl_max = __riscv_vsetvlmax_e32m1();

  for (int m = 0; m < M; ++m) {
    const scalar_t* ptr_A = A + m * lda;
    float* ptr_C = C + m * ldc;

    for (int n = 0; n < N; ++n) {
      const scalar_t* ptr_B = B + n * ldb;

      float sum = 0.0f;
      vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, vl_max);
      int64_t k = 0;

      // Main loop
      for (; k + static_cast<int64_t>(vl_max) <= K; k += vl_max) {
        vfloat32m1_t v_a = load_as_float_rvv(ptr_A + k, vl_max, scratch_a);
        vfloat32m1_t v_b = load_as_float_rvv(ptr_B + k, vl_max, scratch_b);
        v_sum = __riscv_vfmacc_vv_f32m1(v_sum, v_a, v_b, vl_max);
      }

      // Reduce main part
      vfloat32m1_t v_red = __riscv_vfredusum_vs_f32m1_f32m1(v_sum, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl_max);
      sum = __riscv_vfmv_f_s_f32m1_f32(v_red);

      // Tail part
      if (k < K) {
        size_t tail_vl = __riscv_vsetvl_e32m1(K - k);
        vfloat32m1_t v_a = load_as_float_rvv(ptr_A + k, tail_vl, scratch_a);
        vfloat32m1_t v_b = load_as_float_rvv(ptr_B + k, tail_vl, scratch_b);
        vfloat32m1_t v_tail_sum = __riscv_vfmul_vv_f32m1(v_a, v_b, tail_vl);
        vfloat32m1_t v_tail_red =
            __riscv_vfredusum_vs_f32m1_f32m1(v_tail_sum, __riscv_vfmv_v_f_f32m1(0.0f, 1), tail_vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(v_tail_red);
      }

      ptr_C[n] = sum * scale;
    }
  }
}

// Helper: Compute C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
// A is softmax output (float), B is V (scalar_t), C is float accumulator
// c_acc: pre-allocated buffer of size M * N for accumulation
// scratch: pre-allocated buffer of size >= MAX_VL_ELEMENTS for type conversion
template <typename scalar_t>
void gemm_nn_rvv(
    const float* __restrict__ A,
    const scalar_t* __restrict__ B,
    float* __restrict__ c_acc,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    float* __restrict__ scratch) {
  // A is [M, K] (scores)
  // B is [K, N] (values)
  // c_acc is [M, N] (output accumulator in float)

  // Iterate m.
  //   Initialize c_acc[m, :] = 0
  //   Iterate k:
  //     Load A[m, k] (scalar)
  //     Load B[k, :] (vector)
  //     c_acc[m, :] += A[m, k] * B[k, :]

  for (int m = 0; m < M; ++m) {
    float* row_acc = c_acc + m * N;

    // Initialize row to zero
    std::memset(row_acc, 0, N * sizeof(float));

    for (int k = 0; k < K; ++k) {
      float a_val = A[m * lda + k];
      if (a_val < 1e-6f) continue;  // Skip near-zero weights

      const scalar_t* ptr_B = B + k * ldb;

      size_t vl;
      for (int n = 0; n < N; n += vl) {
        vl = __riscv_vsetvl_e32m1(N - n);

        vfloat32m1_t v_a = __riscv_vfmv_v_f_f32m1(a_val, vl);
        vfloat32m1_t v_b = load_as_float_rvv(ptr_B + n, vl, scratch);
        vfloat32m1_t v_c = __riscv_vle32_v_f32m1(row_acc + n, vl);

        v_c = __riscv_vfmacc_vv_f32m1(v_c, v_a, v_b, vl);
        __riscv_vse32_v_f32m1(row_acc + n, v_c, vl);
      }
    }
  }
}

// Helper: Compute C += A @ B (accumulate version)
// A: [M, K] float, B: [K, N] scalar_t, C: [M, N] float
// This version ADDS to existing C values instead of overwriting
// scratch: pre-allocated buffer of size >= MAX_VL_ELEMENTS for type conversion
template <typename scalar_t>
void gemm_nn_acc_rvv(
    const float* __restrict__ A,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float* __restrict__ scratch) {
  for (int m = 0; m < M; ++m) {
    float* row_c = C + m * ldc;

    for (int k = 0; k < K; ++k) {
      float a_val = A[m * lda + k];
      if (a_val < 1e-6f) continue;  // Skip near-zero weights

      const scalar_t* ptr_B = B + k * ldb;

      size_t vl;
      for (int n = 0; n < N; n += vl) {
        vl = __riscv_vsetvl_e32m1(N - n);

        vfloat32m1_t v_b = load_as_float_rvv(ptr_B + n, vl, scratch);
        vfloat32m1_t v_c = __riscv_vle32_v_f32m1(row_c + n, vl);

        v_c = __riscv_vfmacc_vf_f32m1(v_c, a_val, v_b, vl);
        __riscv_vse32_v_f32m1(row_c + n, v_c, vl);
      }
    }
  }
}

template <typename scalar_t, typename index_t>
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
  // Optimized implementation with pre-allocated thread-local buffers
  // and GEMM batch processing for extend part

  int num_groups = num_heads / num_heads_kv;
  constexpr int BLOCK_M = 32;

// Parallel over batches and heads
#pragma omp parallel
  {
    // Thread-local scratch buffers - allocated once per thread
    // These are reused across all (batch, head, block) iterations
    alignas(64) float scratch_a[MAX_VL_ELEMENTS];
    alignas(64) float scratch_b[MAX_VL_ELEMENTS];

    // Pre-allocate working buffers with max possible sizes
    // scores: [BLOCK_M, max_context_len]
    // out_acc: [BLOCK_M, head_size_v]
    std::vector<float> scores_buf(BLOCK_M * max_context_len);
    std::vector<float> out_acc_buf(BLOCK_M * head_size_v);

#pragma omp for collapse(2)
    for (int b = 0; b < batches; ++b) {
      for (int h = 0; h < num_heads; ++h) {
        int head_kv = h / num_groups;
        int seq_len = seq_lens[b];
        int extend_len = extend_seq_lens[b];
        int prefix_len = seq_len - extend_len;
        int start_loc = extend_start_loc[b];
        int req_idx = req_pool_indices[b];

        const scalar_t* q_ptr = q_extend + start_loc * q_strideM + h * q_strideH;
        scalar_t* o_ptr = o_extend + start_loc * q_strideM + h * q_strideH;

        // Process in blocks of M (query tokens)
        for (int m_start = 0; m_start < extend_len; m_start += BLOCK_M) {
          int m_size = std::min(BLOCK_M, extend_len - m_start);
          int total_k_len = prefix_len + extend_len;

          // Use pre-allocated buffer, just clear the portion we need
          float* scores = scores_buf.data();
          std::memset(scores, 0, m_size * total_k_len * sizeof(float));

          // 1. Q @ K_prefix (scattered gather)
          if (prefix_len > 0) {
            for (int i = 0; i < m_size; ++i) {
              const scalar_t* q_vec = q_ptr + (m_start + i) * q_strideM;

              for (int j = 0; j < prefix_len; ++j) {
                int token_idx = req_to_token[req_idx * max_context_len + j];
                const scalar_t* k_vec = k_buffer + token_idx * k_strideN + head_kv * k_strideH;

                // Vectorized dot product using helper function
                float dot = 0.0f;
                size_t vl_max = __riscv_vsetvlmax_e32m1();
                vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, vl_max);

                int k = 0;
                for (; k + static_cast<int>(vl_max) <= head_size; k += vl_max) {
                  vfloat32m1_t v_q = load_as_float_rvv(q_vec + k, vl_max, scratch_a);
                  vfloat32m1_t v_k = load_as_float_rvv(k_vec + k, vl_max, scratch_b);
                  v_sum = __riscv_vfmacc_vv_f32m1(v_sum, v_q, v_k, vl_max);
                }

                vfloat32m1_t v_red = __riscv_vfredusum_vs_f32m1_f32m1(v_sum, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl_max);
                dot = __riscv_vfmv_f_s_f32m1_f32(v_red);

                // Tail part
                if (k < head_size) {
                  size_t tail_vl = __riscv_vsetvl_e32m1(head_size - k);
                  vfloat32m1_t v_q = load_as_float_rvv(q_vec + k, tail_vl, scratch_a);
                  vfloat32m1_t v_k = load_as_float_rvv(k_vec + k, tail_vl, scratch_b);
                  vfloat32m1_t v_tail_sum = __riscv_vfmul_vv_f32m1(v_q, v_k, tail_vl);
                  vfloat32m1_t v_tail_red =
                      __riscv_vfredusum_vs_f32m1_f32m1(v_tail_sum, __riscv_vfmv_v_f_f32m1(0.0f, 1), tail_vl);
                  dot += __riscv_vfmv_f_s_f32m1_f32(v_tail_red);
                }

                scores[i * total_k_len + j] = dot * scaling;
              }
            }
          }

          // 2. Q @ K_extend using GEMM (contiguous, efficient)
          const scalar_t* k_ext_ptr = k_extend + start_loc * ke_strideN + head_kv * ke_strideH;

          gemm_nt_rvv(
              q_ptr + m_start * q_strideM,
              k_ext_ptr,
              scores + prefix_len,
              m_size,
              extend_len,
              head_size,
              q_strideM,
              ke_strideN,
              total_k_len,
              scaling,
              scratch_a,
              scratch_b);

          // 3. Causal Mask & Softmax
          for (int i = 0; i < m_size; ++i) {
            int row_idx = m_start + i;
            int valid_len = prefix_len + row_idx + 1;
            float* row_scores = scores + i * total_k_len;

            // Mask out future tokens
            for (int j = valid_len; j < total_k_len; ++j) {
              row_scores[j] = -std::numeric_limits<float>::infinity();
            }

            // Apply logit cap if needed
            if (logit_cap > 0.0f) {
              for (int j = 0; j < valid_len; ++j) {
                row_scores[j] = logit_cap * std::tanh(row_scores[j] / logit_cap);
              }
            }

            // Softmax: find max
            float max_val = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < valid_len; ++j) {
              max_val = std::max(max_val, row_scores[j]);
            }

            // Exp and sum
            float sum_exp = 0.0f;
            for (int j = 0; j < valid_len; ++j) {
              row_scores[j] = std::exp(row_scores[j] - max_val);
              sum_exp += row_scores[j];
            }

            // Normalize
            float inv_sum = 1.0f / sum_exp;
            for (int j = 0; j < valid_len; ++j) {
              row_scores[j] *= inv_sum;
            }
            // Zero out masked
            for (int j = valid_len; j < total_k_len; ++j) {
              row_scores[j] = 0.0f;
            }
          }

          // 4. Scores @ V
          // Use pre-allocated output accumulator buffer
          float* out_acc = out_acc_buf.data();
          std::memset(out_acc, 0, m_size * head_size_v * sizeof(float));

          // 4a. Scores_prefix @ V_prefix (scattered gather)
          if (prefix_len > 0) {
            for (int i = 0; i < m_size; ++i) {
              float* row_probs = scores + i * total_k_len;
              float* row_acc = out_acc + i * head_size_v;

              for (int j = 0; j < prefix_len; ++j) {
                float prob = row_probs[j];
                if (prob < 1e-6f) continue;

                int token_idx = req_to_token[req_idx * max_context_len + j];
                const scalar_t* v_vec = v_buffer + token_idx * v_strideN + head_kv * v_strideH;

                size_t vl;
                for (int d = 0; d < head_size_v; d += vl) {
                  vl = __riscv_vsetvl_e32m1(head_size_v - d);
                  vfloat32m1_t v_acc = __riscv_vle32_v_f32m1(row_acc + d, vl);
                  vfloat32m1_t v_val = load_as_float_rvv(v_vec + d, vl, scratch_a);
                  v_acc = __riscv_vfmacc_vf_f32m1(v_acc, prob, v_val, vl);
                  __riscv_vse32_v_f32m1(row_acc + d, v_acc, vl);
                }
              }
            }
          }

          // 4b. Scores_extend @ V_extend using GEMM (batch processing)
          // Scores_extend: [m_size, extend_len] at scores[:, prefix_len:]
          // V_extend: [extend_len, head_size_v]
          // Out_acc += Scores_extend @ V_extend
          const scalar_t* v_ext_ptr = v_extend + start_loc * ve_strideN + head_kv * ve_strideH;

          // Use accumulate GEMM: C += A @ B
          gemm_nn_acc_rvv(
              scores + prefix_len,  // A: [m_size, extend_len], but offset in strided buffer
              v_ext_ptr,            // B: [extend_len, head_size_v]
              out_acc,              // C: [m_size, head_size_v]
              m_size,               // M
              head_size_v,          // N
              extend_len,           // K
              total_k_len,          // lda (stride of scores in buffer)
              ve_strideN,           // ldb
              head_size_v,          // ldc
              scratch_a);

          // 5. Convert float accumulator to output (scalar_t)
          for (int i = 0; i < m_size; ++i) {
            scalar_t* out_row = o_ptr + (m_start + i) * q_strideM;
            float* acc_row = out_acc + i * head_size_v;
            for (int d = 0; d < head_size_v; ++d) {
              out_row[d] = static_cast<scalar_t>(acc_row[d]);
            }
          }
        }
      }
    }
  }
}

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

  // Get the actual index dtype from req_to_token
  const auto index_dtype = req_to_token.scalar_type();

  // Use AT_DISPATCH_REDUCED_FLOATING_TYPES for Half and BFloat16 only
  // The RVV Zvfh extension handles FP16 natively
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
          is_prefix_skipped);
    });
  });
}

}  // namespace

#endif  // CPU_CAPABILITY_RVV
