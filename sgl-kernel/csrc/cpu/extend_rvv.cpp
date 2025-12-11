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

// Helper: Store float vector to scalar_t buffer, handling type conversion
template <typename scalar_t>
inline void store_from_float_rvv(scalar_t* ptr, vfloat32m1_t v_val, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vse32_v_f32m1(ptr, v_val, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH
    vfloat16mf2_t v_f16 = __riscv_vfncvt_f_f_w_f16mf2(v_val, vl);
    __riscv_vse16_v_f16mf2(reinterpret_cast<_Float16*>(ptr), v_f16, vl);
#else
    __riscv_vse32_v_f32m1(scratch, v_val, vl);
    for (size_t i = 0; i < vl; ++i) {
      ptr[i] = static_cast<scalar_t>(scratch[i]);
    }
#endif
  } else {
    __riscv_vse32_v_f32m1(scratch, v_val, vl);
    for (size_t i = 0; i < vl; ++i) {
      ptr[i] = static_cast<scalar_t>(scratch[i]);
    }
  }
}

// Helper: Compute C = A @ B^T
// A: [M, K], B: [N, K], C: [M, N]
// lda, ldb, ldc are strides in elements
// scratch_a, scratch_b: pre-allocated buffers of size >= MAX_VL_ELEMENTS for type conversion
template <typename scalar_t_a, typename scalar_t_b>
void gemm_nt_rvv(
    const scalar_t_a* __restrict__ A,
    const scalar_t_b* __restrict__ B,
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
    const scalar_t_a* ptr_A = A + m * lda;
    float* ptr_C = C + m * ldc;

    for (int n = 0; n < N; ++n) {
      const scalar_t_b* ptr_B = B + n * ldb;

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
  // Optimized loop order: m -> n -> k
  // Keeps accumulator (C) in registers to minimize load/store traffic
  for (int m = 0; m < M; ++m) {
    float* row_c = C + m * ldc;

    size_t vl;
    for (int n = 0; n < N; n += vl) {
      vl = __riscv_vsetvl_e32m1(N - n);

      // Load existing C values into accumulator
      vfloat32m1_t v_c = __riscv_vle32_v_f32m1(row_c + n, vl);

      for (int k = 0; k < K; ++k) {
        float a_val = A[m * lda + k];
        if (a_val < 1e-6f) continue;  // Skip near-zero weights

        // Load B row vector
        vfloat32m1_t v_b = load_as_float_rvv(B + k * ldb + n, vl, scratch);

        // Accumulate: v_c += a_val * v_b
        v_c = __riscv_vfmacc_vf_f32m1(v_c, a_val, v_b, vl);
      }

      // Store updated C values
      __riscv_vse32_v_f32m1(row_c + n, v_c, vl);
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
  // Optimized implementation using FlashAttention algorithm (Tiled Q + Tiled K + Online Softmax)
  // This avoids large memory allocation for scores and improves cache locality.

  int num_groups = num_heads / num_heads_kv;
  constexpr int BLOCK_M = 32;
  constexpr int BLOCK_N = 128;  // Tile size for K/V

// Parallel over batches and heads
#pragma omp parallel
  {
    // Thread-local scratch buffers
    alignas(64) float scratch_a[MAX_VL_ELEMENTS];
    alignas(64) float scratch_b[MAX_VL_ELEMENTS];

    // Buffers for FlashAttention
    // scores_block: [BLOCK_M, BLOCK_N]
    std::vector<float> scores_block(BLOCK_M * BLOCK_N);

    // Gather buffers for Paged K/V (Prefix)
    // k_block_buf: [BLOCK_N, head_size]
    // v_block_buf: [BLOCK_N, head_size_v]
    std::vector<float> k_block_buf(BLOCK_N * head_size);
    std::vector<float> v_block_buf(BLOCK_N * head_size_v);

    // Accumulators for Online Softmax
    // o_acc: [BLOCK_M, head_size_v]
    // l_acc: [BLOCK_M] (denominator)
    // m_acc: [BLOCK_M] (max score)
    std::vector<float> o_acc(BLOCK_M * head_size_v);
    std::vector<float> l_acc(BLOCK_M);
    std::vector<float> m_acc(BLOCK_M);

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

        // Loop over Query Blocks (BLOCK_M)
        for (int m_start = 0; m_start < extend_len; m_start += BLOCK_M) {
          int m_size = std::min(BLOCK_M, extend_len - m_start);

          // Initialize accumulators
          std::fill(l_acc.begin(), l_acc.end(), 0.0f);
          std::fill(m_acc.begin(), m_acc.end(), -std::numeric_limits<float>::infinity());
          std::memset(o_acc.data(), 0, m_size * head_size_v * sizeof(float));

          // -------------------------------------------------------
          // Phase 1: Prefix (Paged Memory)
          // -------------------------------------------------------
          if (prefix_len > 0) {
            for (int n_start = 0; n_start < prefix_len; n_start += BLOCK_N) {
              int n_size = std::min(BLOCK_N, prefix_len - n_start);

              // 1. Gather K block into contiguous float buffer
              for (int j = 0; j < n_size; ++j) {
                int token_idx = req_to_token[req_idx * max_context_len + n_start + j];
                const scalar_t* k_src = k_buffer + token_idx * k_strideN + head_kv * k_strideH;
                float* k_dst = k_block_buf.data() + j * head_size;

                size_t vl;
                for (int d = 0; d < head_size; d += vl) {
                  vl = __riscv_vsetvl_e32m1(head_size - d);
                  vfloat32m1_t v_val = load_as_float_rvv(k_src + d, vl, scratch_a);
                  __riscv_vse32_v_f32m1(k_dst + d, v_val, vl);
                }
              }

              // 2. Compute Scores = Q[m_block] @ K_block.T
              // Q: [m_size, head_size], K_block: [n_size, head_size] -> Scores: [m_size, n_size]
              gemm_nt_rvv(
                  q_ptr + m_start * q_strideM,
                  k_block_buf.data(),  // Treated as float* (scalar_t=float)
                  scores_block.data(),
                  m_size,
                  n_size,
                  head_size,
                  q_strideM,
                  head_size,  // ldb for k_block_buf is head_size
                  n_size,     // ldc for scores_block
                  scaling,
                  scratch_a,
                  scratch_b);

              // 3. Gather V block into contiguous float buffer
              for (int j = 0; j < n_size; ++j) {
                int token_idx = req_to_token[req_idx * max_context_len + n_start + j];
                const scalar_t* v_src = v_buffer + token_idx * v_strideN + head_kv * v_strideH;
                float* v_dst = v_block_buf.data() + j * head_size_v;

                size_t vl;
                for (int d = 0; d < head_size_v; d += vl) {
                  vl = __riscv_vsetvl_e32m1(head_size_v - d);
                  vfloat32m1_t v_val = load_as_float_rvv(v_src + d, vl, scratch_a);
                  __riscv_vse32_v_f32m1(v_dst + d, v_val, vl);
                }
              }

              // 4. Update Online Softmax & Accumulate Output
              for (int i = 0; i < m_size; ++i) {
                float* row_scores = scores_block.data() + i * n_size;

                // Apply logit cap
                if (logit_cap > 0.0f) {
                  for (int j = 0; j < n_size; ++j) {
                    row_scores[j] = logit_cap * std::tanh(row_scores[j] / logit_cap);
                  }
                }

                // Find max
                float m_curr = -std::numeric_limits<float>::infinity();
                size_t vl;
                vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(m_curr, 1);
                for (int j = 0; j < n_size; j += vl) {
                  vl = __riscv_vsetvl_e32m1(n_size - j);
                  vfloat32m1_t v_val = __riscv_vle32_v_f32m1(row_scores + j, vl);
                  v_max = __riscv_vfredmax_vs_f32m1_f32m1(v_val, v_max, vl);
                }
                m_curr = __riscv_vfmv_f_s_f32m1_f32(v_max);

                // Update stats
                float m_prev = m_acc[i];
                float m_new = std::max(m_prev, m_curr);
                float alpha = (m_prev == -std::numeric_limits<float>::infinity()) ? 0.0f : std::exp(m_prev - m_new);

                // Compute exp(scores - m_new) and sum
                float l_curr = 0.0f;
                for (int j = 0; j < n_size; ++j) {
                  float val = std::exp(row_scores[j] - m_new);
                  row_scores[j] = val;  // P_ij
                  l_curr += val;
                }

                // Update l_acc
                l_acc[i] = l_acc[i] * alpha + l_curr;
                m_acc[i] = m_new;

                // Rescale O_acc
                float* row_o_acc = o_acc.data() + i * head_size_v;
                if (alpha != 1.0f) {
                  for (int d = 0; d < head_size_v; d += vl) {
                    vl = __riscv_vsetvl_e32m1(head_size_v - d);
                    vfloat32m1_t v_o = __riscv_vle32_v_f32m1(row_o_acc + d, vl);
                    v_o = __riscv_vfmul_vf_f32m1(v_o, alpha, vl);
                    __riscv_vse32_v_f32m1(row_o_acc + d, v_o, vl);
                  }
                }
              }

              // Accumulate P @ V -> O_acc
              // P: [m_size, n_size] (in scores_block), V: [n_size, head_size_v] (in v_block_buf)
              gemm_nn_acc_rvv(
                  scores_block.data(),
                  v_block_buf.data(),
                  o_acc.data(),
                  m_size,
                  head_size_v,
                  n_size,
                  n_size,       // lda
                  head_size_v,  // ldb
                  head_size_v,  // ldc
                  scratch_a);
            }
          }

          // -------------------------------------------------------
          // Phase 2: Extend (Contiguous Memory)
          // -------------------------------------------------------
          for (int n_start = 0; n_start < extend_len; n_start += BLOCK_N) {
            int n_size = std::min(BLOCK_N, extend_len - n_start);

            // 1. Compute Scores = Q[m_block] @ K_ext[n_block].T
            const scalar_t* k_ext_ptr = k_extend + start_loc * ke_strideN + head_kv * ke_strideH + n_start * ke_strideN;

            gemm_nt_rvv(
                q_ptr + m_start * q_strideM,
                k_ext_ptr,
                scores_block.data(),
                m_size,
                n_size,
                head_size,
                q_strideM,
                ke_strideN,
                n_size,
                scaling,
                scratch_a,
                scratch_b);

            // 2. Apply Causal Mask
            // Mask condition: n_idx > m_idx
            // n_idx = n_start + j
            // m_idx = m_start + i
            for (int i = 0; i < m_size; ++i) {
              int m_idx = m_start + i;
              float* row_scores = scores_block.data() + i * n_size;

              // We only need to mask if the block overlaps with the diagonal
              // i.e., if min_n <= m_idx < max_n
              // Optimization: if n_start > m_idx, all masked.
              // If n_start + n_size <= m_idx + 1, none masked.

              if (n_start > m_idx) {
                // All masked
                size_t vl;
                for (int j = 0; j < n_size; j += vl) {
                  vl = __riscv_vsetvl_e32m1(n_size - j);
                  __riscv_vse32_v_f32m1(
                      row_scores + j, __riscv_vfmv_v_f_f32m1(-std::numeric_limits<float>::infinity(), vl), vl);
                }
              } else if (n_start + n_size > m_idx + 1) {
                // Partial mask
                int valid_len = m_idx - n_start + 1;
                size_t vl;
                for (int j = valid_len; j < n_size; j += vl) {
                  vl = __riscv_vsetvl_e32m1(n_size - j);
                  __riscv_vse32_v_f32m1(
                      row_scores + j, __riscv_vfmv_v_f_f32m1(-std::numeric_limits<float>::infinity(), vl), vl);
                }
              }
            }

            // 3. Update Online Softmax & Accumulate Output
            // (Same logic as Phase 1, but V is contiguous)
            const scalar_t* v_ext_ptr = v_extend + start_loc * ve_strideN + head_kv * ve_strideH + n_start * ve_strideN;

            for (int i = 0; i < m_size; ++i) {
              float* row_scores = scores_block.data() + i * n_size;

              // Apply logit cap
              if (logit_cap > 0.0f) {
                for (int j = 0; j < n_size; ++j) {
                  row_scores[j] = logit_cap * std::tanh(row_scores[j] / logit_cap);
                }
              }

              // Find max
              float m_curr = -std::numeric_limits<float>::infinity();
              size_t vl;
              vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(m_curr, 1);
              for (int j = 0; j < n_size; j += vl) {
                vl = __riscv_vsetvl_e32m1(n_size - j);
                vfloat32m1_t v_val = __riscv_vle32_v_f32m1(row_scores + j, vl);
                v_max = __riscv_vfredmax_vs_f32m1_f32m1(v_val, v_max, vl);
              }
              m_curr = __riscv_vfmv_f_s_f32m1_f32(v_max);

              // Update stats
              float m_prev = m_acc[i];
              float m_new = std::max(m_prev, m_curr);
              float alpha = (m_prev == -std::numeric_limits<float>::infinity()) ? 0.0f : std::exp(m_prev - m_new);

              // Compute exp(scores - m_new) and sum
              float l_curr = 0.0f;
              for (int j = 0; j < n_size; ++j) {
                float val = std::exp(row_scores[j] - m_new);
                row_scores[j] = val;  // P_ij
                l_curr += val;
              }

              // Update l_acc
              l_acc[i] = l_acc[i] * alpha + l_curr;
              m_acc[i] = m_new;

              // Rescale O_acc
              float* row_o_acc = o_acc.data() + i * head_size_v;
              if (alpha != 1.0f) {
                for (int d = 0; d < head_size_v; d += vl) {
                  vl = __riscv_vsetvl_e32m1(head_size_v - d);
                  vfloat32m1_t v_o = __riscv_vle32_v_f32m1(row_o_acc + d, vl);
                  v_o = __riscv_vfmul_vf_f32m1(v_o, alpha, vl);
                  __riscv_vse32_v_f32m1(row_o_acc + d, v_o, vl);
                }
              }
            }

            // Accumulate P @ V_ext -> O_acc
            gemm_nn_acc_rvv(
                scores_block.data(),
                v_ext_ptr,
                o_acc.data(),
                m_size,
                head_size_v,
                n_size,
                n_size,       // lda
                ve_strideN,   // ldb
                head_size_v,  // ldc
                scratch_a);
          }

          // -------------------------------------------------------
          // Finalize: O = O_acc / l_acc
          // -------------------------------------------------------
          for (int i = 0; i < m_size; ++i) {
            scalar_t* out_row = o_ptr + (m_start + i) * q_strideM;
            float* acc_row = o_acc.data() + i * head_size_v;
            float inv_l = 1.0f / (l_acc[i] + 1e-6f);  // Avoid div by zero

            size_t vl;
            for (int d = 0; d < head_size_v; d += vl) {
              vl = __riscv_vsetvl_e32m1(head_size_v - d);
              vfloat32m1_t v_val = __riscv_vle32_v_f32m1(acc_row + d, vl);
              v_val = __riscv_vfmul_vf_f32m1(v_val, inv_l, vl);

              // Store with conversion
              store_from_float_rvv(out_row + d, v_val, vl, scratch_a);
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
