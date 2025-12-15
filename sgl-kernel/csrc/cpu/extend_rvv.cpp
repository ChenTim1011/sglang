// RISC-V Vector Extension (RVV) optimized extend attention kernels

#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "common.h"

#if defined(CPU_CAPABILITY_RVV)

#include <riscv_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

// Check for Zvfh (FP16 vector) support
#if defined(__riscv_zvfh) || defined(__riscv_v)
#define HAS_ZVFH 1
#else
#define HAS_ZVFH 0
#endif

namespace {

// Tiling constants adjusted for 32KB L1 Data Cache
constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 32;
constexpr int MAX_HEAD_SIZE = 256;

// Helper: Transpose matrix A [rows, cols] -> B [cols, rows]
template <typename scalar_t>
void transpose_block(const scalar_t* src, scalar_t* dst, int rows, int cols, int src_stride) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      dst[c * rows + r] = src[r * src_stride + c];
    }
  }
}

// GEMM: C = Q @ K^T
template <typename scalar_t>
void gemm_nt_rvv_tiled_transposed(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K_trans,
    float* __restrict__ C,
    int M,
    int N,
    int head_size,
    int q_strideM,
    int ldc,
    float scale) {
  size_t vl;
  // Loop over N columns
  for (int n = 0; n < N; n += vl) {
    vl = __riscv_vsetvl_e16m2(N - n);

    // Process M rows
    int m = 0;
    for (; m < M - 3; m += 4) {
      // Initialize accumulators
      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      // K reduction loop
      for (int k = 0; k < head_size; ++k) {
        const scalar_t* k_ptr = K_trans + k * BLOCK_N + n;

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_k = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr), vl);

          // Load Q scalars
          float q0 = static_cast<float>(Q[(m + 0) * q_strideM + k]);
          float q1 = static_cast<float>(Q[(m + 1) * q_strideM + k]);
          float q2 = static_cast<float>(Q[(m + 2) * q_strideM + k]);
          float q3 = static_cast<float>(Q[(m + 3) * q_strideM + k]);

          acc0 = __riscv_vfwmacc_vf_f32m4(acc0, (_Float16)(q0), v_k, vl);
          acc1 = __riscv_vfwmacc_vf_f32m4(acc1, (_Float16)(q1), v_k, vl);
          acc2 = __riscv_vfwmacc_vf_f32m4(acc2, (_Float16)(q2), v_k, vl);
          acc3 = __riscv_vfwmacc_vf_f32m4(acc3, (_Float16)(q3), v_k, vl);
        } else {
          // Fallback for non-Half types (BFloat16, Float)
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

    // Handle remaining M rows
    for (; m < M; ++m) {
      vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      for (int k = 0; k < head_size; ++k) {
        const scalar_t* k_ptr = K_trans + k * BLOCK_N + n;
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

// GEMM: O += P @ V
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
  // Outer loop over M blocks
  for (int m_base = 0; m_base < M; m_base += 4) {
    int m_count = std::min(4, M - m_base);

    // Loop over head_size_v (Dv) columns
    size_t vl;
    for (int d = 0; d < head_size_v; d += vl) {
      vl = __riscv_vsetvl_e16m2(head_size_v - d);

      // Initialize Accumulators for 4 rows
      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      // Reduction over N
      for (int n = 0; n < N; ++n) {
        // Load V vector: V[n, d:d+vl]
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

      // Add to existing O and Store
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
  // Dynamic head size check
  if (head_size > MAX_HEAD_SIZE || head_size_v > MAX_HEAD_SIZE) {
    printf(
        "RVV Extend Attention: Head size %d exceeds compiled maximum %d. Aborting optimization.\n",
        std::max(head_size, head_size_v),
        MAX_HEAD_SIZE);
    return;
  }

#pragma omp parallel
  {
    // 1. Scores Block [BLOCK_M, BLOCK_N] (FP32)
    alignas(64) float scores_block[BLOCK_M * BLOCK_N];

    // 2. K Tile Buffer Transposed [head_size, BLOCK_N] (scalar_t)
    alignas(64) scalar_t k_trans_buf[MAX_HEAD_SIZE * BLOCK_N];

    // 3. V Tile Buffer [BLOCK_N, head_size_v] (scalar_t)
    alignas(64) scalar_t v_buf[BLOCK_N * MAX_HEAD_SIZE];

    // 4. Online Softmax Accumulators
    alignas(64) float o_acc[BLOCK_M * MAX_HEAD_SIZE];
    alignas(64) float l_acc[BLOCK_M];
    alignas(64) float m_acc[BLOCK_M];

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

        // Loop over Query Blocks (BLOCK_M)
        for (int m_start = 0; m_start < extend_len; m_start += BLOCK_M) {
          int m_size = std::min(BLOCK_M, extend_len - m_start);

          // Initialize accumulators
          std::fill(l_acc, l_acc + m_size, 0.0f);
          std::fill(m_acc, m_acc + m_size, -std::numeric_limits<float>::infinity());
          std::memset(o_acc, 0, m_size * head_size_v * sizeof(float));

          // Phase 1: Prefix (Paged Memory)
          if (prefix_len > 0) {
            for (int n_start = 0; n_start < prefix_len; n_start += BLOCK_N) {
              int n_size = std::min(BLOCK_N, prefix_len - n_start);

              // Gather K directly into k_trans_buf (Transposed: [head_size, n_size])
              for (int j = 0; j < n_size; ++j) {
                int token_idx = req_to_token[req_idx * max_context_len + n_start + j];
                const scalar_t* k_src = k_buffer + token_idx * k_strideN + head_kv * k_strideH;

                // Scatter/Store loop
                for (int d = 0; d < head_size; ++d) {
                  k_trans_buf[d * BLOCK_N + j] = k_src[d];
                }
              }

              // Compute Q @ K.T -> Scores
              gemm_nt_rvv_tiled_transposed(
                  q_ptr + m_start * q_strideM,
                  k_trans_buf,
                  scores_block,
                  m_size,
                  n_size,
                  head_size,
                  q_strideM,
                  BLOCK_N,
                  scaling);

              // Gather V directly into v_buf (Normal: [n_size, head_size_v])
              for (int j = 0; j < n_size; ++j) {
                int token_idx = req_to_token[req_idx * max_context_len + n_start + j];
                const scalar_t* v_src = v_buffer + token_idx * v_strideN + head_kv * v_strideH;

                scalar_t* v_dst = v_buf + j * head_size_v;
                std::memcpy(v_dst, v_src, head_size_v * sizeof(scalar_t));
              }

              // Softmax Update
              for (int i = 0; i < m_size; ++i) {
                float* row_scores = scores_block + i * BLOCK_N;
                if (logit_cap > 0.0f) {
                  for (int j = 0; j < n_size; ++j)
                    row_scores[j] = logit_cap * std::tanh(row_scores[j] / logit_cap);
                }

                float m_curr = -std::numeric_limits<float>::infinity();
                size_t vl;
                vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(m_curr, 1);
                for (int j = 0; j < n_size; j += vl) {
                  vl = __riscv_vsetvl_e32m4(n_size - j);
                  vfloat32m4_t v_val = __riscv_vle32_v_f32m4(row_scores + j, vl);
                  v_max = __riscv_vfredmax_vs_f32m4_f32m1(v_val, v_max, vl);
                }
                m_curr = __riscv_vfmv_f_s_f32m1_f32(v_max);

                float m_prev = m_acc[i];
                float m_new = std::max(m_prev, m_curr);
                float alpha = (m_prev == -std::numeric_limits<float>::infinity()) ? 0.0f : std::exp(m_prev - m_new);

                l_acc[i] = l_acc[i] * alpha;
                m_acc[i] = m_new;

                if (alpha != 1.0f) {
                  float* acc_row = o_acc + i * head_size_v;
                  for (int d = 0; d < head_size_v; d += vl) {
                    vl = __riscv_vsetvl_e32m4(head_size_v - d);
                    vfloat32m4_t v_o = __riscv_vle32_v_f32m4(acc_row + d, vl);
                    v_o = __riscv_vfmul_vf_f32m4(v_o, alpha, vl);
                    __riscv_vse32_v_f32m4(acc_row + d, v_o, vl);
                  }
                }

                float l_curr = 0.0f;
                for (int j = 0; j < n_size; ++j) {
                  float val = std::exp(row_scores[j] - m_new);
                  row_scores[j] = val;
                  l_curr += val;
                }
                l_acc[i] += l_curr;
              }

              // Accumulate P @ V
              gemm_nn_rvv_tiled(scores_block, v_buf, o_acc, m_size, n_size, head_size_v, BLOCK_N, head_size_v);
            }
          }

          // Phase 2: Extend (Contiguous Memory)
          for (int n_start = 0; n_start < extend_len; n_start += BLOCK_N) {
            int n_size = std::min(BLOCK_N, extend_len - n_start);

            // Copy K_extend to k_trans_buf (Transposing)
            const scalar_t* k_ext_ptr = k_extend + start_loc * ke_strideN + head_kv * ke_strideH + n_start * ke_strideN;

            for (int j = 0; j < n_size; ++j) {
              const scalar_t* src_tok = k_ext_ptr + j * ke_strideN;
              for (int d = 0; d < head_size; ++d) {
                k_trans_buf[d * BLOCK_N + j] = src_tok[d];
              }
            }

            // Gemm Q @ K
            gemm_nt_rvv_tiled_transposed(
                q_ptr + m_start * q_strideM,
                k_trans_buf,
                scores_block,
                m_size,
                n_size,
                head_size,
                q_strideM,
                BLOCK_N,
                scaling);

            // Apply Causal Mask
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

            // Copy V Extend to v_buf
            const scalar_t* v_ext_ptr = v_extend + start_loc * ve_strideN + head_kv * ve_strideH + n_start * ve_strideN;
            for (int j = 0; j < n_size; ++j) {
              const scalar_t* src_tok = v_ext_ptr + j * ve_strideN;
              scalar_t* dst_tok = v_buf + j * head_size_v;
              std::memcpy(dst_tok, src_tok, head_size_v * sizeof(scalar_t));
            }

            // Softmax Update
            for (int i = 0; i < m_size; ++i) {
              float* row_scores = scores_block + i * BLOCK_N;
              if (logit_cap > 0.0f) {
                for (int j = 0; j < n_size; ++j)
                  row_scores[j] = logit_cap * std::tanh(row_scores[j] / logit_cap);
              }
              float m_curr = -std::numeric_limits<float>::infinity();
              size_t vl;
              vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(m_curr, 1);
              for (int j = 0; j < n_size; j += vl) {
                vl = __riscv_vsetvl_e32m4(n_size - j);
                vfloat32m4_t v_val = __riscv_vle32_v_f32m4(row_scores + j, vl);
                v_max = __riscv_vfredmax_vs_f32m4_f32m1(v_val, v_max, vl);
              }
              m_curr = __riscv_vfmv_f_s_f32m1_f32(v_max);

              float m_prev = m_acc[i];
              float m_new = std::max(m_prev, m_curr);
              float alpha = (m_prev == -std::numeric_limits<float>::infinity()) ? 0.0f : std::exp(m_prev - m_new);

              l_acc[i] = l_acc[i] * alpha;
              m_acc[i] = m_new;

              if (alpha != 1.0f) {
                float* acc_row = o_acc + i * head_size_v;
                for (int d = 0; d < head_size_v; d += vl) {
                  vl = __riscv_vsetvl_e32m4(head_size_v - d);
                  vfloat32m4_t v_o = __riscv_vle32_v_f32m4(acc_row + d, vl);
                  v_o = __riscv_vfmul_vf_f32m4(v_o, alpha, vl);
                  __riscv_vse32_v_f32m4(acc_row + d, v_o, vl);
                }
              }

              float l_curr = 0.0f;
              for (int j = 0; j < n_size; ++j) {
                float val = std::exp(row_scores[j] - m_new);
                row_scores[j] = val;
                l_curr += val;
              }
              l_acc[i] += l_curr;
            }

            // Accumulate P @ V
            gemm_nn_rvv_tiled(scores_block, v_buf, o_acc, m_size, n_size, head_size_v, BLOCK_N, head_size_v);
          }

          // Finalize: O = O_acc / l_acc
          for (int i = 0; i < m_size; ++i) {
            scalar_t* out_row = o_ptr + (m_start + i) * q_strideM;
            float* acc_row = o_acc + i * head_size_v;
            float inv_l = 1.0f / (l_acc[i] + 1e-6f);

            size_t vl;
            for (int d = 0; d < head_size_v; d += vl) {
              vl = __riscv_vsetvl_e32m4(head_size_v - d);  // LMUL matches acc buffers
              vfloat32m4_t v_val = __riscv_vle32_v_f32m4(acc_row + d, vl);
              v_val = __riscv_vfmul_vf_f32m4(v_val, inv_l, vl);

              // Store with Cast
              if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH
                vfloat16m2_t v_f16 = __riscv_vfncvt_f_f_w_f16m2(v_val, vl);
                __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(out_row + d), v_f16, vl);
#else
                // Fallback
                float temp[128];
                __riscv_vse32_v_f32m4(temp, v_val, vl);
                for (int k = 0; k < vl; ++k)
                  out_row[d + k] = static_cast<scalar_t>(temp[k]);
#endif
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
