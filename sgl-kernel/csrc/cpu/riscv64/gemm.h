#pragma once

#include <ATen/core/Tensor.h>

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "riscv64/vector_helpers.h"

// 4 m4-accumulators × 4 phys-regs = 16 of 32 VRs; leaves 16 for operands.
static constexpr int GEMM_TILE_M = 4;

// Weight packing (RVV block-N format)
at::Tensor convert_weight_packed(at::Tensor& weight);

inline int64_t get_int8_packed_row_size(int64_t K) {
  return K + static_cast<int64_t>(sizeof(int32_t));
}

inline int64_t get_int8_packed_block_size(int64_t K) {
  return rvv_constants::BLOCK_N * get_int8_packed_row_size(K);
}

// GEMM Kernel Declarations

template <typename scalar_t>
void int8_scaled_mm_kernel(
    scalar_t* __restrict__ out,
    const uint8_t* __restrict__ mat1,
    const int8_t* __restrict__ mat2,
    const float* __restrict__ scales1,
    const float* __restrict__ scales2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed);

extern template void int8_scaled_mm_kernel<float>(
    float* out,
    const uint8_t* mat1,
    const int8_t* mat2,
    const float* scales1,
    const float* scales2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed);

extern template void int8_scaled_mm_kernel<at::Half>(
    at::Half* out,
    const uint8_t* mat1,
    const int8_t* mat2,
    const float* scales1,
    const float* scales2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed);

extern template void int8_scaled_mm_kernel<at::BFloat16>(
    at::BFloat16* out,
    const uint8_t* mat1,
    const int8_t* mat2,
    const float* scales1,
    const float* scales2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed);

// TinyGEMM Interface for RVV
template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    float* __restrict__ Ctmp,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);

// TinyGEMM Interface for INT8 (RVV W8A8) - activation is uint8, weight is int8.
template <typename scalar_t>
void tinygemm_kernel(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);

// TinyGEMM Interface matching the shared CPU declaration used by callers that
// include cpu/gemm.h.
template <typename scalar_t>
void tinygemm_kernel(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int32_t* __restrict__ Ctmp,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);

// Inline Attention Kernels (Template Implementations)

#if defined(CPU_CAPABILITY_RVV)
#include <riscv_vector.h>

#include "vector_math.h"

template <typename scalar_t>
inline void gemm_nt_tiled_transposed(
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
  size_t vl_max = __riscv_vsetvlmax_e32m4();

  for (int m_base = 0; m_base < M; m_base += GEMM_TILE_M) {
    int m_count = std::min(GEMM_TILE_M, M - m_base);

    for (int n_base = 0; n_base < N; n_base += vl_max) {
      size_t vl = __riscv_vsetvl_e32m4(N - n_base);

      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      for (int k = 0; k < head_size; ++k) {
        const scalar_t* k_ptr = K_trans + k * block_n + n_base;

        vfloat16m2_t v_k_f16;
        vfloat32m4_t v_k_f32;

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          v_k_f16 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr), vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          v_k_f32 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr), vl);
        } else {
          // scalar_t=float: identity cast, load directly
          v_k_f32 = __riscv_vle32_v_f32m4(reinterpret_cast<const float*>(k_ptr), vl);
        }

        float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
        if (m_count > 0) q0 = static_cast<float>(Q[(m_base + 0) * q_strideM + k]);
        if (m_count > 1) q1 = static_cast<float>(Q[(m_base + 1) * q_strideM + k]);
        if (m_count > 2) q2 = static_cast<float>(Q[(m_base + 2) * q_strideM + k]);
        if (m_count > 3) q3 = static_cast<float>(Q[(m_base + 3) * q_strideM + k]);

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          acc0 = vfwmacc_f16_scalar_to_f32m4(acc0, (_Float16)q0, v_k_f16, vl);
          if (m_count > 1) acc1 = vfwmacc_f16_scalar_to_f32m4(acc1, (_Float16)q1, v_k_f16, vl);
          if (m_count > 2) acc2 = vfwmacc_f16_scalar_to_f32m4(acc2, (_Float16)q2, v_k_f16, vl);
          if (m_count > 3) acc3 = vfwmacc_f16_scalar_to_f32m4(acc3, (_Float16)q3, v_k_f16, vl);
        } else {
          acc0 = __riscv_vfmacc_vf_f32m4(acc0, q0, v_k_f32, vl);
          if (m_count > 1) acc1 = __riscv_vfmacc_vf_f32m4(acc1, q1, v_k_f32, vl);
          if (m_count > 2) acc2 = __riscv_vfmacc_vf_f32m4(acc2, q2, v_k_f32, vl);
          if (m_count > 3) acc3 = __riscv_vfmacc_vf_f32m4(acc3, q3, v_k_f32, vl);
        }
      }

      auto store = [&](int idx, vfloat32m4_t acc) {
        if (idx < m_count) {
          __riscv_vse32_v_f32m4(C + (m_base + idx) * ldc + n_base, __riscv_vfmul_vf_f32m4(acc, scale, vl), vl);
        }
      };

      store(0, acc0);
      store(1, acc1);
      store(2, acc2);
      store(3, acc3);
    }
  }
}

// NOTE: O must be zero-initialized by the caller before the first call.
// This function accumulates (load-add-store) onto O across tiled K iterations.
template <typename scalar_t>
inline void gemm_nn_tiled(
    const float* __restrict__ P,
    const scalar_t* __restrict__ V,
    float* __restrict__ O,
    int M,
    int N,
    int head_size_v,
    int p_strideN,
    int v_strideH) {
  for (int m_base = 0; m_base < M; m_base += GEMM_TILE_M) {
    int m_count = std::min(GEMM_TILE_M, M - m_base);
    size_t vl;
    for (int d = 0; d < head_size_v; d += vl) {
      vl = __riscv_vsetvl_e32m4(head_size_v - d);

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
          // scalar_t=float: identity cast, load directly
          vfloat32m4_t v_v = __riscv_vle32_v_f32m4(reinterpret_cast<const float*>(v_ptr), vl);

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

// INT8 Attention Kernels

template <typename scalar_t>
inline void gemm_nt_tiled_transposed_int8(
    const scalar_t* __restrict__ Q,
    const int8_t* __restrict__ K_trans,
    float* __restrict__ C,
    int M,
    int N,
    int head_size,
    int q_strideM,
    int block_n,
    int ldc,
    float scale,
    float k_scale,
    const float* k_scales_per_token = nullptr) {
  size_t vl_max = __riscv_vsetvlmax_e32m4();

  for (int m_base = 0; m_base < M; m_base += GEMM_TILE_M) {
    int m_count = std::min(GEMM_TILE_M, M - m_base);

    // Prepare Q quantization: Quantize GEMM_TILE_M rows of Q
    TORCH_CHECK(head_size <= MAX_HEAD_SIZE, "head_size ", head_size, " exceeds MAX_HEAD_SIZE ", MAX_HEAD_SIZE);
    int8_t q_quant[GEMM_TILE_M][MAX_HEAD_SIZE];
    float q_scales[GEMM_TILE_M];

    for (int i = 0; i < m_count; ++i) {
      float max_abs = 0.0f;
      const scalar_t* q_row = Q + (m_base + i) * q_strideM;
      for (int k = 0; k < head_size; ++k) {
        max_abs = std::max(max_abs, std::abs(static_cast<float>(q_row[k])));
      }

      q_scales[i] = max_abs / 127.0f;
      float inv_scale = (q_scales[i] > 1e-9f) ? (1.0f / q_scales[i]) : 1.0f;

      for (int k = 0; k < head_size; ++k) {
        float val = static_cast<float>(q_row[k]) * inv_scale;
        q_quant[i][k] = static_cast<int8_t>(std::clamp(std::round(val), -127.0f, 127.0f));
      }
    }

    // GEMM Loop
    for (int n_base = 0; n_base < N; n_base += vl_max) {
      size_t vl = __riscv_vsetvl_e32m4(N - n_base);

      vint32m4_t acc0 = __riscv_vmv_v_x_i32m4(0, vl);
      vint32m4_t acc1 = __riscv_vmv_v_x_i32m4(0, vl);
      vint32m4_t acc2 = __riscv_vmv_v_x_i32m4(0, vl);
      vint32m4_t acc3 = __riscv_vmv_v_x_i32m4(0, vl);

      for (int k = 0; k < head_size; ++k) {
        const int8_t* k_ptr = K_trans + k * block_n + n_base;
        vint8m1_t v_k = __riscv_vle8_v_i8m1(k_ptr, vl);
        vint16m2_t v_k16 = __riscv_vsext_vf2_i16m2(v_k, vl);

        if (m_count > 0) {
          acc0 = __riscv_vwmacc_vx_i32m4(acc0, static_cast<int16_t>(q_quant[0][k]), v_k16, vl);
        }
        if (m_count > 1) {
          acc1 = __riscv_vwmacc_vx_i32m4(acc1, static_cast<int16_t>(q_quant[1][k]), v_k16, vl);
        }
        if (m_count > 2) {
          acc2 = __riscv_vwmacc_vx_i32m4(acc2, static_cast<int16_t>(q_quant[2][k]), v_k16, vl);
        }
        if (m_count > 3) {
          acc3 = __riscv_vwmacc_vx_i32m4(acc3, static_cast<int16_t>(q_quant[3][k]), v_k16, vl);
        }
      }

      // Dequantize and store
      auto store = [&](int idx, vint32m4_t acc_i32) {
        if (idx < m_count) {
          vfloat32m4_t acc_f = __riscv_vfcvt_f_x_v_f32m4(acc_i32, vl);
          if (k_scales_per_token) {
            // Per-token scale: element-wise multiply across the N tile
            float qscale = scale * q_scales[idx];
            acc_f = __riscv_vfmul_vf_f32m4(acc_f, qscale, vl);
            vfloat32m4_t tok_k_scales = __riscv_vle32_v_f32m4(k_scales_per_token + n_base, vl);
            acc_f = __riscv_vfmul_vv_f32m4(acc_f, tok_k_scales, vl);
          } else {
            float combined_scale = scale * q_scales[idx] * k_scale;
            acc_f = __riscv_vfmul_vf_f32m4(acc_f, combined_scale, vl);
          }
          __riscv_vse32_v_f32m4(C + (m_base + idx) * ldc + n_base, acc_f, vl);
        }
      };

      store(0, acc0);
      store(1, acc1);
      store(2, acc2);
      store(3, acc3);
    }
  }
}
// Layout: K_trans[d * block_n + j] = key j, dimension d  (pre-transposed)
// Per-token dequant: C[m][n] = sum_k(Q[m][k] * float(K_trans[k*block_n+n])) * scale * k_scale[n]
template <typename scalar_t>
inline void gemm_nt_tiled_transposed_fp_q_int8_k(
    const scalar_t* __restrict__ Q,
    const int8_t* __restrict__ K_trans,
    float* __restrict__ C,
    int M,
    int N,
    int head_size,
    int q_strideM,
    int block_n,
    int ldc,
    float scale,
    float k_scale,
    const float* k_scales_per_token = nullptr) {
  size_t vl_max = __riscv_vsetvlmax_e32m4();

  for (int m_base = 0; m_base < M; m_base += GEMM_TILE_M) {
    int m_count = std::min(GEMM_TILE_M, M - m_base);

    for (int n_base = 0; n_base < N; n_base += vl_max) {
      size_t vl = __riscv_vsetvl_e32m4(N - n_base);

      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      for (int k = 0; k < head_size; ++k) {
        // Load INT8 keys for this K-dimension slice, widen to FP32
        const int8_t* k_ptr = K_trans + k * block_n + n_base;
        vint8m1_t v_k8 = __riscv_vle8_v_i8m1(k_ptr, vl);
        vint32m4_t v_k32 = __riscv_vsext_vf4_i32m4(v_k8, vl);
        vfloat32m4_t v_kf = __riscv_vfcvt_f_x_v_f32m4(v_k32, vl);

        // Q stays FP — load scalar from BF16/FP16 row, MACC with FP32 K vector
        if (m_count > 0) {
          float q0 = static_cast<float>(Q[(m_base + 0) * q_strideM + k]);
          acc0 = __riscv_vfmacc_vf_f32m4(acc0, q0, v_kf, vl);
        }
        if (m_count > 1) {
          float q1 = static_cast<float>(Q[(m_base + 1) * q_strideM + k]);
          acc1 = __riscv_vfmacc_vf_f32m4(acc1, q1, v_kf, vl);
        }
        if (m_count > 2) {
          float q2 = static_cast<float>(Q[(m_base + 2) * q_strideM + k]);
          acc2 = __riscv_vfmacc_vf_f32m4(acc2, q2, v_kf, vl);
        }
        if (m_count > 3) {
          float q3 = static_cast<float>(Q[(m_base + 3) * q_strideM + k]);
          acc3 = __riscv_vfmacc_vf_f32m4(acc3, q3, v_kf, vl);
        }
      }

      // Store: apply attention scale and per-token K dequant scale
      auto store = [&](int idx, vfloat32m4_t acc_f) {
        if (idx < m_count) {
          if (k_scales_per_token) {
            vfloat32m4_t tok_k_scales = __riscv_vle32_v_f32m4(k_scales_per_token + n_base, vl);
            acc_f = __riscv_vfmul_vf_f32m4(acc_f, scale, vl);
            acc_f = __riscv_vfmul_vv_f32m4(acc_f, tok_k_scales, vl);
          } else {
            acc_f = __riscv_vfmul_vf_f32m4(acc_f, scale * k_scale, vl);
          }
          __riscv_vse32_v_f32m4(C + (m_base + idx) * ldc + n_base, acc_f, vl);
        }
      };

      store(0, acc0);
      store(1, acc1);
      store(2, acc2);
      store(3, acc3);
    }
  }
}

inline void gemm_nn_tiled_int8(
    const float* __restrict__ P,
    const int8_t* __restrict__ V,
    float* __restrict__ O,
    int M,
    int N,
    int head_size_v,
    int p_strideN,
    int v_strideH,
    float v_scale,
    const float* v_scales_per_token = nullptr) {
  // P: [M, N] (probabilities)
  // V: [N, D] (values, int8, row-major)
  // O: [M, D] (output)

  for (int m_base = 0; m_base < M; m_base += GEMM_TILE_M) {
    int m_count = std::min(GEMM_TILE_M, M - m_base);
    size_t vl;
    for (int d = 0; d < head_size_v; d += vl) {
      vl = __riscv_vsetvl_e32m4(head_size_v - d);

      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      for (int n = 0; n < N; ++n) {
        const int8_t* v_ptr = V + n * v_strideH + d;
        vfloat32m4_t v_v_f32 = int8_to_f32m4(v_ptr, vl);

        // Apply per-token scale inside loop (correct for per-token dynamic quant)
        if (v_scales_per_token) {
          v_v_f32 = __riscv_vfmul_vf_f32m4(v_v_f32, v_scales_per_token[n], vl);
        }

        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        if (m_count > 0) {
          p0 = P[(m_base + 0) * p_strideN + n];
        }
        if (m_count > 1) {
          p1 = P[(m_base + 1) * p_strideN + n];
        }
        if (m_count > 2) {
          p2 = P[(m_base + 2) * p_strideN + n];
        }
        if (m_count > 3) {
          p3 = P[(m_base + 3) * p_strideN + n];
        }

        acc0 = __riscv_vfmacc_vf_f32m4(acc0, p0, v_v_f32, vl);
        if (m_count > 1) {
          acc1 = __riscv_vfmacc_vf_f32m4(acc1, p1, v_v_f32, vl);
        }
        if (m_count > 2) {
          acc2 = __riscv_vfmacc_vf_f32m4(acc2, p2, v_v_f32, vl);
        }
        if (m_count > 3) {
          acc3 = __riscv_vfmacc_vf_f32m4(acc3, p3, v_v_f32, vl);
        }
      }

      auto store_o = [&](int idx, vfloat32m4_t acc) {
        if (idx < m_count) {
          float* o_ptr = O + (m_base + idx) * head_size_v + d;
          vfloat32m4_t old_o = __riscv_vle32_v_f32m4(o_ptr, vl);
          // When per-token scales were already applied inside the loop, skip global v_scale
          if (!v_scales_per_token) {
            acc = __riscv_vfmul_vf_f32m4(acc, v_scale, vl);
          }
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

#endif  // CPU_CAPABILITY_RVV
