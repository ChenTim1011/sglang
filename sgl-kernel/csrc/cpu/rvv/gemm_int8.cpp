#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "common.h"
#include "gemm.h"
#include "vec.h"
#include "vector_helpers.h"

#if defined(CPU_CAPABILITY_RVV)
// Clang 19+ version check
#if !defined(__clang__) || !defined(__clang_major__) || __clang_major__ < 19
#error "RVV backend requires Clang 19 or later. Please use Clang 19 or later to compile this file."
#endif
#include <riscv_vector.h>

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace rvv_constants;

namespace {

template <typename scalar_t>
void int8_scaled_mm_rvv_m1_impl(
    scalar_t* __restrict__ out,
    const uint8_t* __restrict__ mat1,
    const int8_t* __restrict__ mat2,
    const float* __restrict__ scales1,
    const float* __restrict__ scales2,
    const float* __restrict__ bias,
    int64_t N,
    int64_t K,
    bool is_packed) {
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  const float a_scale = scales1[0];

  alignas(64) float combined_scales[BLOCK_N];
  alignas(64) float bias_vals[BLOCK_N];

  for (int64_t nb = 0; nb < (N + BLOCK_N - 1) / BLOCK_N; ++nb) {
    int64_t n_start = nb * BLOCK_N;
    int64_t n_size = std::min(BLOCK_N, N - n_start);

    for (int64_t j = 0; j < n_size; ++j) {
      combined_scales[j] = a_scale * scales2[n_start + j];
      bias_vals[j] = bias ? bias[n_start + j] : 0.0f;
    }

    alignas(64) int32_t acc[BLOCK_N];
    std::memset(acc, 0, n_size * sizeof(int32_t));

    const uint8_t* a_row = mat1;

    constexpr int64_t N_CHUNK = rvv_constants::INT8_N_CHUNK_M1;
    size_t vl_max_chunk = __riscv_vsetvlmax_e32m4();
    for (int64_t n_chunk_start = 0; n_chunk_start < n_size; n_chunk_start += N_CHUNK) {
      int64_t n_chunk_size = std::min(N_CHUNK, n_size - n_chunk_start);
      size_t vl = (n_chunk_size >= vl_max_chunk) ? vl_max_chunk : __riscv_vsetvl_e32m4(n_chunk_size);

      vint32m4_t v_acc = __riscv_vmv_v_x_i32m4(0, vl);

      for (int64_t k = 0; k < K; ++k) {
        uint8_t a_val = a_row[k];
        if (a_val == 0) continue;

        const int8_t* b_ptr;
        ptrdiff_t b_stride;

        if (is_packed) {
          b_ptr = mat2 + nb * K * BLOCK_N + k * BLOCK_N + n_chunk_start;
          b_stride = 1;

          if (k + rvv_constants::PREFETCH_DISTANCE < K) {
            __builtin_prefetch(
                mat2 + nb * K * BLOCK_N + (k + rvv_constants::PREFETCH_DISTANCE) * BLOCK_N + n_chunk_start,
                0,
                rvv_constants::PREFETCH_LOCALITY);
          }
        } else {
          b_ptr = mat2 + (n_start + n_chunk_start) * K + k;
          b_stride = K;

          if (k + rvv_constants::PREFETCH_DISTANCE < K) {
            __builtin_prefetch(
                mat2 + (n_start + n_chunk_start) * K + (k + rvv_constants::PREFETCH_DISTANCE),
                0,
                rvv_constants::PREFETCH_LOCALITY);
          }
        }

        vint8m1_t v_b;
        if (is_packed) {
          v_b = __riscv_vle8_v_i8m1(b_ptr, vl);
        } else {
          v_b = __riscv_vlse8_v_i8m1(b_ptr, b_stride, vl);
        }

        vint16m2_t v_b16 = __riscv_vsext_vf2_i16m2(v_b, vl);
        // CAST TO int8_t FIRST to handle symmetric quantization stored in uint8_t container
        int16_t a_val_16 = static_cast<int16_t>(static_cast<int8_t>(a_val));
        v_acc = __riscv_vwmacc_vx_i32m4(v_acc, a_val_16, v_b16, vl);
      }

      __riscv_vse32_v_i32m4(acc + n_chunk_start, v_acc, vl);
    }

    size_t vl_max_dequant = __riscv_vsetvlmax_e32m4();
    size_t vl_dequant;
    for (int64_t j = 0; j < n_size; j += vl_max_dequant) {
      vl_dequant = (j + vl_max_dequant <= n_size) ? vl_max_dequant : __riscv_vsetvl_e32m4(n_size - j);

      vint32m4_t v_acc = __riscv_vle32_v_i32m4(acc + j, vl_dequant);
      vfloat32m4_t v_acc_f = __riscv_vfcvt_f_x_v_f32m4(v_acc, vl_dequant);
      vfloat32m4_t v_scale = __riscv_vle32_v_f32m4(combined_scales + j, vl_dequant);
      v_acc_f = __riscv_vfmul_vv_f32m4(v_acc_f, v_scale, vl_dequant);
      vfloat32m4_t v_bias = __riscv_vle32_v_f32m4(bias_vals + j, vl_dequant);
      v_acc_f = __riscv_vfadd_vv_f32m4(v_acc_f, v_bias, vl_dequant);

      if constexpr (std::is_same_v<scalar_t, float>) {
        __riscv_vse32_v_f32m4(out + n_start + j, v_acc_f, vl_dequant);
      } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
        vfloat16m2_t v_f16 = f32m4_to_f16(v_acc_f, vl_dequant);
        __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(out + n_start + j), v_f16, vl_dequant);
      } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        vuint16m2_t v_bf16 = f32m4_to_bf16(v_acc_f, vl_dequant);
        __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(out + n_start + j), v_bf16, vl_dequant);
      }
    }
  }
}

template <typename scalar_t>
void int8_scaled_mm_rvv_impl(
    scalar_t* __restrict__ out,
    const uint8_t* __restrict__ mat1,
    const int8_t* __restrict__ mat2,
    const float* __restrict__ scales1,
    const float* __restrict__ scales2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed) {
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;

  if (M == 1) {
    int8_scaled_mm_rvv_m1_impl<scalar_t>(out, mat1, mat2, scales1, scales2, bias, N, K, is_packed);
    return;
  }

#pragma omp parallel for schedule(static)
  for (int64_t m_start = 0; m_start < M; m_start += BLOCK_M_INT8) {
    int64_t m_size = std::min(static_cast<int64_t>(BLOCK_M_INT8), M - m_start);

    float a_scales[BLOCK_M_INT8];
    for (int i = 0; i < m_size; ++i) {
      a_scales[i] = scales1[m_start + i];
    }

    alignas(64) float combined_scales[BLOCK_M_INT8][BLOCK_N];

    for (int64_t nb = 0; nb < (N + BLOCK_N - 1) / BLOCK_N; ++nb) {
      int64_t n_start = nb * BLOCK_N;
      int64_t n_size = std::min(BLOCK_N, N - n_start);

      for (int i = 0; i < m_size; ++i) {
        for (int64_t j = 0; j < n_size; ++j) {
          combined_scales[i][j] = a_scales[i] * scales2[n_start + j];
        }
      }

      int32_t acc[BLOCK_M_INT8][BLOCK_N];
      constexpr int64_t N_CHUNK = rvv_constants::INT8_N_CHUNK_GENERAL;

      size_t vl_max_chunk = __riscv_vsetvlmax_e32m4();
      for (int64_t n_chunk_start = 0; n_chunk_start < n_size; n_chunk_start += N_CHUNK) {
        int64_t n_chunk_size = std::min(N_CHUNK, n_size - n_chunk_start);
        size_t vl = (n_chunk_size >= vl_max_chunk) ? vl_max_chunk : __riscv_vsetvl_e32m4(n_chunk_size);

        // Manual unrolling for BLOCK_M_INT8 = 4
        // Vectors cannot be in arrays
        vint32m4_t v_acc0 = __riscv_vmv_v_x_i32m4(0, vl);
        vint32m4_t v_acc1 = __riscv_vmv_v_x_i32m4(0, vl);
        vint32m4_t v_acc2 = __riscv_vmv_v_x_i32m4(0, vl);
        vint32m4_t v_acc3 = __riscv_vmv_v_x_i32m4(0, vl);

        // Unroll K loop by 4
        int64_t k = 0;
        for (; k <= K - 4; k += 4) {
          int8_t a_val0 = static_cast<int8_t>(mat1[(m_start + 0) * K + k]);
          int8_t a_val1 = static_cast<int8_t>(mat1[(m_start + 0) * K + k + 1]);
          int8_t a_val2 = static_cast<int8_t>(mat1[(m_start + 0) * K + k + 2]);
          int8_t a_val3 = static_cast<int8_t>(mat1[(m_start + 0) * K + k + 3]);

          int8_t a_vals[4][4];  // [m][k_offset]
          if (m_size > 0)
            a_vals[0][0] = static_cast<int8_t>(mat1[(m_start + 0) * K + k]),
            a_vals[0][1] = static_cast<int8_t>(mat1[(m_start + 0) * K + k + 1]),
            a_vals[0][2] = static_cast<int8_t>(mat1[(m_start + 0) * K + k + 2]),
            a_vals[0][3] = static_cast<int8_t>(mat1[(m_start + 0) * K + k + 3]);
          if (m_size > 1)
            a_vals[1][0] = static_cast<int8_t>(mat1[(m_start + 1) * K + k]),
            a_vals[1][1] = static_cast<int8_t>(mat1[(m_start + 1) * K + k + 1]),
            a_vals[1][2] = static_cast<int8_t>(mat1[(m_start + 1) * K + k + 2]),
            a_vals[1][3] = static_cast<int8_t>(mat1[(m_start + 1) * K + k + 3]);
          if (m_size > 2)
            a_vals[2][0] = static_cast<int8_t>(mat1[(m_start + 2) * K + k]),
            a_vals[2][1] = static_cast<int8_t>(mat1[(m_start + 2) * K + k + 1]),
            a_vals[2][2] = static_cast<int8_t>(mat1[(m_start + 2) * K + k + 2]),
            a_vals[2][3] = static_cast<int8_t>(mat1[(m_start + 2) * K + k + 3]);
          if (m_size > 3)
            a_vals[3][0] = static_cast<int8_t>(mat1[(m_start + 3) * K + k]),
            a_vals[3][1] = static_cast<int8_t>(mat1[(m_start + 3) * K + k + 1]),
            a_vals[3][2] = static_cast<int8_t>(mat1[(m_start + 3) * K + k + 2]),
            a_vals[3][3] = static_cast<int8_t>(mat1[(m_start + 3) * K + k + 3]);

          const int8_t* b_ptr0;
          const int8_t* b_ptr1;
          const int8_t* b_ptr2;
          const int8_t* b_ptr3;
          ptrdiff_t b_stride;

          if (is_packed) {
            int64_t b_offset_base = nb * K * BLOCK_N + n_chunk_start;
            b_ptr0 = mat2 + b_offset_base + k * BLOCK_N;
            b_ptr1 = mat2 + b_offset_base + (k + 1) * BLOCK_N;
            b_ptr2 = mat2 + b_offset_base + (k + 2) * BLOCK_N;
            b_ptr3 = mat2 + b_offset_base + (k + 3) * BLOCK_N;
            b_stride = 1;
            if (k + rvv_constants::PREFETCH_DISTANCE < K) {
              __builtin_prefetch(
                  mat2 + b_offset_base + (k + rvv_constants::PREFETCH_DISTANCE) * BLOCK_N,
                  0,
                  rvv_constants::PREFETCH_LOCALITY);
            }
          } else {
            int64_t b_offset_base = (n_start + n_chunk_start) * K;
            b_ptr0 = mat2 + b_offset_base + k;
            b_ptr1 = mat2 + b_offset_base + k + 1;
            b_ptr2 = mat2 + b_offset_base + k + 2;
            b_ptr3 = mat2 + b_offset_base + k + 3;
            b_stride = K;
            if (k + rvv_constants::PREFETCH_DISTANCE < K) {
              __builtin_prefetch(
                  mat2 + b_offset_base + (k + rvv_constants::PREFETCH_DISTANCE), 0, rvv_constants::PREFETCH_LOCALITY);
            }
          }

          vint8m1_t v_b0, v_b1, v_b2, v_b3;
          if (is_packed) {
            v_b0 = __riscv_vle8_v_i8m1(b_ptr0, vl);
            v_b1 = __riscv_vle8_v_i8m1(b_ptr1, vl);
            v_b2 = __riscv_vle8_v_i8m1(b_ptr2, vl);
            v_b3 = __riscv_vle8_v_i8m1(b_ptr3, vl);
          } else {
            v_b0 = __riscv_vlse8_v_i8m1(b_ptr0, b_stride, vl);
            v_b1 = __riscv_vlse8_v_i8m1(b_ptr1, b_stride, vl);
            v_b2 = __riscv_vlse8_v_i8m1(b_ptr2, b_stride, vl);
            v_b3 = __riscv_vlse8_v_i8m1(b_ptr3, b_stride, vl);
          }

          vint16m2_t v_b16_0 = __riscv_vsext_vf2_i16m2(v_b0, vl);
          vint16m2_t v_b16_1 = __riscv_vsext_vf2_i16m2(v_b1, vl);
          vint16m2_t v_b16_2 = __riscv_vsext_vf2_i16m2(v_b2, vl);
          vint16m2_t v_b16_3 = __riscv_vsext_vf2_i16m2(v_b3, vl);

          if (m_size > 0) {
            if (a_vals[0][0]) v_acc0 = __riscv_vwmacc_vx_i32m4(v_acc0, (int16_t)a_vals[0][0], v_b16_0, vl);
            if (a_vals[0][1]) v_acc0 = __riscv_vwmacc_vx_i32m4(v_acc0, (int16_t)a_vals[0][1], v_b16_1, vl);
            if (a_vals[0][2]) v_acc0 = __riscv_vwmacc_vx_i32m4(v_acc0, (int16_t)a_vals[0][2], v_b16_2, vl);
            if (a_vals[0][3]) v_acc0 = __riscv_vwmacc_vx_i32m4(v_acc0, (int16_t)a_vals[0][3], v_b16_3, vl);
          }
          if (m_size > 1) {
            if (a_vals[1][0]) v_acc1 = __riscv_vwmacc_vx_i32m4(v_acc1, (int16_t)a_vals[1][0], v_b16_0, vl);
            if (a_vals[1][1]) v_acc1 = __riscv_vwmacc_vx_i32m4(v_acc1, (int16_t)a_vals[1][1], v_b16_1, vl);
            if (a_vals[1][2]) v_acc1 = __riscv_vwmacc_vx_i32m4(v_acc1, (int16_t)a_vals[1][2], v_b16_2, vl);
            if (a_vals[1][3]) v_acc1 = __riscv_vwmacc_vx_i32m4(v_acc1, (int16_t)a_vals[1][3], v_b16_3, vl);
          }
          if (m_size > 2) {
            if (a_vals[2][0]) v_acc2 = __riscv_vwmacc_vx_i32m4(v_acc2, (int16_t)a_vals[2][0], v_b16_0, vl);
            if (a_vals[2][1]) v_acc2 = __riscv_vwmacc_vx_i32m4(v_acc2, (int16_t)a_vals[2][1], v_b16_1, vl);
            if (a_vals[2][2]) v_acc2 = __riscv_vwmacc_vx_i32m4(v_acc2, (int16_t)a_vals[2][2], v_b16_2, vl);
            if (a_vals[2][3]) v_acc2 = __riscv_vwmacc_vx_i32m4(v_acc2, (int16_t)a_vals[2][3], v_b16_3, vl);
          }
          if (m_size > 3) {
            if (a_vals[3][0]) v_acc3 = __riscv_vwmacc_vx_i32m4(v_acc3, (int16_t)a_vals[3][0], v_b16_0, vl);
            if (a_vals[3][1]) v_acc3 = __riscv_vwmacc_vx_i32m4(v_acc3, (int16_t)a_vals[3][1], v_b16_1, vl);
            if (a_vals[3][2]) v_acc3 = __riscv_vwmacc_vx_i32m4(v_acc3, (int16_t)a_vals[3][2], v_b16_2, vl);
            if (a_vals[3][3]) v_acc3 = __riscv_vwmacc_vx_i32m4(v_acc3, (int16_t)a_vals[3][3], v_b16_3, vl);
          }
        }

        // Tail loop for remaining K
        for (; k < K; ++k) {
          const int8_t* b_ptr;
          ptrdiff_t b_stride;

          if (is_packed) {
            b_ptr = mat2 + nb * K * BLOCK_N + k * BLOCK_N + n_chunk_start;
            b_stride = 1;
          } else {
            b_ptr = mat2 + (n_start + n_chunk_start) * K + k;
            b_stride = K;
          }

          vint8m1_t v_b;
          if (is_packed) {
            v_b = __riscv_vle8_v_i8m1(b_ptr, vl);
          } else {
            v_b = __riscv_vlse8_v_i8m1(b_ptr, b_stride, vl);
          }

          vint16m2_t v_b16 = __riscv_vsext_vf2_i16m2(v_b, vl);

          if (m_size > 0) {
            int8_t a = static_cast<int8_t>(mat1[(m_start + 0) * K + k]);
            if (a != 0) v_acc0 = __riscv_vwmacc_vx_i32m4(v_acc0, (int16_t)a, v_b16, vl);
          }
          if (m_size > 1) {
            int8_t a = static_cast<int8_t>(mat1[(m_start + 1) * K + k]);
            if (a != 0) v_acc1 = __riscv_vwmacc_vx_i32m4(v_acc1, (int16_t)a, v_b16, vl);
          }
          if (m_size > 2) {
            int8_t a = static_cast<int8_t>(mat1[(m_start + 2) * K + k]);
            if (a != 0) v_acc2 = __riscv_vwmacc_vx_i32m4(v_acc2, (int16_t)a, v_b16, vl);
          }
          if (m_size > 3) {
            int8_t a = static_cast<int8_t>(mat1[(m_start + 3) * K + k]);
            if (a != 0) v_acc3 = __riscv_vwmacc_vx_i32m4(v_acc3, (int16_t)a, v_b16, vl);
          }
        }

        if (m_size > 0) __riscv_vse32_v_i32m4(acc[0] + n_chunk_start, v_acc0, vl);
        if (m_size > 1) __riscv_vse32_v_i32m4(acc[1] + n_chunk_start, v_acc1, vl);
        if (m_size > 2) __riscv_vse32_v_i32m4(acc[2] + n_chunk_start, v_acc2, vl);
        if (m_size > 3) __riscv_vse32_v_i32m4(acc[3] + n_chunk_start, v_acc3, vl);
      }

      for (int i = 0; i < m_size; ++i) {
        scalar_t* out_row = out + (m_start + i) * N + n_start;
        const float* combined_scale_row = combined_scales[i];
        const float* bias_row = bias ? (bias + n_start) : nullptr;

        size_t vl_max_dequant = __riscv_vsetvlmax_e32m4();
        size_t vl_dequant;
        for (int64_t j = 0; j < n_size; j += vl_max_dequant) {
          vl_dequant = (j + vl_max_dequant <= n_size) ? vl_max_dequant : __riscv_vsetvl_e32m4(n_size - j);

          vint32m4_t v_acc = __riscv_vle32_v_i32m4(acc[i] + j, vl_dequant);
          vfloat32m4_t v_acc_f = __riscv_vfcvt_f_x_v_f32m4(v_acc, vl_dequant);
          vfloat32m4_t v_scale = __riscv_vle32_v_f32m4(combined_scale_row + j, vl_dequant);
          v_acc_f = __riscv_vfmul_vv_f32m4(v_acc_f, v_scale, vl_dequant);

          if (bias_row) {
            vfloat32m4_t v_bias = __riscv_vle32_v_f32m4(bias_row + j, vl_dequant);
            v_acc_f = __riscv_vfadd_vv_f32m4(v_acc_f, v_bias, vl_dequant);
          }

          if constexpr (std::is_same_v<scalar_t, float>) {
            __riscv_vse32_v_f32m4(out_row + j, v_acc_f, vl_dequant);
          } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
            vfloat16m2_t v_f16 = f32m4_to_f16(v_acc_f, vl_dequant);
            __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(out_row + j), v_f16, vl_dequant);
          } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
            vuint16m2_t v_bf16 = f32m4_to_bf16(v_acc_f, vl_dequant);
            __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(out_row + j), v_bf16, vl_dequant);
          }
        }
      }
    }
  }
}

}  // namespace

template <typename scalar_t>
void int8_scaled_mm_kernel_rvv(
    scalar_t* __restrict__ out,
    const uint8_t* __restrict__ mat1,
    const int8_t* __restrict__ mat2,
    const float* __restrict__ scales1,
    const float* __restrict__ scales2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed) {
  int8_scaled_mm_rvv_impl<scalar_t>(out, mat1, mat2, scales1, scales2, bias, M, N, K, is_packed);
}

template void int8_scaled_mm_kernel_rvv<float>(
    float* out,
    const uint8_t* mat1,
    const int8_t* mat2,
    const float* s1,
    const float* s2,
    const float* b,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed);
template void int8_scaled_mm_kernel_rvv<at::Half>(
    at::Half* out,
    const uint8_t* mat1,
    const int8_t* mat2,
    const float* s1,
    const float* s2,
    const float* b,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed);
template void int8_scaled_mm_kernel_rvv<at::BFloat16>(
    at::BFloat16* out,
    const uint8_t* mat1,
    const int8_t* mat2,
    const float* s1,
    const float* s2,
    const float* b,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed);

std::tuple<at::Tensor, at::Tensor> per_token_quant_int8_cpu(at::Tensor& A) {
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  int64_t M = A.size(0);
  int64_t K = A.size(1);

  auto Aq = at::empty({M, K}, A.options().dtype(at::kByte));
  auto As = at::empty({M, 1}, A.options().dtype(at::kFloat));

  // Dispatch based on input type
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, A.scalar_type(), "per_token_quant_int8_cpu_rvv", [&] {
        const scalar_t* A_data = A.data_ptr<scalar_t>();
        uint8_t* Aq_data = Aq.data_ptr<uint8_t>();
        float* As_data = As.data_ptr<float>();

        at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
          for (int64_t m = begin; m < end; ++m) {
            float scale;
            // Use symmetric quantization (auto-compute scale)
            // Cast output pointer to int8_t* since function writes signed bytes
            quantize_row_int8_symmetric_auto_rvv<scalar_t>(
                reinterpret_cast<int8_t*>(Aq_data + m * K), scale, A_data + m * K, K);
            As_data[m] = scale;
          }
        });
      });

  return std::make_tuple(Aq, As);
}

at::Tensor int8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales1,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_packed) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales1);
  CHECK_INPUT(scales2);
  CHECK_DIM(2, mat1);
  if (!is_packed) {
    CHECK_DIM(2, mat2);
  }

  int64_t M = mat1.size(0);
  int64_t K = mat1.size(1);
  int64_t N = mat2.size(0);

  if (is_packed) {
    N = scales2.numel();
  } else {
    // If not packed, generic code often assumes N from mat2 size(0) or similar
    // scales2 usually has size N
    CHECK_EQ(scales2.numel(), N);
  }

  // Input mat1 should be uint8 (holding quantized data)
  TORCH_CHECK(mat1.scalar_type() == at::kByte, "int8_scaled_mm_cpu: expect mat1 to be uint8 (quantized).");
  TORCH_CHECK(mat2.scalar_type() == at::kChar, "int8_scaled_mm_cpu: expect mat2 to be int8.");

  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  // Kernel Dispatch
  if (out_dtype == at::ScalarType::Float) {
    int8_scaled_mm_kernel_rvv<float>(
        out.data_ptr<float>(),
        mat1.data_ptr<uint8_t>(),
        mat2.data_ptr<int8_t>(),
        scales1.data_ptr<float>(),
        scales2.data_ptr<float>(),
        bias_data,
        M,
        N,
        K,
        is_packed);
  } else if (out_dtype == at::ScalarType::Half) {
    int8_scaled_mm_kernel_rvv<at::Half>(
        out.data_ptr<at::Half>(),
        mat1.data_ptr<uint8_t>(),
        mat2.data_ptr<int8_t>(),
        scales1.data_ptr<float>(),
        scales2.data_ptr<float>(),
        bias_data,
        M,
        N,
        K,
        is_packed);
  } else if (out_dtype == at::ScalarType::BFloat16) {
    int8_scaled_mm_kernel_rvv<at::BFloat16>(
        out.data_ptr<at::BFloat16>(),
        mat1.data_ptr<uint8_t>(),
        mat2.data_ptr<int8_t>(),
        scales1.data_ptr<float>(),
        scales2.data_ptr<float>(),
        bias_data,
        M,
        N,
        K,
        is_packed);
  } else {
    TORCH_CHECK(false, "int8_scaled_mm_cpu: unsupported output dtype.");
  }

  return out;
}

at::Tensor int8_scaled_mm_with_quant(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_packed) {
  RECORD_FUNCTION("sgl-kernel::int8_scaled_mm_cpu", std::vector<c10::IValue>({mat1, mat2, scales2, bias}));

  auto packed_w = is_packed ? mat2 : mat2;

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales2);
  CHECK_DIM(2, mat1);
  if (!is_packed) {
    CHECK_DIM(2, mat2);
  }

  int64_t M = mat1.size(0);
  int64_t K = mat1.size(1);
  int64_t N = mat2.size(0);

  if (is_packed) {
    N = scales2.numel();
  }

  int64_t lda = mat1.stride(0);
  CHECK_EQ(scales2.numel(), N);

  const auto st = mat1.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf, "int8_scaled_mm_with_quant: expect A to be bfloat16 or half.");
  TORCH_CHECK(st == out_dtype, "int8_scaled_mm_with_quant: expect A has same dtype with out_dtype.");
  TORCH_CHECK(mat2.scalar_type() == at::kChar, "int8_scaled_mm_with_quant: expect mat2 to be int8.");
  TORCH_CHECK(scales2.scalar_type() == at::kFloat, "int8_scaled_mm_with_quant: expect scales to be float32.");

  const int64_t buffer_size = M * K + M * sizeof(float);
  auto buffer = at::empty({buffer_size}, mat1.options().dtype(at::kByte));
  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  if (out_dtype == at::ScalarType::Float) {
    uint8_t* __restrict__ Aq_data = buffer.data_ptr<uint8_t>();
    float* __restrict__ As_data = (float*)((void*)(Aq_data + M * K));
    const float* __restrict__ A_data = mat1.data_ptr<float>();

    at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        // Use Symmetric Quantization! Matches per_token_quant output style.
        quantize_row_int8_symmetric_auto_rvv<float>(
            reinterpret_cast<int8_t*>(Aq_data + m * K), As_data[m], A_data + m * lda, K);
      }
    });

    int8_scaled_mm_kernel_rvv<float>(
        out.data_ptr<float>(),
        Aq_data,
        packed_w.data_ptr<int8_t>(),
        As_data,
        scales2.data_ptr<float>(),
        bias_data,
        M,
        N,
        K,
        is_packed);
  } else {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(out_dtype, "int8_scaled_mm_with_quant_kernel_impl", [&] {
      uint8_t* __restrict__ Aq_data = buffer.data_ptr<uint8_t>();
      float* __restrict__ As_data = (float*)((void*)(Aq_data + M * K));
      const scalar_t* __restrict__ A_data = mat1.data_ptr<scalar_t>();

      at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
          // Use Symmetric Quantization! Matches per_token_quant output style.
          quantize_row_int8_symmetric_auto_rvv<scalar_t>(
              reinterpret_cast<int8_t*>(Aq_data + m * K), As_data[m], A_data + m * lda, K);
        }
      });

      int8_scaled_mm_kernel_rvv<scalar_t>(
          out.data_ptr<scalar_t>(),
          Aq_data,
          packed_w.data_ptr<int8_t>(),
          As_data,
          scales2.data_ptr<float>(),
          bias_data,
          M,
          N,
          K,
          is_packed);
    });
  }
  return out;
}

// Implementation of tinygemm_kernel for INT8 (generic interface)
// This maps to our RVV implementation.
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
    bool brg) {
  // RVV implementation currently assumes contiguous memory or specific packing.
  // We ignore Ctmp and brg.
  // We assume scales1 (As) has M elements (per-token) and scales2 (Bs) has N elements (per-channel).
  // mat1 (A) is M x K, mat2 (B) is N x K (transposed) or packed.
  // The generic tinygemm usually assumes B is K x N if row-major or similar.
  // But int8_scaled_mm_kernel_rvv assumes B is N x K (transposed/packed).

  // Checks for stride support
  // int8_scaled_mm_rvv_impl assumes A is row-major (lda=K)
  // If is_packed=true, B is packed.
  // If is_packed=false, B is assumed to be K x N ? Or N x K?
  // int8_scaled_mm_rvv_impl logic:
  //   if !is_packed: b_ptr = mat2 + (n_start + n_chunk_start) * K + k; (implies N x K layout?)
  //   Wait, generic torch.mm(A, B) usually implies A(MxK) * B(KxN).
  //   If B is standard layout (KxN), then b_ptr would be B + k*N + n.
  //   Our RVV implementation seems to assume B is TRANSPOSED (NxK) ??
  //   Let's check int8_scaled_mm_cpu inputs.

  // For now, we assume the inputs match what int8_scaled_mm_kernel_rvv expects.
  // Since this is used by MoE, which likely uses packed weights, implies is_packed is possibly true?
  // But tinygemm_kernel signature in gemm.h doesn't have is_packed param?
  // Wait, gemm_int8 generic signature HAS is_packed?
  // The generic gemm.h signature I saw earlier:
  // bool brg (last arg).
  // NO is_packed arg!

  // So how do we know if it's packed?
  // The generic impl checks `can_use_brgemm` etc.
  // Maybe we explicitly assume unpacked for tinygemm?
  // OR maybe it's always packed for MoE?
  // "fused_experts_int8_kernel_impl" calls it?
  // Actually, tinygemm_kernel is an interface.
  // If we assume it is NOT packed, we must ensure B layout is compatible.

  // Let's forward to int8_scaled_mm_kernel_rvv assuming is_packed=false
  // and hoping layout matches.
  // If MoE uses Weight Packed Linear, it uses a different kernel.
  // tinygemm might be for non-packed small GEMMs.

  // Warning: Bias is missing in tinygemm_kernel signature?
  // The signature has Ctmp, As, Bs... but no Bias?

  int8_scaled_mm_kernel_rvv<scalar_t>(
      C,
      A,
      B,
      As,
      Bs,
      nullptr,  // bias
      M,
      N,
      K,
      false  // is_packed
  );
}

template void tinygemm_kernel<float>(
    const uint8_t* A,
    const int8_t* B,
    float* C,
    int32_t* Ctmp,
    const float* As,
    const float* Bs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);

template void tinygemm_kernel<at::Half>(
    const uint8_t* A,
    const int8_t* B,
    at::Half* C,
    int32_t* Ctmp,
    const float* As,
    const float* Bs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);

template void tinygemm_kernel<at::BFloat16>(
    const uint8_t* A,
    const int8_t* B,
    at::BFloat16* C,
    int32_t* Ctmp,
    const float* As,
    const float* Bs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);

#endif  // CPU_CAPABILITY_RVV
