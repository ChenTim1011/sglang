#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "common.h"
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
        int16_t a_val_16 = static_cast<int16_t>(static_cast<uint16_t>(a_val));
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

      for (int i = 0; i < m_size; ++i) {
        size_t vl_max_chunk = __riscv_vsetvlmax_e32m4();
        for (int64_t n_chunk_start = 0; n_chunk_start < n_size; n_chunk_start += N_CHUNK) {
          int64_t n_chunk_size = std::min(N_CHUNK, n_size - n_chunk_start);
          size_t vl = (n_chunk_size >= vl_max_chunk) ? vl_max_chunk : __riscv_vsetvl_e32m4(n_chunk_size);

          vint32m4_t v_acc = __riscv_vmv_v_x_i32m4(0, vl);

          for (int64_t k = 0; k < K; ++k) {
            uint8_t a_val = mat1[(m_start + i) * K + k];
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
            int16_t a_val_16 = static_cast<int16_t>(static_cast<uint16_t>(a_val));
            v_acc = __riscv_vwmacc_vx_i32m4(v_acc, a_val_16, v_b16, vl);
          }

          __riscv_vse32_v_i32m4(acc[i] + n_chunk_start, v_acc, vl);
        }
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
        quantize_row_int8<float>(Aq_data + m * K, As_data[m], A_data + m * lda, K);
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
          quantize_row_int8<scalar_t>(Aq_data + m * K, As_data[m], A_data + m * lda, K);
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

#endif  // CPU_CAPABILITY_RVV
