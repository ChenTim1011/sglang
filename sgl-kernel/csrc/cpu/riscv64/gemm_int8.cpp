#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "common.h"
#include "riscv64/gemm.h"
#include "riscv64/vector_helpers.h"

#if defined(CPU_CAPABILITY_RVV)
#include <riscv_vector.h>

#include <algorithm>
#include <cmath>
using namespace rvv_constants;

namespace {

template <typename scalar_t, bool has_bias, int BLOCK_N>
struct scale_C {
  // Called with vl <= vl_max; operates on exactly vl elements (no inner loop).
  static inline void apply(
      scalar_t* __restrict__ C,
      const int32_t* __restrict__ Ctmp,
      const int32_t* __restrict__ Bcomp,
      const float* __restrict__ bias,
      float As,
      const float* __restrict__ Bs,
      size_t vl) {
    UNUSED(Bcomp);

    vint32m4_t v_acc_i32 = __riscv_vle32_v_i32m4(Ctmp, vl);
    vfloat32m4_t v_acc = __riscv_vfcvt_f_x_v_f32m4(v_acc_i32, vl);

    // Dequantize: C * (As * Bs) — combine scales first for better precision
    vfloat32m4_t v_bs = __riscv_vle32_v_f32m4(Bs, vl);
    vfloat32m4_t v_combined_scale = __riscv_vfmul_vf_f32m4(v_bs, As, vl);
    v_acc = __riscv_vfmul_vv_f32m4(v_acc, v_combined_scale, vl);

    if constexpr (has_bias) {
      vfloat32m4_t v_bias = __riscv_vle32_v_f32m4(bias, vl);
      v_acc = __riscv_vfadd_vv_f32m4(v_acc, v_bias, vl);
    }

    alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
    store_from_float_m4(C, v_acc, vl, scratch);
  }
};

template <typename scalar_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static_assert(BLOCK_M >= 1 && BLOCK_M <= 4, "BLOCK_M must be 1-4");
  static_assert(
      std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>,
      "tinygemm_kernel_nn: only supports float, Half, and BFloat16");

  static inline void apply(
      const int8_t* __restrict__ A,
      const int8_t* __restrict__ B,
      scalar_t* __restrict__ C,
      const float* __restrict__ As,
      const float* __restrict__ Bs,
      const int32_t* __restrict__ Bcomp,
      const float* __restrict__ bias,
      int64_t K,
      int64_t lda,
      int64_t ldc,
      int64_t nb,
      int64_t n_size) {
    UNUSED(Bcomp);
    TORCH_CHECK(n_size >= 0 && n_size <= BLOCK_N);
    if constexpr (has_bias) {
      TORCH_CHECK(bias != nullptr);
    }

    size_t vl_max = __riscv_vsetvlmax_e32m4();
    const int64_t n_start = nb * BLOCK_N;

    // Compute and store per row
    alignas(64) int32_t Ctmp_row[BLOCK_N];

    for (int64_t j = 0; j < n_size; j += vl_max) {
      size_t vl = (j + vl_max <= (size_t)n_size) ? vl_max : __riscv_vsetvl_e32m4(n_size - j);

      // Accumulators in i32
      vint32m4_t v_acc0, v_acc1, v_acc2, v_acc3;
      vint32m4_t* v_acc[4] = {&v_acc0, &v_acc1, &v_acc2, &v_acc3};
      for (int m = 0; m < BLOCK_M; ++m) {
        *v_acc[m] = __riscv_vmv_v_x_i32m4(0, vl);
      }

      // K loop: B layout is [NB, K, BLOCK_N]
      // K-tiling: target B footprint ≈ L1/2; vl_max_m4 = VLEN/8; INT8: KB = L1*4/VLEN.
      // Override L1 assumption: cmake -DRVV_L1_CACHE_KB=N (default 32KB).
      const int8_t* b_ptr_base = B + nb * K * BLOCK_N + j;
      constexpr int64_t KB = rvv_constants::L1_CACHE_BYTES * 4 / __riscv_v_fixed_vlen;
      for (int64_t k_tile = 0; k_tile < K; k_tile += KB) {
        const int64_t k_end = std::min(k_tile + KB, K);
        for (int64_t k = k_tile; k < k_end; ++k) {
          // Prefetch next B block
          if (k + 2 < K) {
            __builtin_prefetch(b_ptr_base + (k + 2) * BLOCK_N, 0, 1);
          }

          const int8_t* b_ptr = b_ptr_base + k * BLOCK_N;
          vint8m1_t v_b_i8 = __riscv_vle8_v_i8m1(b_ptr, vl);
          vint16m2_t v_b_i16 = __riscv_vsext_vf2_i16m2(v_b_i8, vl);

          for (int m = 0; m < BLOCK_M; ++m) {
            int8_t a_val = A[m * lda + k];
            if (a_val == 0) continue;
            *v_acc[m] = __riscv_vwmacc_vx_i32m4(*v_acc[m], static_cast<int16_t>(a_val), v_b_i16, vl);
          }
        }
      }  // k_tile

      // Dequantize and store each row
      for (int m = 0; m < BLOCK_M; ++m) {
        __riscv_vse32_v_i32m4(Ctmp_row, *v_acc[m], vl);
        scale_C<scalar_t, has_bias, BLOCK_N>::apply(
            C + m * ldc + n_start + j,
            Ctmp_row,
            /*Bcomp*/ nullptr,
            has_bias ? (bias + n_start + j) : nullptr,
            As[m],
            Bs + n_start + j,
            vl);
      }
    }
  }
};

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                \
  tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                          \
      B,                                                           \
      C + mb_start * ldc,                                          \
      As + mb_start,                                               \
      Bs,                                                          \
      /*Bcomp*/ nullptr,                                           \
      has_bias ? bias : nullptr,                                   \
      K,                                                           \
      lda,                                                         \
      ldc,                                                         \
      nb,                                                          \
      nb_size);

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldc) {
  // NB_SIZE template arg == BLOCK_N (pack stride); inner loop uses runtime nb_size for tail tiles.
  constexpr int64_t BLOCK_M = GEMM_TILE_M;
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  for (int64_t mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size) {
        case 1:
          LAUNCH_TINYGEMM_KERNEL_NN(1, BLOCK_N);
          break;
        case 2:
          LAUNCH_TINYGEMM_KERNEL_NN(2, BLOCK_N);
          break;
        case 3:
          LAUNCH_TINYGEMM_KERNEL_NN(3, BLOCK_N);
          break;
        case 4:
          LAUNCH_TINYGEMM_KERNEL_NN(4, BLOCK_N);
          break;
        default:
          TORCH_CHECK(false, "Unexpected mb_size: ", mb_size);
      }
    }
  }
}

template <typename scalar_t>
void int8_scaled_mm_kernel_impl(
    scalar_t* __restrict__ out,
    const int8_t* __restrict__ mat1,
    const int8_t* __restrict__ mat2,
    const float* __restrict__ scales1,
    const float* __restrict__ scales2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K) {
  constexpr int64_t BLOCK_M = GEMM_TILE_M;
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    at::parallel_for(0, NB, 0, [&](int64_t begin, int64_t end) {
      for (int64_t nb = begin; nb < end; ++nb) {
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(BLOCK_N, N - nb_start);

        for (int64_t mb = 0; mb < MB; ++mb) {
          int64_t mb_start = mb * BLOCK_M;
          int64_t mb_size = std::min(BLOCK_M, M - mb_start);

          tinygemm_kernel<scalar_t, has_bias>(
              /*      A */ mat1 + mb_start * K,
              /*      B */ mat2 + nb * K * BLOCK_N,
              /*      C */ out + mb_start * N + nb_start,
              /*     As */ scales1 + mb_start,
              /*     Bs */ scales2 + nb_start,
              /*   bias */ has_bias ? (bias + nb_start) : nullptr,
              /*      M */ mb_size,
              /*      N */ nb_size,
              /*      K */ K,
              /*    lda */ K,
              /*    ldc */ N);
        }
      }
    });
  });
}

}  // anonymous namespace

std::tuple<at::Tensor, at::Tensor> per_token_quant_int8_cpu(at::Tensor& A) {
  RECORD_FUNCTION("sgl-kernel::per_token_quant_int8_cpu", std::vector<c10::IValue>({A}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(A);
  CHECK_DIM(2, A);

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t lda = A.stride(0);

  const auto st = A.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf, "per_token_quant_int8: expect A to be bfloat16 or half.");

  // Use int8 (kChar) to match symmetric quantization output range [-128, 127]
  auto Aq = at::empty({M, K}, A.options().dtype(at::kChar));
  auto As = at::empty({M, 1}, A.options().dtype(at::kFloat));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, st, "per_token_quant_int8_cpu", [&] {
    const scalar_t* A_data = A.data_ptr<scalar_t>();
    int8_t* Aq_data = Aq.data_ptr<int8_t>();
    float* As_data = As.data_ptr<float>();

    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        float scale;
        // Symmetric quantization: maps [-max, max] → [-127, 127]
        quantize_row_int8_symmetric_auto<scalar_t>(Aq_data + m * K, scale, A_data + m * lda, K);
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
    // is_packed: true = weight already in RVV block-N format, skip repacking.
    // Declaration uses is_vnni for API consistency with x86 backend;
    // renamed here because RVV has no VNNI hardware — is_vnni is misleading.
    bool is_packed) {
  RECORD_FUNCTION("sgl-kernel::int8_scaled_mm_cpu", std::vector<c10::IValue>({mat1, mat2, scales1, scales2, bias}));

  auto packed_w = is_packed ? mat2 : convert_weight_packed(mat2);

  CHECK_INPUT(mat1);
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
    CHECK_EQ(scales2.numel(), N);
  }

  TORCH_CHECK(
      mat1.scalar_type() == at::kChar || mat1.scalar_type() == at::kByte,
      "int8_scaled_mm_cpu: expect mat1 to be int8 or uint8.");
  TORCH_CHECK(
      mat2.scalar_type() == at::kChar || mat2.scalar_type() == at::kByte,
      "int8_scaled_mm_cpu: expect mat2 to be int8 or uint8.");
  TORCH_CHECK(
      scales1.scalar_type() == at::kFloat && scales2.scalar_type() == at::kFloat,
      "int8_scaled_mm_cpu: expect scales to be float32.");

  at::Tensor scales1_1d = scales1.dim() == 2 ? scales1.squeeze(-1) : scales1;
  at::Tensor scales2_1d = scales2.dim() == 2 ? scales2.squeeze(-1) : scales2;

  TORCH_CHECK(
      scales1_1d.dim() == 1 && scales1_1d.numel() == M,
      "int8_scaled_mm_cpu: scales1 must be 1D with M elements, got shape ",
      scales1.sizes());
  TORCH_CHECK(
      scales2_1d.dim() == 1 && scales2_1d.numel() == N,
      "int8_scaled_mm_cpu: scales2 must be 1D with N elements, got shape ",
      scales2.sizes());

  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_RVV_TYPES(out_dtype, "int8_scaled_mm_kernel_impl", [&]() {
    int8_scaled_mm_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        reinterpret_cast<const int8_t*>(mat1.data_ptr()),
        reinterpret_cast<const int8_t*>(packed_w.data_ptr()),
        scales1_1d.data_ptr<float>(),
        scales2_1d.data_ptr<float>(),
        bias_data,
        M,
        N,
        K);
  });

  return out;
}

at::Tensor int8_scaled_mm_with_quant(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    // is_packed: true = weight already in RVV block-N format, skip repacking.
    // Declaration uses is_vnni for API consistency with x86 backend;
    // renamed here because RVV has no VNNI hardware — is_vnni is misleading.
    bool is_packed) {
  RECORD_FUNCTION("sgl-kernel::int8_scaled_mm_cpu", std::vector<c10::IValue>({mat1, mat2, scales2, bias}));

  auto packed_w = is_packed ? mat2 : convert_weight_packed(mat2);

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

  // Round M*K up to a float boundary so As_data (placed after Aq) is float-aligned.
  // alignof(float) == sizeof(float) == 4 on all supported targets.
  constexpr int64_t kFloatAlign = static_cast<int64_t>(alignof(float));
  const int64_t aq_bytes = ((M * K + kFloatAlign - 1) / kFloatAlign) * kFloatAlign;
  const int64_t buffer_size = aq_bytes + M * static_cast<int64_t>(sizeof(float));
  auto buffer = at::empty({buffer_size}, mat1.options().dtype(at::kChar));
  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_RVV_TYPES(out_dtype, "int8_scaled_mm_with_quant_kernel_impl", [&]() {
    int8_t* __restrict__ buffer_ptr = buffer.data_ptr<int8_t>();
    int8_t* __restrict__ Aq_data = buffer_ptr;
    float* __restrict__ As_data = reinterpret_cast<float*>(buffer_ptr + aq_bytes);
    const scalar_t* __restrict__ A_data = mat1.data_ptr<scalar_t>();

    at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        quantize_row_int8_symmetric_auto<scalar_t>(Aq_data + m * K, As_data[m], A_data + m * lda, K);
      }
    });

    int8_scaled_mm_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        Aq_data,
        packed_w.data_ptr<int8_t>(),
        As_data,
        scales2.data_ptr<float>(),
        bias_data,
        M,
        N,
        K);
  });
  return out;
}

template <typename scalar_t>
void tinygemm_kernel(
    const int8_t* __restrict__ A,
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
    bool brg) {
  TORCH_CHECK(!brg, "RVV does not use brgemm");
  TORCH_CHECK(lda == K, "tinygemm_kernel: A must be contiguous (lda == K)");
  TORCH_CHECK(ldc == N, "tinygemm_kernel: C must be contiguous (ldc == N)");

  UNUSED(ldb);
  tinygemm_kernel<scalar_t, false>(
      A,
      B,
      C,
      As,
      Bs,
      nullptr,  // bias
      M,
      N,
      K,
      lda,
      ldc);
}

#define INSTANTIATE_TINYGEMM_TEMPLATE_RVV(TYPE) \
  template void tinygemm_kernel<TYPE>(          \
      const int8_t* __restrict__ A,             \
      const int8_t* __restrict__ B,             \
      TYPE* __restrict__ C,                     \
      const float* __restrict__ As,             \
      const float* __restrict__ Bs,             \
      int64_t M,                                \
      int64_t N,                                \
      int64_t K,                                \
      int64_t lda,                              \
      int64_t ldb,                              \
      int64_t ldc,                              \
      bool brg)

INSTANTIATE_TINYGEMM_TEMPLATE_RVV(at::BFloat16);
INSTANTIATE_TINYGEMM_TEMPLATE_RVV(at::Half);

// Provide uint8_t version for files that include generic gemm.h (e.g., moe_int8.cpp)
// This is a minimal wrapper that casts uint8_t to int8_t (RVV uses symmetric quantization)
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
  UNUSED(Ctmp);
  tinygemm_kernel<scalar_t>(reinterpret_cast<const int8_t*>(A), B, C, As, Bs, M, N, K, lda, ldb, ldc, brg);
}

template void tinygemm_kernel<at::BFloat16>(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    at::BFloat16* __restrict__ C,
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

template void tinygemm_kernel<at::Half>(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    at::Half* __restrict__ C,
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

#endif  // CPU_CAPABILITY_RVV
