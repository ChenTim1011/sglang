#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "riscv64/gemm.h"

#include <ATen/core/Tensor.h>

#include <cstdint>

#include "common.h"
#include "vector_helpers.h"
#include "vector_math.h"

#if defined(CPU_CAPABILITY_RVV)
// Clang 19+ version check
#if !defined(__clang__) || !defined(__clang_major__) || __clang_major__ < 19
#error "RVV backend requires Clang 19 or later. Please use Clang 19 or later to compile this file."
#endif

#include <omp.h>
#include <riscv_vector.h>

#include <algorithm>
#include <cstring>

template <typename scalar_t>
void pack_weight(scalar_t* packed_w, const scalar_t* orig_w, int64_t N, int64_t K) {
  constexpr int64_t BN = rvv_constants::BLOCK_N;
  int64_t NB = (N + BN - 1) / BN;

#pragma omp parallel for collapse(2)
  for (int64_t nb = 0; nb < NB; ++nb) {
    for (int64_t k = 0; k < K; ++k) {
      int64_t n_start = nb * BN;
      int64_t n_size = std::min(BN, N - n_start);

      scalar_t* dst = packed_w + nb * K * BN + k * BN;

      if constexpr (std::is_same_v<scalar_t, float>) {
        size_t vl;
        for (int64_t j = 0; j < n_size; j += vl) {
          vl = __riscv_vsetvl_e32m4(n_size - j);
          vfloat32m4_t v_data = __riscv_vlse32_v_f32m4(orig_w + (n_start + j) * K + k, K * sizeof(float), vl);
          __riscv_vse32_v_f32m4(dst + j, v_data, vl);
        }
        if (n_size < BN) {
          size_t vl_pad = __riscv_vsetvl_e32m4(BN - n_size);
          vfloat32m4_t v_zero = __riscv_vfmv_v_f_f32m4(0.0f, vl_pad);
          __riscv_vse32_v_f32m4(dst + n_size, v_zero, vl_pad);
        }
      } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
        size_t vl;
        for (int64_t j = 0; j < n_size; j += vl) {
          vl = __riscv_vsetvl_e16m2(n_size - j);
          vfloat16m2_t v_data = __riscv_vlse16_v_f16m2(
              reinterpret_cast<const _Float16*>(orig_w + (n_start + j) * K + k), K * sizeof(_Float16), vl);
          __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(dst + j), v_data, vl);
        }
        if (n_size < BN) {
          size_t vl_pad = __riscv_vsetvl_e16m2(BN - n_size);
          vfloat16m2_t v_zero = __riscv_vfmv_v_f_f16m2((_Float16)0.0f, vl_pad);
          __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(dst + n_size), v_zero, vl_pad);
        }
      } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        size_t vl;
        for (int64_t j = 0; j < n_size; j += vl) {
          vl = __riscv_vsetvl_e16m2(n_size - j);
          vuint16m2_t v_data = __riscv_vlse16_v_u16m2(
              reinterpret_cast<const uint16_t*>(orig_w + (n_start + j) * K + k), K * sizeof(uint16_t), vl);
          __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(dst + j), v_data, vl);
        }
        if (n_size < BN) {
          size_t vl_pad = __riscv_vsetvl_e16m2(BN - n_size);
          vuint16m2_t v_zero = __riscv_vmv_v_x_u16m2(0, vl_pad);
          __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(dst + n_size), v_zero, vl_pad);
        }
      } else {
        // Scalar fallback
        for (int64_t j = 0; j < n_size; ++j) {
          dst[j] = orig_w[(n_start + j) * K + k];
        }
        for (int64_t j = n_size; j < BN; ++j) {
          dst[j] = static_cast<scalar_t>(0);
        }
      }
    }
  }
}

template <typename scalar_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static_assert(BLOCK_M >= 1 && BLOCK_M <= 4, "BLOCK_M must be 1-4");
  static_assert(
      std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>,
      "tinygemm_kernel_nn: only supports float, Half, and BFloat16");

  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      scalar_t* __restrict__ C,
      const float* __restrict__ bias,
      int64_t K,
      int64_t lda,
      int64_t ldc,
      int64_t n_size) {
    size_t vl_max = __riscv_vsetvlmax_e32m4();

    // Process N dimension in chunks of vl_max
    for (int64_t j = 0; j < n_size; j += vl_max) {
      size_t vl = (j + vl_max <= n_size) ? vl_max : __riscv_vsetvl_e32m4(n_size - j);

      // Initialize accumulators for BLOCK_M rows
      vfloat32m4_t v_acc0, v_acc1, v_acc2, v_acc3;
      vfloat32m4_t* v_acc[4] = {&v_acc0, &v_acc1, &v_acc2, &v_acc3};

      // Initialize with bias or zero
      for (int m = 0; m < BLOCK_M; ++m) {
        if constexpr (has_bias) {
          *v_acc[m] = __riscv_vle32_v_f32m4(bias + j, vl);
        } else {
          *v_acc[m] = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        }
      }

      // K-loop: accumulate in registers
      const scalar_t* b_ptr_base = B + j;

      for (int64_t k = 0; k < K; ++k) {
        // Prefetch next B block
        if (k + 2 < K) {
          __builtin_prefetch(b_ptr_base + (k + 2) * BLOCK_N, 0, 1);
        }

        const scalar_t* b_ptr = b_ptr_base + k * BLOCK_N;

        // Load B vector (same for all M rows)
        vfloat32m4_t v_b;
        if constexpr (std::is_same_v<scalar_t, float>) {
          v_b = __riscv_vle32_v_f32m4(b_ptr, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_b16 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(b_ptr), vl);
          v_b = __riscv_vfwcvt_f_f_v_f32m4(v_b16, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          v_b = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(b_ptr), vl);
        } else {
          // Should not reach here due to static_assert, but provide compile-time error
          static_assert(
              std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, at::Half> ||
                  std::is_same_v<scalar_t, at::BFloat16>,
              "Unsupported scalar type");
        }

        // FMA for each row
        for (int m = 0; m < BLOCK_M; ++m) {
          float a_val = static_cast<float>(A[m * lda + k]);
          *v_acc[m] = __riscv_vfmacc_vf_f32m4(*v_acc[m], a_val, v_b, vl);
        }
      }

      // Store results
      for (int m = 0; m < BLOCK_M; ++m) {
        scalar_t* c_ptr = C + m * ldc + j;

        if constexpr (std::is_same_v<scalar_t, float>) {
          __riscv_vse32_v_f32m4(c_ptr, *v_acc[m], vl);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_f16 = f32m4_to_f16(*v_acc[m], vl);
          __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(c_ptr), v_f16, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          vuint16m2_t v_bf16 = f32m4_to_bf16(*v_acc[m], vl);
          __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(c_ptr), v_bf16, vl);
        } else {
          // Should not reach here due to static_assert
          static_assert(
              std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, at::Half> ||
                  std::is_same_v<scalar_t, at::BFloat16>,
              "Unsupported scalar type");
        }
      }
    }
  }
};

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                \
  tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                          \
      B + nb_start * K,                                            \
      C + mb_start * ldc + nb_start,                               \
      has_bias ? bias + nb_start : nullptr,                        \
      K,                                                           \
      lda,                                                         \
      ldc,                                                         \
      nb_size);

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldc) {
  // pattern: 1-4-16, N = 16, 32, 48, 64
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x11:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 16);
          break;
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 0x13:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 48);
          break;
        case 0x14:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 64);
          break;
        // mb_size = 2
        case 0x21:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 16);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 32);
          break;
        case 0x23:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 48);
          break;
        case 0x24:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 64);
          break;
        // mb_size = 3
        case 0x31:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 16);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 32);
          break;
        case 0x33:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 48);
          break;
        case 0x34:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 64);
          break;
        // mb_size = 4
        case 0x41:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 16);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 32);
          break;
        case 0x43:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 48);
          break;
        case 0x44:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 64);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, " x ", nb_size);
      }
    }
  }
}

template <typename scalar_t>
void weight_packed_linear_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const scalar_t* __restrict__ mat2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
      loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(N - nb_start, BLOCK_N);

        tinygemm_kernel<scalar_t, has_bias>(
            mat1 + mb_start * mat1_strideM,
            mat2 + nb_start * K,  // pointer adjusted for NB block
            out + mb_start * out_strideM + nb_start,
            bias ? bias + nb_start : nullptr,
            mb_size,
            nb_size,
            K,
            mat1_strideM,
            out_strideM);
      });
    });
  });
}

template <typename scalar_t>
void weight_packed_linear_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const float* __restrict__ mat2,
    const float* __restrict__ bias,
    const scalar_t* __restrict__ post_mul_mat,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM) {
  const size_t vl_max = __riscv_vsetvlmax_e32m4();
  const bool two_chunks = (static_cast<size_t>(N) > vl_max);

#pragma omp parallel for if (M > 1)
  for (int64_t m = 0; m < M; ++m) {
    const scalar_t* a_row = mat1 + m * mat1_strideM;
    scalar_t* c_row = out + m * out_strideM;

    for (int64_t n_start = 0; n_start < N; n_start += vl_max) {
      size_t vl = __riscv_vsetvl_e32m4(N - n_start);
      alignas(64) float linear_row[rvv_constants::MAX_VL_ELEMENTS_M4];

      // Initialize accumulator with bias or zero
      vfloat32m4_t v_acc =
          (bias != nullptr) ? __riscv_vle32_v_f32m4(bias + n_start, vl) : __riscv_vfmv_v_f_f32m4(0.0f, vl);

      // K-loop: broadcast A scalar, load contiguous floats from B row
      for (int64_t k = 0; k < K; ++k) {
        const float a_val = static_cast<float>(a_row[k]);
        const float* b_row = mat2 + k * N + n_start;
        v_acc = __riscv_vfmacc_vf_f32m4(v_acc, a_val, __riscv_vle32_v_f32m4(b_row, vl), vl);
      }

      __riscv_vse32_v_f32m4(linear_row, v_acc, vl);

      // Post-process
      if (post_mul_mat != nullptr) {
        scalar_sigmoid_and_mul<scalar_t, false>(
            c_row + n_start, linear_row, nullptr, post_mul_mat + m * out_strideM + n_start, vl);
      } else {
        copy_stub(c_row + n_start, linear_row, vl);
      }
    }
  }
}

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
    bool brg) {
  UNUSED(Ctmp);
  UNUSED(ldb);
  TORCH_CHECK(!brg, "RVV does not use brgemm");
  // Forward to internal implementation (no bias)
  tinygemm_kernel<scalar_t, false>(A, B, C, nullptr, M, N, K, lda, ldc);
}

#define INSTANTIATE_TINYGEMM_TEMPLATE(TYPE) \
  template void tinygemm_kernel<TYPE>(      \
      const TYPE* __restrict__ A,           \
      const TYPE* __restrict__ B,           \
      TYPE* __restrict__ C,                 \
      float* __restrict__ Ctmp,             \
      int64_t M,                            \
      int64_t N,                            \
      int64_t K,                            \
      int64_t lda,                          \
      int64_t ldb,                          \
      int64_t ldc,                          \
      bool brg)

INSTANTIATE_TINYGEMM_TEMPLATE(float);
INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half);

// Implementation
at::Tensor convert_weight_packed(at::Tensor& weight) {
  CHECK_INPUT(weight);

  const int64_t ndim = weight.ndimension();
  TORCH_CHECK(ndim == 2 || ndim == 3, "expect weight to be 2d or 3d, got ", ndim, "d tensor.");

  constexpr int64_t BLOCK_N_RVV = rvv_constants::BLOCK_N;
  if (ndim == 2 && (weight.size(0) < BLOCK_N_RVV || weight.size(0) % BLOCK_N_RVV != 0)) {
    // for 2D weight and small OC shape, we use fma linear path, which needs transpose not pack
    return weight.to(at::kFloat).t().contiguous();
  }

  const auto st = weight.scalar_type();
  const int64_t E = ndim == 3 ? weight.size(0) : 1;
  const int64_t OC = ndim == 3 ? weight.size(1) : weight.size(0);
  const int64_t IC = ndim == 3 ? weight.size(2) : weight.size(1);

  const int64_t NB = div_up(OC, BLOCK_N_RVV);

  auto packed_weight = at::empty({}, weight.options());
  const int64_t stride = OC * IC;

  TORCH_CHECK(
      st == at::kBFloat16 || st == at::kHalf || st == at::kChar, "expect weight to be bfloat16, float16, int8 ");

  CPU_DISPATCH_PACKED_TYPES_RVV(st, [&] {
    const int64_t packed_block_size = IC * BLOCK_N_RVV;
    const int64_t packed_size_per_expert = NB * packed_block_size;
    // Preserve shape [E, NB, packed_block_size] / [NB, packed_block_size] so MoE kernel
    if (ndim == 3) {
      packed_weight.resize_({E, NB, packed_block_size});
    } else {
      packed_weight.resize_({NB, packed_block_size});
    }
    packed_t* packed_data = packed_weight.data_ptr<packed_t>();
    const packed_t* w_data = weight.data_ptr<packed_t>();

    // parallel on {E, NB}
    at::parallel_for(0, E * NB, 0, [&](int64_t begin, int64_t end) {
      int64_t e{0}, nb{0};
      data_index_init(begin, e, E, nb, NB);

      for (int64_t i = begin; i < end; ++i) {
        UNUSED(i);

        int64_t n = nb * BLOCK_N_RVV;
        int64_t n_size = std::min(BLOCK_N_RVV, OC - n);
        pack_weight<packed_t>(
            packed_data + e * packed_size_per_expert + nb * packed_block_size,
            w_data + e * stride + n * IC,
            n_size,
            IC);

        // move to the next index
        data_index_step(e, E, nb, NB);
      }
    });
  });
  return packed_weight;
}

at::Tensor
weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2, const std::optional<at::Tensor>& bias, bool is_vnni) {
  RECORD_FUNCTION("sgl-kernel::weight_packed_linear", std::vector<c10::IValue>({mat1, mat2, bias}));

  bool is_packed = is_vnni;
  auto packed_w = is_packed ? mat2 : convert_weight_packed(mat2);
  bool use_fma_gemm = false;
  if (packed_w.scalar_type() == at::kFloat) {
    use_fma_gemm = true;
  }

  int64_t M = mat1.size(0);
  int64_t K = mat1.size(1);
  int64_t N;
  if (use_fma_gemm) {
    N = packed_w.size(1);
  } else {
    if (is_packed) {
      // RVV packed: 2D [NB, IC*BLOCK_N] or 3D [E, NB, block_size]; logical OC = NB*BLOCK_N
      if (packed_w.dim() == 2) {
        N = packed_w.size(0) * block_size_n();
      } else if (packed_w.dim() == 3) {
        N = packed_w.size(1) * block_size_n();
      } else {
        N = (mat2.dim() == 1) ? (mat2.size(0) / K) : (mat2.dim() == 3 ? mat2.size(1) : mat2.size(0));
      }
    } else {
      N = (mat2.dim() == 3) ? mat2.size(1) : mat2.size(0);
    }
  }

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_DIM(2, mat1);
  if (!is_packed) {
    CHECK_DIM(2, mat2);
    if (!use_fma_gemm) {
      CHECK_EQ(mat1.size(1), mat2.size(1));
    }
  }

  auto dispatch_type = mat1.scalar_type();
  auto out = at::empty({M, N}, mat1.options());
  // strides
  int64_t out_strideM = out.stride(0);
  int64_t mat1_strideM = mat1.stride(0);

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_RVV_TYPES(dispatch_type, "weight_packed_linear_kernel_impl", [&] {
    if (use_fma_gemm) {
      weight_packed_linear_kernel_impl<scalar_t>(
          out.data_ptr<scalar_t>(),
          mat1.data_ptr<scalar_t>(),
          packed_w.data_ptr<float>(),
          bias_data,
          nullptr,
          M,
          N,
          K,
          mat1_strideM,
          out_strideM);
    } else {
      weight_packed_linear_kernel_impl<scalar_t>(
          out.data_ptr<scalar_t>(),
          mat1.data_ptr<scalar_t>(),
          packed_w.data_ptr<scalar_t>(),
          bias_data,
          M,
          N,
          K,
          mat1_strideM,
          out_strideM);
    }
  });

  return out;
}

// mat1         : [M, K]
// mat2         : [K, 1]
// post_mul_mat : [M, K]
// bias         : [N]
// out          : [M, N]
//
at::Tensor fused_linear_sigmoid_mul(
    at::Tensor& mat1,
    at::Tensor& mat2,
    const std::optional<at::Tensor>& bias,
    bool is_vnni,
    const at::Tensor& post_mul_mat) {
  RECORD_FUNCTION("sgl-kernel::fused_linear_sigmoid_mul", std::vector<c10::IValue>({mat1, mat2, bias, post_mul_mat}));

  bool is_packed = is_vnni;
  auto packed_w = is_packed ? mat2 : convert_weight_packed(mat2);

  int64_t M = mat1.size(0);
  int64_t K = mat1.size(1);
  int64_t N = is_packed ? mat2.size(1) : (mat2.dim() == 3 ? mat2.size(1) : mat2.size(0));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_DIM(2, mat1);
  if (!is_packed) {
    CHECK_DIM(2, mat2);
    TORCH_CHECK(mat2.size(1) == K, "K dimension mismatch: mat1 K=", K, " vs mat2 K=", mat2.size(1));
  }

  int64_t out_strideM = post_mul_mat.size(1);
  int64_t mat1_strideM = mat1.stride(0);
  auto dispatch_type = mat1.scalar_type();
  auto out = at::empty({M, out_strideM}, mat1.options());

  TORCH_CHECK(
      N == 1 && out_strideM % 32 == 0,
      "post_mul_mat tensor size(1) should be 32 dividable, and the mat2 OC=1 (Mx1 as linear output shape)");

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_RVV_TYPES(dispatch_type, "fused_linear_sigmoid_mul", [&] {
    if (packed_w.scalar_type() != at::kFloat) {
      auto float_w = packed_w.to(at::kFloat);
      weight_packed_linear_kernel_impl<scalar_t>(
          out.data_ptr<scalar_t>(),
          mat1.data_ptr<scalar_t>(),
          float_w.data_ptr<float>(),
          bias_data,
          post_mul_mat.data_ptr<scalar_t>(),
          M,
          N,
          K,
          mat1_strideM,
          out_strideM);
    } else {
      weight_packed_linear_kernel_impl<scalar_t>(
          out.data_ptr<scalar_t>(),
          mat1.data_ptr<scalar_t>(),
          packed_w.data_ptr<float>(),
          bias_data,
          post_mul_mat.data_ptr<scalar_t>(),
          M,
          N,
          K,
          mat1_strideM,
          out_strideM);
    }
  });

  return out;
}

#endif  // CPU_CAPABILITY_RVV
