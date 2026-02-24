#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "common.h"
#include "riscv64/gemm.h"
#include "vec.h"

// Shared moe API uses these; not in riscv64/gemm.h
namespace {
inline int64_t get_row_size(int64_t K, bool use_int8_w8a8) {
  return use_int8_w8a8 ? K + static_cast<int64_t>(sizeof(int32_t)) : K;
}
constexpr int64_t kCPUQuantUNQUANT = 0;
constexpr int64_t kCPUQuantINT8_W8A8 = 1;
constexpr int64_t kCPUQuantFP8_W8A16 = 2;
constexpr int64_t kCPUQuantINT4_W4A8 = 3;
static inline void check_moe_scales(
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size) {
  if (use_int8_w8a8) {
    TORCH_CHECK(w1_scale.has_value(), "missing w1_scale for int8 w8a8.");
    TORCH_CHECK(w2_scale.has_value(), "missing w2_scale for int8 w8a8.");
  }
  if (use_fp8_w8a16) {
    TORCH_CHECK(w1_scale.has_value(), "missing w1_scale for fp8 w8a16.");
    TORCH_CHECK(w2_scale.has_value(), "missing w2_scale for fp8 w8a16.");
    TORCH_CHECK(block_size.has_value(), "missing block_size for fp8 w8a16.");
    TORCH_CHECK(block_size.value().size() == 2, "expect block_size.size() to be 2.");
  }
}
}  // namespace

#if defined(CPU_CAPABILITY_RVV)
#include <riscv_vector.h>

#include "riscv64/vector_helpers.h"
#include "riscv64/vector_math.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

namespace {

// copy_mul_stub: read scalar_t, scale by weight (float), write scalar_t.
// Used in Stage 2 to scatter GEMM output with per-token topk weights.
template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, float weight, int64_t size) {
#if defined(CPU_CAPABILITY_RVV)
  size_t vl;
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
  vfloat32m4_t v_w = __riscv_vfmv_v_f_f32m4(weight, __riscv_vsetvlmax_e32m4());
  for (int64_t d = 0; d < size; d += vl) {
    vl = __riscv_vsetvl_e32m4(size - d);
    vfloat32m4_t v_in = load_as_float_m4(input + d, vl, scratch);
    vfloat32m4_t v_out = __riscv_vfmul_vv_f32m4(v_in, v_w, vl);
    store_from_float_m4(out + d, v_out, vl, scratch);
  }
#else
  for (int64_t d = 0; d < size; ++d)
    out[d] = static_cast<scalar_t>(static_cast<float>(input[d]) * weight);
#endif
}

// sum_stub: accumulate [topk, K] -> [K] in float32 precision.
template <typename scalar_t>
inline void sum_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t topk, int64_t K) {
  if (topk == 1) {
    copy_stub(out, input, K);
    return;
  }
#if defined(CPU_CAPABILITY_RVV)
  size_t vl;
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
  for (int64_t d = 0; d < K; d += vl) {
    vl = __riscv_vsetvl_e32m4(K - d);
    vfloat32m4_t v_sum = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    for (int64_t t = 0; t < topk; ++t) {
      vfloat32m4_t v_x = load_as_float_m4(input + t * K + d, vl, scratch);
      v_sum = __riscv_vfadd_vv_f32m4(v_sum, v_x, vl);
    }
    store_from_float_m4(out + d, v_sum, vl, scratch);
  }
#else
  for (int64_t d = 0; d < K; ++d) {
    float s = 0.f;
    for (int64_t t = 0; t < topk; ++t)
      s += static_cast<float>(input[t * K + d]);
    out[d] = static_cast<scalar_t>(s);
  }
#endif
}

// add_mul_stub: out = in_float + in2_scalar * scale
template <typename scalar_t>
inline void add_mul_stub(
    scalar_t* __restrict__ out,
    const float* __restrict__ input,
    const scalar_t* __restrict__ input2,
    float scale,
    int64_t size) {
#if defined(CPU_CAPABILITY_RVV)
  size_t vl;
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
  vfloat32m4_t v_s = __riscv_vfmv_v_f_f32m4(scale, __riscv_vsetvlmax_e32m4());
  for (int64_t d = 0; d < size; d += vl) {
    vl = __riscv_vsetvl_e32m4(size - d);
    vfloat32m4_t v_x = __riscv_vle32_v_f32m4(input + d, vl);
    vfloat32m4_t v_y = load_as_float_m4(input2 + d, vl, scratch);
    vfloat32m4_t v_out = __riscv_vfmacc_vv_f32m4(v_x, v_s, v_y, vl);
    store_from_float_m4(out + d, v_out, vl, scratch);
  }
#else
  for (int64_t d = 0; d < size; ++d)
    out[d] = static_cast<scalar_t>(input[d] + static_cast<float>(input2[d]) * scale);
#endif
}

// silu_and_mul: for each row m, out[m*N+j] = silu(C0[m*n_size+j]) * C1[m*n_size+j]
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
template <typename scalar_t>
inline void silu_and_mul(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ C0,
    const scalar_t* __restrict__ C1,
    int64_t m_size,
    int64_t N,
    int64_t n_size) {
#if defined(CPU_CAPABILITY_RVV)
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
  for (int64_t m = 0; m < m_size; ++m) {
    scalar_t* __restrict__ out_row = output + m * N;
    const scalar_t* c0_row = C0 + m * n_size;
    const scalar_t* c1_row = C1 + m * n_size;
    size_t vl;
    for (int64_t j = 0; j < n_size; j += vl) {
      vl = __riscv_vsetvl_e32m4(n_size - j);
      vfloat32m4_t v_x = load_as_float_m4(c0_row + j, vl, scratch);
      vfloat32m4_t v_y = load_as_float_m4(c1_row + j, vl, scratch);
      vfloat32m4_t v_one = __riscv_vfmv_v_f_f32m4(1.0f, vl);
      vfloat32m4_t v_neg = __riscv_vfneg_v_f32m4(v_x, vl);
      vfloat32m4_t v_exp = vfexp_f32m4(v_neg, vl);
      vfloat32m4_t v_denom = __riscv_vfadd_vv_f32m4(v_one, v_exp, vl);
      vfloat32m4_t v_sig = __riscv_vfdiv_vv_f32m4(v_x, v_denom, vl);
      vfloat32m4_t v_out = __riscv_vfmul_vv_f32m4(v_sig, v_y, vl);
      store_from_float_m4(out_row + j, v_out, vl, scratch);
    }
  }
#else
  for (int64_t m = 0; m < m_size; ++m) {
    for (int64_t j = 0; j < n_size; ++j) {
      float x = static_cast<float>(C0[m * n_size + j]);
      float y = static_cast<float>(C1[m * n_size + j]);
      float silu = x / (1.0f + std::exp(-x));
      output[m * N + j] = static_cast<scalar_t>(silu * y);
    }
  }
#endif
}

// moe_align_block_size: architecture-neutral token routing
template <int BLOCK_M>
int moe_align_block_size(
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ total_cnts,
    int32_t* __restrict__ cumsums,
    int32_t* __restrict__ offsets,
    int num_experts,
    int numel,
    int num_threads) {
#define T_INDEX(tt) total_cnts + (tt) * num_experts

  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    int32_t* __restrict__ local_cnts = T_INDEX(tid + 1);
    for (int i = begin; i < end; ++i)
      local_cnts[topk_ids[i]]++;
  });

  using iVec = at::vec::Vectorized<int32_t>;
  for (int t = 0; t < num_threads; ++t) {
    at::vec::map2<int32_t>(
        [](iVec x, iVec y) { return x + y; }, T_INDEX(t + 1), T_INDEX(t + 1), T_INDEX(t), num_experts);
  }

  int32_t* total_cnts_t_1 = T_INDEX(num_threads);
  cumsums[0] = 0;
  for (int e = 0; e < num_experts; ++e) {
    cumsums[e + 1] = cumsums[e] + div_up(total_cnts_t_1[e], BLOCK_M) * BLOCK_M;
    for (int k = cumsums[e]; k < cumsums[e + 1]; k += BLOCK_M)
      expert_ids[k / BLOCK_M] = e;
  }
  int num_tokens_post_pad = cumsums[num_experts];

  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    int32_t* __restrict__ t_offsets = T_INDEX(tid);
    for (int i = begin; i < end; ++i) {
      int32_t expert_id = topk_ids[i];
      sorted_ids[cumsums[expert_id] + t_offsets[expert_id]] = i;
      t_offsets[expert_id]++;
    }
  });

  int32_t* total_cnts_t_2 = T_INDEX(num_threads - 1);
  for (int e = 0; e < num_experts; ++e)
    TORCH_CHECK(total_cnts_t_1[e] == total_cnts_t_2[e]);

  auto sorted_id_size = [=](const int32_t* ptr) {
    for (int d = 0; d < BLOCK_M; ++d)
      if (ptr[d] == numel) return d;
    return BLOCK_M;
  };

  offsets[0] = 0;
  const int num_token_blocks = num_tokens_post_pad / BLOCK_M;
  at::parallel_for(0, num_token_blocks, GRAIN_SIZE / BLOCK_M, [&](int begin, int end) {
    for (int mb = begin; mb < end; ++mb)
      offsets[mb + 1] = sorted_id_size(sorted_ids + mb * BLOCK_M);
  });
  for (int mb = 0; mb < num_token_blocks; ++mb)
    offsets[mb + 1] += offsets[mb];
  TORCH_CHECK(offsets[num_token_blocks] == numel);
  return num_tokens_post_pad;
#undef T_INDEX
}

// Main FP16/BF16 MoE kernels
template <typename scalar_t>
void fused_experts_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ packed_w1,
    const scalar_t* __restrict__ packed_w2,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  TORCH_CHECK(N % BLOCK_N == 0, "N must be a multiple of BLOCK_N=", BLOCK_N, " for RVV MoE, got N=", N);

  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  const int64_t stride_e = 2 * N * K;
  const int64_t stride_nb = BLOCK_N * K;

  // Stage 1: ic1 = silu(A @ w1[:N,:]) * (A @ w1[N:,:])
  at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    scalar_t* A = A_tmp + tid * BLOCK_M * K;
    float* Ctmp = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    scalar_t C0_scratch[BLOCK_M * BLOCK_N];
    scalar_t C1_scratch[BLOCK_M * BLOCK_N];

    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t mb = idx / NB;
      int64_t nb = idx % NB;

      int64_t m_size = offsets[mb + 1] - offsets[mb];
      if (m_size == 0) continue;

      int64_t n_size = std::min(BLOCK_N, N - nb * BLOCK_N);
      int32_t expert_id = expert_ids[mb];

      const scalar_t* B0 = packed_w1 + expert_id * stride_e + nb * stride_nb;
      const scalar_t* B1 = packed_w1 + expert_id * stride_e + (nb + NB) * stride_nb;

      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t token_idx = A_ids[m] / topk;
        copy_stub(A + m * K, input + token_idx * K, K);
      }

      tinygemm_kernel(A, B0, C0_scratch, Ctmp, m_size, n_size, K, K, BLOCK_N, n_size, false);
      tinygemm_kernel(A, B1, C1_scratch, Ctmp, m_size, n_size, K, K, BLOCK_N, n_size, false);

      int64_t offset = offsets[mb];
      silu_and_mul(ic1 + offset * N + nb * BLOCK_N, C0_scratch, C1_scratch, m_size, N, n_size);
    }
  });

  // Stage 2: ic2 += (ic1 @ w2) * topk_weights (scatter by sorted index)
  const int64_t OC = K;
  const int64_t IC = N;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_e2 = OC * IC;
  const int64_t stride_nb2 = BLOCK_N * IC;

  at::parallel_for(0, MB * NB2, 0, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    float* Ctmp = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    scalar_t C_scratch[BLOCK_M * BLOCK_N];

    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t mb = idx / NB2;
      int64_t nb2 = idx % NB2;

      int64_t m_size = offsets[mb + 1] - offsets[mb];
      if (m_size == 0) continue;

      int64_t n_size = std::min(BLOCK_N, OC - nb2 * BLOCK_N);
      int32_t expert_id = expert_ids[mb];

      const scalar_t* A = ic1 + offsets[mb] * IC;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      const scalar_t* B = packed_w2 + expert_id * stride_e2 + nb2 * stride_nb2;

      tinygemm_kernel(A, B, C_scratch, Ctmp, m_size, n_size, IC, IC, BLOCK_N, n_size, false);

      for (int64_t m = 0; m < m_size; ++m) {
        int32_t sorted_idx = A_ids[m];
        float weight = topk_weights[sorted_idx];
        copy_mul_stub(ic2 + sorted_idx * OC + nb2 * BLOCK_N, C_scratch + m * n_size, weight, n_size);
      }
    }
  });

  // Stage 3: output = sum over topk of ic2
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m)
      sum_stub(output + m * K, ic2 + m * topk * K, topk, K);
  });
}

template <typename scalar_t>
void shared_expert_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    float* __restrict__ C_tmp,
    scalar_t* __restrict__ input,
    const scalar_t* __restrict__ packed_w1,
    const scalar_t* __restrict__ packed_w2,
    const scalar_t* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  TORCH_CHECK(N % BLOCK_N == 0, "N must be a multiple of BLOCK_N=", BLOCK_N, " for RVV shared expert");

  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  const int64_t stride_nb = BLOCK_N * K;

  // Stage 1: ic1 = silu(input @ w1[:N]) * (input @ w1[N:])
  at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    float* Ctmp = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    scalar_t C0_scratch[BLOCK_M * BLOCK_N];
    scalar_t C1_scratch[BLOCK_M * BLOCK_N];

    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t mb = idx / NB;
      int64_t nb = idx % NB;

      int64_t m_size = std::min(BLOCK_M, M - mb * BLOCK_M);
      int64_t n_size = std::min(BLOCK_N, N - nb * BLOCK_N);

      const scalar_t* A = input + mb * BLOCK_M * K;
      const scalar_t* B0 = packed_w1 + nb * stride_nb;
      const scalar_t* B1 = packed_w1 + (nb + NB) * stride_nb;

      tinygemm_kernel(A, B0, C0_scratch, Ctmp, m_size, n_size, K, K, BLOCK_N, n_size, false);
      tinygemm_kernel(A, B1, C1_scratch, Ctmp, m_size, n_size, K, K, BLOCK_N, n_size, false);

      silu_and_mul(ic1 + mb * BLOCK_M * N + nb * BLOCK_N, C0_scratch, C1_scratch, m_size, N, n_size);
    }
  });

  // Stage 2: output = ic1 @ w2 + fused_experts_out * routed_scaling_factor
  const int64_t OC = K;
  const int64_t IC = N;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_nb2 = BLOCK_N * IC;

  at::parallel_for(0, MB * NB2, 0, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    float* Ctmp = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    float* C_float = Ctmp + BLOCK_M * BLOCK_N;
    scalar_t C_scratch[BLOCK_M * BLOCK_N];

    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t mb = idx / NB2;
      int64_t nb2 = idx % NB2;

      int64_t m_size = std::min(BLOCK_M, M - mb * BLOCK_M);
      int64_t n_size = std::min(BLOCK_N, OC - nb2 * BLOCK_N);

      const scalar_t* A = ic1 + mb * BLOCK_M * IC;
      const scalar_t* B = packed_w2 + nb2 * stride_nb2;

      tinygemm_kernel(A, B, C_scratch, Ctmp, m_size, n_size, IC, IC, BLOCK_N, n_size, false);

      scalar_t* out_row = output + mb * BLOCK_M * OC + nb2 * BLOCK_N;
      const scalar_t* fused_row = fused_experts_out + mb * BLOCK_M * OC + nb2 * BLOCK_N;

      for (int64_t m = 0; m < m_size; ++m) {
        copy_stub(C_float, C_scratch + m * n_size, n_size);
        add_mul_stub(out_row + m * OC, C_float, fused_row + m * OC, routed_scaling_factor, n_size);
      }
    }
  });
}

// INT8/INT4 stubs (linker satisfaction only — not implemented)
template <typename scalar_t>
void fused_experts_int8_kernel_impl(
    scalar_t*,
    scalar_t*,
    scalar_t*,
    uint8_t*,
    float*,
    uint8_t*,
    float*,
    const scalar_t*,
    const int8_t*,
    const int8_t*,
    const float*,
    const float*,
    const float*,
    const int32_t*,
    const int32_t*,
    const int32_t*,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t) {
  TORCH_CHECK(false, "fused_experts_int8 (w8a8) is not yet implemented for RISC-V");
}

template <typename scalar_t>
void shared_expert_int8_kernel_impl(
    scalar_t*,
    scalar_t*,
    float*,
    uint8_t*,
    float*,
    const scalar_t*,
    const int8_t*,
    const int8_t*,
    const float*,
    const float*,
    const scalar_t*,
    float,
    int64_t,
    int64_t,
    int64_t) {
  TORCH_CHECK(false, "shared_expert_int8 (w8a8) is not yet implemented for RISC-V");
}

template <typename scalar_t>
void fused_experts_int4_w4a8_kernel_impl(
    scalar_t*,
    scalar_t*,
    scalar_t*,
    scalar_t*,
    uint8_t*,
    uint8_t*,
    float*,
    int32_t*,
    float*,
    int8_t*,
    const scalar_t*,
    const uint8_t*,
    const uint8_t*,
    const int8_t*,
    const int8_t*,
    const float*,
    const float*,
    int,
    const float*,
    const int32_t*,
    const int32_t*,
    const int32_t*,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t) {
  TORCH_CHECK(false, "INT4_W4A8 MoE is not supported on RISC-V");
}

#define INST_MOE_STUBS(T)                               \
  template void fused_experts_int8_kernel_impl<T>(      \
      T*,                                               \
      T*,                                               \
      T*,                                               \
      uint8_t*,                                         \
      float*,                                           \
      uint8_t*,                                         \
      float*,                                           \
      const T*,                                         \
      const int8_t*,                                    \
      const int8_t*,                                    \
      const float*,                                     \
      const float*,                                     \
      const float*,                                     \
      const int32_t*,                                   \
      const int32_t*,                                   \
      const int32_t*,                                   \
      int64_t,                                          \
      int64_t,                                          \
      int64_t,                                          \
      int64_t,                                          \
      int64_t,                                          \
      int64_t);                                         \
  template void shared_expert_int8_kernel_impl<T>(      \
      T*,                                               \
      T*,                                               \
      float*,                                           \
      uint8_t*,                                         \
      float*,                                           \
      const T*,                                         \
      const int8_t*,                                    \
      const int8_t*,                                    \
      const float*,                                     \
      const float*,                                     \
      const T*,                                         \
      float,                                            \
      int64_t,                                          \
      int64_t,                                          \
      int64_t);                                         \
  template void fused_experts_int4_w4a8_kernel_impl<T>( \
      T*,                                               \
      T*,                                               \
      T*,                                               \
      T*,                                               \
      uint8_t*,                                         \
      uint8_t*,                                         \
      float*,                                           \
      int32_t*,                                         \
      float*,                                           \
      int8_t*,                                          \
      const T*,                                         \
      const uint8_t*,                                   \
      const uint8_t*,                                   \
      const int8_t*,                                    \
      const int8_t*,                                    \
      const float*,                                     \
      const float*,                                     \
      int,                                              \
      const float*,                                     \
      const int32_t*,                                   \
      const int32_t*,                                   \
      const int32_t*,                                   \
      int64_t,                                          \
      int64_t,                                          \
      int64_t,                                          \
      int64_t,                                          \
      int64_t,                                          \
      int64_t)

INST_MOE_STUBS(at::BFloat16);
INST_MOE_STUBS(at::Half);

}  // anonymous namespace

// Public API
at::Tensor fused_experts_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    int64_t moe_comp_method,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<at::Tensor>& w1_zero,
    const std::optional<at::Tensor>& w2_zero,
    const std::optional<std::vector<int64_t>> block_size,
    bool is_vnni) {
  RECORD_FUNCTION(
      "sgl-kernel::fused_experts_cpu", std::vector<c10::IValue>({hidden_states, w1, w2, topk_weights, topk_ids}));

  bool is_packed = is_vnni;
  auto packed_w1 = is_packed ? w1 : convert_weight_packed(w1);
  auto packed_w2 = is_packed ? w2 : convert_weight_packed(w2);

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  const auto st = hidden_states.scalar_type();
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(w1);
  CHECK_INPUT(w2);
  CHECK_EQ(topk_weights.sizes(), topk_ids.sizes());
  CHECK_DIM(2, hidden_states);
  if (moe_comp_method == kCPUQuantINT4_W4A8 && is_packed) {
    CHECK_DIM(4, w1);
    CHECK_DIM(4, w2);
  } else if (!is_packed) {
    CHECK_DIM(3, w1);
    CHECK_DIM(3, w2);
  } else {
    // RVV packed: w1/w2 are [E, NB, block_size]; same ndim as unpacked
    TORCH_CHECK(
        w1.ndimension() == 3 && w2.ndimension() == 3,
        "when is_packed, w1 and w2 must be 3D [E, NB, block_size]; got w1.ndim=",
        w1.ndimension(),
        " w2.ndim=",
        w2.ndimension());
  }
  CHECK_DIM(2, topk_weights);
  CHECK_DIM(2, topk_ids);
  CHECK_EQ(topk_ids.scalar_type(), at::kInt);

  auto topk_weights_ = topk_weights.to(at::kFloat);
  CHECK_EQ(topk_weights_.scalar_type(), at::kFloat);

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t N;
  int64_t E;
  if (is_packed && moe_comp_method != kCPUQuantINT4_W4A8) {
    // RVV packed: w1 [E, NB, K*BLOCK_N], w2 [E, NB2, N*BLOCK_N]; derive E and N from packed shape
    E = w1.size(0);
    const int64_t NB_w1 = w1.size(1);
    N = (NB_w1 * BLOCK_N) / 2;
  } else {
    N = (moe_comp_method == kCPUQuantINT4_W4A8 && w1_scale.has_value())
            ? w1_scale.value().size(1) * w1_scale.value().size(3) / 2
            : w1.size(1) / 2;
    E = w1.size(0);
  }
  int64_t topk = topk_weights_.size(1);

  int64_t packed_K = get_row_size(K, moe_comp_method == kCPUQuantINT8_W8A8);
  int64_t packed_N = get_row_size(N, moe_comp_method == kCPUQuantINT8_W8A8);

  CHECK_EQ(w2.size(0), E);
  if (moe_comp_method != kCPUQuantINT4_W4A8) {
    if (!is_packed) {
      CHECK_EQ(w2.size(1), K);
      CHECK_EQ(packed_w1.size(2), packed_K);
      CHECK_EQ(packed_w2.size(2), packed_N);
    }
    // when is_packed, packed layout is [E, NB, K*BLOCK_N] / [E, NB2, N*BLOCK_N]; no need to check last dim vs
    // packed_K/packed_N
  }
  check_moe_scales(
      moe_comp_method == kCPUQuantINT8_W8A8, moe_comp_method == kCPUQuantFP8_W8A16, w1_scale, w2_scale, block_size);

  at::Tensor out_hidden_states = inplace ? hidden_states : at::empty_like(hidden_states);

  int num_threads = at::get_num_threads();
  int64_t max_num_tokens_padded = M * topk + E * (BLOCK_M - 1);
  int64_t max_num_blocks = div_up(max_num_tokens_padded, BLOCK_M);
  auto buffer = at::empty(
      {max_num_tokens_padded + max_num_blocks + (num_threads + 1) * E + (E + 1) + (max_num_blocks + 1)},
      topk_ids.options());

  int32_t* __restrict__ sorted_ids = buffer.data_ptr<int32_t>();
  int32_t* __restrict__ expert_ids = sorted_ids + max_num_tokens_padded;
  int32_t* __restrict__ total_cnts = expert_ids + max_num_blocks;
  int32_t* __restrict__ cumsums = total_cnts + (num_threads + 1) * E;
  int32_t* __restrict__ offsets = cumsums + (E + 1);

  int64_t numel = M * topk;
  at::parallel_for(0, max_num_blocks, GRAIN_SIZE / BLOCK_M, [&](int64_t begin, int64_t end) {
    int64_t m_start = begin * BLOCK_M;
    int64_t m_size = std::min((end - begin) * BLOCK_M, max_num_tokens_padded - m_start);
    fill_stub(sorted_ids + m_start, static_cast<int32_t>(numel), m_size);
    fill_stub(expert_ids + begin, static_cast<int32_t>(E), end - begin);
  });
  at::parallel_for(0, (num_threads + 1) * E + (E + 1), GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    fill_stub(total_cnts + begin, 0, end - begin);
  });

  int64_t num_tokens_post_pad = moe_align_block_size<BLOCK_M>(
      sorted_ids, expert_ids, topk_ids.data_ptr<int32_t>(), total_cnts, cumsums, offsets, E, numel, num_threads);

  int64_t buffer_size_nbytes = max_num_tokens_padded * N * 2 + M * topk * K * 2 + num_threads * BLOCK_M * K * 2 +
                               num_threads * 2 * BLOCK_M * BLOCK_N * sizeof(float);
  if (moe_comp_method == kCPUQuantINT8_W8A8) {
    buffer_size_nbytes += std::max(M * K, M * topk * N) + M * topk * sizeof(float);
  }
  if (moe_comp_method == kCPUQuantFP8_W8A16 || moe_comp_method == kCPUQuantINT4_W4A8) {
    TORCH_CHECK(false, "FP8 and INT4 MoE are not supported on RISC-V");
  }

  auto buffer2 = at::empty({buffer_size_nbytes}, hidden_states.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_experts_kernel_impl", [&] {
    scalar_t* __restrict__ intermediate_cache1 = (scalar_t*)((void*)(buffer2.data_ptr<int8_t>()));
    scalar_t* __restrict__ intermediate_cache2 = intermediate_cache1 + max_num_tokens_padded * N;
    scalar_t* __restrict__ A_tmp = intermediate_cache2 + M * topk * K;
    float* __restrict__ C_tmp = (float*)((void*)(A_tmp + num_threads * BLOCK_M * K));

    if (moe_comp_method == kCPUQuantINT8_W8A8) {
      fused_experts_int8_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache1,
          intermediate_cache2,
          nullptr,
          C_tmp,
          nullptr,
          nullptr,
          hidden_states.data_ptr<scalar_t>(),
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          topk_weights_.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad);
    } else {
      fused_experts_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache1,
          intermediate_cache2,
          A_tmp,
          C_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<scalar_t>(),
          packed_w2.data_ptr<scalar_t>(),
          topk_weights_.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad);
    }
  });
  return out_hidden_states;
}

at::Tensor shared_expert_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& fused_experts_out,
    double routed_scaling_factor,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    bool is_vnni) {
  RECORD_FUNCTION("sgl-kernel::shared_expert_cpu", std::vector<c10::IValue>({hidden_states, w1, w2}));

  bool is_packed = is_vnni;
  auto packed_w1 = is_packed ? w1 : convert_weight_packed(w1);
  auto packed_w2 = is_packed ? w2 : convert_weight_packed(w2);

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  const auto st = hidden_states.scalar_type();
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(fused_experts_out);
  CHECK_INPUT(w1);
  CHECK_INPUT(w2);
  CHECK_DIM(2, hidden_states);
  TORCH_CHECK(
      w1.ndimension() == 2 && w2.ndimension() == 2,
      "w1 and w2 must be 2D (unpacked [OC, K] or packed [NB, block_size])");
  CHECK_EQ(hidden_states.sizes(), fused_experts_out.sizes());

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t N;
  if (is_packed && !use_int8_w8a8) {
    // RVV packed: w1 is [NB, K*BLOCK_N] with 2*N = NB*BLOCK_N (gate+up)
    N = (w1.size(0) * BLOCK_N) / 2;
  } else {
    N = w1.size(0) / 2;
  }

  int64_t packed_K = get_row_size(K, use_int8_w8a8);
  int64_t packed_N = get_row_size(N, use_int8_w8a8);

  if (!is_packed) {
    CHECK_EQ(w2.size(0), K);
    CHECK_EQ(packed_w1.size(1), packed_K);
    CHECK_EQ(packed_w2.size(1), packed_N);
  }
  // when is_packed, packed w1/w2 are [NB, K*BLOCK_N] / [NB2, N*BLOCK_N]
  check_moe_scales(use_int8_w8a8, use_fp8_w8a16, w1_scale, w2_scale, block_size);

  at::Tensor out_hidden_states = inplace ? hidden_states : at::empty_like(hidden_states);

  int num_threads = at::get_num_threads();
  int64_t buffer_size_nbytes = M * N * 2 + num_threads * 2 * BLOCK_M * BLOCK_N * sizeof(float);
  if (use_int8_w8a8) buffer_size_nbytes += std::max(M * K, M * N) + M * sizeof(float);
  if (use_fp8_w8a16) TORCH_CHECK(false, "FP8 shared expert is not supported on RISC-V");

  auto buffer = at::empty({buffer_size_nbytes}, hidden_states.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "shared_expert_kernel_impl", [&] {
    scalar_t* __restrict__ intermediate_cache1 = (scalar_t*)((void*)(buffer.data_ptr<int8_t>()));
    float* __restrict__ C_tmp = (float*)((void*)(intermediate_cache1 + M * N));

    if (use_int8_w8a8) {
      shared_expert_int8_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache1,
          C_tmp,
          nullptr,
          nullptr,
          hidden_states.data_ptr<scalar_t>(),
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          fused_experts_out.data_ptr<scalar_t>(),
          static_cast<float>(routed_scaling_factor),
          M,
          N,
          K);
    } else {
      shared_expert_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache1,
          C_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<scalar_t>(),
          packed_w2.data_ptr<scalar_t>(),
          fused_experts_out.data_ptr<scalar_t>(),
          static_cast<float>(routed_scaling_factor),
          M,
          N,
          K);
    }
  });
  return out_hidden_states;
}
