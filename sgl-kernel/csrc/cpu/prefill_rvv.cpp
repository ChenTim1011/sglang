// RISC-V Vector Extension (RVV) optimized prefill/cache-write kernels
// This file contains RVV specific implementations for writing K/V data to cache

#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "common.h"

#if defined(CPU_CAPABILITY_RVV)

#include <riscv_vector.h>

// Check for Zvfh (FP16 vector) support
#if defined(__riscv_zvfh) || defined(__riscv_v)
#define HAS_ZVFH 1
#else
#define HAS_ZVFH 0
#endif

namespace {

// Helper: Load float vector from scalar_t buffer, handling type conversion
template <typename scalar_t>
inline vfloat32m1_t load_as_float_rvv(const scalar_t* ptr, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return __riscv_vle32_v_f32m1(ptr, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if HAS_ZVFH
    vfloat16mf2_t v_f16 = __riscv_vle16_v_f16mf2(reinterpret_cast<const _Float16*>(ptr), vl);
    return __riscv_vfwcvt_f_f_v_f32m1(v_f16, vl);
#else
    for (size_t i = 0; i < vl; ++i) {
      scratch[i] = static_cast<float>(ptr[i]);
    }
    return __riscv_vle32_v_f32m1(scratch, vl);
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    for (size_t i = 0; i < vl; ++i) {
      scratch[i] = static_cast<float>(ptr[i]);
    }
    return __riscv_vle32_v_f32m1(scratch, vl);
  } else {
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

template <typename scalar_t, typename index_t>
void cache_store_kernel_rvv_impl(
    const scalar_t* __restrict__ k_new,
    const scalar_t* __restrict__ v_new,
    scalar_t* __restrict__ k_buffer,
    scalar_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    const int64_t* __restrict__ extend_start_loc,
    const int64_t* __restrict__ extend_seq_lens,
    int num_seqs,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    int max_context_len,
    int max_total_num_tokens) {
#pragma omp parallel
  {
    alignas(64) float scratch[512];  // Thread-local scratch

#pragma omp for
    for (int b = 0; b < num_seqs; ++b) {
      int64_t req_idx = req_pool_indices[b];
      int64_t seq_len = seq_lens[b];
      int64_t new_len = extend_seq_lens[b];
      int64_t start_loc = extend_start_loc[b];
      int64_t prefix_len = seq_len - new_len;

      if (new_len <= 0) continue;

      const scalar_t* k_src_base = k_new + start_loc * num_heads_kv * head_size;
      const scalar_t* v_src_base = v_new + start_loc * num_heads_kv * head_size_v;

      for (int i = 0; i < new_len; ++i) {
        int64_t token_idx = req_to_token[req_idx * max_context_len + prefix_len + i];
        TORCH_CHECK(token_idx >= 0 && token_idx < max_total_num_tokens, "token index out of scope in cache store");

        for (int h = 0; h < num_heads_kv; ++h) {
          // Copy K
          const scalar_t* k_src = k_src_base + (i * num_heads_kv + h) * head_size;
          scalar_t* k_dst = k_buffer + token_idx * k_strideN + h * k_strideH;

          size_t vl;
          for (int d = 0; d < head_size; d += vl) {
            vl = __riscv_vsetvl_e32m1(head_size - d);
            vfloat32m1_t v_val = load_as_float_rvv(k_src + d, vl, scratch);
            store_from_float_rvv(k_dst + d, v_val, vl, scratch);
          }

          // Copy V
          const scalar_t* v_src = v_src_base + (i * num_heads_kv + h) * head_size_v;
          scalar_t* v_dst = v_buffer + token_idx * v_strideN + h * v_strideH;

          for (int d = 0; d < head_size_v; d += vl) {
            vl = __riscv_vsetvl_e32m1(head_size_v - d);
            vfloat32m1_t v_val = load_as_float_rvv(v_src + d, vl, scratch);
            store_from_float_rvv(v_dst + d, v_val, vl, scratch);
          }
        }
      }
    }
  }
}

}  // namespace

void prefill_cache_kernel_rvv(
    at::Tensor& k_new,
    at::Tensor& v_new,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_start_loc,
    at::Tensor& extend_seq_lens) {
  int num_seqs = seq_lens.size(0);
  int num_heads_kv = k_new.size(1);
  int head_size = k_new.size(2);
  int head_size_v = v_new.size(2);

  int k_strideN = k_buffer.stride(0);
  int k_strideH = k_buffer.stride(1);
  int v_strideN = v_buffer.stride(0);
  int v_strideH = v_buffer.stride(1);

  int max_context_len = req_to_token.size(1);
  int max_total_num_tokens = k_buffer.size(0);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, k_new.scalar_type(), "prefill_cache_kernel_rvv", [&] {
        if (req_to_token.scalar_type() == at::ScalarType::Int) {
          cache_store_kernel_rvv_impl<scalar_t, int32_t>(
              k_new.data_ptr<scalar_t>(),
              v_new.data_ptr<scalar_t>(),
              k_buffer.data_ptr<scalar_t>(),
              v_buffer.data_ptr<scalar_t>(),
              req_to_token.data_ptr<int32_t>(),
              req_pool_indices.data_ptr<int64_t>(),
              seq_lens.data_ptr<int64_t>(),
              extend_start_loc.data_ptr<int64_t>(),
              extend_seq_lens.data_ptr<int64_t>(),
              num_seqs,
              num_heads_kv,
              head_size,
              head_size_v,
              k_strideN,
              k_strideH,
              v_strideN,
              v_strideH,
              max_context_len,
              max_total_num_tokens);
        } else {
          cache_store_kernel_rvv_impl<scalar_t, int64_t>(
              k_new.data_ptr<scalar_t>(),
              v_new.data_ptr<scalar_t>(),
              k_buffer.data_ptr<scalar_t>(),
              v_buffer.data_ptr<scalar_t>(),
              req_to_token.data_ptr<int64_t>(),
              req_pool_indices.data_ptr<int64_t>(),
              seq_lens.data_ptr<int64_t>(),
              extend_start_loc.data_ptr<int64_t>(),
              extend_seq_lens.data_ptr<int64_t>(),
              num_seqs,
              num_heads_kv,
              head_size,
              head_size_v,
              k_strideN,
              k_strideH,
              v_strideN,
              v_strideH,
              max_context_len,
              max_total_num_tokens);
        }
      });
}

#endif
