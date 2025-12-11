#include "common.h"

#if defined(CPU_CAPABILITY_RVV)
void prefill_cache_kernel_rvv(
    at::Tensor& k_new,
    at::Tensor& v_new,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_start_loc,
    at::Tensor& extend_seq_lens);
#define prefill_cache_kernel_dispatch prefill_cache_kernel_rvv
#else
// Fallback or error if not RVV
namespace {
void prefill_cache_kernel_fallback(
    at::Tensor& k_new,
    at::Tensor& v_new,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_start_loc,
    at::Tensor& extend_seq_lens) {
  TORCH_CHECK(false, "prefill_cache_kernel not implemented for non-RVV CPU");
}
}  // namespace
#define prefill_cache_kernel_dispatch prefill_cache_kernel_fallback
#endif

void prefill_cache_cpu(
    at::Tensor& k_new,
    at::Tensor& v_new,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_start_loc,
    at::Tensor& extend_seq_lens) {
  prefill_cache_kernel_dispatch(
      k_new, v_new, k_buffer, v_buffer, req_to_token, req_pool_indices, seq_lens, extend_start_loc, extend_seq_lens);
}
