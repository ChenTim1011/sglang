#include "common.h"
#include "gemm.h"
#include "vec.h"

#if defined(CPU_CAPABILITY_RVV)
// Clang 19+ version check
#if !defined(__clang__) || !defined(__clang_major__) || __clang_major__ < 19
#error "RVV backend requires Clang 19 or later. Please use Clang 19 or later to compile this file."
#endif

#include <riscv_vector.h>

#include "vector_helpers.h"
#include "vector_math.h"

template <typename scalar_t, typename index_t>
void decode_attention_kernel_rvv_int8(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);

template <typename scalar_t, typename index_t>
void decode_attention_grouped_kernel_rvv_int8(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const void* __restrict__ k_buffer,
    const void* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    float k_scale,
    float v_scale);
#endif

namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ src, int64_t size) {
  copy_stub_rvv<scalar_t>(out, src, size);
}

// Symmetric quantization (signed int8) for KV cache
template <typename scalar_t>
inline void quantize_row_int8_symmetric_with_scale(
    int8_t* __restrict__ Aq, const scalar_t* __restrict__ A, int64_t K, float scale) {
  quantize_row_int8_symmetric_rvv<scalar_t>(Aq, A, K, scale);
}

template <typename scalar_t>
inline void quantize_row_int8_symmetric_auto(
    int8_t* __restrict__ Aq, float& scale_out, const scalar_t* __restrict__ A, int64_t K, float eps = 1e-7) {
  float amax = 0.f;
  for (int64_t k = 0; k < K; ++k) {
    const float val = static_cast<float>(A[k]);
    amax = std::max(amax, std::abs(val));
  }

  amax = std::max(amax, eps);
  const float scale = amax / 127.0f;
  const float inv_scale = 127.0f / amax;

  for (int64_t k = 0; k < K; ++k) {
    const float val = static_cast<float>(A[k]) * inv_scale;
    int32_t quantized = (int32_t)(std::round(val));
    quantized = std::max(-128, std::min(127, quantized));
    Aq[k] = (int8_t)(quantized);
  }

  scale_out = scale;
}

template <typename buffer_t>
void decode_set_kv_buffer_int8_copy(
    buffer_t* __restrict__ k_buffer,
    buffer_t* __restrict__ v_buffer,
    const buffer_t* __restrict__ key,
    const buffer_t* __restrict__ value,
    const int64_t* __restrict__ loc,
    int64_t batches,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    int64_t nk_strideN,
    int64_t nk_strideH,
    int64_t nv_strideN,
    int64_t nv_strideH) {
  at::parallel_for(0, batches * num_heads_kv, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, head_kv_id{0};
    data_index_init(begin, bs, batches, head_kv_id, num_heads_kv);

    for (int64_t i = begin; i < end; i++) {
      int64_t loc_val = loc[bs];
      buffer_t* k_buffer_ptr = k_buffer + loc_val * k_strideN + head_kv_id * k_strideH;
      const buffer_t* new_key_ptr = key + bs * nk_strideN + head_kv_id * nk_strideH;
      copy_stub_rvv<buffer_t>(k_buffer_ptr, new_key_ptr, head_size);

      buffer_t* v_buffer_ptr = v_buffer + loc_val * v_strideN + head_kv_id * v_strideH;
      const buffer_t* new_value_ptr = value + bs * nv_strideN + head_kv_id * nv_strideH;
      copy_stub_rvv<buffer_t>(v_buffer_ptr, new_value_ptr, head_size_v);

      data_index_step(bs, batches, head_kv_id, num_heads_kv);
    }
  });
}

template <typename scalar_t, typename buffer_t>
void decode_set_kv_buffer_int8_quantize(
    buffer_t* __restrict__ k_buffer,
    buffer_t* __restrict__ v_buffer,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ loc,
    int64_t batches,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    int64_t nk_strideN,
    int64_t nk_strideH,
    int64_t nv_strideN,
    int64_t nv_strideH,
    float k_scale,
    float v_scale) {
  at::parallel_for(0, batches * num_heads_kv, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, head_kv_id{0};
    data_index_init(begin, bs, batches, head_kv_id, num_heads_kv);

    for (int64_t i = begin; i < end; i++) {
      int64_t loc_val = loc[bs];
      int8_t* k_buffer_ptr = reinterpret_cast<int8_t*>(k_buffer + loc_val * k_strideN + head_kv_id * k_strideH);
      const scalar_t* new_key_ptr = key + bs * nk_strideN + head_kv_id * nk_strideH;

      int8_t* v_buffer_ptr = reinterpret_cast<int8_t*>(v_buffer + loc_val * v_strideN + head_kv_id * v_strideH);
      const scalar_t* new_value_ptr = value + bs * nv_strideN + head_kv_id * nv_strideH;

      if (k_scale != 1.0f && k_scale > 0.0f) {
        quantize_row_int8_symmetric_with_scale<scalar_t>(k_buffer_ptr, new_key_ptr, head_size, k_scale);
      } else {
        float computed_k_scale;
        quantize_row_int8_symmetric_auto<scalar_t>(k_buffer_ptr, computed_k_scale, new_key_ptr, head_size);
      }

      if (v_scale != 1.0f && v_scale > 0.0f) {
        quantize_row_int8_symmetric_with_scale<scalar_t>(v_buffer_ptr, new_value_ptr, head_size_v, v_scale);
      } else {
        float computed_v_scale;
        quantize_row_int8_symmetric_auto<scalar_t>(v_buffer_ptr, computed_v_scale, new_value_ptr, head_size_v);
      }

      data_index_step(bs, batches, head_kv_id, num_heads_kv);
    }
  });
}

}  // namespace

void decode_attention_int8_cpu(
    at::Tensor& query,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& output,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& loc,
    at::Tensor& attn_logits,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    double sm_scale,
    double logit_cap,
    double k_scale,
    double v_scale) {
  RECORD_FUNCTION(
      "sgl-kernel::decode_attention_int8_cpu_rvv",
      std::vector<c10::IValue>(
          {query, output, k_buffer, v_buffer, attn_logits, req_to_token, req_pool_indices, seq_lens}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(value);
  CHECK_DIM(3, query);
  CHECK_DIM(3, k_buffer);
  CHECK_DIM(3, v_buffer);
  CHECK_DIM(3, key);
  CHECK_DIM(3, value);
  CHECK_DIM(1, loc);
  TORCH_CHECK(
      k_buffer.scalar_type() == at::kByte || k_buffer.scalar_type() == at::kChar,
      "decode_int8: k_buffer must be int8/uint8, got ",
      k_buffer.scalar_type());
  TORCH_CHECK(
      v_buffer.scalar_type() == at::kByte || v_buffer.scalar_type() == at::kChar,
      "decode_int8: v_buffer must be int8/uint8, got ",
      v_buffer.scalar_type());

  int64_t num_seqs = seq_lens.size(0);
  int64_t max_num_reqs = req_to_token.size(0);
  int64_t max_context_len = req_to_token.size(1);
  int64_t max_total_num_tokens = k_buffer.size(0);

  int64_t num_heads = query.size(1);
  int64_t num_heads_kv = k_buffer.size(1);
  int64_t head_size = query.size(2);
  int64_t head_size_v = v_buffer.size(2);

  int64_t num_kv_splits = attn_logits.size(2);

  CHECK_EQ(loc.numel(), num_seqs);
  CHECK_EQ(attn_logits.size(0), num_seqs);
  CHECK_EQ(attn_logits.size(1), num_heads);
  CHECK_EQ(attn_logits.size(3), head_size_v + 1);
  CHECK_EQ(attn_logits.scalar_type(), at::kFloat);

  int64_t q_strideM = query.stride(0);
  int64_t q_strideH = query.stride(1);
  int64_t k_strideN = k_buffer.stride(0);
  int64_t k_strideH = k_buffer.stride(1);
  int64_t v_strideN = v_buffer.stride(0);
  int64_t v_strideH = v_buffer.stride(1);
  int64_t nk_strideN = key.stride(0);
  int64_t nk_strideH = key.stride(1);
  int64_t nv_strideN = value.stride(0);
  int64_t nv_strideH = value.stride(1);

  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "decode: expect req_to_token to be int32 or int64, got ",
      index_dtype);

  void* k_buffer_data = k_buffer.data_ptr();
  void* v_buffer_data = v_buffer.data_ptr();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "decode_attention_int8_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "decode_attention_indices", [&] {
      // Update KV buffer with INT8 quantization
      if (key.scalar_type() == at::kByte || key.scalar_type() == at::kChar) {
        // Input already INT8: just copy
        decode_set_kv_buffer_int8_copy<int8_t>(
            (int8_t*)k_buffer_data,
            (int8_t*)v_buffer_data,
            (int8_t*)key.data_ptr(),
            (int8_t*)value.data_ptr(),
            loc.data_ptr<int64_t>(),
            num_seqs,
            num_heads_kv,
            head_size,
            head_size_v,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            nk_strideN,
            nk_strideH,
            nv_strideN,
            nv_strideH);
      } else {
        AT_DISPATCH_REDUCED_FLOATING_TYPES(key.scalar_type(), "decode_set_kv_buffer_quantize", [&] {
          // Quantize FP input to INT8 (symmetric)
          decode_set_kv_buffer_int8_quantize<scalar_t, int8_t>(
              (int8_t*)k_buffer_data,
              (int8_t*)v_buffer_data,
              key.data_ptr<scalar_t>(),
              value.data_ptr<scalar_t>(),
              loc.data_ptr<int64_t>(),
              num_seqs,
              num_heads_kv,
              head_size,
              head_size_v,
              k_strideN,
              k_strideH,
              v_strideN,
              v_strideH,
              nk_strideN,
              nk_strideH,
              nv_strideN,
              nv_strideH,
              (float)k_scale,
              (float)v_scale);
        });
      }

#if defined(CPU_CAPABILITY_RVV)
      // Dispatch to INT8-aware RVV kernel
      if (num_heads == num_heads_kv) {
        // MHA : num_heads == num_heads_kv
        decode_attention_kernel_rvv_int8<scalar_t, index_t>(
            output.data_ptr<scalar_t>(),
            attn_logits.data_ptr<float>(),
            query.data_ptr<scalar_t>(),
            (const void*)k_buffer_data,
            (const void*)v_buffer_data,
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            num_seqs,
            num_heads,
            head_size,
            head_size_v,
            num_kv_splits,
            q_strideM,
            q_strideH,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            sm_scale,
            logit_cap,
            max_num_reqs,
            max_context_len,
            max_total_num_tokens,
            (float)k_scale,
            (float)v_scale);
      } else {
        // GQA : num_heads != num_heads_kv
        decode_attention_grouped_kernel_rvv_int8<scalar_t, index_t>(
            output.data_ptr<scalar_t>(),
            attn_logits.data_ptr<float>(),
            query.data_ptr<scalar_t>(),
            (const void*)k_buffer_data,
            (const void*)v_buffer_data,
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            num_seqs,
            num_heads,
            num_heads_kv,
            head_size,
            head_size_v,
            num_kv_splits,
            q_strideM,
            q_strideH,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            sm_scale,
            logit_cap,
            max_num_reqs,
            max_context_len,
            max_total_num_tokens,
            (float)k_scale,
            (float)v_scale);
      }
#else
      TORCH_CHECK(false, "decode_int8: RVV capability required for INT8 kernel.");
#endif
    });
  });
}
