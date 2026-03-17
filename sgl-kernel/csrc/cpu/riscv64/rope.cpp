#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include <cmath>
#include <cstdint>

#include "common.h"
#include "riscv64/vector_helpers.h"

namespace {

// 3D kernel: non-neox, out-of-place (query_out, key_out)
// query shape: [num_tokens, num_heads, head_size]
// key shape:   [num_tokens, 1, head_size]  (num_kv_heads == 1)
// Rotation: adjacent pairs (h, h+1) with cos[h/2], sin[h/2]
// Cache layout: [cos(rotary_dim/2) | sin(rotary_dim/2)] per position
template <typename scalar_t>
void rotary_embedding_3D_kernel_impl(
    scalar_t* __restrict__ query_out,
    scalar_t* __restrict__ key_out,
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t rotary_dim,
    int64_t query_stride_s,
    int64_t query_out_stride_s,
    int64_t key_out_stride_s,
    int64_t key_stride_s,
    int64_t query_stride_h,
    int64_t query_out_stride_h) {
  int64_t COFF = rotary_dim / 2;
  at::parallel_for(0, num_tokens * num_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t seq{0}, head_id{0};
    data_index_init(begin, seq, num_tokens, head_id, num_heads);
    for (int64_t i = begin; i < end; ++i) {
      int64_t in_offset_q = seq * query_stride_s + head_id * query_stride_h;
      int64_t out_offset_q = seq * query_out_stride_s + head_id * query_out_stride_h;
      int64_t out_offset_k = seq * key_out_stride_s;
      int64_t p = positions[seq];
      scalar_t* sin_start = cos_sin_cache + p * rotary_dim + COFF;
      scalar_t* cos_start = cos_sin_cache + p * rotary_dim;

      // Apply rotary pos emb to query — adjacent pairs
      for (int64_t h = 0; h < rotary_dim; h += 2) {
        float cos_val = static_cast<float>(cos_start[h >> 1]);
        float sin_val = static_cast<float>(sin_start[h >> 1]);
        float in1 = static_cast<float>(query[in_offset_q + h]);
        float in2 = static_cast<float>(query[in_offset_q + h + 1]);
        query_out[out_offset_q + h] = static_cast<scalar_t>(in1 * cos_val - in2 * sin_val);
        query_out[out_offset_q + h + 1] = static_cast<scalar_t>(in2 * cos_val + in1 * sin_val);
      }

      // Apply rotary pos emb to key (single head)
      for (int64_t h = 0; h < rotary_dim; h += 2) {
        float cos_val = static_cast<float>(cos_start[h >> 1]);
        float sin_val = static_cast<float>(sin_start[h >> 1]);
        int64_t k_pe_offset = seq * key_stride_s;
        float in1_k = static_cast<float>(key[k_pe_offset + h]);
        float in2_k = static_cast<float>(key[k_pe_offset + h + 1]);
        key_out[out_offset_k + h] = static_cast<scalar_t>(in1_k * cos_val - in2_k * sin_val);
        key_out[out_offset_k + h + 1] = static_cast<scalar_t>(in2_k * cos_val + in1_k * sin_val);
      }

      data_index_step(seq, num_tokens, head_id, num_heads);
    }
  });
}

// Neox 4D kernel: in-place
// Paired indices: (j, embed_dim + j) — front half / back half
// Cache layout: [cos(embed_dim) | sin(embed_dim)] per position
template <typename scalar_t>
void rotary_embedding_neox_4D_kernel_impl(
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t rotary_dim,
    int64_t query_stride_b,
    int64_t query_stride_s,
    int64_t query_stride_h,
    int64_t key_stride_b,
    int64_t key_stride_s,
    int64_t key_stride_h,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t batch_size,
    int64_t seq_len) {
  int64_t embed_dim = rotary_dim / 2;

  // scratch is per-call-site (inside lambda) to avoid data race under OMP parallel
  auto compute_loop = [&](scalar_t* cache_ptr, scalar_t* qk, int64_t token_head) {
    alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
    size_t vl = 0;
    for (int64_t j = 0; j < embed_dim; j += vl) {
      vl = __riscv_vsetvl_e32m4(embed_dim - j);
      int64_t x_index = j;
      int64_t y_index = embed_dim + j;

      // Load cos, sin from cache
      vfloat32m4_t v_cos = load_as_float_m4(cache_ptr + x_index, vl, scratch);
      vfloat32m4_t v_sin = load_as_float_m4(cache_ptr + y_index, vl, scratch);

      // Load x (front half) and y (back half) of head
      vfloat32m4_t v_x = load_as_float_m4(qk + token_head + x_index, vl, scratch);
      vfloat32m4_t v_y = load_as_float_m4(qk + token_head + y_index, vl, scratch);

      // out_x = x * cos - y * sin
      vfloat32m4_t v_out_x = __riscv_vfmul_vv_f32m4(v_x, v_cos, vl);
      v_out_x = __riscv_vfnmsac_vv_f32m4(v_out_x, v_y, v_sin, vl);

      // out_y = y * cos + x * sin
      vfloat32m4_t v_out_y = __riscv_vfmul_vv_f32m4(v_y, v_cos, vl);
      v_out_y = __riscv_vfmacc_vv_f32m4(v_out_y, v_x, v_sin, vl);

      store_from_float_m4(qk + token_head + x_index, v_out_x, vl, scratch);
      store_from_float_m4(qk + token_head + y_index, v_out_y, vl, scratch);
    }
  };

#pragma omp parallel for collapse(2)
  for (int64_t bs = 0; bs < batch_size; ++bs) {
    for (int64_t seq = 0; seq < seq_len; ++seq) {
      int64_t pos = positions[bs * seq_len + seq];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;

      for (int64_t i = 0; i < num_heads; ++i) {
        int64_t token_head = bs * query_stride_b + seq * query_stride_s + i * query_stride_h;
        compute_loop(cache_ptr, query, token_head);
      }

      for (int64_t i = 0; i < num_kv_heads; ++i) {
        int64_t token_head = bs * key_stride_b + seq * key_stride_s + i * key_stride_h;
        compute_loop(cache_ptr, key, token_head);
      }
    }
  }
}
// Non-neox 4D kernel: in-place
// Paired indices: (2j, 2j+1) — adjacent pairs
// Cache layout: [cos(embed_dim) | sin(embed_dim)] per position
template <typename scalar_t>
void rotary_embedding_4D_kernel_impl(
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t rotary_dim,
    int64_t query_stride_b,
    int64_t query_stride_s,
    int64_t query_stride_h,
    int64_t key_stride_b,
    int64_t key_stride_s,
    int64_t key_stride_h,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t batch_size,
    int64_t seq_len) {
  int64_t embed_dim = rotary_dim / 2;

  at::parallel_for(0, batch_size * seq_len * num_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t bs = {0}, seq = {0}, i = {0};
    data_index_init(begin, bs, batch_size, seq, seq_len, i, num_heads);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t pos = positions[bs * seq_len + seq];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;
      scalar_t* cos_cache_ptr = cache_ptr;
      scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      int64_t token_head = bs * query_stride_b + seq * query_stride_s + i * query_stride_h;
      scalar_t* head_query = token_head + query;
      for (int64_t j = 0; j < embed_dim; j += 1) {
        int64_t x_index = 2 * j;
        int64_t y_index = 2 * j + 1;

        float cos_val = static_cast<float>(cos_cache_ptr[j]);
        float sin_val = static_cast<float>(sin_cache_ptr[j]);

        float x = static_cast<float>(head_query[x_index]);
        float y = static_cast<float>(head_query[y_index]);

        head_query[x_index] = static_cast<scalar_t>(x * cos_val - y * sin_val);
        head_query[y_index] = static_cast<scalar_t>(y * cos_val + x * sin_val);
      }
      data_index_step(bs, batch_size, seq, seq_len, i, num_heads);
    }
  });

  at::parallel_for(0, batch_size * seq_len * num_kv_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t bs = {0}, seq = {0}, i = {0};
    data_index_init(begin, bs, batch_size, seq, seq_len, i, num_kv_heads);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t pos = positions[bs * seq_len + seq];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;
      scalar_t* cos_cache_ptr = cache_ptr;
      scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      int64_t token_head = bs * key_stride_b + seq * key_stride_s + i * key_stride_h;
      scalar_t* head_key = key + token_head;
      for (int64_t j = 0; j < embed_dim; j += 1) {
        int64_t x_index = 2 * j;
        int64_t y_index = 2 * j + 1;

        float cos_val = static_cast<float>(cos_cache_ptr[j]);
        float sin_val = static_cast<float>(sin_cache_ptr[j]);

        float x = static_cast<float>(head_key[x_index]);
        float y = static_cast<float>(head_key[y_index]);

        head_key[x_index] = static_cast<scalar_t>(x * cos_val - y * sin_val);
        head_key[y_index] = static_cast<scalar_t>(y * cos_val + x * sin_val);
      }
      data_index_step(bs, batch_size, seq, seq_len, i, num_kv_heads);
    }
  });
}

}  // namespace

// Top-level dispatch: rotary_embedding_cpu
std::tuple<at::Tensor, at::Tensor> rotary_embedding_cpu(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    bool is_neox) {
  RECORD_FUNCTION("sgl-kernel::rotary_embedding_cpu", std::vector<c10::IValue>({query, key}));
  CHECK_DIM(1, positions);
  const auto input_dim = query.dim();
  const auto input_dtype = query.scalar_type();
  TORCH_CHECK(
      input_dim == 2 || input_dim == 3 || input_dim == 4,
      " Query/Key must be 2D [num_tokens, num_heads*head_size] or 3D [num_tokens, num_heads, head_size] or 4D "
      "[batch_size, seq_len, num_heads, head_size] tensor");
  CHECK_DIM(2, cos_sin_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);

  int64_t rotary_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even, got ", rotary_dim);
  if (input_dim == 3) {
    CHECK_EQ(query.size(-1), rotary_dim);
    CHECK_EQ(key.size(1), 1);
  }

  int64_t num_tokens = positions.numel();
  if (input_dim <= 3) {
    CHECK_EQ(key.size(0), num_tokens);
    CHECK_EQ(query.size(0), num_tokens);
  }

  TORCH_CHECK(positions.scalar_type() == at::kLong, "expect positions to be int64, got ", positions.scalar_type());
  TORCH_CHECK(input_dtype == key.scalar_type(), "query and key must have the same data type");
  TORCH_CHECK(input_dtype == cos_sin_cache.scalar_type(), "query and cos_sin_cache must have the same data type");

  int64_t num_heads = input_dim == 2 ? query.size(-1) / head_size : query.size(-2);
  int64_t num_kv_heads = input_dim == 2 ? key.size(-1) / head_size : key.size(-2);
  int64_t key_stride_s = key.stride(0);
  int64_t query_stride_s = query.stride(0);

  int64_t query_stride_h = input_dim == 2 ? head_size : query.stride(-2);
  int64_t key_stride_h = input_dim == 2 ? head_size : key.stride(-2);
  at::Tensor query_out = at::empty_like(query);
  at::Tensor key_out = at::empty_like(key);
  int64_t query_out_stride_s = query_out.stride(0);
  int64_t key_out_stride_s = key_out.stride(0);
  int64_t query_out_stride_h = input_dim == 3 ? query_out.stride(1) : -1;
  int64_t batch_size = 1;
  int64_t seq_len = num_tokens;
  int64_t query_stride_b = 0;
  int64_t key_stride_b = 0;
  if (input_dim == 4) {
    batch_size = query.size(0);
    seq_len = query.size(1);
    query_stride_b = query.stride(0);
    key_stride_b = key.stride(0);
    query_stride_s = query.stride(1);
    key_stride_s = key.stride(1);
    CHECK_EQ(batch_size, key.size(0));
    CHECK_EQ(seq_len, key.size(1));
    CHECK_EQ(key.size(0) * key.size(1), num_tokens);
    CHECK_EQ(query.size(0) * query.size(1), num_tokens);
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input_dtype, "rotary_embedding_cpu", [&] {
    if (input_dim == 2 || input_dim == 4) {
      if (is_neox) {
        rotary_embedding_neox_4D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            query_stride_b,
            query_stride_s,
            query_stride_h,
            key_stride_b,
            key_stride_s,
            key_stride_h,
            num_heads,
            num_kv_heads,
            head_size,
            batch_size,
            seq_len);
      } else {
        rotary_embedding_4D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            query_stride_b,
            query_stride_s,
            query_stride_h,
            key_stride_b,
            key_stride_s,
            key_stride_h,
            num_heads,
            num_kv_heads,
            head_size,
            batch_size,
            seq_len);
      }
      query_out = query;
      key_out = key;

    } else {
      TORCH_CHECK(
          is_neox == false, " Query/Key with 3D [num_tokens, num_heads, head_size] does not support neox rope yet");
      rotary_embedding_3D_kernel_impl<scalar_t>(
          query_out.data_ptr<scalar_t>(),
          key_out.data_ptr<scalar_t>(),
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          num_tokens,
          num_heads,
          num_kv_heads,
          head_size,
          rotary_dim,
          query_stride_s,
          query_out_stride_s,
          key_out_stride_s,
          key_stride_s,
          query_stride_h,
          query_out_stride_h);
    }
  });
  return std::make_tuple(query_out, key_out);
}
