/*
 * 1. Tuning Constants: Block sizes (BLOCK_N, BLOCK_M) for different kernels.
 * 2. Tuning Logic: Runtime selection of optimal tiling parameters.
 * 3. Common Definitions: Shared constants between different GEMM implementations.
 */

#pragma once

#include <ATen/core/Tensor.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>

// SpacemiT K1: 256-bit VLEN configuration
namespace rvv_constants {
constexpr float ZERO_THRESHOLD = 1e-8f;
constexpr int64_t BLOCK_N = 64;
constexpr int64_t BLOCK_M_RVV = 4;
constexpr int64_t BLOCK_K_RVV = 64;
constexpr int64_t BLOCK_M_INT8 = 4;
constexpr int64_t TILE_M = 4;
constexpr int64_t TILE_N = 4;
constexpr int64_t TILE_K = 128;
constexpr int64_t INT8_N_CHUNK_M1 = 32;
constexpr int64_t INT8_N_CHUNK_GENERAL = 16;
constexpr size_t MAX_VL_ELEMENTS = 8;
constexpr int PREFETCH_DISTANCE = 2;
constexpr int PREFETCH_LOCALITY = 1;
constexpr int DEFAULT_BLOCK_M = 16;
constexpr int DEFAULT_BLOCK_N = 32;
}  // namespace rvv_constants

namespace {
inline int get_env_block_m() {
  static const int val = [] {
    const char* env = std::getenv("SGL_TUNE_BLOCK_M");
    return (env) ? std::atoi(env) : 0;
  }();
  return val;
}

inline int get_env_block_n() {
  static const int val = [] {
    const char* env = std::getenv("SGL_TUNE_BLOCK_N");
    return (env) ? std::atoi(env) : 0;
  }();
  return val;
}
}  // namespace

inline int get_optimal_block_m(int head_size, int head_size_v) {
  int block_m = get_env_block_m();
  if (block_m > 0) return block_m;

  if (head_size <= 64 && head_size_v <= 64) return 32;
  if (head_size <= 128 && head_size_v <= 128) return 24;
  return rvv_constants::DEFAULT_BLOCK_M;
}

template <typename scalar_t>
inline int get_optimal_block_m(int head_size, int head_size_v) {
  return get_optimal_block_m(head_size, head_size_v);
}

inline int get_optimal_block_n(int head_size, int head_size_v) {
  int block_n = get_env_block_n();
  if (block_n > 0) return block_n;

  if (head_size <= 64 && head_size_v <= 64) return 64;
  if (head_size <= 128 && head_size_v <= 128) return 48;
  return rvv_constants::DEFAULT_BLOCK_N;
}

template <typename scalar_t>
inline int get_optimal_block_n(int head_size, int head_size_v) {
  return get_optimal_block_n(head_size, head_size_v);
}

template <typename scalar_t>
void weight_packed_linear_kernel_impl_rvv(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const scalar_t* __restrict__ mat2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed = false);

template <typename scalar_t>
void pack_weight_rvv(scalar_t* packed_w, const scalar_t* orig_w, int64_t N, int64_t K);

extern template void pack_weight_rvv<float>(float* packed_w, const float* orig_w, int64_t N, int64_t K);
extern template void
pack_weight_rvv<at::BFloat16>(at::BFloat16* packed_w, const at::BFloat16* orig_w, int64_t N, int64_t K);
extern template void pack_weight_rvv<at::Half>(at::Half* packed_w, const at::Half* orig_w, int64_t N, int64_t K);
extern template void pack_weight_rvv<int8_t>(int8_t* packed_w, const int8_t* orig_w, int64_t N, int64_t K);

extern template void weight_packed_linear_kernel_impl_rvv<float>(
    float* out,
    const float* mat1,
    const float* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed);

extern template void weight_packed_linear_kernel_impl_rvv<at::BFloat16>(
    at::BFloat16* out,
    const at::BFloat16* mat1,
    const at::BFloat16* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed);

extern template void weight_packed_linear_kernel_impl_rvv<at::Half>(
    at::Half* out,
    const at::Half* mat1,
    const at::Half* mat2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    bool is_packed);

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
    bool is_packed);

extern template void int8_scaled_mm_kernel_rvv<float>(
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

extern template void int8_scaled_mm_kernel_rvv<at::Half>(
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

extern template void int8_scaled_mm_kernel_rvv<at::BFloat16>(
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
