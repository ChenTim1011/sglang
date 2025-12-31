// RISC-V Vector Extension (RVV) common helper functions

#ifndef SGL_KERNEL_RVV_VECTOR_HELPERS_H
#define SGL_KERNEL_RVV_VECTOR_HELPERS_H

#if defined(CPU_CAPABILITY_RVV)

#include <ATen/core/Tensor.h>
#include <riscv_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "vec.h"

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
// SpacemiT K1: 256-bit VLEN = 8 FP32 elements
constexpr size_t MAX_VL_ELEMENTS = 8;
constexpr int PREFETCH_DISTANCE = 2;
constexpr int PREFETCH_LOCALITY = 1;
}  // namespace rvv_constants

template <typename scalar_t>
inline void copy_stub_rvv(scalar_t* __restrict__ out, const scalar_t* __restrict__ src, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = bVec::size();
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    bVec out_bvec = bVec::loadu(src + d);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = src[d];
  }
}

template <>
inline void copy_stub_rvv<int8_t>(int8_t* __restrict__ out, const int8_t* __restrict__ src, int64_t size) {
  size_t vl;
  for (int64_t d = 0; d < size; d += vl) {
    vl = __riscv_vsetvl_e8m1(size - d);
    vint8m1_t v_data = __riscv_vle8_v_i8m1(src + d, vl);
    __riscv_vse8_v_i8m1(out + d, v_data, vl);
  }
}

template <>
inline void copy_stub_rvv<uint8_t>(uint8_t* __restrict__ out, const uint8_t* __restrict__ src, int64_t size) {
  size_t vl;
  for (int64_t d = 0; d < size; d += vl) {
    vl = __riscv_vsetvl_e8m1(size - d);
    vuint8m1_t v_data = __riscv_vle8_v_u8m1(src + d, vl);
    __riscv_vse8_v_u8m1(out + d, v_data, vl);
  }
}

// Symmetric INT8 quantization for KV cache: produces signed int8 [-128, 127]
// No +128 offset, compatible with vle8_v_i8m1 (signed load)
template <typename scalar_t>
inline void
quantize_row_int8_symmetric_rvv(int8_t* __restrict__ Aq, const scalar_t* __restrict__ A, int64_t K, float scale) {
  const float inv_scale = 1.0f / scale;
  for (int64_t k = 0; k < K; ++k) {
    const float val = static_cast<float>(A[k]) * inv_scale;
    int32_t quantized = (int32_t)(std::round(val));
    quantized = std::max(-128, std::min(127, quantized));
    Aq[k] = (int8_t)(quantized);
  }
}

template <typename scalar_t>
inline void transpose_block_rvv(const scalar_t* src, scalar_t* dst, int rows, int cols, int src_stride) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      dst[c * rows + r] = src[r * src_stride + c];
    }
  }
}

// BF16 to FP32: manual implementation (hardware lacks Zvfbfmin)
// Shift-widening: BF16 and FP32 share exponent field, left-shift by 16 bits
inline vfloat32m4_t bf16_to_f32m4(const uint16_t* ptr, size_t vl) {
  vuint16m2_t v_bf16 = __riscv_vle16_v_u16m2(ptr, vl);
  vuint32m4_t v_u32 = __riscv_vzext_vf2_u32m4(v_bf16, vl);
  v_u32 = __riscv_vsll_vx_u32m4(v_u32, 16, vl);
  return __riscv_vreinterpret_v_u32m4_f32m4(v_u32);
}

inline vfloat32m8_t bf16_to_f32m8(const uint16_t* ptr, size_t vl) {
  vuint16m4_t v_bf16 = __riscv_vle16_v_u16m4(ptr, vl);
  vuint32m8_t v_u32 = __riscv_vzext_vf2_u32m8(v_bf16, vl);
  v_u32 = __riscv_vsll_vx_u32m8(v_u32, 16, vl);
  return __riscv_vreinterpret_v_u32m8_f32m8(v_u32);
}

// FP32 to BF16: extract upper 16 bits (sign + exp + upper 7 mantissa)
inline vuint16m2_t f32m4_to_bf16(vfloat32m4_t v_f32, size_t vl) {
  vuint32m4_t v_u32 = __riscv_vreinterpret_v_f32m4_u32m4(v_f32);
  return __riscv_vnsrl_wx_u16m2(v_u32, 16, vl);
}

inline vuint16m4_t f32m8_to_bf16(vfloat32m8_t v_f32, size_t vl) {
  vuint32m8_t v_u32 = __riscv_vreinterpret_v_f32m8_u32m8(v_f32);
  return __riscv_vnsrl_wx_u16m4(v_u32, 16, vl);
}

inline vfloat16m2_t f32m4_to_f16(vfloat32m4_t v_f32, size_t vl) {
  return __riscv_vfncvt_f_f_w_f16m2(v_f32, vl);
}

inline vfloat16m4_t f32m8_to_f16(vfloat32m8_t v_f32, size_t vl) {
  return __riscv_vfncvt_f_f_w_f16m4(v_f32, vl);
}

inline vfloat32m4_t f16_to_f32m4(const _Float16* ptr, size_t vl) {
  vfloat16m2_t v_f16 = __riscv_vle16_v_f16m2(ptr, vl);
  return __riscv_vfwcvt_f_f_v_f32m4(v_f16, vl);
}

inline vfloat32m8_t f16_to_f32m8(const _Float16* ptr, size_t vl) {
  vfloat16m4_t v_f16 = __riscv_vle16_v_f16m4(ptr, vl);
  return __riscv_vfwcvt_f_f_v_f32m8(v_f16, vl);
}

inline float rvv_reduce_sum_f32(const float* data, int64_t len) {
  size_t vl = __riscv_vsetvl_e32m8(len);
  if (vl == static_cast<size_t>(len)) {
    vfloat32m8_t vdata = __riscv_vle32_v_f32m8(data, vl);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
    vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m8_f32m1(vdata, vzero, vl);
    return __riscv_vfmv_f_s_f32m1_f32(vsum);
  }
  float sum = 0.f;
  for (int64_t i = 0; i < len; ++i) {
    sum += data[i];
  }
  return sum;
}

inline float rvv_reduce_max_f32(const float* data, int64_t len) {
  if (len <= 0) {
    return -std::numeric_limits<float>::infinity();
  }
  size_t vl = __riscv_vsetvl_e32m8(len);
  if (vl == static_cast<size_t>(len)) {
    vfloat32m8_t vdata = __riscv_vle32_v_f32m8(data, vl);
    vfloat32m1_t vmin = __riscv_vfmv_s_f_f32m1(-std::numeric_limits<float>::infinity(), 1);
    vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m8_f32m1(vdata, vmin, vl);
    return __riscv_vfmv_f_s_f32m1_f32(vmax);
  }
  float max_val = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < len; ++i) {
    max_val = std::max(max_val, data[i]);
  }
  return max_val;
}

inline float reduce_sum_f32m8(vfloat32m8_t v_acc, size_t vl_max) {
  vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  vfloat32m1_t vred = __riscv_vfredusum_vs_f32m8_f32m1(v_acc, vzero, vl_max);
  return __riscv_vfmv_f_s_f32m1_f32(vred);
}

inline float reduce_sum_f32m4(vfloat32m4_t v_acc, size_t vl_max) {
  vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  vfloat32m1_t vred = __riscv_vfredusum_vs_f32m4_f32m1(v_acc, vzero, vl_max);
  return __riscv_vfmv_f_s_f32m1_f32(vred);
}

inline float reduce_sum_f32m1(vfloat32m1_t v_acc, size_t vl_max) {
  vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m1_t vred = __riscv_vfredusum_vs_f32m1_f32m1(v_acc, vzero, vl_max);
  return __riscv_vfmv_f_s_f32m1_f32(vred);
}

// INT8 to FP16: INT8 -> INT16 -> FP16
inline vfloat16m2_t int8_to_f16m2(const int8_t* ptr, size_t vl) {
  vint8m1_t v_i8 = __riscv_vle8_v_i8m1(ptr, vl);
  vint16m2_t v_i16 = __riscv_vsext_vf2_i16m2(v_i8, vl);
  return __riscv_vfcvt_f_x_v_f16m2(v_i16, vl);
}

// INT8 to FP32: INT8 -> INT16 -> INT32 -> FP32 (LMUL=2 -> LMUL=8)
inline vfloat32m8_t int8_to_f32m8(const int8_t* ptr, size_t vl) {
  vint8m2_t v_i8 = __riscv_vle8_v_i8m2(ptr, vl);
  vint16m4_t v_i16 = __riscv_vsext_vf2_i16m4(v_i8, vl);
  vint32m8_t v_i32 = __riscv_vsext_vf2_i32m8(v_i16, vl);
  return __riscv_vfcvt_f_x_v_f32m8(v_i32, vl);
}

// INT8 to FP32: INT8 -> INT16 -> INT32 -> FP32 (LMUL=1 -> LMUL=4)
inline vfloat32m4_t int8_to_f32m4(const int8_t* ptr, size_t vl) {
  vint8m1_t v_i8 = __riscv_vle8_v_i8m1(ptr, vl);
  vint16m2_t v_i16 = __riscv_vsext_vf2_i16m2(v_i8, vl);
  vint32m4_t v_i32 = __riscv_vsext_vf2_i32m4(v_i16, vl);
  return __riscv_vfcvt_f_x_v_f32m4(v_i32, vl);
}

inline _Float16 reduce_sum_f16m4(vfloat16m4_t v_acc, size_t vl_max) {
  vfloat16m1_t vzero = __riscv_vfmv_v_f_f16m1((_Float16)0.0f, 1);
  vfloat16m1_t vred = __riscv_vfredusum_vs_f16m4_f16m1(v_acc, vzero, vl_max);
  return __riscv_vfmv_f_s_f16m1_f16(vred);
}

template <typename scalar_t>
inline vfloat32m1_t load_as_float_m1(const scalar_t* ptr, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return __riscv_vle32_v_f32m1(ptr, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh) || defined(__riscv_v)
    vfloat16mf2_t v_f16 = __riscv_vle16_v_f16mf2(reinterpret_cast<const _Float16*>(ptr), vl);
    return __riscv_vfwcvt_f_f_v_f32m1(v_f16, vl);
#else
    for (size_t i = 0; i < vl; ++i) {
      scratch[i] = static_cast<float>(ptr[i]);
    }
    return __riscv_vle32_v_f32m1(scratch, vl);
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint16mf2_t v_bf16 = __riscv_vle16_v_u16mf2(reinterpret_cast<const uint16_t*>(ptr), vl);
    vuint32m1_t v_u32 = __riscv_vzext_vf2_u32m1(v_bf16, vl);
    v_u32 = __riscv_vsll_vx_u32m1(v_u32, 16, vl);
    return __riscv_vreinterpret_v_u32m1_f32m1(v_u32);
  } else {
    for (size_t i = 0; i < vl; ++i) {
      scratch[i] = static_cast<float>(ptr[i]);
    }
    return __riscv_vle32_v_f32m1(scratch, vl);
  }
}

template <typename scalar_t>
inline vfloat32m4_t load_as_float_m4(const scalar_t* ptr, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return __riscv_vle32_v_f32m4(ptr, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
    vfloat16m2_t v_f16 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(ptr), vl);
    return __riscv_vfwcvt_f_f_v_f32m4(v_f16, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    return bf16_to_f32m4(reinterpret_cast<const uint16_t*>(ptr), vl);
  } else {
    for (size_t i = 0; i < vl; ++i) {
      scratch[i] = static_cast<float>(ptr[i]);
    }
    return __riscv_vle32_v_f32m4(scratch, vl);
  }
}

template <typename scalar_t>
inline void store_from_float_m1(scalar_t* ptr, vfloat32m1_t v, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vse32_v_f32m1(ptr, v, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh) || defined(__riscv_v)
    vfloat16mf2_t v_f16 = __riscv_vfncvt_f_f_w_f16mf2(v, vl);
    __riscv_vse16_v_f16mf2(reinterpret_cast<_Float16*>(ptr), v_f16, vl);
#else
    __riscv_vse32_v_f32m1(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i) {
      ptr[i] = static_cast<scalar_t>(scratch[i]);
    }
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint32m1_t v_u32 = __riscv_vreinterpret_v_f32m1_u32m1(v);
    vuint16mf2_t v_bf16 = __riscv_vnsrl_wx_u16mf2(v_u32, 16, vl);
    __riscv_vse16_v_u16mf2(reinterpret_cast<uint16_t*>(ptr), v_bf16, vl);
  } else {
    __riscv_vse32_v_f32m1(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i) {
      ptr[i] = static_cast<scalar_t>(scratch[i]);
    }
  }
}

template <typename scalar_t>
inline void store_from_float_m4(scalar_t* ptr, vfloat32m4_t v, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vse32_v_f32m4(ptr, v, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
    vfloat16m2_t v_f16 = f32m4_to_f16(v, vl);
    __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(ptr), v_f16, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint16m2_t v_bf16 = f32m4_to_bf16(v, vl);
    __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(ptr), v_bf16, vl);
  } else {
    __riscv_vse32_v_f32m4(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i) {
      ptr[i] = static_cast<scalar_t>(scratch[i]);
    }
  }
}

template <typename scalar_t>
inline vfloat32m4_t load_strided_as_float_m4(const scalar_t* ptr, ptrdiff_t stride_byte, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return __riscv_vlse32_v_f32m4(ptr, stride_byte, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
    vfloat16m2_t v_f16 = __riscv_vlse16_v_f16m2(reinterpret_cast<const _Float16*>(ptr), stride_byte, vl);
    return __riscv_vfwcvt_f_f_v_f32m4(v_f16, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint16m2_t v_bf16 = __riscv_vlse16_v_u16m2(reinterpret_cast<const uint16_t*>(ptr), stride_byte, vl);
    vuint32m4_t v_u32 = __riscv_vzext_vf2_u32m4(v_bf16, vl);
    v_u32 = __riscv_vsll_vx_u32m4(v_u32, 16, vl);
    return __riscv_vreinterpret_v_u32m4_f32m4(v_u32);
  } else {
    for (size_t i = 0; i < vl; ++i) {
      const scalar_t* elem_ptr =
          reinterpret_cast<const scalar_t*>(reinterpret_cast<const char*>(ptr) + i * stride_byte);
      scratch[i] = static_cast<float>(*elem_ptr);
    }
    return __riscv_vle32_v_f32m4(scratch, vl);
  }
}

constexpr int DEFAULT_BLOCK_M = 16;
constexpr int DEFAULT_BLOCK_N = 32;
constexpr int MAX_HEAD_SIZE = 256;

// Supports SGL_TUNE_BLOCK_M environment variable override for tuning
template <typename scalar_t>
inline int get_optimal_block_m(int head_size, int head_size_v) {
  const char* env_block_m = std::getenv("SGL_TUNE_BLOCK_M");
  if (env_block_m != nullptr) {
    int block_m = std::atoi(env_block_m);
    if (block_m > 0) {
      return block_m;
    }
  }

  if (head_size <= 64 && head_size_v <= 64) {
    return 32;
  } else if (head_size <= 128 && head_size_v <= 128) {
    return 24;
  } else {
    return DEFAULT_BLOCK_M;
  }
}

// Supports SGL_TUNE_BLOCK_N environment variable override for tuning
template <typename scalar_t>
inline int get_optimal_block_n(int head_size, int head_size_v) {
  const char* env_block_n = std::getenv("SGL_TUNE_BLOCK_N");
  if (env_block_n != nullptr) {
    int block_n = std::atoi(env_block_n);
    if (block_n > 0) {
      return block_n;
    }
  }

  if (head_size <= 64 && head_size_v <= 64) {
    return 64;
  } else if (head_size <= 128 && head_size_v <= 128) {
    return 48;
  } else {
    return DEFAULT_BLOCK_N;
  }
}

#endif  // CPU_CAPABILITY_RVV

#endif  // SGL_KERNEL_RVV_VECTOR_HELPERS_H
