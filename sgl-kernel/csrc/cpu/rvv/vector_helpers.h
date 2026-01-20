/*
 * 1. VLEN Utilities: Querying hardware vector length.
 * 2. Type Conversion: Converting between BF16/FP16/INT8 and FP32.
 * 3. Load/Store Helpers: Unified interfaces for loading/storing vector registers,
 *    handling necessary type conversions transparently.
 * 4. Memory Operations: Basic memory manipulation like copy and transpose.
 */

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

#ifndef MAX_HEAD_SIZE
#define MAX_HEAD_SIZE 256
#endif

#define AT_DISPATCH_RVV_TYPES(TYPE, NAME, ...)                         \
  [&] {                                                                \
    at::ScalarType _st = TYPE;                                         \
    switch (_st) {                                                     \
      case at::ScalarType::Float: {                                    \
        using scalar_t = float;                                        \
        return __VA_ARGS__();                                          \
      }                                                                \
      case at::ScalarType::Half: {                                     \
        using scalar_t = at::Half;                                     \
        return __VA_ARGS__();                                          \
      }                                                                \
      case at::ScalarType::BFloat16: {                                 \
        using scalar_t = at::BFloat16;                                 \
        return __VA_ARGS__();                                          \
      }                                                                \
      default:                                                         \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'"); \
    }                                                                  \
  }()

// =============================================================================
// VLEN Utilities
// =============================================================================

inline size_t get_vlenb_internal() {
  size_t vlenb;
  asm volatile("csrr %0, vlenb" : "=r"(vlenb));
  // Fallback if CSR read fails
  if (vlenb == 0) {
    vlenb = __riscv_vsetvlmax_e8m1();
  }
  return vlenb;
}

inline int64_t get_rvv_vlenb() {
  return static_cast<int64_t>(get_vlenb_internal());
}

inline int64_t get_rvv_vlen() {
  return get_rvv_vlenb() * 8;
}

inline bool check_vlen_alignment(int64_t size_bytes) {
  return (size_bytes % get_rvv_vlenb()) == 0;
}

// =============================================================================
// Type Conversion Helpers (BF16 Only - Complex logic kept)
// =============================================================================

// --- BF16 to FP32 ---
// Shift-widening: BF16 and FP32 share exponent field, left-shift by 16 bits

inline vfloat32m1_t bf16_to_f32m1(const uint16_t* ptr, size_t vl) {
  vuint16mf2_t v_bf16 = __riscv_vle16_v_u16mf2(ptr, vl);
  vuint32m1_t v_u32 = __riscv_vzext_vf2_u32m1(v_bf16, vl);
  v_u32 = __riscv_vsll_vx_u32m1(v_u32, 16, vl);
  return __riscv_vreinterpret_v_u32m1_f32m1(v_u32);
}

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

inline vfloat32m4_t int8_to_f32m4(const int8_t* ptr, size_t vl) {
  vint8m1_t v_i8 = __riscv_vle8_v_i8m1(ptr, vl);
  vint16m2_t v_i16 = __riscv_vsext_vf2_i16m2(v_i8, vl);
  vint32m4_t v_i32 = __riscv_vsext_vf2_i32m4(v_i16, vl);
  return __riscv_vfcvt_f_x_v_f32m4(v_i32, vl);
}

// --- FP32 to BF16 ---
// Extract upper 16 bits (sign + exp + upper 7 mantissa)

inline vfloat16m2_t f32m4_to_f16(vfloat32m4_t v, size_t vl) {
  return __riscv_vfncvt_f_f_w_f16m2(v, vl);
}

inline vuint16m2_t f32m4_to_bf16(vfloat32m4_t v_f32, size_t vl) {
  vuint32m4_t v_u32 = __riscv_vreinterpret_v_f32m4_u32m4(v_f32);
  return __riscv_vnsrl_wx_u16m2(v_u32, 16, vl);
}

inline vuint16m4_t f32m8_to_bf16(vfloat32m8_t v_f32, size_t vl) {
  vuint32m8_t v_u32 = __riscv_vreinterpret_v_f32m8_u32m8(v_f32);
  return __riscv_vnsrl_wx_u16m4(v_u32, 16, vl);
}

// =============================================================================
// Load / Store Helpers
// =============================================================================

template <typename scalar_t>
inline vfloat32m1_t load_as_float_m1(const scalar_t* ptr, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return __riscv_vle32_v_f32m1(ptr, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh) || defined(__riscv_v)
    vfloat16mf2_t v_f16 = __riscv_vle16_v_f16mf2(reinterpret_cast<const _Float16*>(ptr), vl);
    return __riscv_vfwcvt_f_f_v_f32m1(v_f16, vl);
#else
    for (size_t i = 0; i < vl; ++i)
      scratch[i] = static_cast<float>(ptr[i]);
    return __riscv_vle32_v_f32m1(scratch, vl);
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    return bf16_to_f32m1(reinterpret_cast<const uint16_t*>(ptr), vl);
  } else {
    for (size_t i = 0; i < vl; ++i)
      scratch[i] = static_cast<float>(ptr[i]);
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
  } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
    return int8_to_f32m4(reinterpret_cast<const int8_t*>(ptr), vl);
  } else {
    for (size_t i = 0; i < vl; ++i)
      scratch[i] = static_cast<float>(ptr[i]);
    return __riscv_vle32_v_f32m4(scratch, vl);
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
    // Manual strided load + convert for BF16
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
    for (size_t i = 0; i < vl; ++i)
      ptr[i] = static_cast<scalar_t>(scratch[i]);
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint32m1_t v_u32 = __riscv_vreinterpret_v_f32m1_u32m1(v);
    vuint16mf2_t v_bf16 = __riscv_vnsrl_wx_u16mf2(v_u32, 16, vl);
    __riscv_vse16_v_u16mf2(reinterpret_cast<uint16_t*>(ptr), v_bf16, vl);
  } else {
    __riscv_vse32_v_f32m1(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i)
      ptr[i] = static_cast<scalar_t>(scratch[i]);
  }
}

template <typename scalar_t>
inline void store_from_float_m4(scalar_t* ptr, vfloat32m4_t v, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vse32_v_f32m4(ptr, v, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
    vfloat16m2_t v_f16 = __riscv_vfncvt_f_f_w_f16m2(v, vl);
    __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(ptr), v_f16, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint16m2_t v_bf16 = f32m4_to_bf16(v, vl);
    __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(ptr), v_bf16, vl);
  } else {
    __riscv_vse32_v_f32m4(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i)
      ptr[i] = static_cast<scalar_t>(scratch[i]);
  }
}

template <typename scalar_t>
inline void store_from_float_m8(scalar_t* ptr, vfloat32m8_t v, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vse32_v_f32m8(ptr, v, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
    vfloat16m4_t v_f16 = __riscv_vfncvt_f_f_w_f16m4(v, vl);
    __riscv_vse16_v_f16m4(reinterpret_cast<_Float16*>(ptr), v_f16, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint16m4_t v_bf16 = f32m8_to_bf16(v, vl);
    __riscv_vse16_v_u16m4(reinterpret_cast<uint16_t*>(ptr), v_bf16, vl);
  } else {
    __riscv_vse32_v_f32m8(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i)
      ptr[i] = static_cast<scalar_t>(scratch[i]);
  }
}

// =============================================================================
// Memory Operations (Copy, Transpose)
// =============================================================================

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
  for (; d < size; ++d)
    out[d] = src[d];
}

// Byte-wise specialized copy (Handles both int8_t and uint8_t efficiently)
template <typename T>
inline void copy_stub_rvv_byte(T* __restrict__ out, const T* __restrict__ src, int64_t size) {
  size_t vl;
  for (int64_t d = 0; d < size; d += vl) {
    vl = __riscv_vsetvl_e8m1(size - d);
    vint8m1_t v_data = __riscv_vle8_v_i8m1(reinterpret_cast<const int8_t*>(src + d), vl);
    __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(out + d), v_data, vl);
  }
}

template <>
inline void copy_stub_rvv<int8_t>(int8_t* __restrict__ out, const int8_t* __restrict__ src, int64_t size) {
  copy_stub_rvv_byte(out, src, size);
}

template <>
inline void copy_stub_rvv<uint8_t>(uint8_t* __restrict__ out, const uint8_t* __restrict__ src, int64_t size) {
  copy_stub_rvv_byte(out, src, size);
}

template <typename scalar_t>
inline void transpose_block_rvv(const scalar_t* src, scalar_t* dst, int rows, int cols, int src_stride) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      dst[c * rows + r] = src[r * src_stride + c];
    }
  }
}

// =============================================================================
// Quantization Operations
// =============================================================================

// Static Quantization (Scale Provided)
template <typename scalar_t>
inline void
quantize_row_int8_symmetric_rvv(int8_t* __restrict__ Aq, const scalar_t* __restrict__ A, int64_t K, float scale) {
  const float inv_scale = 1.0f / scale;
  size_t vl;
  float scratch[128];  // Max VL for m4 @ 256b is 32, so 128 is safe

  for (int64_t k = 0; k < K; k += vl) {
    vl = __riscv_vsetvl_e32m4(K - k);

    // Load as FP32
    vfloat32m4_t v_val = load_as_float_m4(A + k, vl, scratch);

    // Scale
    vfloat32m4_t v_scaled = __riscv_vfmul_vf_f32m4(v_val, inv_scale, vl);

    // Convert to Int32 (Round-to-nearest-even typically)
    vint32m4_t v_i32 = __riscv_vfcvt_x_f_v_i32m4(v_scaled, vl);

    // Narrowing Saturation: Int32 -> Int16 -> Int8
    vint16m2_t v_i16 = __riscv_vnclip_wx_i16m2(v_i32, 0, __RISCV_VXRM_RNU, vl);
    vint8m1_t v_i8 = __riscv_vnclip_wx_i8m1(v_i16, 0, __RISCV_VXRM_RNU, vl);

    __riscv_vse8_v_i8m1(Aq + k, v_i8, vl);
  }
}

// Asymmetric Quantization (for GEMM)
// Maps [-max, max] to [1, 255] (0 maps to 128)
// Matches logic: Aq = round(x / scale) + 128
template <typename scalar_t>
inline void quantize_row_int8_asymmetric_rvv(
    int8_t* __restrict__ Aq, float& scale_out, const scalar_t* __restrict__ A, int64_t K, float eps = 1e-7) {
  float max_val = 0.f;
  size_t vl;
  float scratch[128];

  // Pass 1: Find Max Absolute Value
  vfloat32m4_t v_max_acc = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());

  for (int64_t k = 0; k < K; k += vl) {
    vl = __riscv_vsetvl_e32m4(K - k);
    vfloat32m4_t v_val = load_as_float_m4(A + k, vl, scratch);
    vfloat32m4_t v_abs = __riscv_vfsgnjx_vv_f32m4(v_val, v_val, vl);  // Abs
    v_max_acc = __riscv_vfmax_vv_f32m4(v_max_acc, v_abs, vl);
  }

  vfloat32m1_t v_max_scalar =
      __riscv_vfredmax_vs_f32m4_f32m1(v_max_acc, __riscv_vfmv_s_f_f32m1(0.0f, 1), __riscv_vsetvlmax_e32m4());
  max_val = __riscv_vfmv_f_s_f32m1_f32(v_max_scalar);

  max_val = std::max(max_val, eps);
  const float scale = max_val / 127.0f;
  const float inv_scale = 127.0f / max_val;

  scale_out = scale;

  // Pass 2: Quantize
  // val * inv_scale -> int -> +128

  for (int64_t k = 0; k < K; k += vl) {
    vl = __riscv_vsetvl_e32m4(K - k);
    vfloat32m4_t v_val = load_as_float_m4(A + k, vl, scratch);
    vfloat32m4_t v_scaled = __riscv_vfmul_vf_f32m4(v_val, inv_scale, vl);

    // Round to nearest integer
    vint32m4_t v_i32 = __riscv_vfcvt_x_f_v_i32m4(v_scaled, vl);

    // Add 128
    vint32m4_t v_off = __riscv_vadd_vx_i32m4(v_i32, 128, vl);

    // Saturation to u8 (0-255)
    // Since we used signed int32, we clip to [0, 255] then cast.
    // However, v_i32 is approx [-127, 127]. +128 -> [1, 255].
    // vnclip_wx_u8 would be nice but we have i32.
    // Narrow i32 -> i16 -> u8

    vint16m2_t v_i16 = __riscv_vnclip_wx_i16m2(v_off, 0, __RISCV_VXRM_RNU, vl);
    vuint16m2_t v_u16 = __riscv_vreinterpret_v_i16m2_u16m2(v_i16);
    vuint8m1_t v_u8 = __riscv_vnclipu_wx_u8m1(v_u16, 0, __RISCV_VXRM_RNU, vl);

    __riscv_vse8_v_u8m1(reinterpret_cast<uint8_t*>(Aq + k), v_u8, vl);
  }
}

// Dynamic Quantization (Compute Scale -> Static Quantize)
template <typename scalar_t>
inline void quantize_row_int8_symmetric_auto_rvv(
    int8_t* __restrict__ Aq, float& scale_out, const scalar_t* __restrict__ A, int64_t K, float eps = 1e-7) {
  float max_val = 0.f;
  size_t vl;
  float scratch[128];

  // Pass 1: Find Max Absolute Value
  vfloat32m4_t v_max_acc = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());

  for (int64_t k = 0; k < K; k += vl) {
    vl = __riscv_vsetvl_e32m4(K - k);
    vfloat32m4_t v_val = load_as_float_m4(A + k, vl, scratch);
    vfloat32m4_t v_abs = __riscv_vfsgnjx_vv_f32m4(v_val, v_val, vl);  // Abs
    v_max_acc = __riscv_vfmax_vv_f32m4(v_max_acc, v_abs, vl);
  }

  vfloat32m1_t v_max_scalar =
      __riscv_vfredmax_vs_f32m4_f32m1(v_max_acc, __riscv_vfmv_s_f_f32m1(0.0f, 1), __riscv_vsetvlmax_e32m4());
  max_val = __riscv_vfmv_f_s_f32m1_f32(v_max_scalar);

  max_val = std::max(max_val, eps);
  const float scale = max_val / 127.0f;

  // Pass 2: Quantize
  quantize_row_int8_symmetric_rvv(Aq, A, K, scale);

  scale_out = scale;
}

template <typename scalar_t>
inline void
quantize_and_copy_rvv(int8_t* __restrict__ dst, const scalar_t* __restrict__ src, int64_t size, float scale) {
  if (scale != 1.0f && scale > 0.0f) {
    quantize_row_int8_symmetric_rvv<scalar_t>(dst, src, size, scale);
  } else {
    float computed_scale;
    quantize_row_int8_symmetric_auto_rvv<scalar_t>(dst, computed_scale, src, size);
  }
}

#endif  // CPU_CAPABILITY_RVV

#endif  // SGL_KERNEL_RVV_VECTOR_HELPERS_H
