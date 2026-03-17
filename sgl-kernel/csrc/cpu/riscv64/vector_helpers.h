#ifndef SGL_KERNEL_RVV_VECTOR_HELPERS_H
#define SGL_KERNEL_RVV_VECTOR_HELPERS_H

#if defined(CPU_CAPABILITY_RVV)

#include <ATen/core/Tensor.h>
#include <riscv_vector.h>

#include <algorithm>
#include <limits>

#include "vector_math.h"

// Type trait to detect INT8-quantized KV types (used by decode and extend kernels).
template <typename T>
struct is_quantized : std::false_type {};
template <>
struct is_quantized<int8_t> : std::true_type {};

namespace rvv_constants {
inline constexpr int64_t BLOCK_N = 64;
// MAX_VL_ELEMENTS_M{4,8}: maximum element count per vsetvlmax call for each LMUL.
// Computed from __riscv_v_fixed_vlen (set by -mrvv-vector-bits=N in CMakeLists).
// Example: VLEN=256 → M4=32, M8=64; VLEN=512 → M4=64, M8=128.
inline constexpr size_t MAX_VL_ELEMENTS_M4 = __riscv_v_fixed_vlen / 8;
inline constexpr size_t MAX_VL_ELEMENTS_M8 = __riscv_v_fixed_vlen / 4;
}  // namespace rvv_constants

// Returns VLENB (vector register length in bytes) at runtime via vsetvlmax.
// e8m1: LMUL=1, element=8-bit → vl == VLEN/8 == vlenb.
inline int64_t rvv_get_vlenb() {
  return static_cast<int64_t>(__riscv_vsetvlmax_e8m1());
}

// Fixed-width vector type: width follows __riscv_v_fixed_vlen (set at compile time via -mrvv-vector-bits=N).
// Enables stack arrays of vector type (e.g., vf32m1_t arr[N]) in kernels that need them.
typedef vfloat32m1_t vf32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

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

// Dispatch macro for packed types
#define CPU_DISPATCH_PACKED_TYPES_RVV(TYPE, ...)                 \
  [&] {                                                          \
    switch (TYPE) {                                              \
      case at::ScalarType::Float: {                              \
        using packed_t = float;                                  \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::BFloat16: {                           \
        using packed_t = at::BFloat16;                           \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Half: {                               \
        using packed_t = at::Half;                               \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Char: {                               \
        using packed_t = int8_t;                                 \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Float8_e4m3fn: {                      \
        using packed_t = at::Float8_e4m3fn;                      \
        return __VA_ARGS__();                                    \
      }                                                          \
      default:                                                   \
        TORCH_CHECK(false, "Unsupported floating data type.\n"); \
    }                                                            \
  }()

// Type Conversion Helpers

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

// Load / Store Helpers

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
inline void store_from_float_m2(scalar_t* ptr, vfloat32m2_t v, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vse32_v_f32m2(ptr, v, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
    vfloat16m1_t v_f16 = __riscv_vfncvt_f_f_w_f16m1(v, vl);
    __riscv_vse16_v_f16m1(reinterpret_cast<_Float16*>(ptr), v_f16, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint32m2_t v_u32 = __riscv_vreinterpret_v_f32m2_u32m2(v);
    vuint16m1_t v_bf16 = __riscv_vnsrl_wx_u16m1(v_u32, 16, vl);
    __riscv_vse16_v_u16m1(reinterpret_cast<uint16_t*>(ptr), v_bf16, vl);
  } else {
    __riscv_vse32_v_f32m2(scratch, v, vl);
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

// Memory Operations (Copy, Fill, Transpose)

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, float val, int64_t size) {
  size_t vl = 0;  // Initialized to silence -Wuninitialized; always set before use in loop body.

  if constexpr (std::is_same_v<scalar_t, float>) {
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e32m4(size - d);
      vfloat32m4_t v_val = __riscv_vfmv_v_f_f32m4(val, vl);
      __riscv_vse32_v_f32m4(out + d, v_val, vl);
    }
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
    // FP16: use hardware narrowing convert
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e16m2(size - d);
      vfloat32m4_t v_f32 = __riscv_vfmv_v_f_f32m4(val, vl);
      vfloat16m2_t v_f16 = __riscv_vfncvt_f_f_w_f16m2(v_f32, vl);
      __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(out + d), v_f16, vl);
    }
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    // BF16: use bit-shift method (BF16 is upper 16 bits of FP32)
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e16m2(size - d);
      vfloat32m4_t v_f32 = __riscv_vfmv_v_f_f32m4(val, vl);
      vuint16m2_t v_bf16 = f32m4_to_bf16(v_f32, vl);
      __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(out + d), v_bf16, vl);
    }
  } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
    // int32: used by MoE for sorted_ids, expert_ids, total_cnts
    int32_t ival = static_cast<int32_t>(val);
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e32m4(size - d);
      vint32m4_t v_val = __riscv_vmv_v_x_i32m4(ival, vl);
      __riscv_vse32_v_i32m4(out + d, v_val, vl);
    }
  }
}

// Copy Stub Functions - Multiple Overloads

// 1. Same-type copy: scalar_t -> scalar_t
template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ src, int64_t size) {
  size_t vl = 0;  // Initialized to silence -Wuninitialized; always set before use in loop body.
  if constexpr (std::is_same_v<scalar_t, float>) {
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e32m4(size - d);
      vfloat32m4_t v_data = __riscv_vle32_v_f32m4(src + d, vl);
      __riscv_vse32_v_f32m4(out + d, v_data, vl);
    }
  } else {
    // FP16/BF16
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e16m4(size - d);
      vuint16m4_t v_data = __riscv_vle16_v_u16m4(reinterpret_cast<const uint16_t*>(src + d), vl);
      __riscv_vse16_v_u16m4(reinterpret_cast<uint16_t*>(out + d), v_data, vl);
    }
  }
}

// 2. Type conversion with scale: (float * scale) -> scalar_t
template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ acc, float s, int64_t size) {
  size_t vl;
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];

  for (int64_t d = 0; d < size; d += vl) {
    vl = __riscv_vsetvl_e32m4(size - d);
    vfloat32m4_t v_acc = __riscv_vle32_v_f32m4(acc + d, vl);
    vfloat32m4_t v_scaled = __riscv_vfmul_vf_f32m4(v_acc, s, vl);
    store_from_float_m4(out + d, v_scaled, vl, scratch);
  }
}

// 3. float -> scalar_t (no scale) — disabled when scalar_t=float to avoid ambiguity with overload 1
template <typename scalar_t, std::enable_if_t<!std::is_same_v<scalar_t, float>, int> = 0>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  size_t vl_max = __riscv_vsetvlmax_e32m4();
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];

  int64_t d = 0;
  // Process in chunks of vl_max
  for (; d + static_cast<int64_t>(vl_max) <= size; d += vl_max) {
    vfloat32m4_t v_data = __riscv_vle32_v_f32m4(input + d, vl_max);
    store_from_float_m4(out + d, v_data, vl_max, scratch);
  }

  // Handle remaining elements
  if (d < size) {
    size_t tail_vl = __riscv_vsetvl_e32m4(size - d);
    vfloat32m4_t v_data = __riscv_vle32_v_f32m4(input + d, tail_vl);
    store_from_float_m4(out + d, v_data, tail_vl, scratch);
  }
}

// 4. scalar_t -> float — disabled when scalar_t=float to avoid ambiguity with overload 1
template <typename scalar_t, std::enable_if_t<!std::is_same_v<scalar_t, float>, int> = 0>
inline void copy_stub(float* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  size_t vl_max = __riscv_vsetvlmax_e32m4();
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];

  int64_t d = 0;
  // Process in chunks of vl_max
  for (; d + static_cast<int64_t>(vl_max) <= size; d += vl_max) {
    vfloat32m4_t v_f32 = load_as_float_m4(input + d, vl_max, scratch);
    __riscv_vse32_v_f32m4(out + d, v_f32, vl_max);
  }

  // Handle remaining elements
  if (d < size) {
    size_t tail_vl = __riscv_vsetvl_e32m4(size - d);
    vfloat32m4_t v_f32 = load_as_float_m4(input + d, tail_vl, scratch);
    __riscv_vse32_v_f32m4(out + d, v_f32, tail_vl);
  }
}

// INT8 specialization for same-type copy
inline void copy_stub_int8(int8_t* __restrict__ out, const int8_t* __restrict__ src, int64_t size) {
  size_t vl;
  for (int64_t d = 0; d < size; d += vl) {
    vl = __riscv_vsetvl_e8m4(size - d);
    vint8m4_t v_data = __riscv_vle8_v_i8m4(src + d, vl);
    __riscv_vse8_v_i8m4(out + d, v_data, vl);
  }
}

template <>
inline void copy_stub<int8_t>(int8_t* __restrict__ out, const int8_t* __restrict__ src, int64_t size) {
  copy_stub_int8(out, src, size);
}

// Scalar sigmoid and multiply with vector
template <typename scalar_t, bool has_bias>
inline void scalar_sigmoid_and_mul(
    scalar_t* __restrict__ out,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    const scalar_t* __restrict__ mul,
    int SIZE) {
  // Scalar sigmoid: compute sigmoid(input[0] + bias[0])
  float x_val;
  if constexpr (has_bias) {
    assert(bias != nullptr);
    x_val = input[0] + bias[0];
  } else {
    x_val = input[0];
  }

  // Sigmoid: 1 / (1 + exp(-x))
  float sigmoid_val = 1.0f / (1.0f + std::exp(-x_val));
  size_t vl_max = __riscv_vsetvlmax_e32m4();

  // Vector multiply: mul * sigmoid
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];

  for (int d = 0; d < SIZE; d += vl_max) {
    size_t vl = (d + static_cast<int64_t>(vl_max) <= SIZE) ? vl_max : __riscv_vsetvl_e32m4(SIZE - d);

    // Load mul as float
    vfloat32m4_t v_mul = load_as_float_m4(mul + d, vl, scratch);

    // Multiply by sigmoid
    vfloat32m4_t v_result = __riscv_vfmul_vf_f32m4(v_mul, sigmoid_val, vl);

    // Store back as scalar_t
    store_from_float_m4(out + d, v_result, vl, scratch);
  }
}

// Quantization Operations

// Symmetric Quantization (Scale Provided)
template <typename scalar_t>
inline void
quantize_row_int8_symmetric(int8_t* __restrict__ Aq, const scalar_t* __restrict__ A, int64_t K, float scale) {
  // Guard against zero or negative scale to prevent NaN/Inf
  const float safe_scale = (scale > 1e-9f) ? scale : 1e-9f;
  const float inv_scale = 1.0f / safe_scale;
  size_t vl;
  alignas(64) float scratch[MAX_HEAD_SIZE];

  for (int64_t k = 0; k < K; k += vl) {
    vl = __riscv_vsetvl_e32m4(K - k);

    // Load as FP32
    vfloat32m4_t v_val = load_as_float_m4(A + k, vl, scratch);

    // Scale
    vfloat32m4_t v_scaled = __riscv_vfmul_vf_f32m4(v_val, inv_scale, vl);

    // Convert to Int32 (Round-to-nearest-even)
    vint32m4_t v_i32 = __riscv_vfcvt_x_f_v_i32m4(v_scaled, vl);

    // Narrowing Saturation: Int32 -> Int16 -> Int8
    vint16m2_t v_i16 = __riscv_vnclip_wx_i16m2(v_i32, 0, __RISCV_VXRM_RNU, vl);
    vint8m1_t v_i8 = __riscv_vnclip_wx_i8m1(v_i16, 0, __RISCV_VXRM_RNU, vl);

    __riscv_vse8_v_i8m1(Aq + k, v_i8, vl);
  }
}

// Dynamic Quantization (Compute Scale -> Symmetric Quantize)
template <typename scalar_t>
inline void quantize_row_int8_symmetric_auto(
    int8_t* __restrict__ Aq, float& scale_out, const scalar_t* __restrict__ A, int64_t K, float eps = 1e-7) {
  float max_val = 0.f;
  size_t vl;
  alignas(64) float scratch[MAX_HEAD_SIZE];

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
  quantize_row_int8_symmetric(Aq, A, K, scale);

  scale_out = scale;
}

template <typename scalar_t>
inline void quantize_and_copy(
    int8_t* __restrict__ dst, const scalar_t* __restrict__ src, int64_t size, float scale, float* scale_out = nullptr) {
  if (scale != 1.0f && scale > 0.0f) {
    quantize_row_int8_symmetric<scalar_t>(dst, src, size, scale);
    if (scale_out) *scale_out = scale;
  } else {
    float computed_scale;
    quantize_row_int8_symmetric_auto<scalar_t>(dst, computed_scale, src, size);
    if (scale_out) *scale_out = computed_scale;
  }
}

struct AlignedArena {
  char* ptr;

  explicit AlignedArena(void* base, size_t align = 64) {
    auto addr = reinterpret_cast<uintptr_t>(base);
    ptr = reinterpret_cast<char*>((addr + align - 1) & ~(align - 1));
  }

  template <typename T>
  T* alloc(size_t count, size_t align = 64) {
    auto addr = reinterpret_cast<uintptr_t>(ptr);
    ptr = reinterpret_cast<char*>((addr + align - 1) & ~(align - 1));
    T* result = reinterpret_cast<T*>(ptr);
    ptr += count * sizeof(T);
    return result;
  }
};

#endif  // CPU_CAPABILITY_RVV

#endif  // SGL_KERNEL_RVV_VECTOR_HELPERS_H
