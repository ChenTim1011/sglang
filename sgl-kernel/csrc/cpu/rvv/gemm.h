#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>

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
extern template void
pack_weight_rvv<at::Float8_e4m3fn>(at::Float8_e4m3fn* packed_w, const at::Float8_e4m3fn* orig_w, int64_t N, int64_t K);

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
