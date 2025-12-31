// Polynomial approximation for exp(x) = 2^(x*log2(e))
// Reference: veclibm (https://github.com/rivosinc/veclibm)

#ifndef SGL_KERNEL_RVV_VECTOR_MATH_H_
#define SGL_KERNEL_RVV_VECTOR_MATH_H_

#if defined(CPU_CAPABILITY_RVV)

#include <riscv_vector.h>

#include <cmath>
#include <limits>

namespace {

constexpr float RVV_EXP_C0 = 1.0f;
constexpr float RVV_EXP_C1 = 0.69314718056f;
constexpr float RVV_EXP_C2 = 0.24022650695f;
constexpr float RVV_EXP_C3 = 0.05550410866f;
constexpr float RVV_EXP_C4 = 0.00961812910f;
constexpr float RVV_EXP_C5 = 0.00133335581f;
constexpr float RVV_LOG2_E = 1.44269504089f;

inline vfloat32m4_t vfexp_f32m4(vfloat32m4_t vx, size_t vl) {
  vfloat32m4_t v_log2e = __riscv_vfmv_v_f_f32m4(RVV_LOG2_E, vl);
  vfloat32m4_t vz = __riscv_vfmul_vv_f32m4(vx, v_log2e, vl);
  vint32m4_t vn_int = __riscv_vfcvt_x_f_v_i32m4(vz, vl);
  vfloat32m4_t vn = __riscv_vfcvt_f_x_v_f32m4(vn_int, vl);
  vfloat32m4_t vf = __riscv_vfsub_vv_f32m4(vz, vn, vl);

  vfloat32m4_t v_c1 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C1, vl);
  vfloat32m4_t v_c2 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C2, vl);
  vfloat32m4_t v_c3 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C3, vl);
  vfloat32m4_t v_c4 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C4, vl);
  vfloat32m4_t v_c5 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C5, vl);
  vfloat32m4_t v_1 = __riscv_vfmv_v_f_f32m4(1.0f, vl);

  vfloat32m4_t poly = __riscv_vfmadd_vv_f32m4(v_c5, vf, v_c4, vl);
  poly = __riscv_vfmadd_vv_f32m4(poly, vf, v_c3, vl);
  poly = __riscv_vfmadd_vv_f32m4(poly, vf, v_c2, vl);
  poly = __riscv_vfmadd_vv_f32m4(poly, vf, v_c1, vl);
  poly = __riscv_vfmadd_vv_f32m4(poly, vf, v_1, vl);

  vint32m4_t v_127 = __riscv_vmv_v_x_i32m4(127, vl);
  vint32m4_t v_exp = __riscv_vadd_vv_i32m4(vn_int, v_127, vl);
  v_exp = __riscv_vsll_vx_i32m4(v_exp, 23, vl);
  vfloat32m4_t v_pow2n = __riscv_vreinterpret_v_i32m4_f32m4(v_exp);
  return __riscv_vfmul_vv_f32m4(poly, v_pow2n, vl);
}

inline vfloat32m8_t vfexp_f32m8(vfloat32m8_t vx, size_t vl) {
  vfloat32m8_t v_log2e = __riscv_vfmv_v_f_f32m8(RVV_LOG2_E, vl);
  vfloat32m8_t vz = __riscv_vfmul_vv_f32m8(vx, v_log2e, vl);
  vint32m8_t vn_int = __riscv_vfcvt_x_f_v_i32m8(vz, vl);
  vfloat32m8_t vn = __riscv_vfcvt_f_x_v_f32m8(vn_int, vl);
  vfloat32m8_t vf = __riscv_vfsub_vv_f32m8(vz, vn, vl);

  vfloat32m8_t v_c1 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C1, vl);
  vfloat32m8_t v_c2 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C2, vl);
  vfloat32m8_t v_c3 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C3, vl);
  vfloat32m8_t v_c4 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C4, vl);
  vfloat32m8_t v_c5 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C5, vl);
  vfloat32m8_t v_1 = __riscv_vfmv_v_f_f32m8(1.0f, vl);

  vfloat32m8_t poly = __riscv_vfmadd_vv_f32m8(v_c5, vf, v_c4, vl);
  poly = __riscv_vfmadd_vv_f32m8(poly, vf, v_c3, vl);
  poly = __riscv_vfmadd_vv_f32m8(poly, vf, v_c2, vl);
  poly = __riscv_vfmadd_vv_f32m8(poly, vf, v_c1, vl);
  poly = __riscv_vfmadd_vv_f32m8(poly, vf, v_1, vl);

  vint32m8_t v_127 = __riscv_vmv_v_x_i32m8(127, vl);
  vint32m8_t v_exp = __riscv_vadd_vv_i32m8(vn_int, v_127, vl);
  v_exp = __riscv_vsll_vx_i32m8(v_exp, 23, vl);
  vfloat32m8_t v_pow2n = __riscv_vreinterpret_v_i32m8_f32m8(v_exp);
  return __riscv_vfmul_vv_f32m8(poly, v_pow2n, vl);
}

inline vfloat32m4_t vftanh_f32m4(vfloat32m4_t vx, size_t vl) {
  vx = __riscv_vfmax_vf_f32m4(vx, -9.0f, vl);
  vx = __riscv_vfmin_vf_f32m4(vx, 9.0f, vl);
  vfloat32m4_t v2x = __riscv_vfmul_vf_f32m4(vx, 2.0f, vl);
  vfloat32m4_t vex = vfexp_f32m4(v2x, vl);
  vfloat32m4_t v_numerator = __riscv_vfsub_vf_f32m4(vex, 1.0f, vl);
  vfloat32m4_t v_denom = __riscv_vfadd_vf_f32m4(vex, 1.0f, vl);
  return __riscv_vfdiv_vv_f32m4(v_numerator, v_denom, vl);
}

inline vfloat32m8_t vftanh_f32m8(vfloat32m8_t vx, size_t vl) {
  vx = __riscv_vfmax_vf_f32m8(vx, -9.0f, vl);
  vx = __riscv_vfmin_vf_f32m8(vx, 9.0f, vl);

  vfloat32m8_t v2x = __riscv_vfmul_vf_f32m8(vx, 2.0f, vl);
  vfloat32m8_t vex = vfexp_f32m8(v2x, vl);
  vfloat32m8_t v_numerator = __riscv_vfsub_vf_f32m8(vex, 1.0f, vl);
  vfloat32m8_t v_denom = __riscv_vfadd_vf_f32m8(vex, 1.0f, vl);
  return __riscv_vfdiv_vv_f32m8(v_numerator, v_denom, vl);
}

inline float exp_and_sum_rvv(float* __restrict__ scores, int n_size, float m_i) {
  if (n_size <= 0) return 0.0f;
  if (!std::isfinite(m_i)) {
    size_t vl_max = __riscv_vsetvlmax_e32m4();
    for (int j = 0; j < n_size; j += vl_max) {
      size_t vl = __riscv_vsetvl_e32m4(n_size - j);
      __riscv_vse32_v_f32m4(scores + j, __riscv_vfmv_v_f_f32m4(0.0f, vl), vl);
    }
    return 0.0f;
  }

  size_t vl_max = __riscv_vsetvlmax_e32m4();
  float total_sum = 0.0f;

  for (int j = 0; j < n_size; j += vl_max) {
    size_t vl = __riscv_vsetvl_e32m4(n_size - j);
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(scores + j, vl);
    vx = __riscv_vfsub_vf_f32m4(vx, m_i, vl);
    vx = __riscv_vfmax_vf_f32m4(vx, -88.0f, vl);
    vfloat32m4_t vex = vfexp_f32m4(vx, vl);
    __riscv_vse32_v_f32m4(scores + j, vex, vl);
    vfloat32m1_t v_partial = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    v_partial = __riscv_vfredusum_vs_f32m4_f32m1(vex, v_partial, vl);
    total_sum += __riscv_vfmv_f_s_f32m1_f32(v_partial);
  }
  return std::isfinite(total_sum) ? total_sum : 0.0f;
}

}  // namespace

#endif  // CPU_CAPABILITY_RVV

#endif  // SGL_KERNEL_RVV_VECTOR_MATH_H_
