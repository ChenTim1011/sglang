#ifndef SGL_KERNEL_RVV_VECTOR_MATH_H_
#define SGL_KERNEL_RVV_VECTOR_MATH_H_

#if defined(CPU_CAPABILITY_RVV)

#include <riscv_vector.h>

#include <limits>

namespace {

// Polynomial approximation: exp(x) = 2^(x*log2(e))
// Reference: veclibm (https://github.com/rivosinc/veclibm)
constexpr float RVV_EXP_C0 = 1.0f;
constexpr float RVV_EXP_C1 = 0.69314718056f;
constexpr float RVV_EXP_C2 = 0.24022650695f;
constexpr float RVV_EXP_C3 = 0.05550410866f;
constexpr float RVV_EXP_C4 = 0.00961812910f;
constexpr float RVV_EXP_C5 = 0.00133335581f;
constexpr float RVV_LOG2_E = 1.44269504089f;

// Generic Exp template for LMUL={1,2,4,8}
template <int LMUL>
struct RVVExpImpl;

// Specialization: LMUL=4
template <>
struct RVVExpImpl<4> {
  using VFloat = vfloat32m4_t;
  using VInt = vint32m4_t;

  static inline VFloat compute(VFloat vx, size_t vl) {
    vx = __riscv_vfmax_vf_f32m4(vx, -87.0f, vl);  // Clamp to avoid denormals
    vx = __riscv_vfmin_vf_f32m4(vx, 88.0f, vl);

    VFloat vz = __riscv_vfmul_vf_f32m4(vx, RVV_LOG2_E, vl);
    VInt vn_int = __riscv_vfcvt_x_f_v_i32m4(vz, vl);
    VFloat vn = __riscv_vfcvt_f_x_v_f32m4(vn_int, vl);
    VFloat vf = __riscv_vfsub_vv_f32m4(vz, vn, vl);

    // Horner's method: ((((C5*f + C4)*f + C3)*f + C2)*f + C1)*f + C0
    // vfmadd(vd, vs1, vs2) = vd * vs1 + vs2 — vd is the multiplicand
    VFloat vC0 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C0, vl);
    VFloat vC1 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C1, vl);
    VFloat vC2 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C2, vl);
    VFloat vC3 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C3, vl);
    VFloat vC4 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C4, vl);

    VFloat poly = __riscv_vfmv_v_f_f32m4(RVV_EXP_C5, vl);
    poly = __riscv_vfmadd_vv_f32m4(poly, vf, vC4, vl);  // C5*f + C4
    poly = __riscv_vfmadd_vv_f32m4(poly, vf, vC3, vl);  // (C5*f+C4)*f + C3
    poly = __riscv_vfmadd_vv_f32m4(poly, vf, vC2, vl);  // ...
    poly = __riscv_vfmadd_vv_f32m4(poly, vf, vC1, vl);
    poly = __riscv_vfmadd_vv_f32m4(poly, vf, vC0, vl);

    VInt v_exp = __riscv_vadd_vx_i32m4(vn_int, 127, vl);
    v_exp = __riscv_vsll_vx_i32m4(v_exp, 23, vl);
    VFloat v_pow2n = __riscv_vreinterpret_v_i32m4_f32m4(v_exp);

    return __riscv_vfmul_vv_f32m4(poly, v_pow2n, vl);
  }
};

// Specialization: LMUL=8
template <>
struct RVVExpImpl<8> {
  using VFloat = vfloat32m8_t;
  using VInt = vint32m8_t;

  static inline VFloat compute(VFloat vx, size_t vl) {
    vx = __riscv_vfmax_vf_f32m8(vx, -87.0f, vl);
    vx = __riscv_vfmin_vf_f32m8(vx, 88.0f, vl);

    VFloat vz = __riscv_vfmul_vf_f32m8(vx, RVV_LOG2_E, vl);
    VInt vn_int = __riscv_vfcvt_x_f_v_i32m8(vz, vl);
    VFloat vn = __riscv_vfcvt_f_x_v_f32m8(vn_int, vl);
    VFloat vf = __riscv_vfsub_vv_f32m8(vz, vn, vl);

    // Horner's method: ((((C5*f + C4)*f + C3)*f + C2)*f + C1)*f + C0
    VFloat vC0 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C0, vl);
    VFloat vC1 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C1, vl);
    VFloat vC2 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C2, vl);
    VFloat vC3 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C3, vl);
    VFloat vC4 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C4, vl);

    VFloat poly = __riscv_vfmv_v_f_f32m8(RVV_EXP_C5, vl);
    poly = __riscv_vfmadd_vv_f32m8(poly, vf, vC4, vl);
    poly = __riscv_vfmadd_vv_f32m8(poly, vf, vC3, vl);
    poly = __riscv_vfmadd_vv_f32m8(poly, vf, vC2, vl);
    poly = __riscv_vfmadd_vv_f32m8(poly, vf, vC1, vl);
    poly = __riscv_vfmadd_vv_f32m8(poly, vf, vC0, vl);

    VInt v_exp = __riscv_vadd_vx_i32m8(vn_int, 127, vl);
    v_exp = __riscv_vsll_vx_i32m8(v_exp, 23, vl);
    VFloat v_pow2n = __riscv_vreinterpret_v_i32m8_f32m8(v_exp);

    return __riscv_vfmul_vv_f32m8(poly, v_pow2n, vl);
  }
};

// Wrapper functions (delegate to template implementations)
inline vfloat32m4_t vfexp_f32m4(vfloat32m4_t vx, size_t vl) {
  return RVVExpImpl<4>::compute(vx, vl);
}

inline vfloat32m8_t vfexp_f32m8(vfloat32m8_t vx, size_t vl) {
  return RVVExpImpl<8>::compute(vx, vl);
}

// Fast reciprocal: ~1/vd via vfrec7 + one Newton-Raphson step (~14-bit accuracy).
// NR: r <- r * (2 - d * r).  Safe for any d > 0 (which holds for all our use sites).
inline vfloat32m4_t vrec_f32m4(vfloat32m4_t vd, size_t vl) {
  vfloat32m4_t vr = __riscv_vfrec7_v_f32m4(vd, vl);
  vfloat32m4_t vdr = __riscv_vfmul_vv_f32m4(vd, vr, vl);
  vfloat32m4_t vcorr = __riscv_vfrsub_vf_f32m4(vdr, 2.0f, vl);  // 2 - d*r
  return __riscv_vfmul_vv_f32m4(vr, vcorr, vl);
}

inline vfloat32m8_t vrec_f32m8(vfloat32m8_t vd, size_t vl) {
  vfloat32m8_t vr = __riscv_vfrec7_v_f32m8(vd, vl);
  vfloat32m8_t vdr = __riscv_vfmul_vv_f32m8(vd, vr, vl);
  vfloat32m8_t vcorr = __riscv_vfrsub_vf_f32m8(vdr, 2.0f, vl);
  return __riscv_vfmul_vv_f32m8(vr, vcorr, vl);
}

// tanh(x) = (e^2x - 1) / (e^2x + 1).  Clamped to ±9 (tanh saturates beyond that).
// Replace vfdiv with vrec_f32m4: saves ~10-20 cycle division latency.
inline vfloat32m4_t vftanh_f32m4(vfloat32m4_t vx, size_t vl) {
  vx = __riscv_vfmax_vf_f32m4(vx, -9.0f, vl);
  vx = __riscv_vfmin_vf_f32m4(vx, 9.0f, vl);
  vfloat32m4_t v2x = __riscv_vfmul_vf_f32m4(vx, 2.0f, vl);
  vfloat32m4_t vex = vfexp_f32m4(v2x, vl);
  vfloat32m4_t v_num = __riscv_vfsub_vf_f32m4(vex, 1.0f, vl);
  vfloat32m4_t v_denom = __riscv_vfadd_vf_f32m4(vex, 1.0f, vl);
  return __riscv_vfmul_vv_f32m4(v_num, vrec_f32m4(v_denom, vl), vl);
}

inline vfloat32m8_t vftanh_f32m8(vfloat32m8_t vx, size_t vl) {
  vx = __riscv_vfmax_vf_f32m8(vx, -9.0f, vl);
  vx = __riscv_vfmin_vf_f32m8(vx, 9.0f, vl);
  vfloat32m8_t v2x = __riscv_vfmul_vf_f32m8(vx, 2.0f, vl);
  vfloat32m8_t vex = vfexp_f32m8(v2x, vl);
  vfloat32m8_t v_num = __riscv_vfsub_vf_f32m8(vex, 1.0f, vl);
  vfloat32m8_t v_denom = __riscv_vfadd_vf_f32m8(vex, 1.0f, vl);
  return __riscv_vfmul_vv_f32m8(v_num, vrec_f32m8(v_denom, vl), vl);
}

// Polynomial approximation: erf(x)
// Abramowitz & Stegun 7.1.26, max error 1.5e-7
// erf(x) = sign(x) * (1 - poly(t) * exp(-x^2))
// where t = 1 / (1 + 0.3275911 * |x|)
// poly(t) = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t  (Horner form)
// erf saturates to ±1 for |x| >= 4; |x| is clamped to 4 to avoid exp underflow.
constexpr float RVV_ERF_P = 0.3275911f;
constexpr float RVV_ERF_A1 = 0.254829592f;
constexpr float RVV_ERF_A2 = -0.284496736f;
constexpr float RVV_ERF_A3 = 1.421413741f;
constexpr float RVV_ERF_A4 = -1.453152027f;
constexpr float RVV_ERF_A5 = 1.061405429f;

inline vfloat32m4_t vferf_f32m4(vfloat32m4_t vx, size_t vl) {
  // |x|, clamped to 4 (erf saturates to ±1 beyond this)
  vfloat32m4_t vabs = __riscv_vfabs_v_f32m4(vx, vl);
  vabs = __riscv_vfmin_vf_f32m4(vabs, 4.0f, vl);

  // t = 1 / (1 + p * |x|)
  vfloat32m4_t vone = __riscv_vfmv_v_f_f32m4(1.0f, vl);
  vfloat32m4_t vdenom = __riscv_vfmacc_vf_f32m4(vone, RVV_ERF_P, vabs, vl);
  vfloat32m4_t vt = vrec_f32m4(vdenom, vl);  // ~1/(1 + p*|x|), replaces vfdiv

  // Horner's method: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
  // vfmadd(vd, vs1, vs2) = vd * vs1 + vs2
  vfloat32m4_t vpoly = __riscv_vfmv_v_f_f32m4(RVV_ERF_A5, vl);
  vpoly = __riscv_vfmadd_vv_f32m4(vpoly, vt, __riscv_vfmv_v_f_f32m4(RVV_ERF_A4, vl), vl);
  vpoly = __riscv_vfmadd_vv_f32m4(vpoly, vt, __riscv_vfmv_v_f_f32m4(RVV_ERF_A3, vl), vl);
  vpoly = __riscv_vfmadd_vv_f32m4(vpoly, vt, __riscv_vfmv_v_f_f32m4(RVV_ERF_A2, vl), vl);
  vpoly = __riscv_vfmadd_vv_f32m4(vpoly, vt, __riscv_vfmv_v_f_f32m4(RVV_ERF_A1, vl), vl);
  vpoly = __riscv_vfmul_vv_f32m4(vpoly, vt, vl);  // final *t gives a1*t + ... + a5*t^5

  // exp(-x^2)
  vfloat32m4_t vx2 = __riscv_vfmul_vv_f32m4(vabs, vabs, vl);
  vfloat32m4_t vexp = vfexp_f32m4(__riscv_vfneg_v_f32m4(vx2, vl), vl);

  // result = 1 - poly * exp(-x^2); vfnmsac(vd, vs1, vs2) = vd - vs1 * vs2
  vfloat32m4_t vresult = __riscv_vfnmsac_vv_f32m4(vone, vpoly, vexp, vl);

  // Apply sign: erf(x) has the same sign as x.
  // vfsgnjx(vd, vs) = vd with sign = sign(vd) XOR sign(vs).
  // vresult >= 0 (sign bit = 0), so result sign = sign(vx).
  return __riscv_vfsgnjx_vv_f32m4(vresult, vx, vl);
}

inline vfloat32m8_t vferf_f32m8(vfloat32m8_t vx, size_t vl) {
  vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(vx, vl);
  vabs = __riscv_vfmin_vf_f32m8(vabs, 4.0f, vl);

  vfloat32m8_t vone = __riscv_vfmv_v_f_f32m8(1.0f, vl);
  vfloat32m8_t vdenom = __riscv_vfmacc_vf_f32m8(vone, RVV_ERF_P, vabs, vl);
  vfloat32m8_t vt = vrec_f32m8(vdenom, vl);  // ~1/(1 + p*|x|), replaces vfdiv

  vfloat32m8_t vpoly = __riscv_vfmv_v_f_f32m8(RVV_ERF_A5, vl);
  vpoly = __riscv_vfmadd_vv_f32m8(vpoly, vt, __riscv_vfmv_v_f_f32m8(RVV_ERF_A4, vl), vl);
  vpoly = __riscv_vfmadd_vv_f32m8(vpoly, vt, __riscv_vfmv_v_f_f32m8(RVV_ERF_A3, vl), vl);
  vpoly = __riscv_vfmadd_vv_f32m8(vpoly, vt, __riscv_vfmv_v_f_f32m8(RVV_ERF_A2, vl), vl);
  vpoly = __riscv_vfmadd_vv_f32m8(vpoly, vt, __riscv_vfmv_v_f_f32m8(RVV_ERF_A1, vl), vl);
  vpoly = __riscv_vfmul_vv_f32m8(vpoly, vt, vl);

  vfloat32m8_t vx2 = __riscv_vfmul_vv_f32m8(vabs, vabs, vl);
  vfloat32m8_t vexp = vfexp_f32m8(__riscv_vfneg_v_f32m8(vx2, vl), vl);

  vfloat32m8_t vresult = __riscv_vfnmsac_vv_f32m8(vone, vpoly, vexp, vl);
  return __riscv_vfsgnjx_vv_f32m8(vresult, vx, vl);
}

// Softmax helper: in-place exp and sum (no normalization)
// scores[i] = exp(scores[i] - max), returns sum = Σ scores[i]
// NOTE: Does NOT normalize scores. Callers use unnormalized exp values
// for the online softmax algorithm (FlashAttention-style).
inline float exp_and_sum(float* __restrict__ scores, int n_size, float m_i) {
  if (n_size <= 0) return 0.0f;

  size_t vl_max = __riscv_vsetvlmax_e32m4();
  float total_sum = 0.0f;

  for (int j = 0; j < n_size; j += vl_max) {
    size_t vl = __riscv_vsetvl_e32m4(n_size - j);
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(scores + j, vl);
    vx = __riscv_vfsub_vf_f32m4(vx, m_i, vl);
    vfloat32m4_t vex = vfexp_f32m4(vx, vl);
    __riscv_vse32_v_f32m4(scores + j, vex, vl);

    vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m4_f32m1(vex, vzero, vl);
    total_sum += __riscv_vfmv_f_s_f32m1_f32(vsum);
  }

  return total_sum;
}

inline float rvv_reduce_max_f32(const float* data, int64_t len) {
  if (len <= 0) return -std::numeric_limits<float>::infinity();

  float max_val = -std::numeric_limits<float>::infinity();
  int64_t remaining = len;

  while (remaining > 0) {
    size_t vl = __riscv_vsetvl_e32m8(remaining);
    vfloat32m8_t vdata = __riscv_vle32_v_f32m8(data + (len - remaining), vl);

    vfloat32m1_t vcurrent_max = __riscv_vfmv_s_f_f32m1(max_val, 1);
    vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m8_f32m1(vdata, vcurrent_max, vl);
    max_val = __riscv_vfmv_f_s_f32m1_f32(vmax);

    remaining -= vl;
  }

  return max_val;
}

// Reduction Wrappers for Different register configurations
inline float reduce_sum_f32m4(vfloat32m4_t v_acc, size_t vl_max) {
  vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  vfloat32m1_t vred = __riscv_vfredusum_vs_f32m4_f32m1(v_acc, vzero, vl_max);
  return __riscv_vfmv_f_s_f32m1_f32(vred);
}

inline float reduce_sum_f32m1(vfloat32m1_t v_acc, size_t vl_max) {
  vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  vfloat32m1_t vred = __riscv_vfredusum_vs_f32m1_f32m1(v_acc, vzero, vl_max);
  return __riscv_vfmv_f_s_f32m1_f32(vred);
}

// Widening Multiply-Accumulate Support

// FP16 Vector-Vector -> FP32 Accumulator
inline vfloat32m4_t vfwmacc_f16_to_f32m4(vfloat32m4_t vd, vfloat16m2_t vs1, vfloat16m2_t vs2, size_t vl) {
#if defined(__riscv_zvfh)
  return __riscv_vfwmacc_vv_f32m4_tu(vd, vs1, vs2, vl);
#else
  vfloat32m4_t vs1_f32 = __riscv_vfwcvt_f_f_v_f32m4(vs1, vl);
  vfloat32m4_t vs2_f32 = __riscv_vfwcvt_f_f_v_f32m4(vs2, vl);
  return __riscv_vfmacc_vv_f32m4_tu(vd, vs1_f32, vs2_f32, vl);
#endif
}

// FP16 Scalar-Vector -> FP32 Accumulator
inline vfloat32m4_t vfwmacc_f16_scalar_to_f32m4(vfloat32m4_t vd, _Float16 scalar, vfloat16m2_t vs2, size_t vl) {
#if defined(__riscv_zvfh)
  return __riscv_vfwmacc_vf_f32m4_tu(vd, scalar, vs2, vl);
#else
  vfloat32m4_t vs2_f32 = __riscv_vfwcvt_f_f_v_f32m4(vs2, vl);
  return __riscv_vfmacc_vf_f32m4_tu(vd, (float)scalar, vs2_f32, vl);
#endif
}

}  // namespace

#endif  // CPU_CAPABILITY_RVV

#endif  // SGL_KERNEL_RVV_VECTOR_MATH_H_
