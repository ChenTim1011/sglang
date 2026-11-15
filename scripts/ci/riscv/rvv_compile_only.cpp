#include <riscv_vector.h>

#include <cstddef>

// Compiler/assembler smoke only. This object is never loaded or executed.
extern "C" void rvv_compile_only(const float *input, float *output,
                                 std::size_t n) {
  while (n != 0) {
    const std::size_t vl = __riscv_vsetvl_e32m1(n);
    auto values = __riscv_vle32_v_f32m1(input, vl);
    auto squared = __riscv_vfmul_vv_f32m1(values, values, vl);
    auto result = __riscv_vfadd_vf_f32m1(squared, 1.0f, vl);
    __riscv_vse32_v_f32m1(output, result, vl);
    input += vl;
    output += vl;
    n -= vl;
  }
}
