#include <ATen/ATen.h>
#include <torch/all.h>

#if defined(CPU_CAPABILITY_RVV)
#include <riscv_vector.h>
#endif

namespace {

#if defined(CPU_CAPABILITY_RVV)

inline size_t get_vlenb_csr() {
  size_t vlenb;
  asm volatile("csrr %0, vlenb" : "=r"(vlenb));
  return vlenb;
}

inline size_t get_vlenb_vsetvl() {
  size_t vl_max = __riscv_vsetvlmax_e8m1();
  return vl_max;
}

size_t get_vlenb() {
  size_t vlenb = get_vlenb_csr();
  if (vlenb == 0) {
    return get_vlenb_vsetvl();
  }
  return vlenb;
}

size_t get_vlen() {
  return get_vlenb() * 8;
}

bool is_vlen_aligned(size_t size_bytes) {
  size_t vlenb = get_vlenb();
  return (size_bytes % vlenb) == 0;
}

template <typename T>
bool is_vlen_aligned_elements(size_t num_elements) {
  size_t vlenb = get_vlenb();
  size_t size_bytes = num_elements * sizeof(T);
  return (size_bytes % vlenb) == 0;
}

size_t round_up_to_vlen(size_t size_bytes) {
  size_t vlenb = get_vlenb();
  return ((size_bytes + vlenb - 1) / vlenb) * vlenb;
}

#else
size_t get_vlenb() {
  return 0;
}
size_t get_vlen() {
  return 0;
}
bool is_vlen_aligned(size_t size_bytes) {
  return true;
}
template <typename T>
bool is_vlen_aligned_elements(size_t num_elements) {
  return true;
}
size_t round_up_to_vlen(size_t size_bytes) {
  return size_bytes;
}

#endif

}  // namespace

int64_t get_rvv_vlenb() {
  return static_cast<int64_t>(get_vlenb());
}

int64_t get_rvv_vlen() {
  return static_cast<int64_t>(get_vlen());
}

bool check_vlen_alignment(int64_t size_bytes) {
  return is_vlen_aligned(static_cast<size_t>(size_bytes));
}
