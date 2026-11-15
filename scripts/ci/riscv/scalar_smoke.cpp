#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

extern "C" void scalar_kernel(const float *input, float *output,
                              std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    output[i] = input[i] * input[i] + 1.0f;
  }
}

int main() {
  constexpr std::array<std::size_t, 7> sizes{0, 1, 7, 8, 9, 4096, 4099};
  constexpr float sentinel = -12345.0f;
  for (std::size_t n : sizes) {
    std::vector<float> input(n);
    std::vector<float> output(n + 2, sentinel);
    for (std::size_t i = 0; i < n; ++i) {
      input[i] = (static_cast<int>(i % 29) - 14) * 0.25f;
    }
    scalar_kernel(input.data(), output.data() + 1, n);
    if (output.front() != sentinel || output.back() != sentinel) {
      std::cerr << "Out-of-bounds write for n=" << n << '\n';
      return 1;
    }
    for (std::size_t i = 0; i < n; ++i) {
      const double x = input[i];
      const double expected = x * x + 1.0;
      if (std::abs(output[i + 1] - expected) > 1e-6) {
        std::cerr << "Numerical mismatch for n=" << n << ", i=" << i << '\n';
        return 1;
      }
    }
  }
  std::cout << "Scalar execution: 7 sizes passed\n";
}
