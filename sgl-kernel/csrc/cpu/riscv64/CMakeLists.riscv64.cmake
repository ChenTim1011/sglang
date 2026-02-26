# RISC-V build configuration.
# Included from parent CMakeLists.txt when CMAKE_SYSTEM_PROCESSOR matches riscv64.
# Sets: PLAT_LIB_DIR, RISCV64_SKIP_NUMA, ARCH_LINK_LIBS; defines riscv64_filter_sources()

set(PLAT_LIB_DIR "/usr/lib/riscv64-linux-gnu")
set(RISCV64_SKIP_NUMA TRUE)

# Top-level sources to exclude (replaced by riscv64/*.cpp or not supported)
set(RISCV64_SKIP_TOPLEVEL
  activation decode extend gemm gemm_int8 norm rope moe moe_int8 moe_fp8
  shm interface gemm_int4 moe_int4 flash_attn
)

macro(riscv64_filter_sources _sources_var)
  set(_arch_dirs "x86_64|aarch64|riscv64")
  list(FILTER ${_sources_var} EXCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/(${_arch_dirs})/")
  foreach(_basename IN LISTS RISCV64_SKIP_TOPLEVEL)
    list(FILTER ${_sources_var} EXCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/${_basename}\\.cpp$")
  endforeach()
  file(GLOB_RECURSE _arch_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/riscv64/*.cpp")
  list(APPEND ${_sources_var} ${_arch_sources})
  message(STATUS "RISC-V: skipped top-level, included riscv64/*.cpp")
endmacro()

# Compiler checks
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(WARNING "RISC-V build requires Clang for RVV intrinsics. Current: ${CMAKE_CXX_COMPILER_ID}")
else()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.0)
    message(FATAL_ERROR "Clang >= 19.0 required for RVV. Found: ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
endif()

# RISC-V compile options (override -march=native from common options)
add_compile_options(-march=rv64gcv_zvfh -mabi=lp64d -mrvv-vector-bits=256)
add_compile_definitions(CPU_CAPABILITY_RVV)
add_compile_definitions(SGLANG_RISCV_NO_SHM)
add_compile_definitions(SGLANG_RISCV_NO_NUMA)

message(STATUS "Build Architecture: riscv64 (RVV enabled)")

# Link libraries: PyTorch + OpenMP (no NUMA on RISC-V)
set(ARCH_LINK_LIBS ${TORCH_LIBRARIES})
set(OMP_LIB_PATH "$ENV{RISCV_OMP_LIB_PATH}")
if(EXISTS "${OMP_LIB_PATH}")
  list(APPEND ARCH_LINK_LIBS "${OMP_LIB_PATH}")
else()
  find_library(OMP_LIB omp)
  if(OMP_LIB)
    list(APPEND ARCH_LINK_LIBS ${OMP_LIB})
  endif()
endif()
