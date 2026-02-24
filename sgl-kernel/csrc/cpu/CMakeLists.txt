cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sgl_kernel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PY_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Build all cpp files in current dir; arch-specific subdirs added later
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(ARCH_DIRS "x86_64|aarch64|riscv64")
list(FILTER SOURCES EXCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/(${ARCH_DIRS})/")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
  set(RISCV64_SKIP_TOPLEVEL
    # Replaced by riscv64/*.cpp
    decode extend gemm gemm_int8 moe
    # Not supported on RISC-V (no implementation)
    shm interface moe_int8 gemm_int4 moe_int4 flash_attn
  )
  foreach(basename IN LISTS RISCV64_SKIP_TOPLEVEL)
    list(FILTER SOURCES EXCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/${basename}\\.cpp$")
  endforeach()
  message(STATUS "RISC-V: skipped top-level sources: ${RISCV64_SKIP_TOPLEVEL}")
endif()

# Platform-specific source and library directory
set(MY_ARCH_DIR "")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(PLAT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
    set(MY_ARCH_DIR "x86_64")
    add_compile_options(-march=native)
    message(STATUS "Build Architecture: x86_64")

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(PLAT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
    set(MY_ARCH_DIR "aarch64")
    message(STATUS "Build Architecture: aarch64")

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    set(PLAT_LIB_DIR "/usr/lib/riscv64-linux-gnu")
    set(MY_ARCH_DIR "riscv64")
    # RISC-V specific compiler checks
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(WARNING "RISC-V build requires Clang for RVV intrinsics. Current: ${CMAKE_CXX_COMPILER_ID}")
    else()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.0)
            message(FATAL_ERROR "Clang >= 19.0 required for RVV. Found: ${CMAKE_CXX_COMPILER_VERSION}")
        endif()
    endif()

    # RISC-V Compile Definitions & Options
    add_compile_options(-march=rv64gcv_zvfh -mabi=lp64d -mrvv-vector-bits=256)
    add_compile_definitions(CPU_CAPABILITY_RVV)
    add_compile_definitions(SGLANG_RISCV_NO_SHM)
    add_compile_definitions(SGLANG_RISCV_NO_NUMA)
    message(STATUS "Build Architecture: riscv64 (RVV enabled)")

else()
    set(PLAT_LIB_DIR "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
    message(STATUS "Unknown Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

link_directories(${PLAT_LIB_DIR})

# Include arch-specific sources
if(MY_ARCH_DIR)
    file(GLOB_RECURSE ARCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${MY_ARCH_DIR}/*.cpp")
    list(APPEND SOURCES ${ARCH_SOURCES})
    message(STATUS "Included ${MY_ARCH_DIR} specialized sources.")
endif()

# Conda library path support
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
    link_directories(${CONDA_LIB_DIR})
    include_directories("$ENV{CONDA_PREFIX}/include")

    find_library(NUMA_LIB numa HINTS "${CONDA_LIB_DIR}")
else()
    if(DEFINED ENV{VIRTUAL_ENV})
        link_directories("$ENV{VIRTUAL_ENV}/lib")
        include_directories("$ENV{VIRTUAL_ENV}/include")
    endif()
    find_library(NUMA_LIB numa)
endif()

# Check for Libnuma (Skip on RISC-V)
if(NUMA_LIB)
    message(STATUS "Found libnuma: ${NUMA_LIB}")
else()
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
        message(FATAL_ERROR "libnuma not found. Please install libnuma/numactl.")
    else()
        message(STATUS "Skipping libnuma for RISC-V.")
    endif()
endif()

# SGLang internal environment vars
if(NOT DEFINED ENV{SGLANG_CPU_FP8_CVT_FTZ})
    set(ENV{SGLANG_CPU_FP8_CVT_FTZ} "1")
endif()

if("$ENV{SGLANG_CPU_FP8_CVT_FTZ}" STREQUAL "1")
    add_compile_definitions(SGLANG_CPU_FP8_CVT_FTZ)
endif()

# -------------------------------------------------------------------------
# Target Definition
# -------------------------------------------------------------------------
add_compile_options(-O3 -Wno-unknown-pragmas -fopenmp)

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

# Link libraries based on architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    # RISC-V: Link PyTorch and OpenMP
    target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES})

    # OpenMP for RISC-V
    set(OMP_LIB_PATH "$ENV{RISCV_OMP_LIB_PATH}")
    if(EXISTS "${OMP_LIB_PATH}")
        target_link_libraries(common_ops PRIVATE "${OMP_LIB_PATH}")
    else()
        find_library(OMP_LIB omp)
        if(OMP_LIB)
            target_link_libraries(common_ops PRIVATE ${OMP_LIB})
        endif()
    endif()
else()
    # x86_64 / aarch64: Link PyTorch and NUMA
    target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} ${NUMA_LIB})
endif()

target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})

install(TARGETS common_ops LIBRARY DESTINATION sgl_kernel)
