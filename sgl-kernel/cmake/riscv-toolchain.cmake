# CMake toolchain file for RISC-V cross-compilation
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/riscv-toolchain.cmake ...

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Toolchain paths (set via environment or CMake variables)
if(DEFINED ENV{TOOLCHAIN_BIN})
    set(TOOLCHAIN_BIN $ENV{TOOLCHAIN_BIN})
else()
    set(TOOLCHAIN_BIN "/home/juiting/sdxl_on_shark-turbine/iree/y/toolchain/clang/linux/RISCV/bin")
endif()

if(DEFINED ENV{SYSROOT})
    set(CMAKE_SYSROOT $ENV{SYSROOT})
else()
    set(CMAKE_SYSROOT "/home/juiting/sdxl_on_shark-turbine/iree/y/toolchain/clang/linux/RISCV/sysroot")
endif()

# Compilers
# Priority: Use environment variables (CC/CXX) if set, otherwise use TOOLCHAIN_BIN defaults
if(DEFINED ENV{CC} AND EXISTS $ENV{CC})
    set(CMAKE_C_COMPILER $ENV{CC})
else()
    # Try clang-19 first, then clang-18, then clang
    if(EXISTS "${TOOLCHAIN_BIN}/clang-19")
        set(CMAKE_C_COMPILER "${TOOLCHAIN_BIN}/clang-19")
    elseif(EXISTS "${TOOLCHAIN_BIN}/clang-18")
        set(CMAKE_C_COMPILER "${TOOLCHAIN_BIN}/clang-18")
    else()
        set(CMAKE_C_COMPILER "${TOOLCHAIN_BIN}/clang")
    endif()
endif()

if(DEFINED ENV{CXX} AND EXISTS $ENV{CXX})
    set(CMAKE_CXX_COMPILER $ENV{CXX})
else()
    # Try clang++-19 first, then clang++-18, then clang++
    if(EXISTS "${TOOLCHAIN_BIN}/clang++-19")
        set(CMAKE_CXX_COMPILER "${TOOLCHAIN_BIN}/clang++-19")
    elseif(EXISTS "${TOOLCHAIN_BIN}/clang++-18")
        set(CMAKE_CXX_COMPILER "${TOOLCHAIN_BIN}/clang++-18")
    else()
        set(CMAKE_CXX_COMPILER "${TOOLCHAIN_BIN}/clang++")
    endif()
endif()

set(CMAKE_AR "${TOOLCHAIN_BIN}/llvm-ar")
set(CMAKE_RANLIB "${TOOLCHAIN_BIN}/llvm-ranlib")
set(CMAKE_STRIP "${TOOLCHAIN_BIN}/llvm-strip")

# Build flags
# Priority: Use environment variables if set (from clang19_rvv_env.sh), otherwise use defaults
if(DEFINED ENV{CFLAGS})
    set(CMAKE_C_FLAGS_INIT "$ENV{CFLAGS}")
else()
    set(CMAKE_C_FLAGS_INIT "--target=riscv64-unknown-linux-gnu -march=rv64gcv -mabi=lp64d --sysroot=${CMAKE_SYSROOT}")
endif()

if(DEFINED ENV{CXXFLAGS})
    set(CMAKE_CXX_FLAGS_INIT "$ENV{CXXFLAGS}")
else()
    set(CMAKE_CXX_FLAGS_INIT "--target=riscv64-unknown-linux-gnu -march=rv64gcv -mabi=lp64d --sysroot=${CMAKE_SYSROOT}")
endif()

if(DEFINED ENV{LDFLAGS})
    set(CMAKE_EXE_LINKER_FLAGS_INIT "$ENV{LDFLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_INIT "$ENV{LDFLAGS}")
else()
    set(CMAKE_EXE_LINKER_FLAGS_INIT "--target=riscv64-unknown-linux-gnu --sysroot=${CMAKE_SYSROOT} -latomic")
    set(CMAKE_SHARED_LINKER_FLAGS_INIT "--target=riscv64-unknown-linux-gnu --sysroot=${CMAKE_SYSROOT} -latomic")
endif()

# Search paths
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Python (for cross-compilation, we may need to specify)
# For now, let CMake find it, but we'll handle PyTorch separately
if(DEFINED ENV{PYTHON_RISCV})
    set(Python3_EXECUTABLE $ENV{PYTHON_RISCV})
endif()

# PyTorch path (set via environment)
if(DEFINED ENV{PYTORCH_CROSS_PREFIX})
    set(TORCH_PY_PREFIX $ENV{PYTORCH_CROSS_PREFIX})
    message(STATUS "Using PyTorch from: ${TORCH_PY_PREFIX}")
endif()

