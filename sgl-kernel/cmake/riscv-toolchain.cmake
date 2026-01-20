# CMake toolchain file for RISC-V cross-compilation
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/riscv-toolchain.cmake ...

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Toolchain paths
if(DEFINED ENV{TOOLCHAIN_BIN})
    set(TOOLCHAIN_BIN $ENV{TOOLCHAIN_BIN})
else()
    set(TOOLCHAIN_BIN "")
endif()

if(DEFINED ENV{SYSROOT})
    set(CMAKE_SYSROOT $ENV{SYSROOT})
else()
    set(CMAKE_SYSROOT "")
endif()

# Compilers
# Use environment variables (CC/CXX) if set, otherwise use TOOLCHAIN_BIN defaults
if(DEFINED ENV{CC} AND EXISTS $ENV{CC})
    set(CMAKE_C_COMPILER $ENV{CC})
else()
    # Assuming Clang >= 19
    # If TOOLCHAIN_BIN is set, use it. Otherwise rely on PATH.
    if(TOOLCHAIN_BIN)
        set(_BIN_PREFIX "${TOOLCHAIN_BIN}/")
    else()
        set(_BIN_PREFIX "")
    endif()

    if(EXISTS "${_BIN_PREFIX}clang-19")
        set(CMAKE_C_COMPILER "${_BIN_PREFIX}clang-19")
    elseif(EXISTS "${_BIN_PREFIX}clang-20")
        set(CMAKE_C_COMPILER "${_BIN_PREFIX}clang-20")
    elseif(EXISTS "${_BIN_PREFIX}clang-21")
        set(CMAKE_C_COMPILER "${_BIN_PREFIX}clang-21")
    else()
        if(TOOLCHAIN_BIN)
            # If explicit path given, use it
            set(CMAKE_C_COMPILER "${_BIN_PREFIX}clang")
        else()
            # Fallback to PATH search or generic name
            set(CMAKE_C_COMPILER "clang")
        endif()
    endif()
endif()

if(DEFINED ENV{CXX} AND EXISTS $ENV{CXX})
    set(CMAKE_CXX_COMPILER $ENV{CXX})
else()
    if(TOOLCHAIN_BIN)
        set(_BIN_PREFIX "${TOOLCHAIN_BIN}/")
    else()
        set(_BIN_PREFIX "")
    endif()

    if(EXISTS "${_BIN_PREFIX}clang++-19")
        set(CMAKE_CXX_COMPILER "${_BIN_PREFIX}clang++-19")
    elseif(EXISTS "${_BIN_PREFIX}clang++-20")
        set(CMAKE_CXX_COMPILER "${_BIN_PREFIX}clang++-20")
    elseif(EXISTS "${_BIN_PREFIX}clang++-21")
        set(CMAKE_CXX_COMPILER "${_BIN_PREFIX}clang++-21")
    else()
        if(TOOLCHAIN_BIN)
            set(CMAKE_CXX_COMPILER "${_BIN_PREFIX}clang++")
        else()
            set(CMAKE_CXX_COMPILER "clang++")
        endif()
    endif()
endif()

set(CMAKE_AR "${TOOLCHAIN_BIN}/llvm-ar")
set(CMAKE_RANLIB "${TOOLCHAIN_BIN}/llvm-ranlib")
set(CMAKE_STRIP "${TOOLCHAIN_BIN}/llvm-strip")

# Build flags
# Priority: Use environment variables if set (from clang_rvv_env.sh), otherwise use defaults
if(DEFINED ENV{CFLAGS})
    set(CMAKE_C_FLAGS_INIT "$ENV{CFLAGS}")
else()
    set(CMAKE_C_FLAGS_INIT "--target=riscv64-unknown-linux-gnu -march=rv64gcv_zvfh -mabi=lp64d --sysroot=${CMAKE_SYSROOT}")
endif()

if(DEFINED ENV{CXXFLAGS})
    set(CMAKE_CXX_FLAGS_INIT "$ENV{CXXFLAGS}")
else()
    set(CMAKE_CXX_FLAGS_INIT "--target=riscv64-unknown-linux-gnu -march=rv64gcv_zvfh -mabi=lp64d --sysroot=${CMAKE_SYSROOT}")
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
if(DEFINED ENV{PYTHON_RISCV})
    set(Python3_EXECUTABLE $ENV{PYTHON_RISCV})
endif()

# PyTorch path (set via environment)
if(DEFINED ENV{PYTORCH_CROSS_PREFIX})
    set(TORCH_PY_PREFIX $ENV{PYTORCH_CROSS_PREFIX})
    message(STATUS "Using PyTorch from: ${TORCH_PY_PREFIX}")
endif()
