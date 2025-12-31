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
)

# Python include directories - handle cross-compilation
if(CMAKE_SYSROOT)
    # For cross-compilation, prefer sysroot Python headers
    if(EXISTS "${CMAKE_SYSROOT}/usr/include/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
        include_directories("${CMAKE_SYSROOT}/usr/include/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
        message(STATUS "Using Python headers from sysroot: ${CMAKE_SYSROOT}/usr/include/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
    endif()
    # Also add sysroot usr/include for general headers
    if(EXISTS "${CMAKE_SYSROOT}/usr/include")
        include_directories("${CMAKE_SYSROOT}/usr/include")
    endif()
endif()

# Add system Python include directories (fallback)
include_directories(${Python_INCLUDE_DIRS})

# Python include directories - handle cross-compilation
if(CMAKE_SYSROOT)
    # For cross-compilation, prefer sysroot Python headers
    if(EXISTS "${CMAKE_SYSROOT}/usr/include/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
        include_directories("${CMAKE_SYSROOT}/usr/include/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
        message(STATUS "Using Python headers from sysroot: ${CMAKE_SYSROOT}/usr/include/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
    endif()
    # Also add sysroot usr/include for general headers
    if(EXISTS "${CMAKE_SYSROOT}/usr/include")
        include_directories("${CMAKE_SYSROOT}/usr/include")
    endif()
endif()

# Add system Python include directories (fallback)
include_directories(${Python_INCLUDE_DIRS})

# Platform-specific source and library directory
set(MY_ARCH_DIR "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(PLAT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
    set(MY_ARCH_DIR "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(PLAT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
    set(MY_ARCH_DIR "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le|ppc64")
    set(PLAT_LIB_DIR "/usr/lib/powerpc64le-linux-gnu")
    set(MY_ARCH_DIR "ppc64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|riscv32")
    set(PLAT_LIB_DIR "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
    # Check if RVV is supported (Clang 19+)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --version
            OUTPUT_VARIABLE CLANG_VERSION_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        string(REGEX MATCH "clang version ([0-9]+)" _ "${CLANG_VERSION_OUTPUT}")
        if(CMAKE_MATCH_1)
            set(CLANG_MAJOR_VERSION "${CMAKE_MATCH_1}")
            if(CLANG_MAJOR_VERSION GREATER_EQUAL 19)
                add_compile_definitions(CPU_CAPABILITY_RVV)
                set(MY_ARCH_DIR "rvv")
            endif()
        endif()
    endif()
else()
    set(PLAT_LIB_DIR "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
endif()
link_directories(${PLAT_LIB_DIR})
if(MY_ARCH_DIR)
    file(GLOB_RECURSE ARCH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${MY_ARCH_DIR}/*.cpp")
    list(APPEND SOURCES ${ARCH_SOURCES})
endif()

# Conda library path support
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Using Conda lib dir: ${CONDA_LIB_DIR}")
    link_directories(${CONDA_LIB_DIR})
    set(CONDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
    include_directories(${CONDA_INCLUDE_DIR})

    # Look for libnuma in Conda's lib directory
    find_library(NUMA_LIB numa HINTS "${CONDA_LIB_DIR}")
    if(NUMA_LIB)
        message(STATUS "Found libnuma: ${NUMA_LIB}")
    else()
        message(FATAL_ERROR "libnuma not found in Conda environment at ${CONDA_LIB_DIR}\n"
                            "Please install it using: conda install libnuma numactl\n")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV" OR CMAKE_CXX_FLAGS MATCHES "--target=riscv64")
    # RISC-V cross-compilation: skip conda, use sysroot
    message(STATUS "RISC-V cross-compilation: Skipping conda libraries (x86_64)")
    set(NUMA_LIB "")
elseif(CMAKE_SYSROOT)
    # Cross-compilation: search in sysroot
    message(STATUS "Cross-compilation: Searching libraries in ${CMAKE_SYSROOT}")
    find_library(NUMA_LIB numa 
        HINTS 
        "${CMAKE_SYSROOT}/usr/lib/riscv64-linux-gnu"
        "${CMAKE_SYSROOT}/usr/lib64"
        "${CMAKE_SYSROOT}/usr/lib"
        "${CMAKE_SYSROOT}/lib"
    )
    if(NUMA_LIB)
        message(STATUS "Found libnuma: ${NUMA_LIB}")
    else()
        message(WARNING "libnuma not found in sysroot. Continuing without it.")
        set(NUMA_LIB "")
    endif()
else()
    # System libraries
    find_library(NUMA_LIB numa)
    if(NUMA_LIB)
        message(STATUS "Found libnuma: ${NUMA_LIB}")
    endif()
endif()

# Build all cpp files in current and sub dirs
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Exclude all arch dependent files, then add back only files for this arch
set(ARCH_DIRS "x86_64|aarch64|ppc64|rvv")
list(FILTER SOURCES EXCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/(${ARCH_DIRS})/")

if(NOT DEFINED ENV{SGLANG_CPU_FP8_CVT_FTZ})
    set(ENV{SGLANG_CPU_FP8_CVT_FTZ} "1")
endif()

if("$ENV{SGLANG_CPU_FP8_CVT_FTZ}" STREQUAL "1")
    message(STATUS "Enabling macro: SGLANG_CPU_FP8_CVT_FTZ")
    add_compile_definitions(SGLANG_CPU_FP8_CVT_FTZ)
endif()

add_compile_options(
    -O3
    -Wno-unknown-pragmas
    -march=native
    -fopenmp
)

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} ${NUMA_LIB})
target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})

install(TARGETS common_ops
    LIBRARY DESTINATION sgl_kernel
)
