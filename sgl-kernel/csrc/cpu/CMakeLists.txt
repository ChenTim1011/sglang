cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sgl_kernel)

# Suppress CMake developer warnings
if(POLICY CMP0174)
    cmake_policy(SET CMP0174 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# Handle PyTorch path for cross-compilation
# For cross-compilation, we can't execute Python to find PyTorch (ABI mismatch)
# So we allow manual specification via TORCH_PY_PREFIX environment variable
if(DEFINED ENV{TORCH_PY_PREFIX})
    # Cross-compilation: use manually specified PyTorch path
    set(TORCH_PY_PREFIX $ENV{TORCH_PY_PREFIX})
    message(STATUS "Cross-compilation mode: Using PyTorch from ${TORCH_PY_PREFIX}")
    # PyTorch wheel structure: torch/share/cmake/Torch/TorchConfig.cmake
    # TORCH_INSTALL_PREFIX should be the torch/ directory (parent of share/)
    # Check if TORCH_PY_PREFIX already points to torch directory (has lib/libtorch.so)
    if(EXISTS "${TORCH_PY_PREFIX}/lib/libtorch.so")
        # TORCH_PY_PREFIX is already the torch directory
        set(TORCH_INSTALL_PREFIX "${TORCH_PY_PREFIX}")
        message(STATUS "TORCH_PY_PREFIX is torch directory, using as-is")
    else()
        # TORCH_PY_PREFIX is the parent directory (e.g., site-packages), add /torch
        set(TORCH_INSTALL_PREFIX "${TORCH_PY_PREFIX}/torch")
        message(STATUS "TORCH_PY_PREFIX is parent directory, adding /torch")
    endif()
    message(STATUS "Setting TORCH_INSTALL_PREFIX to: ${TORCH_INSTALL_PREFIX}")
    
    # Set Torch_DIR and Caffe2_DIR
    set(TORCH_CMAKE_BASE "${TORCH_INSTALL_PREFIX}/share/cmake")
    set(TORCH_CMAKE_DIR "${TORCH_CMAKE_BASE}/Torch")
    set(CAFFE2_CMAKE_DIR "${TORCH_CMAKE_BASE}/Caffe2")
    
    if(EXISTS "${TORCH_CMAKE_DIR}/TorchConfig.cmake")
        message(STATUS "Found TorchConfig.cmake at: ${TORCH_CMAKE_DIR}")
        set(Torch_DIR "${TORCH_CMAKE_DIR}")
        # Also add base cmake directory to prefix path for Caffe2 and other dependencies
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_BASE}")
    else()
        # Fallback: try adding to CMAKE_PREFIX_PATH
        message(STATUS "TorchConfig.cmake not found at ${TORCH_CMAKE_DIR}, trying CMAKE_PREFIX_PATH")
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_BASE}")
    endif()
    
    # Set Caffe2_DIR if it exists
    if(EXISTS "${CAFFE2_CMAKE_DIR}/Caffe2Config.cmake")
        message(STATUS "Found Caffe2Config.cmake at: ${CAFFE2_CMAKE_DIR}")
        set(Caffe2_DIR "${CAFFE2_CMAKE_DIR}")
    endif()
    
    # Set TORCH_INSTALL_PREFIX as CMake variable (TorchConfig.cmake checks both ENV and variable)
    # TorchConfig.cmake checks: if(DEFINED ENV{TORCH_INSTALL_PREFIX}) or uses computed path
    # We set it as both to be safe
    set(ENV{TORCH_INSTALL_PREFIX} "${TORCH_INSTALL_PREFIX}")
    # Also set as CMake cache variable
    set(TORCH_INSTALL_PREFIX "${TORCH_INSTALL_PREFIX}" CACHE PATH "PyTorch installation prefix")
    
    # Verify library exists and manually set TORCH_LIBRARY if found
    # This is needed for cross-compilation where find_library may not work correctly
    # We set it as CACHE FORCE to prevent find_library from overwriting it
    if(EXISTS "${TORCH_INSTALL_PREFIX}/lib/libtorch.so")
        message(STATUS "Found libtorch.so at: ${TORCH_INSTALL_PREFIX}/lib/libtorch.so")
        # Manually set TORCH_LIBRARY with FORCE to override find_library
        set(TORCH_LIBRARY "${TORCH_INSTALL_PREFIX}/lib/libtorch.so" CACHE FILEPATH "PyTorch library" FORCE)
        message(STATUS "Manually set TORCH_LIBRARY to: ${TORCH_LIBRARY}")
    else()
        message(WARNING "libtorch.so not found at: ${TORCH_INSTALL_PREFIX}/lib/libtorch.so")
        message(STATUS "Available libraries in ${TORCH_INSTALL_PREFIX}/lib:")
        file(GLOB LIB_FILES "${TORCH_INSTALL_PREFIX}/lib/*.so")
        foreach(lib ${LIB_FILES})
            message(STATUS "  - ${lib}")
        endforeach()
    endif()
    
    find_package(Torch REQUIRED)
else()
    # Native build: use Python to find PyTorch
execute_process(
    COMMAND ${Python_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PY_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYTHON_TORCH_RESULT
        ERROR_QUIET
)

    if(PYTHON_TORCH_RESULT EQUAL 0 AND TORCH_PY_PREFIX)
        message(STATUS "Native build: Found PyTorch at ${TORCH_PY_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)
    else()
        message(FATAL_ERROR "Could not find PyTorch. For cross-compilation, set TORCH_PY_PREFIX environment variable.\n"
                            "Example: export TORCH_PY_PREFIX=/path/to/site-packages")
    endif()
endif()

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}
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

# Platform-specific library directory
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(PLAT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(PLAT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le|ppc64")
    set(PLAT_LIB_DIR "/usr/lib/powerpc64le-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV")
    # For RISC-V, use sysroot if available
    if(CMAKE_SYSROOT)
        set(PLAT_LIB_DIR "${CMAKE_SYSROOT}/usr/lib/riscv64-linux-gnu")
        # Also check sysroot lib64
        if(EXISTS "${CMAKE_SYSROOT}/usr/lib64")
            link_directories("${CMAKE_SYSROOT}/usr/lib64")
        endif()
    else()
        set(PLAT_LIB_DIR "/usr/lib/riscv64-linux-gnu")
    endif()
else()
    set(PLAT_LIB_DIR "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
endif()
link_directories(${PLAT_LIB_DIR})

# Library search: Conda or sysroot
# For RISC-V cross-compilation, skip conda libraries (they're x86_64)
if((DEFINED ENV{CONDA_PREFIX} AND NOT CMAKE_SYSROOT) AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV" OR CMAKE_CXX_FLAGS MATCHES "--target=riscv64"))
    # Native build with Conda (not RISC-V)
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
    if(DEFINED ENV{VIRTUAL_ENV})
        set(VENV_LIB_DIR "$ENV{VIRTUAL_ENV}/lib")
        message(STATUS "Using venv lib dir: ${VENV_LIB_DIR}")
        link_directories(${VENV_LIB_DIR})
        set(VENV_INCLUDE_DIR "$ENV{VIRTUAL_ENV}/include")
        include_directories(${VENV_INCLUDE_DIR})
    endif()
    # Look for libnuma in system env paths
    find_library(NUMA_LIB numa)
    if(NUMA_LIB)
        message(STATUS "Found libnuma: ${NUMA_LIB}")
    else()
        message(FATAL_ERROR "libnuma not found in system environment\n"
                            "Please install it using: apt-get install libnuma numactl\n")
    endif()
endif()

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
# Exclude shm.cpp for RISC-V or when cross-compiling with --target flag
# shm.cpp contains x86-specific intrinsics (immintrin.h) that are not available on RISC-V
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV" OR CMAKE_CXX_FLAGS MATCHES "--target=riscv64")
    list(FILTER SOURCES EXCLUDE REGEX ".*shm\\.cpp$")
    message(STATUS "Excluding shm.cpp (contains x86-specific intrinsics)")
    # Define macro to indicate SHM collectives are not available
    add_compile_definitions(SGLANG_RISCV_NO_SHM_COLLECTIVES)
endif()

if(NOT DEFINED ENV{SGLANG_CPU_FP8_CVT_FTZ})
    set(ENV{SGLANG_CPU_FP8_CVT_FTZ} "1")
endif()

if("$ENV{SGLANG_CPU_FP8_CVT_FTZ}" STREQUAL "1")
    message(STATUS "Enabling macro: SGLANG_CPU_FP8_CVT_FTZ")
    add_compile_definitions(SGLANG_CPU_FP8_CVT_FTZ)
endif()

# RISC-V Vector Extension support
# Check if we're building for RISC-V
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV")
    message(STATUS "Building for RISC-V with Vector Extension support")
    
    # --- START: Enforce Clang 19 requirement ---
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "RISC-V RVV backend requires Clang compiler. "
                            "Current compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
    endif()
    
    # Check Clang version (must be >= 19)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        OUTPUT_VARIABLE CLANG_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    # Extract major version number (e.g., "19" from "clang version 19.1.0")
    string(REGEX MATCH "clang version ([0-9]+)\\.[0-9]+" _ "${CLANG_VERSION_OUTPUT}")
    if(CMAKE_MATCH_1)
        set(CLANG_MAJOR_VERSION "${CMAKE_MATCH_1}")
        if(CLANG_MAJOR_VERSION LESS 19)
            message(FATAL_ERROR "RISC-V RVV backend requires Clang 19 or later. "
                                "Current version: ${CLANG_VERSION_OUTPUT}\n"
                                "Please use Clang 19.1.0 or later.\n"
                                "You can set CXX environment variable to point to Clang 19:\n"
                                "  export CXX=/path/to/clang19/bin/clang++")
        endif()
        message(STATUS "✓ Clang version check passed: ${CLANG_VERSION_OUTPUT} (major: ${CLANG_MAJOR_VERSION})")
    else()
        # Try to get version from CMAKE_CXX_COMPILER_VERSION
        if(CMAKE_CXX_COMPILER_VERSION)
            string(REGEX MATCH "^([0-9]+)" CLANG_MAJOR_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
            if(CLANG_MAJOR_VERSION AND CLANG_MAJOR_VERSION LESS 19)
                message(FATAL_ERROR "RISC-V RVV backend requires Clang 19 or later. "
                                    "Current version: ${CMAKE_CXX_COMPILER_VERSION}\n"
                                    "Please use Clang 19.1.0 or later.")
            endif()
            message(STATUS "✓ Clang version check passed: ${CMAKE_CXX_COMPILER_VERSION} (major: ${CLANG_MAJOR_VERSION})")
        else()
            message(WARNING "Could not determine Clang version from --version output.")
            message(WARNING "Version output: ${CLANG_VERSION_OUTPUT}")
            message(WARNING "Proceeding with caution. Please ensure you are using Clang 19.1.0 or later.")
        endif()
    endif()
    # --- END: Enforce Clang 19 requirement ---
    
    # Remove any existing -march flags to avoid conflicts
    # Then add RISC-V specific options
    add_compile_options(
        -O3
        -Wno-unknown-pragmas
        -Wno-unused-value
        -Wno-macro-redefined
        -march=rv64gcv
        -mabi=lp64d
        -fopenmp
    )
    # Define CPU_CAPABILITY_RVV for RISC-V Vector Extension
    add_compile_definitions(CPU_CAPABILITY_RVV)
    
    # Ensure riscv_vector.h can be found
    # Clang 19 includes it in its internal include directory
    # We need to explicitly add it for cross-compilation
    get_filename_component(CLANG_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(CLANG_ROOT "${CLANG_BIN_DIR}" DIRECTORY)
    
    # Try to get Clang version from compiler (full version string)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        OUTPUT_VARIABLE CLANG_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    # Extract full version (e.g., "19.1.0" from "clang version 19.1.0")
    string(REGEX MATCH "clang version ([0-9]+\\.[0-9]+\\.[0-9]+)" _ "${CLANG_VERSION_OUTPUT}")
    if(CMAKE_MATCH_1)
        set(CLANG_VERSION "${CMAKE_MATCH_1}")
    else()
        # Fallback to CMAKE_CXX_COMPILER_VERSION
        set(CLANG_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
    endif()
    
    # Try standard path first: ${CLANG_ROOT}/lib/clang/${CLANG_VERSION}/include
    set(CLANG_INCLUDE_DIR "${CLANG_ROOT}/lib/clang/${CLANG_VERSION}/include")
    
    # Also try alternative paths if standard path doesn't exist
    if(NOT EXISTS "${CLANG_INCLUDE_DIR}/riscv_vector.h")
        # Try finding it in any version subdirectory
        file(GLOB CLANG_VERSION_DIRS "${CLANG_ROOT}/lib/clang/*/include")
        foreach(CLANG_INC_DIR ${CLANG_VERSION_DIRS})
            if(EXISTS "${CLANG_INC_DIR}/riscv_vector.h")
                set(CLANG_INCLUDE_DIR "${CLANG_INC_DIR}")
                break()
            endif()
        endforeach()
    endif()
    
    if(EXISTS "${CLANG_INCLUDE_DIR}/riscv_vector.h")
        message(STATUS "✓ Found riscv_vector.h at: ${CLANG_INCLUDE_DIR}/riscv_vector.h")
        # Use include_directories with BEFORE to ensure it's searched first
        include_directories(BEFORE "${CLANG_INCLUDE_DIR}")
        message(STATUS "  Added Clang include directory: ${CLANG_INCLUDE_DIR}")
    else()
        message(WARNING "⚠ riscv_vector.h not found. Searched:")
        message(WARNING "    - ${CLANG_ROOT}/lib/clang/${CLANG_VERSION}/include")
        message(WARNING "    - ${CLANG_ROOT}/lib/clang/*/include")
        message(WARNING "  Clang root: ${CLANG_ROOT}, Version: ${CLANG_VERSION}")
        message(WARNING "  This may cause compilation errors. Please ensure Clang 19.1.0 is properly installed.")
    endif()
else()
    # Non-RISC-V platforms (x86, ARM, etc.)
    add_compile_options(
        -O3
        -Wno-unknown-pragmas
        -fopenmp
    )
    # Only use -march=native for native builds (not cross-compilation)
    # AND only if not using Clang with --target flag (which indicates cross-compilation)
    if(NOT CMAKE_CROSSCOMPILING AND NOT CMAKE_CXX_FLAGS MATCHES "--target")
        add_compile_options(-march=native)
    endif()
endif()

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

# Suppress warnings for RISC-V builds
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV")
    target_compile_options(common_ops PRIVATE
        -Wno-macro-redefined
        -Wno-unused-value
    )
endif()

# Link libraries (NUMA_LIB may be empty for cross-compilation)
# For RISC-V cross-compilation, skip x86_64 libraries
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV" OR CMAKE_CXX_FLAGS MATCHES "--target=riscv64")
    # RISC-V cross-compilation: only link RISC-V compatible libraries
    # Skip NUMA_LIB if it's x86_64 (from conda/system)
    if(NUMA_LIB)
        # Check if NUMA_LIB is in conda path (likely x86_64)
        if(NUMA_LIB MATCHES "conda|miniconda")
            message(STATUS "Skipping x86_64 NUMA_LIB for RISC-V: ${NUMA_LIB}")
            target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES})
        else()
            # NUMA_LIB might be RISC-V compatible, use it
            target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} ${NUMA_LIB})
        endif()
    else()
        target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES})
    endif()
else()
    # Native build: use NUMA_LIB as-is
    if(NUMA_LIB)
        target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} ${NUMA_LIB})
    else()
        target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES})
    endif()
endif()

target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})

# For RISC-V, ensure Clang include directory is added to target
# We need to add it as a compile option to ensure it's in the compiler command line
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(DEFINED CLANG_INCLUDE_DIR AND EXISTS "${CLANG_INCLUDE_DIR}/riscv_vector.h")
        target_include_directories(common_ops PRIVATE "${CLANG_INCLUDE_DIR}")
        # Also add as compile option to ensure it's in the command line
        target_compile_options(common_ops PRIVATE "-I${CLANG_INCLUDE_DIR}")
        message(STATUS "Added Clang include directory to target: ${CLANG_INCLUDE_DIR}")
    endif()
endif()

install(TARGETS common_ops
    LIBRARY DESTINATION sgl_kernel
)
