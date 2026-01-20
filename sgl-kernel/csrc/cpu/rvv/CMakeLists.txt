project(sgl_kernel_rvv)

# Find Python again since we are in a new scope and it might not have been found yet
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)


# PyTorch Discovery
set(IS_RISCV_BUILD ON)

if(DEFINED ENV{TORCH_PY_PREFIX})
    # --- Cross-Compilation Path ---
    set(TORCH_PY_PREFIX $ENV{TORCH_PY_PREFIX})
    if(EXISTS "${TORCH_PY_PREFIX}/lib/libtorch.so")
        set(TORCH_INSTALL_PREFIX "${TORCH_PY_PREFIX}")
    else()
        set(TORCH_INSTALL_PREFIX "${TORCH_PY_PREFIX}/torch")
    endif()
    message(STATUS "Cross-compiling with PyTorch at: ${TORCH_INSTALL_PREFIX}")

    # Set CMake search paths
    set(TORCH_CMAKE_BASE "${TORCH_INSTALL_PREFIX}/share/cmake")
    if(EXISTS "${TORCH_CMAKE_BASE}/Torch/TorchConfig.cmake")
        set(Torch_DIR "${TORCH_CMAKE_BASE}/Torch")
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_BASE}")
    endif()
    if(EXISTS "${TORCH_CMAKE_BASE}/Caffe2/Caffe2Config.cmake")
        set(Caffe2_DIR "${TORCH_CMAKE_BASE}/Caffe2")
    endif()

    set(TORCH_LIBRARY "${TORCH_INSTALL_PREFIX}/lib/libtorch.so" CACHE FILEPATH "PyTorch library" FORCE)
    find_package(Torch REQUIRED)
else()
    # --- Standard / Native Path ---
    execute_process(
        COMMAND ${Python_EXECUTABLE}
                -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_PY_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "PyTorch Prefix: ${TORCH_PY_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
    find_package(Torch REQUIRED)
endif()


# Compiler Flags for RVV
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
     message(WARNING "RISC-V build usually requires Clang for V-extension intrinsics. You are using: ${CMAKE_CXX_COMPILER_ID}")
else()
     if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.0)
          message(FATAL_ERROR "SGLang RVV backend requires Clang >= 19.0 for modern vector intrinsics. "
                              "Found Clang version: ${CMAKE_CXX_COMPILER_VERSION}. "
                              "Please unset 'CC'/'CXX' or point them to Clang 19+.")
     endif()
endif()
add_compile_options(-O3 -march=rv64gcv_zvfh -mabi=lp64d -fopenmp)

# Definitions specific to RISC-V
add_compile_definitions(CPU_CAPABILITY_RVV)
add_compile_definitions(SGLANG_RISCV_NO_SHM)
add_compile_definitions(SGLANG_RISCV_NO_NUMA)

# Optional FTZ optimization
if(NOT DEFINED ENV{SGLANG_CPU_FP8_CVT_FTZ})
    set(ENV{SGLANG_CPU_FP8_CVT_FTZ} "1")
endif()
if("$ENV{SGLANG_CPU_FP8_CVT_FTZ}" STREQUAL "1")
    add_compile_definitions(SGLANG_CPU_FP8_CVT_FTZ)
endif()


# Source File Selection
# Start with parent directory sources
file(GLOB PARENT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../*.cpp")
file(GLOB MAMBA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../mamba/*.cpp")
file(GLOB MODEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.cpp")
list(APPEND SOURCES ${PARENT_SOURCES} ${MAMBA_SOURCES} ${MODEL_SOURCES})

# Exclude unsupported generic files
list(FILTER SOURCES EXCLUDE REGEX "shm\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX "interface\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX "numa_utils\\.cpp$")

# Collect RVV specific sources
file(GLOB RVV_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Merge and Override
if(RVV_SOURCES)
    list(APPEND SOURCES ${RVV_SOURCES})
    foreach(rvv_src ${RVV_SOURCES})
        get_filename_component(ver_name ${rvv_src} NAME)
        # Check if there is a corresponding generic file in the parent directory to override
        string(REPLACE "." "\\." ver_name_esc "${ver_name}")
        list(FILTER SOURCES EXCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/../${ver_name_esc}$")
    endforeach()
endif()


# Include Directories
# We need to explicitly add the parent's parent (csrc root) and other paths
# since we are now in a subdirectory
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CONDA_INCLUDE_DIR}
    ${VENV_INCLUDE_DIR}
)

if(CMAKE_SYSROOT)
    file(GLOB PY_INC "${CMAKE_SYSROOT}/usr/include/python*")
    if(PY_INC)
        include_directories(${PY_INC})
    endif()
endif()

# Check for riscv_vector.h path for Clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    get_filename_component(CLANG_ROOT "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(CLANG_ROOT "${CLANG_ROOT}" DIRECTORY)
    file(GLOB_RECURSE RVV_HEADER "${CLANG_ROOT}/lib/clang/*/include/riscv_vector.h")
    if(RVV_HEADER)
        get_filename_component(RVV_HEADER_DIR "${RVV_HEADER}" DIRECTORY)
        include_directories(BEFORE "${RVV_HEADER_DIR}")
    endif()
endif()


# Build Target
Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

# RISC-V OpenMP Support
# Strategies to find OpenMP library for RISC-V Cross-Compilation:
# 1. Check env var RISCV_OMP_LIB_PATH (useful if libomp is built outside sysroot)
# 2. Standard find_library (works if libomp is installed in sysroot)

set(OMP_LIB_PATH "$ENV{RISCV_OMP_LIB_PATH}")

if(EXISTS "${OMP_LIB_PATH}")
    message(STATUS "Found OpenMP lib via RISCV_OMP_LIB_PATH: ${OMP_LIB_PATH}")
    target_link_libraries(common_ops PRIVATE "${OMP_LIB_PATH}")
else()
    # Fallback to default search
    find_library(OMP_LIB omp)
    if(OMP_LIB)
        message(STATUS "Found OpenMP lib via find_library: ${OMP_LIB}")
        target_link_libraries(common_ops PRIVATE ${OMP_LIB})
    else()
        message(WARNING "Could not find libomp. Linking might fail. \
        Consider setting RISCV_OMP_LIB_PATH environment variable.")
    endif()
endif()

# RISC-V build: Link PyTorch
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES})
target_compile_options(common_ops PRIVATE -fopenmp)

target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})

install(TARGETS common_ops LIBRARY DESTINATION sgl_kernel)
