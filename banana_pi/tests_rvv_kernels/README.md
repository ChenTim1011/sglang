# RVV Kernel Tests and Benchmarks

This directory contains scripts for running RVV kernel unit tests and performance benchmarks from the `sgl-kernel` directory.

## Overview

The `tests_benchs_rvv_kernels.sh` script runs pytest-based unit tests and performance benchmarks for the RVV backend implementation. These tests validate kernel correctness and measure performance at the kernel level.

## Script

### `tests_benchs_rvv_kernels.sh`

**Purpose**: Run RVV kernel unit tests and benchmarks from the `sgl-kernel` directory

**What it does**:
1. **Sets up environment**
   - Sources environment configuration (`~/.local_riscv_env/env.sh`)
   - Activates virtual environment
   - Configures OpenMP library paths (`LD_PRELOAD`, `LD_LIBRARY_PATH`)
   - Sets Python path for sglang
   - Navigates to `sgl-kernel/` directory

2. **Runs unit tests** (pytest)
   - Decode attention tests (FP16/BF16, INT8)
   - Extend attention tests (FP16/BF16, INT8)
   - GEMM tests (FP16/BF16, INT8)
   - Backend integration tests

3. **Runs benchmarks**
   - Performance benchmarks for each kernel
   - Optional quick mode for faster testing

**Test Location**: Tests run from `~/.local_riscv_env/workspace/sglang/sgl-kernel/` directory

**Test Files**:
- Unit tests: `sgl-kernel/tests/test_rvv_*.py`
- Benchmarks: `sgl-kernel/benchmark/bench_rvv_*.py`

## Usage

### Basic Usage

```bash
# Run all tests and benchmarks
./tests_benchs_rvv_kernels.sh

# Show help
./tests_benchs_rvv_kernels.sh --help
```

### Test Selection

```bash
# Run only tests (skip benchmarks)
./tests_benchs_rvv_kernels.sh --test-only

# Run only benchmarks (skip tests)
./tests_benchs_rvv_kernels.sh --bench-only

# Run specific kernel
./tests_benchs_rvv_kernels.sh --decode
./tests_benchs_rvv_kernels.sh --extend
./tests_benchs_rvv_kernels.sh --gemm
```

### Granular Control

```bash
# Run only decode attention tests
./tests_benchs_rvv_kernels.sh --test-decode

# Run only decode attention benchmarks
./tests_benchs_rvv_kernels.sh --bench-decode

# Run only extend attention tests
./tests_benchs_rvv_kernels.sh --test-extend

# Run only extend attention benchmarks
./tests_benchs_rvv_kernels.sh --bench-extend

# Run only GEMM tests
./tests_benchs_rvv_kernels.sh --test-gemm

# Run only GEMM benchmarks
./tests_benchs_rvv_kernels.sh --bench-gemm

# Run only backend integration tests
./tests_benchs_rvv_kernels.sh --test-backend
```

### Performance Options

```bash
# Quick mode (fewer iterations, faster)
./tests_benchs_rvv_kernels.sh --quick

# Combine options
./tests_benchs_rvv_kernels.sh --decode --test-only --quick
```

## Prerequisites

**⚠️ Important: Before running tests, you must set up the environment:**

```bash
# Set OpenMP library paths
export LD_PRELOAD=~/.local/lib/libomp.so
export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH

# Activate virtual environment
source ~/.local_riscv_env/workspace/venv_sglang/bin/activate
```

**Note**: The script automatically sets up these environment variables, but you may need to ensure:
- `~/.local_riscv_env/env.sh` exists (created by `setup_banana_pi.sh`)
- Virtual environment is set up at `~/.local_riscv_env/workspace/venv_sglang/`
- `sgl-kernel` directory exists at `~/.local_riscv_env/workspace/sglang/sgl-kernel/`

## Test vs. End-to-End Tests

This script runs **kernel-level unit tests** and benchmarks. For **end-to-end application tests**, see:
- `banana_pi/tests/run_tests.sh` - Runs end-to-end tests in `banana_pi/tests/` directory

**Difference**:
- **Kernel tests** (`tests_benchs_rvv_kernels.sh`): Validate individual kernel functions (decode, extend, GEMM) using pytest
- **End-to-end tests** (`tests/run_tests.sh`): Validate complete INT8 KV cache implementation, model accuracy, memory bandwidth

## Output

The script provides:
- Colored output for easy reading
- Test pass/fail summary
- Benchmark results
- Final summary with test counts

## Troubleshooting

### Environment Not Found

If you see:
```
ERROR: ~/.local_riscv_env/env.sh not found
```

**Solution**: Run `setup_banana_pi.sh` from the `install/` directory to set up the environment.

### Virtual Environment Not Found

If you see:
```
ERROR: Virtual environment not found
```

**Solution**: Ensure `setup_banana_pi.sh` completed successfully and created the virtual environment.

### Test Files Not Found

If pytest cannot find test files:
- Ensure `sgl-kernel` is cloned and built
- Check that test files exist in `sgl-kernel/tests/`
- Verify you're running from the correct location

### OpenMP Errors

If you see OpenMP-related errors:
- Verify `~/.local/lib/libomp.so` exists
- Check `LD_PRELOAD` and `LD_LIBRARY_PATH` are set correctly
- The script sets these automatically, but you can set them manually if needed

## Related Documentation

- `banana_pi/README.md` - Overall project documentation
- `banana_pi/tests/README.md` - End-to-end tests documentation
- `banana_pi/install/README.md` - Installation scripts documentation
