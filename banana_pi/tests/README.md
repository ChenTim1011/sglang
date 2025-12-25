# Tests Directory

This directory contains test scripts for validating and benchmarking the RVV backend implementation for SGLang.

## Test Files

### `test_end_to_end_int8.py`
**Purpose**: End-to-end validation of INT8 KV cache implementation

**What it tests**:
- Prefill (Extend) phase with INT8 KV cache
- Decode phase with INT8 KV cache
- Accuracy comparison between FP16 and INT8 (cosine similarity, MSE, max error)
- Performance metrics (TTFT, ITL, TPOT, throughput)

**Usage**:
```bash
python test_end_to_end_int8.py
python test_end_to_end_int8.py --num-requests 4 --seq-len 256
```

---

### `test_model_accuracy_int8.py`
**Purpose**: Model-level accuracy validation for INT8 quantization

**What it tests**:
- Attention kernel accuracy (FP16 vs INT8 output comparison)
- Simulated model perplexity and token prediction accuracy
- Quantization impact on model quality

**Usage**:
```bash
python test_model_accuracy_int8.py
python test_model_accuracy_int8.py --num-heads 32 --head-dim 64
```

---

### `test_parametrized_int8.py`
**Purpose**: Parameterized testing across multiple configurations

**What it tests**:
- Decode attention accuracy across various configurations:
  - Sequence lengths: 32, 64, 128, 256, 512, 1024
  - Batch sizes: 1, 2, 4, 8
  - Head dimensions: 32, 64, 128
  - Number of heads: 8, 16, 32

**Usage**:
```bash
python test_parametrized_int8.py
python test_parametrized_int8.py --quick  # Fewer configurations
```

---

### `test_memory_bandwidth.py`
**Purpose**: Memory bandwidth analysis for INT8 vs FP16 KV cache

**What it tests**:
- Estimated memory bandwidth usage
- Decoding throughput (tokens/second)
- Correlation between bandwidth reduction and throughput improvement
- Memory efficiency gains from INT8 quantization

**Usage**:
```bash
python test_memory_bandwidth.py
python test_memory_bandwidth.py --seq-len 2048 --batch-size 8
```

---

### `profile_paged_attention.py`
**Purpose**: Performance profiling for paged attention with INT8 KV cache

**What it tests**:
- Paged attention performance (simulated paged KV cache)
- FP16 vs INT8 throughput comparison
- Memory access patterns and efficiency

**Usage**:
```bash
python profile_paged_attention.py
```

---

### `test_vlen_alignment.py`
**Purpose**: VLEN (vector length) alignment validation

**What it tests**:
- RVV vector length bindings
- Head dimension alignment with VLEN
- Backend logic for different VLEN configurations

**Usage**:
```bash
python test_vlen_alignment.py
```

---

### `test_utils.py`
**Purpose**: Shared utility functions for testing

**What it provides**:
- `measure_with_statistics()`: Statistical measurement with mean, std, CI, min, max, median
- `check_system_state()`: System state checking (CPU frequency, utilization, memory)
- `generate_fair_kv_buffers()`: Fair KV buffer generation for FP16/INT8 comparison
- `compare_tensors_fair()`: Fair tensor comparison with cosine similarity, MSE, max error

**Usage**: Imported by other test scripts

---

## Running Tests

### Test Scripts Overview

This directory contains the **end-to-end test runner** (`run_tests.sh`).

**For kernel-level unit tests and benchmarks**, see `banana_pi/tests_rvv_kernels/` directory.

**Difference**:
- **`run_tests.sh`** (this directory): Application-level end-to-end validation
  - Tests: `test_end_to_end_int8.py`, `test_model_accuracy_int8.py`, etc.
  - Purpose: Validate INT8 KV cache implementation, model accuracy, memory bandwidth
  - Location: Runs in `banana_pi/tests/` directory

- **`tests_benchs_rvv_kernels.sh`** (`tests_rvv_kernels/` directory): Kernel-level unit tests and benchmarks
  - Tests: `sgl-kernel/tests/test_rvv_*.py` (pytest)
  - Benchmarks: `sgl-kernel/benchmark/bench_rvv_*.py`
  - Purpose: Validate kernel correctness and measure performance
  - Location: Runs in `sgl-kernel/` directory

### Using the End-to-End Test Runner (`run_tests.sh`)

The `run_tests.sh` script provides a convenient way to run all or specific end-to-end tests:

```bash
# Show help
./run_tests.sh --help

# List all available tests
./run_tests.sh --list

# Run all tests
./run_tests.sh

# Run specific test
./run_tests.sh --end-to-end
./run_tests.sh --model-accuracy
./run_tests.sh --parametrized

# Run multiple tests
./run_tests.sh --end-to-end --model-accuracy

# Run with custom arguments
./run_tests.sh --end-to-end -- --num-requests 4 --seq-len 256

# Run in quick mode (fewer iterations)
./run_tests.sh --quick
```

**Note**: The script automatically sets up:
- `LD_PRELOAD=~/.local/lib/libomp.so`
- `LD_LIBRARY_PATH=~/.local/lib`
- Virtual environment activation

### Running Tests Manually

To run tests individually:

```bash
# Basic tests
python test_end_to_end_int8.py
python test_model_accuracy_int8.py
python test_parametrized_int8.py --quick

# Performance analysis
python test_memory_bandwidth.py
python profile_paged_attention.py

# System validation
python test_vlen_alignment.py
```

---

### Kernel-Level Tests

For kernel-level unit tests and benchmarks, see the `tests_rvv_kernels/` directory:
- Location: `banana_pi/tests_rvv_kernels/`
- Script: `tests_benchs_rvv_kernels.sh`
- Documentation: `banana_pi/tests_rvv_kernels/README.md`

---

## Dependencies

All tests require:
- `torch` (PyTorch)
- `sgl_kernel` (compiled SGLang kernel with RVV support)
- `numpy` (optional, for better statistics)
- `test_utils` (local module)

Some tests may require:
- `psutil` (for system state checking)
- `requests` (for HTTP-based benchmarks)

---

## Notes

- All tests use `test_utils.py` for consistent statistical analysis and fair comparisons
- Tests are designed to be run on RISC-V hardware (Banana Pi K1)
- INT8 tests ensure symmetric quantization (signed int8, no +128 offset) for KV cache
- Performance tests include warmup iterations and statistical analysis for reliable results
