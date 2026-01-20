# Tests Directory

This directory contains test scripts for testing vlen and memory profiling.
## Test Files

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

### Running Tests Manually

To run tests individually:

```bash
# Performance analysis
python test_memory_bandwidth.py

# System validation
python test_vlen_alignment.py
```
---

### Kernel-Level Tests

For kernel-level unit tests and benchmarks, see the `tests_rvv_kernels/` directory:
- Location: `banana_pi/tests_rvv_kernels/`
- Script: `tests_benchs_rvv_kernels.sh`
- Documentation: `banana_pi/tests_rvv_kernels/README.md`
