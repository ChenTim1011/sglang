# Banana Pi RISC-V Setup and Testing

This directory contains scripts and test suites for building, deploying, and testing the RVV backend implementation for SGLang on Banana Pi BPI-F3 SpacemiT K1.

## Directory Structure

### (RVV)`test_tinyllama_rvv/`
**Purpose**: TinyLlama 1.1B model testing and backend comparison

**Contents**:
- `benchmark_rvv_backends.py` - Compare RVV vs TORCH_NATIVE backend performance
- `test_tinyllama_rvv.py` - Interactive chat interface with TinyLlama
- `test_environment.py` - Environment validation script
- Configuration files are generated dynamically by scripts (no static config file needed)
- `triton_stub.py` & `vllm_stub.py` - Stub modules for RISC-V compatibility

**Usage**: See `test_tinyllama_rvv/README.md` for detailed documentation.

---

### (RVV)`tests/`
**Purpose**: End-to-end validation and benchmarking on Banana Pi

**Contents**:
- `test_end_to_end_int8.py` - End-to-end INT8 KV cache validation
- `test_model_accuracy_int8.py` - Model accuracy tests (perplexity, token prediction)
- `test_parametrized_int8.py` - Parameterized testing across configurations
- `test_memory_bandwidth.py` - Memory bandwidth analysis
- `profile_paged_attention.py` - Paged attention performance profiling
- `test_vlen_alignment.py` - VLEN alignment validation
- `test_utils.py` - Shared utility functions
- `run_tests.sh` - Test runner for end-to-end tests (runs Python test scripts)

**Usage**: See `tests/README.md` for detailed documentation.

**Note**: This directory contains application-level end-to-end tests. For kernel-level unit tests, see `tests_rvv_kernels/`.

---

### (RVV)`tests_rvv_kernels/`
**Purpose**: Kernel-level unit tests and benchmarks

**Contents**:
- `tests_benchs_rvv_kernels.sh` - Test runner for sgl-kernel unit tests and benchmarks (runs pytest)

**Usage**: See `tests_rvv_kernels/README.md` for detailed documentation.

**Note**:
- Runs pytest tests and benchmarks from `sgl-kernel/` directory
- Tests individual kernel functions (decode, extend, GEMM)
- Must be run on Banana Pi (not from x86_64 host)

---

## Scripts

### `install/` (X86 host)
**Purpose**: Installation and deployment scripts

**Contents**:
- `build_and_deploy_sgl-kernel.sh` - Build and deploy sgl-kernel wheel to Banana Pi
- `setup_banana_pi.sh` - Set up SGLang environment on Banana Pi remotely
- `build_and_deploy_sgl-kernel.config.sh` - User-specific configuration (optional)

**Usage**: See `install/README.md` for detailed documentation.

**Note**: All scripts in this directory are designed to run from an **x86_64 host** machine.

---

## Workflow

### Initial Setup
1. **Build and deploy kernel** (from x86_64 host):
   ```bash
   cd install
   ./build_and_deploy_sgl-kernel.sh --yes
   ```

2. **Set up SGLang environment** (from x86_64 host):
   ```bash
   cd install
   ./setup_banana_pi.sh --yes
   ```

### Testing

**⚠️ Important: Before running any tests, you must set up the environment variables:**

```bash
# Set OpenMP library paths
export LD_PRELOAD=~/.local/lib/libomp.so
export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH

# Activate virtual environment
source ~/.local_riscv_env/workspace/venv_sglang/bin/activate
```

**Note**: These environment variables are required for all tests and benchmarks. You can add them to your `~/.bashrc` to make them persistent:
```bash
echo 'export LD_PRELOAD=~/.local/lib/libomp.so' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'source ~/.local_riscv_env/workspace/venv_sglang/bin/activate' >> ~/.bashrc
```

1. **Run sgl-kernel unit tests and benchmarks** (on Banana Pi):
   ```bash
   cd tests_rvv_kernels
   ./tests_benchs_rvv_kernels.sh
   ```

2. **Run end-to-end tests** (on Banana Pi):
   ```bash
   cd tests
   ./run_tests.sh
   ```

3. **Run TinyLlama benchmark** (on Banana Pi):
   ```bash
   cd test_tinyllama_rvv
   python benchmark_rvv_backends.py
   ```

---

## Dependencies

### On x86_64 Host (for building)
- CMake (>=3.25)
- Ninja build system
- Git
- Python 3 with `build`, `scikit-build-core`, `wheel`
- Clang 19+ (or let script build it)
- SSH/SCP access to Banana Pi

### On Banana Pi (for running)
- Python 3
- Virtual environment (created by `setup_banana_pi.sh`)
- OpenMP library (`libomp.so` - installed by `setup_banana_pi.sh`)
- SGLang dependencies (installed by `setup_banana_pi.sh`)

---

## Configuration

### Environment Variables
- `BANANA_PI_USER` - Banana Pi username (default: `jtchen`)
- `BANANA_PI_HOST` - Banana Pi host/IP (default: `140.114.78.64`)
- `GITHUB_TOKEN` - GitHub Personal Access Token (for private repos)
- `CLANG19_TOOLCHAIN_DIR` - Custom Clang 19 installation path
- `TORCH_PY_PREFIX` - Custom PyTorch RISC-V installation path

### Configuration Files
- `build_and_deploy_sgl-kernel.config.sh` - User-specific build configuration (optional)
- Configuration files are generated dynamically (see `benchmark_rvv_backends.py`)

---

## Troubleshooting

### Build Issues
- **Clang 19 not found**: Run `build_and_deploy_sgl-kernel.sh` without `--skip-clang-build`
- **PyTorch RISC-V not found**: Script will download automatically, or set `TORCH_PY_PREFIX`
- **Cross-compilation fails**: Check Clang 19 installation and RISC-V sysroot

### Deployment Issues
- **SSH connection fails**: Check SSH keys or use password authentication
- **Wheel transfer fails**: Verify network connectivity and disk space
- **Installation fails**: Check virtual environment and Python version

### Runtime Issues
- **OpenMP errors**: Verify `~/.local/lib/libomp.so` exists and `LD_PRELOAD` is set
- **Import errors**: Check virtual environment is activated and dependencies installed
- **Backend not found**: Ensure RVV backend is registered in SGLang's attention registry

---

## Quick Reference

| Task | Command |
|------|---------|
| Build and deploy kernel (x86 host) | `cd install && ./build_and_deploy_sgl-kernel.sh --yes` |
| Set up Banana Pi (x86 host) | `cd install && ./setup_banana_pi.sh --yes` |
| **Set environment** (required before tests, on Banana Pi) | `export LD_PRELOAD=~/.local/lib/libomp.so && export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH && source ~/.local_riscv_env/workspace/venv_sglang/bin/activate` |
| Run sgl-kernel unit tests (Banana Pi) | `cd tests_rvv_kernels && ./tests_benchs_rvv_kernels.sh` |
| Run end-to-end tests (Banana Pi) | `cd tests && ./run_tests.sh` |
| Compare backends (Banana Pi) | `cd test_tinyllama_rvv && python benchmark_rvv_backends.py` |

---

## Documentation

- `install/README.md` - Installation and deployment scripts documentation
- `test_tinyllama_rvv/README.md` - TinyLlama testing documentation
- `tests/README.md` - End-to-end testing documentation
- `tests_rvv_kernels/README.md` - Kernel unit tests and benchmarks documentation
