# Banana Pi RISC-V Setup and Testing

This directory contains scripts and test suites for building, deploying, and testing the RVV backend implementation for SGLang on Banana Pi BPI-F3 SpacemiT K1.

## Directory Structure

### (RVV)`test_tinyllama_rvv/`
**Purpose**: End-to-end testing of TinyLlama 1.1B model with RVV backend on Banana Pi

**Contents**:
- `benchmark_rvv_backends.py` - Compare RVV vs TORCH_NATIVE backend performance
- `test_tinyllama_rvv.py` - Interactive chat interface with TinyLlama
- `manage_rvv_env.py` - Consolidated environment manager (installs stubs, verifies env)
- `launch_server_rvv.py` - Shared server launcher wrapper
- `bench_endtoend.sh` - End-to-end benchmark script (moved from `tests_rvv_kernels/`)
- Stub modules (installed by `manage_rvv_env.py`)

**Usage**: See `test_tinyllama_rvv/README.md` for detailed documentation.

---

### (RVV)`tests/`
**Purpose**: vlen tests and memory profiling on Banana Pi

**Contents**:
- `test_memory_bandwidth.py` - Memory bandwidth analysis
- `test_vlen.py` - vlen tests
- `test_utils.py` - Shared utility functions

**Usage**: See `tests/README.md` for detailed documentation.

---

### (RVV)`tests_rvv_kernels/`
**Purpose**: Kernel-level unit tests and benchmarks

**Contents**:
- `tests_benchs_rvv_kernels.sh` - Test runner for sgl-kernel unit tests and benchmarks

**Usage**: See `tests_rvv_kernels/README.md` for detailed documentation.

---

## Scripts

### `install/` (X86 host)
**Purpose**: Installation and deployment scripts

**Contents**:
- `build_and_deploy_sgl-kernel.sh` - Build and deploy sgl-kernel wheel to Banana Pi
- `setup_banana_pi.sh` - Set up SGLang environment on Banana Pi remotely
- `build_and_deploy_sgl-kernel.config.sh` - User-specific configuration (optional)

**Usage**: See `install/README.md` for detailed documentation.

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

**⚠️ Important: Before running any tests, source the environment settings:**

```bash
# Sourcing this script sets up LD_PRELOAD, LD_LIBRARY_PATH, and the virtual environment
source banana_pi/environment_setting.sh
```

1. **Run TinyLlama benchmark** (on Banana Pi):
   ```bash
   cd test_tinyllama_rvv
   # Check environment
   python3 manage_rvv_env.py
   # Run benchmarks
   python3 benchmark_rvv_backends.py
   # Run end-to-end suite
   ./bench_endtoend.sh --bench-serving
   ```

2. **Run sgl-kernel tests** (on Banana Pi):
   ```bash
   cd tests_rvv_kernels
   ./tests_benchs_rvv_kernels.sh
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
- OpenMP library (`libomp.so`)
- SGLang dependencies

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
