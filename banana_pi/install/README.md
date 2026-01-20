# Installation Scripts

This directory contains scripts for building, deploying, and setting up the SGLang RVV backend on Banana Pi.

## Scripts

### `build_and_deploy_sgl-kernel.sh` (X86 host)
**Purpose**: Complete automation for building and deploying sgl-kernel wheel to Banana Pi

**What it does**:
1. **Build Clang 19 RISC-V toolchain** (if needed)
   - Compiles Clang 19 with RISC-V Vector Extension support
   - Installs to `~/tools/clang19-riscv`
   - Required for cross-compiling RISC-V code

2. **Build sgl-kernel wheel**
   - Cross-compiles sgl-kernel for RISC-V architecture
   - Uses Clang 19 with RVV support
   - Downloads PyTorch RISC-V from GitHub Releases (if needed)
   - Creates wheel file: `sgl_kernel-*.whl`

3. **Transfer to Banana Pi**
   - Uploads wheel file via SCP
   - Syncs `banana_pi/` directory to remote server

4. **Install on Banana Pi**
   - Installs wheel in virtual environment
   - Verifies installation and imports

**Usage**:
```bash
# Full workflow (interactive)
./build_and_deploy_sgl-kernel.sh

# Non-interactive (skip confirmations)
./build_and_deploy_sgl-kernel.sh --yes

# Skip specific steps
./build_and_deploy_sgl-kernel.sh --skip-clang-build --skip-build
./build_and_deploy_sgl-kernel.sh --skip-transfer --skip-install

# Custom target
./build_and_deploy_sgl-kernel.sh --user jtchen --host 140.114.78.64
```

**Prerequisites**:
- CMake, Ninja, Git (for Clang build)
- SSH access to Banana Pi
- GitHub token (for private repository access)

---

### `setup_banana_pi.sh` (X86 host)
**Purpose**: Set up SGLang environment on Banana Pi remotely

**What it does**:
1. **Clone/Update SGLang repository**
   - Clones `pllab-sglang/rvv_backend` branch
   - Handles merge conflicts and untracked files
   - Sets up workspace at `~/.local_riscv_env/workspace/sglang`

2. **Download and install wheels**
   - Downloads Python wheels from GitHub Releases
   - Installs PyTorch, NumPy, and other dependencies
   - Sets up OpenMP library (`libomp.so`)

3. **Install Python dependencies**
   - Installs packages from `wheel_builder` (pre-compiled for RISC-V)
   - Installs additional packages from PyPI
   - Installs SGLang in development mode

4. **Configure environment**
   - Sets up virtual environment
   - Configures OpenMP library paths
   - Installs triton/vllm stubs for subprocess compatibility

**Usage**:
```bash
# Interactive setup
./setup_banana_pi.sh

# Non-interactive
./setup_banana_pi.sh --yes

# Skip wheel installation (if already installed)
./setup_banana_pi.sh --skip-wheels

# Custom target
./setup_banana_pi.sh --user changeuser --host changehost
```

**Prerequisites**:
- SSH access to Banana Pi
- GitHub token (for private repository and releases)

---

### `build_and_deploy_sgl-kernel.config.sh` (Optional)
**Purpose**: User-specific configuration file for `build_and_deploy_sgl-kernel.sh`

**What it does**:
- Allows users to set custom paths for Clang toolchain, sysroot, etc.
- Automatically loaded by `build_and_deploy_sgl-kernel.sh` if present
- This file is ignored by git, so personal paths won't be committed

**Usage**:
```bash
# Copy example and customize
cp build_and_deploy_sgl-kernel.config.sh.example build_and_deploy_sgl-kernel.config.sh

# Edit with your paths
nano build_and_deploy_sgl-kernel.config.sh
```

**Example configuration**:
```bash
#!/bin/bash
# Alternative Clang toolchain directory
ALT_CLANG_TOOLCHAIN_DIR="/path/to/clang/toolchain"

# RISC-V sysroot path
RISCV_SYSROOT="/path/to/riscv/sysroot"
```

---

## Workflow

### Initial Setup (from x86_64 host)

1. **Build and deploy kernel**:
   ```bash
   cd install
   ./build_and_deploy_sgl-kernel.sh --yes
   ```

2. **Set up SGLang environment**:
   ```bash
   cd install
   ./setup_banana_pi.sh --yes
   ```

### After Setup (on Banana Pi)

After setup is complete, you can run tests on Banana Pi:
- Kernel-level tests: `tests_rvv_kernels/tests_benchs_rvv_kernels.sh`
- End-to-end tests: `tests/run_tests.sh`
- Model tests: `test_tinyllama_rvv/benchmark_rvv_backends.py`

---

## Notes

- All scripts in this directory are designed to run from an **x86_64 host** machine
- They use SSH/SCP to interact with the Banana Pi remotely
- The scripts handle environment setup, dependency installation, and configuration automatically
- For testing scripts that run on Banana Pi, see the `tests/` and `test_tinyllama_rvv/` directories
