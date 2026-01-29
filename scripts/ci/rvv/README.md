# GitHub Actions Self-hosted Runner for RVV Testing

This directory contains scripts and configuration for running GitHub Actions self-hosted runner on Banana Pi BPI-F3 (RISC-V 64) for SGLang RVV 1.0 regression testing.

## Overview

- **Runner**: GitHub Actions self-hosted runner on BPI-F3
- **Container**: Podman-based execution (rootless, no sudo)
- **Image**: `docker.io/juitingchen/sglang-rvv:v1.0`
- **Label**: `rvv-bpi-f3`

## Quick Start (BPI-F3 Setup)

### 1. Deploy Runner

```bash
# SSH to BPI-F3 with proper LD_LIBRARY_PATH
LD_LIBRARY_PATH= ssh jtchen@140.114.78.64

# Get runner token from:
# https://github.com/ChenTim1011/sglang/settings/actions/runners/new

# Download and run setup script
cd ~/sglang
bash scripts/ci/rvv/setup_bpi_f3_runner.sh <YOUR_TOKEN>

# Start runner
cd ~/actions-runner
./run.sh
```

### 2. Verify Runner

Visit: https://github.com/ChenTim1011/sglang/settings/actions/runners

You should see **BPI-F3-RVV-Runner** with status "Idle" (green).

### 3. Test Workflow

1. Visit: https://github.com/ChenTim1011/sglang/actions/workflows/pr-test-rvv.yml
2. Click "Run workflow"
3. Select branch: `rvv_backend`
4. Choose test level: `basic`
5. Click "Run workflow"

Monitor execution in the Actions UI and on BPI-F3:
```bash
# On BPI-F3
tail -f ~/actions-runner/_diag/Runner_*.log
watch -n 2 'podman ps -a'
```

## Scripts

### `setup_bpi_f3_runner.sh`
Downloads, configures, and registers GitHub Actions runner on BPI-F3.

**Usage**:
```bash
bash setup_bpi_f3_runner.sh <RUNNER_TOKEN>
```

### `rvv_ci_start_container.sh`
Starts Podman container with SGLang RVV image.

**Environment Variables**:
- `GITHUB_WORKSPACE`: Workspace directory (auto-set by GitHub Actions)
- `PODMAN_IMAGE`: Container image (default: `docker.io/juitingchen/sglang-rvv:v1.0`)
- `CONTAINER_NAME`: Container name (default: `ci_sglang_rvv`)

### `rvv_ci_install_dependency.sh`
Installs Python dependencies and builds `sgl-kernel` inside container.

### `rvv_ci_exec.sh`
Wrapper for executing commands inside container.

**Usage**:
```bash
bash rvv_ci_exec.sh -w /workspace/sglang/test python3 -m pytest test_rvv.py
```

### `verify_rvv_support.py`
Verifies RISC-V CPU, RVV extension, and `sgl_kernel` availability.

**Usage**:
```bash
python3 verify_rvv_support.py
```

## Workflow

`.github/workflows/pr-test-rvv.yml` defines 4 test jobs:

1. **rvv-basic-test**: Verify RVV support (always runs)
2. **rvv-kernel-unit-test**: Run `sgl-kernel` RVV kernel tests
3. **rvv-backend-integration-test**: Run SGLang RVV backend tests
4. **rvv-e2e-test**: End-to-end inference with Llama 1B (manual only)

### Manual Trigger Options

- `basic`: Only verification
- `kernel`: Verification + kernel tests
- `integration`: Verification + kernel + backend tests
- `full`: All tests including E2E

### Auto-Trigger on PR

Workflow auto-triggers on PRs that modify:
- `python/sglang/srt/layers/attention/rvv_backend.py`
- `sgl-kernel/csrc/cpu/rvv/**`
- `test/srt/cpu/rvv/**`
- `scripts/ci/rvv/**`

## Troubleshooting

### Runner Not Connecting

```bash
# Check runner logs
cat ~/actions-runner/_diag/Runner_*.log

# Re-register runner
cd ~/actions-runner
./config.sh remove
bash ~/sglang/scripts/ci/rvv/setup_bpi_f3_runner.sh <NEW_TOKEN>
```

### Container Fails to Start

```bash
# Check Podman status
podman ps -a
podman logs ci_sglang_rvv

# Remove stale container
podman rm -f ci_sglang_rvv

# Pull latest image
podman pull docker.io/juitingchen/sglang-rvv:v1.0
```

### RVV Not Detected

```bash
# Check CPU info
cat /proc/cpuinfo | grep isa

# Should show: isa : rv64imafdcv or similar with 'v' extension
```

## Security

**Public Repository Considerations**:

This workflow runs on a **public fork** (`ChenTim1011/sglang`). Since GitHub does not allow converting forks to private, we rely on:

1. **Manual trigger only** for testing (no auto-run from external PRs)
2. **Path-based filtering** (only runs when RVV code changes)
3. **Container isolation** (`--security-opt no-new-privileges`, restricted networking)
4. **Resource limits** (8 CPUs, 16GB RAM max)

**Recommended**: When pushing to upstream (`sgl-project/sglang`), ensure PR gate workflow is enabled to require `run-ci` label and manual approval.

## Maintenance

### Updating Runner

```bash
# SSH to BPI-F3
cd ~/actions-runner
./svc.sh stop  # If running as service
rm -rf *
bash ~/sglang/scripts/ci/rvv/setup_bpi_f3_runner.sh <NEW_TOKEN>
```

### Updating Container Image

When `docker.io/juitingchen/sglang-rvv:v1.0` is updated:

```bash
# On BPI-F3
podman pull docker.io/juitingchen/sglang-rvv:v1.0

# No other changes needed; workflow will use new image
```

## Contact

For issues with BPI-F3 runner, contact: jtchen@140.114.78.64
