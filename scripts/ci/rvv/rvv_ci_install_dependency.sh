#!/bin/bash
# Install dependencies for RVV CI testing inside container
# Should be run inside the Podman container
#
# Usage:
#   podman exec ci_sglang_rvv bash scripts/ci/rvv/rvv_ci_install_dependency.sh

set -e

CONTAINER_NAME="${CONTAINER_NAME:-ci_sglang_rvv}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if running inside container
if [ ! -f /.dockerenv ] && [ ! -f /run/.containerenv ]; then
    log_warn "Not running inside container. Executing via podman exec..."
    podman exec "${CONTAINER_NAME}" bash /workspace/sglang/scripts/ci/rvv/rvv_ci_install_dependency.sh
    exit $?
fi

log_info "Installing RVV CI dependencies..."

# Update package lists
log_info "Updating package lists..."
apt-get update -qq

# Install build dependencies
log_info "Installing build essentials..."
apt-get install -y -qq \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3-pip \
    python3-dev

# Upgrade pip
log_info "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install Python dependencies
log_info "Installing Python dependencies..."
cd /workspace/sglang/python
python3 -m pip install -e . -v

# Build and install sgl-kernel
log_info "Building sgl-kernel..."
cd /workspace/sglang/sgl-kernel

# Build with CPU backend
python3 setup.py build_ext --inplace
python3 -m pip install -e . -v

# Verify installation
log_info "Verifying RVV support..."
python3 -c "
import platform
import sys

print(f'Python: {sys.version}')
print(f'Platform: {platform.machine()}')

# Check CPU info
with open('/proc/cpuinfo', 'r') as f:
    for line in f:
        if 'isa' in line.lower():
            print(f'ISA: {line.strip()}')
            break

# Try importing sgl_kernel
try:
    import sgl_kernel
    print('✓ sgl_kernel imported successfully')
except ImportError as e:
    print(f'✗ Failed to import sgl_kernel: {e}')
    sys.exit(1)

# Check for RVV functions
try:
    # Check if RVV kernels are available
    import torch
    print(f'✓ PyTorch version: {torch.__version__}')
except Exception as e:
    print(f'Warning: PyTorch check failed: {e}')
"

log_info "Dependency installation complete!"
