#!/bin/bash
# Start Podman container for RVV CI testing
#
# Environment Variables:
#   GITHUB_WORKSPACE: Workspace directory to mount
#   PODMAN_IMAGE: Container image to use (default: localhost/sglang-rvv:latest)
#   CONTAINER_NAME: Name for the container (default: ci_sglang_rvv)
#
# The image must be pre-built on the runner using docker/rvv.Dockerfile.
# If the image is not found locally, this script will build it automatically
# (expect ~30-60 minutes on first run).

set -e

# Abort early if not running on RISC-V hardware (prevents accidental x86 runs)
if ! uname -m | grep -q "riscv64"; then
    echo "ERROR: This script must run on RISC-V (riscv64). Current arch: $(uname -m)"
    exit 1
fi

# Configuration
PODMAN_IMAGE="${PODMAN_IMAGE:-localhost/sglang-rvv:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-ci_sglang_rvv}"
WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Stop and remove existing container if running
if podman ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    log_warn "Container ${CONTAINER_NAME} already exists. Removing..."
    podman rm -f "${CONTAINER_NAME}" || true
fi

log_info "Starting Podman container: ${CONTAINER_NAME}"
log_info "Image: ${PODMAN_IMAGE}"
log_info "Workspace: ${WORKSPACE}"

# Ensure the image exists locally; build from source if not
if ! podman image exists "${PODMAN_IMAGE}"; then
    log_warn "Image ${PODMAN_IMAGE} not found locally. Building from docker/rvv.Dockerfile (~30-60 min)..."
    podman build \
        --format docker \
        -t "${PODMAN_IMAGE}" \
        -f "${WORKSPACE}/docker/rvv.Dockerfile" \
        "${WORKSPACE}"
    log_info "Image build complete."
else
    log_info "Using cached image: ${PODMAN_IMAGE}"
fi

# Start container with appropriate settings
log_info "Starting container..."
podman run -dt \
    --name "${CONTAINER_NAME}" \
    --ipc=host \
    -v "${WORKSPACE}:/workspace/sglang:z" \
    -v "${HF_HOME}:/root/.cache/huggingface:z" \
    --security-opt no-new-privileges \
    --cpus=8 \
    --memory=16g \
    --network slirp4netns \
    "${PODMAN_IMAGE}"

if [ $? -ne 0 ]; then
    echo "Failed to start container!"
    exit 1
fi

log_info "Container started successfully!"

# Verify container is running
if ! podman ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "ERROR: Container failed to start properly!"
    podman logs "${CONTAINER_NAME}"
    exit 1
fi

# Show container info
log_info "Container info:"
podman exec "${CONTAINER_NAME}" cat /proc/cpuinfo | grep -E '(processor|isa)' | head -10

log_info "Setup complete. Container ${CONTAINER_NAME} is ready."
