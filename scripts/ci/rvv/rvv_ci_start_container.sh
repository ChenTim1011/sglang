#!/bin/bash
# Start Podman container for RVV CI testing
# Modeled after scripts/ci/amd/amd_ci_start_container.sh
#
# Environment Variables:
#   GITHUB_WORKSPACE: Workspace directory to mount (required)
#   PODMAN_IMAGE: Container image to use (default: docker.io/juitingchen/sglang-rvv:v1.0)
#   CONTAINER_NAME: Name for the container (default: ci_sglang_rvv)

set -e

# Configuration
PODMAN_IMAGE="${PODMAN_IMAGE:-docker.io/juitingchen/sglang-rvv:v1.0}"
CONTAINER_NAME="${CONTAINER_NAME:-ci_sglang_rvv}"
WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"

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

# Pull latest image
log_info "Pulling container image..."
podman pull "${PODMAN_IMAGE}"

# Start container with appropriate settings
log_info "Starting container..."
podman run -dt \
    --name "${CONTAINER_NAME}" \
    --ipc=host \
    -v "${WORKSPACE}:/workspace/sglang:z" \
    --device /dev/dri \
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
