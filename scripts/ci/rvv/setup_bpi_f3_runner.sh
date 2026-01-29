#!/bin/bash
# Setup script for GitHub Actions Self-hosted Runner on Banana Pi BPI-F3
# RISC-V 64-bit architecture, no sudo required
#
# Prerequisites:
# - RISC-V 64 system (Banana Pi BPI-F3)
# - Network access to GitHub
# - User account with podman configured
#
# Usage:
#   1. Get runner token from: https://github.com/ChenTim1011/sglang/settings/actions/runners/new
#   2. Run: bash setup_bpi_f3_runner.sh <YOUR_TOKEN>

set -e

# Configuration
RUNNER_VERSION="2.321.0"
RUNNER_NAME="BPI-F3-RVV-Runner"
RUNNER_LABEL="rvv-bpi-f3"
REPO_URL="https://github.com/ChenTim1011/sglang"
RUNNER_DIR="${HOME}/actions-runner"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if token provided
if [ -z "$1" ]; then
    log_error "Runner token not provided!"
    echo "Usage: $0 <RUNNER_TOKEN>"
    echo ""
    echo "Get your token from:"
    echo "  https://github.com/ChenTim1011/sglang/settings/actions/runners/new"
    exit 1
fi

RUNNER_TOKEN="$1"

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "riscv64" ]]; then
    log_warn "Architecture is $ARCH, expected riscv64. Continuing anyway..."
fi

# Create runner directory
log_info "Creating runner directory at ${RUNNER_DIR}..."
mkdir -p "${RUNNER_DIR}"
cd "${RUNNER_DIR}"

# Download runner
RUNNER_PACKAGE="actions-runner-linux-riscv64-${RUNNER_VERSION}.tar.gz"
DOWNLOAD_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_PACKAGE}"

if [ ! -f "${RUNNER_PACKAGE}" ]; then
    log_info "Downloading GitHub Actions Runner v${RUNNER_VERSION} for RISC-V 64..."
    curl -o "${RUNNER_PACKAGE}" -L "${DOWNLOAD_URL}"
else
    log_info "Runner package already downloaded."
fi

# Verify download
if [ ! -f "${RUNNER_PACKAGE}" ]; then
    log_error "Failed to download runner package!"
    exit 1
fi

# Extract runner
log_info "Extracting runner..."
tar xzf "./${RUNNER_PACKAGE}"

# Verify extraction
if [ ! -f "./config.sh" ]; then
    log_error "Runner extraction failed! config.sh not found."
    exit 1
fi

# Configure runner
log_info "Configuring runner..."
log_info "  Repository: ${REPO_URL}"
log_info "  Name: ${RUNNER_NAME}"
log_info "  Labels: ${RUNNER_LABEL}"

./config.sh \
    --url "${REPO_URL}" \
    --token "${RUNNER_TOKEN}" \
    --name "${RUNNER_NAME}" \
    --labels "${RUNNER_LABEL}" \
    --work _work \
    --unattended \
    --replace

if [ $? -ne 0 ]; then
    log_error "Runner configuration failed!"
    exit 1
fi

log_info "Runner configured successfully!"
echo ""
log_info "================================================"
log_info "Runner Setup Complete!"
log_info "================================================"
echo ""
log_info "To START the runner:"
log_info "  cd ${RUNNER_DIR}"
log_info "  ./run.sh"
echo ""
log_info "To run in BACKGROUND:"
log_info "  nohup ./run.sh > runner.log 2>&1 &"
echo ""
log_info "To run with AUTO-RESTART (recommended for production):"
log_info "  # Create systemd user service"
log_info "  mkdir -p ~/.config/systemd/user/"
log_info "  cat > ~/.config/systemd/user/github-runner.service << 'EOF'
[Unit]
Description=GitHub Actions Runner
After=network.target

[Service]
Type=simple
WorkingDirectory=${RUNNER_DIR}
ExecStart=${RUNNER_DIR}/run.sh
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF"
log_info "  systemctl --user daemon-reload"
log_info "  systemctl --user enable github-runner.service"
log_info "  systemctl --user start github-runner.service"
echo ""
log_info "To VERIFY runner status:"
log_info "  Visit: https://github.com/ChenTim1011/sglang/settings/actions/runners"
echo ""
