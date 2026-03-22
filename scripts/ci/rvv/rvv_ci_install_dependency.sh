#!/bin/bash
# Install/reinstall sglang and sgl-kernel from checkout inside the RVV CI container.
# Modelled after scripts/ci/amd/amd_ci_install_dependency.sh
#
# Runs from the CI host (outside the container) via podman exec.
#
# Usage:
#   bash scripts/ci/rvv/rvv_ci_install_dependency.sh [OPTIONS]
#
# Options:
#   --skip-kernel-build    Skip rebuilding sgl-kernel C++ extension (use image pre-built)
#   --skip-sglang-build    Skip reinstalling sglang python package (use image pre-built)
#
# Environment Variables:
#   CONTAINER_NAME: Target container name (default: ci_sglang_rvv)

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ci_sglang_rvv}"
SKIP_KERNEL_BUILD=""
SKIP_SGLANG_BUILD=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-kernel-build) SKIP_KERNEL_BUILD="1"; shift ;;
    --skip-sglang-build) SKIP_SGLANG_BUILD="1"; shift ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "  --skip-kernel-build    Skip rebuilding sgl-kernel C++ extension"
      echo "  --skip-sglang-build    Skip reinstalling sglang python package"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

WORKSPACE="/workspace/sglang"

echo "[RVV-CI] Installing dependencies in container: ${CONTAINER_NAME}"

# ── sglang python package ────────────────────────────────────────────────────
if [[ -n "${SKIP_SGLANG_BUILD}" ]]; then
  echo "[RVV-CI] Skipping sglang python install (--skip-sglang-build)"
else
  echo "[RVV-CI] Reinstalling sglang from checkout..."
  podman exec "${CONTAINER_NAME}" pip uninstall sglang -y || true
  # Clear stale bytecode so fresh checkout is used
  podman exec "${CONTAINER_NAME}" \
    find "${WORKSPACE}/python" -name "*.pyc" -delete || true
  podman exec "${CONTAINER_NAME}" \
    find "${WORKSPACE}/python" -name "__pycache__" -type d -exec rm -rf {} + || true
  podman exec -w "${WORKSPACE}/python" "${CONTAINER_NAME}" pip install -e .
fi

# ── sgl-kernel C++ extension ─────────────────────────────────────────────────
if [[ -n "${SKIP_KERNEL_BUILD}" ]]; then
  echo "[RVV-CI] Skipping sgl-kernel build (--skip-kernel-build)"
else
  echo "[RVV-CI] Rebuilding sgl-kernel from checkout..."
  podman exec "${CONTAINER_NAME}" pip uninstall sgl-kernel -y || true
  podman exec -w "${WORKSPACE}/sgl-kernel" "${CONTAINER_NAME}" pip install -e .
fi

# ── Verify RVV support ────────────────────────────────────────────────────────
echo "[RVV-CI] Verifying RVV support..."
podman exec "${CONTAINER_NAME}" \
  python3 "${WORKSPACE}/scripts/ci/rvv/verify_rvv_support.py"

echo "[RVV-CI] Dependency installation complete!"
