#!/bin/bash
# Install/reinstall sglang and sgl-kernel from checkout inside the RVV CI container.
#
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
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache}"
PIP_INSTALL_RETRY="${PIP_INSTALL_RETRY:-3}"

install_with_retry() {
  local desc="$1"
  shift
  local attempt
  for attempt in $(seq 1 "$PIP_INSTALL_RETRY"); do
    echo "[RVV-CI] ${desc} (attempt ${attempt}/${PIP_INSTALL_RETRY})"
    if "$@"; then
      return 0
    fi
    if [[ "$attempt" -lt "$PIP_INSTALL_RETRY" ]]; then
      echo "[RVV-CI] Failed, retrying in 5 seconds..."
      sleep 5
    fi
  done
  echo "[RVV-CI] ${desc} failed after ${PIP_INSTALL_RETRY} attempts"
  return 1
}

verify_rvv_support() {
  echo "[RVV-CI] Verifying RVV support..."
  podman exec -i "${CONTAINER_NAME}" python3 - <<'PY'
import os
import platform
import sys

import torch

if "riscv" not in platform.machine().lower():
    print("✗ Not RISC-V CPU")
    sys.exit(1)

if os.getenv("SGLANG_DISABLE_RVV_KERNELS", "").lower() in ("1", "true", "yes"):
    print("✗ SGLANG_DISABLE_RVV_KERNELS is set")
    sys.exit(1)

try:
    torch.ops.sgl_kernel.weight_packed_linear
except AttributeError:
    print("✗ RVV kernels not available: weight_packed_linear not registered")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error probing RVV kernels: {e}")
    sys.exit(1)

print("✓ RVV support verified")
PY
}

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
podman exec "${CONTAINER_NAME}" mkdir -p "${PIP_CACHE_DIR}" || true

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
  install_with_retry "Installing sglang (editable)" \
    podman exec -w "${WORKSPACE}/python" "${CONTAINER_NAME}" \
    pip install --cache-dir "${PIP_CACHE_DIR}" -e .
fi

# ── sgl-kernel C++ extension ─────────────────────────────────────────────────
if [[ -n "${SKIP_KERNEL_BUILD}" ]]; then
  echo "[RVV-CI] Skipping sgl-kernel build (--skip-kernel-build)"
else
  echo "[RVV-CI] Rebuilding sgl-kernel from checkout..."
  podman exec "${CONTAINER_NAME}" pip uninstall sgl-kernel -y || true
  install_with_retry "Installing sgl-kernel (editable)" \
    podman exec -w "${WORKSPACE}/sgl-kernel" "${CONTAINER_NAME}" \
    pip install --cache-dir "${PIP_CACHE_DIR}" -e .
fi

# ── Verify RVV support ────────────────────────────────────────────────────────
verify_rvv_support

echo "[RVV-CI] Dependency installation complete!"
