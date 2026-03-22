#!/bin/bash
# Execute commands inside the RVV CI container.
# Modelled after scripts/ci/amd/amd_ci_exec.sh
#
# Usage:
#   bash rvv_ci_exec.sh [-w WORKDIR] [-e KEY=VAL] COMMAND [ARGS...]
#
# Environment Variables:
#   CONTAINER_NAME: Target container name (default: ci_sglang_rvv)

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ci_sglang_rvv}"
WORKDIR="/workspace/sglang"

declare -A ENV_MAP=(
  [SGLANG_IS_IN_CI]=1
  [SGLANG_IS_IN_CI_RVV]=1
)

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workdir)
      WORKDIR="$2"
      shift 2
      ;;
    -e)
      IFS="=" read -r key val <<< "$2"
      ENV_MAP["$key"]="$val"
      shift 2
      ;;
    --)
      shift; break
      ;;
    *)
      break
      ;;
  esac
done

# Build ENV_ARGS
ENV_ARGS=()
for key in "${!ENV_MAP[@]}"; do
  ENV_ARGS+=("-e" "$key=${ENV_MAP[$key]}")
done

podman exec \
  -w "$WORKDIR" \
  "${ENV_ARGS[@]}" \
  "$CONTAINER_NAME" "$@"
