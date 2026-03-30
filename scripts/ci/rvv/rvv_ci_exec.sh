#!/bin/bash
# Execute commands inside the RVV CI container.
#
#
# Usage:
#   bash rvv_ci_exec.sh [-w WORKDIR] [-e KEY=VAL] COMMAND [ARGS...]
#
# Environment Variables:
#   CONTAINER_NAME: Target container name (default: ci_sglang_rvv)

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ci_sglang_rvv}"
WORKDIR="/workspace/sglang"

usage() {
  cat <<'EOF'
Usage:
  bash rvv_ci_exec.sh [-w WORKDIR] [-e KEY=VAL] COMMAND [ARGS...]

Examples:
  bash rvv_ci_exec.sh python3 -m unittest -v
  bash rvv_ci_exec.sh -w /workspace/sglang/python pip install -e .
  bash rvv_ci_exec.sh -e HF_TOKEN=xxx python3 script.py
EOF
}

declare -A ENV_MAP=(
  [SGLANG_IS_IN_CI]=1
  [SGLANG_IS_IN_CI_RVV]=1
)

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workdir)
      if [[ $# -lt 2 ]]; then
        echo "[RVV-CI] ERROR: Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      WORKDIR="$2"
      shift 2
      ;;
    -e)
      if [[ $# -lt 2 || "$2" != *=* ]]; then
        echo "[RVV-CI] ERROR: -e expects KEY=VAL" >&2
        usage >&2
        exit 2
      fi
      IFS="=" read -r key val <<< "$2"
      ENV_MAP["$key"]="$val"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
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

if [[ $# -eq 0 ]]; then
  echo "[RVV-CI] ERROR: No command provided" >&2
  usage >&2
  exit 2
fi

# Retry once in offline mode when
# failures look like network/model-download issues.
if podman exec \
  -w "$WORKDIR" \
  "${ENV_ARGS[@]}" \
  "$CONTAINER_NAME" "$@"; then
  exit 0
else
  FIRST_EXIT_CODE=$?
fi

echo "[RVV-CI] First attempt failed with exit code $FIRST_EXIT_CODE"

if [[ "$FIRST_EXIT_CODE" -eq 1 || "$FIRST_EXIT_CODE" -eq 137 || "$FIRST_EXIT_CODE" -eq 255 ]]; then
  echo "[RVV-CI] Exit code indicates test/runtime failure; skip retry"
  exit $FIRST_EXIT_CODE
fi

RETRY_ENV_ARGS=("${ENV_ARGS[@]}")
if [[ -z "${ENV_MAP[HF_HUB_OFFLINE]+x}" ]]; then
  RETRY_ENV_ARGS+=("-e" "HF_HUB_OFFLINE=1")
fi

podman exec \
  -w "$WORKDIR" \
  "${RETRY_ENV_ARGS[@]}" \
  "$CONTAINER_NAME" "$@"
