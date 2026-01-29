#!/bin/bash
# Execute commands inside RVV CI container
# Modeled after scripts/ci/amd/amd_ci_exec.sh
#
# Usage:
#   bash rvv_ci_exec.sh [-w WORKDIR] COMMAND [ARGS...]
#
# Example:
#   bash rvv_ci_exec.sh -w /workspace/sglang/test python3 -m pytest test_rvv.py

set -e

CONTAINER_NAME="${CONTAINER_NAME:-ci_sglang_rvv}"
WORKDIR=""

# Parse arguments
while getopts "w:" opt; do
    case $opt in
        w)
            WORKDIR="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

shift $((OPTIND-1))

# Build podman exec command
EXEC_CMD="podman exec"

if [ -n "$WORKDIR" ]; then
    EXEC_CMD="$EXEC_CMD -w $WORKDIR"
fi

EXEC_CMD="$EXEC_CMD $CONTAINER_NAME"

# Execute command
$EXEC_CMD "$@"
exit $?
