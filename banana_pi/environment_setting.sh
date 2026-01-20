#!/bin/bash
# Environment Setup Script for Banana Pi RVV Testing
# Usage: source environment_setting.sh
# This script MUST be sourced to affect the current shell (cd, export, activate).

echo "Setting up RVV Environment..."

# 1. Define Paths
WORKSPACE_DIR="$HOME/.local_riscv_env/workspace"
SGLANG_DIR="$WORKSPACE_DIR/sglang"
RVV_TEST_DIR="$SGLANG_DIR/banana_pi/test_tinyllama_rvv"
VENV_ACTIVATE="$WORKSPACE_DIR/venv_sglang/bin/activate"

# 2. Run Setup Stubs & Verify Environment
if [ -f "$RVV_TEST_DIR/manage_rvv_env.py" ]; then
    echo "Running manage_rvv_env.py..."

    if [ -f "$VENV_ACTIVATE" ]; then
        source "$VENV_ACTIVATE"
    else
        echo "Virtual environment not found at $VENV_ACTIVATE"
        return 1
    fi

    python3 "$RVV_TEST_DIR/manage_rvv_env.py" --action all
else
    echo "manage_rvv_env.py not found at $RVV_TEST_DIR"
fi

# 4. Export Variables
export LD_PRELOAD="$HOME/.local/lib/libomp.so"
echo "rw  LD_PRELOAD set to $LD_PRELOAD"

echo "Environment Ready!"
