#!/bin/bash
# ==============================================================================
# RVV Kernel Test & Benchmark Runner
# ==============================================================================
#
# This script runs all RVV kernel tests and benchmarks on Banana Pi.
#
# Usage:
#   ./run_rvv_tests.sh [options]
#
# Options:
#   --test-only       Run only tests (skip benchmarks)
#   --bench-only      Run only benchmarks (skip tests)
#   --gemm            Run only GEMM tests/benchmarks
#   --decode          Run only decode attention tests/benchmarks
#   --extend          Run only extend attention tests/benchmarks
#   --quick           Run quick tests with fewer iterations
#   --help            Show this help message
#
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
RUN_TESTS=true
RUN_BENCHMARKS=true
RUN_GEMM=true
RUN_DECODE=true
RUN_EXTEND=true
QUICK_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-only)
            RUN_BENCHMARKS=false
            shift
            ;;
        --bench-only)
            RUN_TESTS=false
            shift
            ;;
        --gemm)
            RUN_DECODE=false
            RUN_EXTEND=false
            shift
            ;;
        --decode)
            RUN_GEMM=false
            RUN_EXTEND=false
            shift
            ;;
        --extend)
            RUN_GEMM=false
            RUN_DECODE=false
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Environment Setup
# ==============================================================================

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}RVV Kernel Test & Benchmark Runner${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Source environment
if [[ -f ~/.local_riscv_env/env.sh ]]; then
    source ~/.local_riscv_env/env.sh
else
    echo -e "${RED}ERROR: ~/.local_riscv_env/env.sh not found${NC}"
    exit 1
fi

# Activate virtual environment
if [[ -f $RISCV_WORKSPACE/venv_sglang/bin/activate ]]; then
    source $RISCV_WORKSPACE/venv_sglang/bin/activate
else
    echo -e "${RED}ERROR: Virtual environment not found${NC}"
    exit 1
fi

# Set library paths
export LD_PRELOAD=~/.local/lib/libomp.so
export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH

# Navigate to sgl-kernel directory
cd $RISCV_WORKSPACE/sglang/sgl-kernel

echo -e "${GREEN}Environment:${NC}"
echo "  Platform: $(uname -m)"
echo "  Python: $(python --version 2>&1)"
echo "  Working dir: $(pwd)"
echo ""

# Track results
TESTS_PASSED=0
TESTS_FAILED=0
declare -A BENCHMARK_RESULTS

# ==============================================================================
# Test Functions
# ==============================================================================

run_test() {
    local name=$1
    local test_file=$2

    echo -e "${YELLOW}Running: $name${NC}"
    if python -m pytest "$test_file" -v --tb=short 2>&1; then
        echo -e "${GREEN}✓ $name PASSED${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ $name FAILED${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

run_benchmark() {
    local name=$1
    local bench_cmd=$2

    echo -e "${YELLOW}Running: $name${NC}"
    local output
    output=$(eval "$bench_cmd" 2>&1) || true
    echo "$output"

    # Store result for summary
    BENCHMARK_RESULTS["$name"]="$output"
}

# ==============================================================================
# Tests
# ==============================================================================

if [[ "$RUN_TESTS" == "true" ]]; then
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Running Tests${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""

    if [[ "$RUN_DECODE" == "true" ]]; then
        run_test "RVV Decode Attention CPU" "tests/test_rvv_decode_attention_cpu.py" || true
    fi

    if [[ "$RUN_EXTEND" == "true" ]]; then
        run_test "RVV Extend Attention CPU" "tests/test_rvv_extend_attention_cpu.py" || true
    fi

    if [[ "$RUN_GEMM" == "true" ]]; then
        run_test "RVV GEMM" "tests/test_rvv_gemm.py" || true
    fi

    echo ""
    echo -e "${BLUE}Test Summary:${NC}"
    echo -e "  ${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "  ${RED}Failed: $TESTS_FAILED${NC}"
    echo ""
fi

# ==============================================================================
# Benchmarks
# ==============================================================================

if [[ "$RUN_BENCHMARKS" == "true" ]]; then
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Running Benchmarks${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""

    BENCHMARK_DIR=$(pwd)/benchmark
    if [ ! -d "$BENCHMARK_DIR" ]; then
        echo -e "${RED}ERROR: Benchmark directory not found: $BENCHMARK_DIR${NC}"
    else
        # Build benchmark command based on QUICK_MODE
        BENCH_ARGS=""
        if [[ "$QUICK_MODE" == "true" ]]; then
            BENCH_ARGS="--quick"
        fi

        if [[ "$RUN_DECODE" == "true" ]]; then
            run_benchmark "RVV Decode Attention" \
                "cd $BENCHMARK_DIR && python bench_rvv_decode_attention.py $BENCH_ARGS"
        fi

        if [[ "$RUN_EXTEND" == "true" ]]; then
            run_benchmark "RVV Extend Attention" \
                "cd $BENCHMARK_DIR && python bench_rvv_extend_attention.py $BENCH_ARGS"
        fi

        if [[ "$RUN_GEMM" == "true" ]]; then
            run_benchmark "RVV GEMM" \
                "cd $BENCHMARK_DIR && python bench_rvv_gemm.py $BENCH_ARGS"
        fi
    fi
fi

# ==============================================================================
# Final Summary
# ==============================================================================

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Final Summary${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

if [[ "$RUN_TESTS" == "true" ]]; then
    echo -e "${GREEN}Tests:${NC}"
    echo "  Passed: $TESTS_PASSED"
    echo "  Failed: $TESTS_FAILED"
    echo ""
fi

if [[ "$RUN_BENCHMARKS" == "true" ]]; then
    echo -e "${GREEN}Benchmarks completed. See output above for detailed results.${NC}"
    echo ""
fi

echo -e "${GREEN}Done!${NC}"
