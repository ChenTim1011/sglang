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
#   --prefill         Run only prefill attention tests/benchmarks
#   --test-decode     Run only decode attention tests
#   --bench-decode    Run only decode attention benchmarks
#   --test-extend     Run only extend attention tests
#   --bench-extend    Run only extend attention benchmarks
#   --test-prefill    Run only prefill attention tests
#   --bench-prefill   Run only prefill attention benchmarks
#   --test-backend    Run only backend integration tests
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
RUN_PREFILL=true
QUICK_MODE=false
RUN_BACKEND=false

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
            RUN_PREFILL=false
            shift
            ;;
        --decode)
            RUN_GEMM=false
            RUN_EXTEND=false
            RUN_PREFILL=false
            shift
            ;;
        --extend)
            RUN_GEMM=false
            RUN_DECODE=false
            RUN_PREFILL=false
            shift
            ;;
        --prefill)
            RUN_GEMM=false
            RUN_DECODE=false
            RUN_EXTEND=false
            shift
            ;;
        --test-decode)
            RUN_TESTS=true
            RUN_BENCHMARKS=false
            RUN_GEMM=false
            RUN_EXTEND=false
            RUN_PREFILL=false
            RUN_DECODE=true
            shift
            ;;
        --bench-decode)
            RUN_TESTS=false
            RUN_BENCHMARKS=true
            RUN_GEMM=false
            RUN_EXTEND=false
            RUN_PREFILL=false
            RUN_DECODE=true
            shift
            ;;
        --test-extend)
            RUN_TESTS=true
            RUN_BENCHMARKS=false
            RUN_GEMM=false
            RUN_DECODE=false
            RUN_PREFILL=false
            RUN_EXTEND=true
            shift
            ;;
        --bench-extend)
            RUN_TESTS=false
            RUN_BENCHMARKS=true
            RUN_GEMM=false
            RUN_DECODE=false
            RUN_PREFILL=false
            RUN_EXTEND=true
            shift
            ;;
        --test-prefill)
            RUN_TESTS=true
            RUN_BENCHMARKS=false
            RUN_GEMM=false
            RUN_DECODE=false
            RUN_EXTEND=false
            RUN_PREFILL=true
            shift
            ;;
        --bench-prefill)
            RUN_TESTS=false
            RUN_BENCHMARKS=true
            RUN_GEMM=false
            RUN_DECODE=false
            RUN_EXTEND=false
            RUN_PREFILL=true
            shift
            ;;
        --test-backend)
            # run only backend integration tests
            RUN_TESTS=true
            RUN_BENCHMARKS=false
            RUN_GEMM=false
            RUN_DECODE=false
            RUN_EXTEND=false
            RUN_PREFILL=false
            RUN_BACKEND=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --help)
            head -40 "$0" | tail -35
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
if [[ -f ~/.local_riscv_env/workspace/venv_sglang/bin/activate ]]; then
    source ~/.local_riscv_env/workspace/venv_sglang/bin/activate
else
    echo -e "${RED}ERROR: Virtual environment not found${NC}"
    exit 1
fi

# Set library paths
export LD_PRELOAD=~/.local/lib/libomp.so
export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=~/.local_riscv_env/workspace/sglang/python:$PYTHONPATH

# Navigate to sgl-kernel directory
cd ~/.local_riscv_env/workspace/sglang/sgl-kernel

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
        run_test "RVV Decode Attention (FP16/BF16)" "tests/test_rvv_decode.py" || true
        run_test "RVV Decode Attention (INT8)" "tests/test_rvv_decode_int8.py" || true
    fi

    if [[ "$RUN_EXTEND" == "true" ]]; then
        run_test "RVV Extend Attention (FP16/BF16)" "tests/test_rvv_extend.py" || true
        run_test "RVV Extend Attention (INT8)" "tests/test_rvv_extend_int8.py" || true
    fi

    if [[ "$RUN_PREFILL" == "true" ]]; then
        run_test "RVV Prefill Attention CPU" "tests/test_rvv_prefill_attention_cpu.py" || true
    fi

    if [[ "$RUN_GEMM" == "true" ]]; then
        run_test "RVV GEMM (FP16/BF16)" "tests/test_rvv_gemm.py" || true
        run_test "RVV GEMM (INT8)" "tests/test_rvv_gemm_int8.py" || true
    fi

    # Run Backend Integration Tests (Python Layer)
    if [[ "$RUN_BACKEND" == "true" ]]; then
        run_test "RVV Backend Integration" "../test/srt/test_rvv_backend.py" || true
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
            run_benchmark "RVV Decode Attention (FP16)" \
                "cd $BENCHMARK_DIR && python bench_rvv_decode.py $BENCH_ARGS"
            run_benchmark "RVV Decode Attention (INT8)" \
                "cd $BENCHMARK_DIR && python bench_rvv_decode_int8.py $BENCH_ARGS"
        fi

        if [[ "$RUN_EXTEND" == "true" ]]; then
            run_benchmark "RVV Extend Attention (FP16)" \
                "cd $BENCHMARK_DIR && python bench_rvv_extend.py $BENCH_ARGS"
            run_benchmark "RVV Extend Attention (INT8)" \
                "cd $BENCHMARK_DIR && python bench_rvv_extend_int8.py $BENCH_ARGS"
        fi

        if [[ "$RUN_PREFILL" == "true" ]]; then
            run_benchmark "RVV Prefill Attention" \
                "cd $BENCHMARK_DIR && python bench_rvv_prefill_attention.py $BENCH_ARGS"
        fi

        if [[ "$RUN_GEMM" == "true" ]]; then
            run_benchmark "RVV GEMM (FP16/BF16)" \
                "cd $BENCHMARK_DIR && python bench_rvv_gemm.py $BENCH_ARGS"
            run_benchmark "RVV GEMM (INT8)" \
                "cd $BENCHMARK_DIR && python bench_rvv_gemm_int8.py $BENCH_ARGS"
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
