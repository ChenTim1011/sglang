#!/bin/bash
# ==============================================================================
# Tests Directory Test Runner
# ==============================================================================
#
# This script runs test scripts in the banana_pi/tests directory.
#
# Usage:
#   ./run_tests.sh [options] [test_name] [-- python_args...]
#
# Examples:
#   ./run_tests.sh                          # Run all tests
#   ./run_tests.sh --end-to-end              # Run end-to-end test only
#   ./run_tests.sh --end-to-end -- --num-requests 4  # Run with custom args
#   ./run_tests.sh --quick                   # Run all tests in quick mode
#   ./run_tests.sh --list                    # List all available tests
#
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Test files mapping
declare -A TEST_FILES=(
    ["end-to-end"]="test_end_to_end_int8.py"
    ["model-accuracy"]="test_model_accuracy_int8.py"
    ["parametrized"]="test_parametrized_int8.py"
    ["memory-bandwidth"]="test_memory_bandwidth.py"
    ["paged-attention"]="profile_paged_attention.py"
    ["vlen-alignment"]="test_vlen_alignment.py"
)

# Default settings
RUN_ALL=true
QUICK_MODE=false
SELECTED_TESTS=()
PYTHON_ARGS=()
SHOW_HELP=false
LIST_TESTS=false

# Function to print colored messages
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

log_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
${CYAN}Tests Directory Test Runner${NC}

${GREEN}Usage:${NC}
    ./run_tests.sh [options] [test_name] [-- python_args...]

${GREEN}Options:${NC}
    --help, -h              Show this help message
    --list                  List all available tests
    --quick                  Run tests in quick mode (fewer iterations)
    --all                    Run all tests (default)

${GREEN}Test Selection:${NC}
    --end-to-end            Run end-to-end INT8 test
    --model-accuracy        Run model accuracy test
    --parametrized          Run parametrized test
    --memory-bandwidth      Run memory bandwidth analysis
    --paged-attention       Run paged attention profiling
    --vlen-alignment        Run VLEN alignment test

${GREEN}Examples:${NC}
    # Run all tests
    ./run_tests.sh

    # Run specific test
    ./run_tests.sh --end-to-end

    # Run with custom arguments
    ./run_tests.sh --end-to-end -- --num-requests 4 --seq-len 256

    # Run multiple tests
    ./run_tests.sh --end-to-end --model-accuracy

    # Run in quick mode
    ./run_tests.sh --quick

    # List available tests
    ./run_tests.sh --list

${GREEN}Environment:${NC}
    This script automatically sets up:
    - LD_PRELOAD=~/.local/lib/libomp.so
    - LD_LIBRARY_PATH=~/.local/lib
    - Virtual environment activation

${GREEN}Available Tests:${NC}
EOF
    for test_name in "${!TEST_FILES[@]}"; do
        echo "    --${test_name}    ${TEST_FILES[$test_name]}"
    done
}

# Function to list tests
list_tests() {
    echo -e "${CYAN}Available Tests:${NC}"
    echo ""
    for test_name in "${!TEST_FILES[@]}"; do
        file="${TEST_FILES[$test_name]}"
        if [ -f "$file" ]; then
            echo -e "  ${GREEN}✓${NC} --${test_name}"
            echo "     File: $file"
        else
            echo -e "  ${RED}✗${NC} --${test_name}"
            echo "     File: $file (not found)"
        fi
        echo ""
    done
}

# Function to setup environment
setup_environment() {
    log_step "Setting up environment..."

    # Set OpenMP library paths
    if [ -f ~/.local/lib/libomp.so ]; then
        export LD_PRELOAD=~/.local/lib/libomp.so
        export LD_LIBRARY_PATH=~/.local/lib:${LD_LIBRARY_PATH:-}
        log_info "✓ OpenMP library configured"
    else
        log_warn "libomp.so not found at ~/.local/lib/libomp.so"
        log_warn "  Some tests may fail without OpenMP support"
    fi

    # Activate virtual environment if exists
    VENV_PATH="$HOME/.local_riscv_env/workspace/venv_sglang/bin/activate"
    if [ -f "$VENV_PATH" ]; then
        source "$VENV_PATH"
        log_info "✓ Virtual environment activated"
    else
        log_warn "Virtual environment not found at $VENV_PATH"
        log_warn "  Using system Python (may have missing dependencies)"
    fi

    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python not found!"
        exit 1
    fi

    log_info "✓ Using Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
    echo ""
}

# Function to run a single test
run_test() {
    local test_name=$1
    local test_file="${TEST_FILES[$test_name]}"
    local args=("${@:2}")

    if [ ! -f "$test_file" ]; then
        log_error "Test file not found: $test_file"
        return 1
    fi

    log_test "Running: $test_name ($test_file)"

    # Add --quick flag if quick mode is enabled
    if [ "$QUICK_MODE" = true ]; then
        if [[ ! " ${args[@]} " =~ " --quick " ]] && [[ ! " ${args[@]} " =~ " -q " ]]; then
            args+=("--quick")
        fi
    fi

    # Run the test
    set +e  # Don't exit on error for individual tests
    if [ ${#args[@]} -eq 0 ]; then
        $PYTHON_CMD "$test_file"
    else
        $PYTHON_CMD "$test_file" "${args[@]}"
    fi
    local exit_code=$?
    set -e

    echo ""
    if [ $exit_code -eq 0 ]; then
        log_info "✓ Test passed: $test_name"
    else
        log_error "✗ Test failed: $test_name (exit code: $exit_code)"
    fi
    echo ""

    return $exit_code
}

# Parse command line arguments
PARSING_PYTHON_ARGS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        --list)
            LIST_TESTS=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --end-to-end|--end_to_end)
            RUN_ALL=false
            SELECTED_TESTS+=("end-to-end")
            shift
            ;;
        --model-accuracy|--model_accuracy)
            RUN_ALL=false
            SELECTED_TESTS+=("model-accuracy")
            shift
            ;;
        --parametrized)
            RUN_ALL=false
            SELECTED_TESTS+=("parametrized")
            shift
            ;;
        --memory-bandwidth|--memory_bandwidth)
            RUN_ALL=false
            SELECTED_TESTS+=("memory-bandwidth")
            shift
            ;;
        --paged-attention|--paged_attention)
            RUN_ALL=false
            SELECTED_TESTS+=("paged-attention")
            shift
            ;;
        --vlen-alignment|--vlen_alignment)
            RUN_ALL=false
            SELECTED_TESTS+=("vlen-alignment")
            shift
            ;;
        --)
            PARSING_PYTHON_ARGS=true
            shift
            ;;
        *)
            if [ "$PARSING_PYTHON_ARGS" = true ]; then
                PYTHON_ARGS+=("$1")
            else
                log_warn "Unknown option: $1"
                log_info "Use --help for usage information"
                exit 1
            fi
            shift
            ;;
    esac
done

# Show help if requested
if [ "$SHOW_HELP" = true ]; then
    show_help
    exit 0
fi

# List tests if requested
if [ "$LIST_TESTS" = true ]; then
    list_tests
    exit 0
fi

# Setup environment
setup_environment

# Determine which tests to run
if [ "$RUN_ALL" = true ]; then
    SELECTED_TESTS=("${!TEST_FILES[@]}")
fi

# If no tests selected, run all
if [ ${#SELECTED_TESTS[@]} -eq 0 ]; then
    SELECTED_TESTS=("${!TEST_FILES[@]}")
fi

# Print summary
echo "============================================================"
log_step "Test Execution Summary"
echo "============================================================"
log_info "Tests to run: ${#SELECTED_TESTS[@]}"
for test_name in "${SELECTED_TESTS[@]}"; do
    echo "  - $test_name (${TEST_FILES[$test_name]})"
done
if [ "$QUICK_MODE" = true ]; then
    log_info "Mode: Quick (fewer iterations)"
fi
if [ ${#PYTHON_ARGS[@]} -gt 0 ]; then
    log_info "Python arguments: ${PYTHON_ARGS[*]}"
fi
echo "============================================================"
echo ""

# Run tests
FAILED_TESTS=()
PASSED_TESTS=()

for test_name in "${SELECTED_TESTS[@]}"; do
    if run_test "$test_name" "${PYTHON_ARGS[@]}"; then
        PASSED_TESTS+=("$test_name")
    else
        FAILED_TESTS+=("$test_name")
    fi
done

# Print final summary
echo ""
echo "============================================================"
log_step "Final Summary"
echo "============================================================"
log_info "Total tests: ${#SELECTED_TESTS[@]}"
log_info "Passed: ${#PASSED_TESTS[@]}"
if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    log_error "Failed: ${#FAILED_TESTS[@]}"
    echo ""
    log_error "Failed tests:"
    for test_name in "${FAILED_TESTS[@]}"; do
        echo "  - $test_name"
    done
else
    log_info "Failed: 0"
fi
echo "============================================================"

# Exit with error if any test failed
if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi
