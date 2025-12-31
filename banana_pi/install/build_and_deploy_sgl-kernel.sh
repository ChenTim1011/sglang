#!/bin/bash
# Complete automation script: Build Clang 19, build sglang-kernel wheel, transfer to Banana Pi, and install
# Usage: ./build_and_deploy_sgl-kernel.sh [OPTIONS]
#
# This script automates the complete workflow:
#   0. Check/Build Clang 19 RISC-V toolchain (if needed)
#   1. Build sglang-kernel wheel for RISC-V on x86_64 host
#   2. Transfer wheel file to Banana Pi via SCP
#   3. Install wheel on Banana Pi via SSH
#   4. Verify installation and run tests (optional)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get banana_pi directory (parent of install/)
BANANA_PI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load user-specific configuration if exists (for personal paths)
# Users can create build_and_deploy_sgl-kernel.config.sh to set their own paths
CONFIG_FILE="${SCRIPT_DIR}/build_and_deploy_sgl-kernel.config.sh"
if [ -f "${CONFIG_FILE}" ]; then
    echo "Loading user configuration from ${CONFIG_FILE}..."
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
fi

# Default values
BANANA_PI_USER="${BANANA_PI_USER:-jtchen}"
BANANA_PI_HOST="${BANANA_PI_HOST:-140.114.78.64}"
SGL_KERNEL_DIR="${SGL_KERNEL_DIR:-}"
REMOTE_BANANA_PI_DIR="${REMOTE_BANANA_PI_DIR:-~/.local_riscv_env/workspace/sglang/banana_pi}"
SSH_BIN="${SSH_BIN:-/usr/bin/ssh}"
SCP_BIN="${SCP_BIN:-/usr/bin/scp}"
SKIP_CONFIRM=false
SKIP_CLANG_BUILD="${SKIP_CLANG_BUILD:-}"
SKIP_BUILD="${SKIP_BUILD:-}"
SKIP_TRANSFER="${SKIP_TRANSFER:-}"
SKIP_INSTALL="${SKIP_INSTALL:-}"
# PyTorch RISC-V GitHub Release configuration
PYTORCH_RELEASE_TAG="${PYTORCH_RELEASE_TAG:-v1.1}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

ssh_cmd() {
    if "${SSH_BIN}" "$@"; then
        return 0
    fi
    env LD_LIBRARY_PATH= "${SSH_BIN}" "$@"
}

scp_cmd() {
    if "${SCP_BIN}" "$@"; then
        return 0
    fi
    env LD_LIBRARY_PATH= "${SCP_BIN}" "$@"
}

# Function to check system prerequisites
check_prerequisites() {
    local missing_tools=()
    local optional_tools=()

    # Check for required build tools
    for tool in cmake ninja git python3 tar; do
        if ! command -v "${tool}" >/dev/null 2>&1; then
            missing_tools+=("${tool}")
        fi
    done

    if [ ${#missing_tools[@]} -gt 0 ]; then
        echo "❌ ERROR: Missing required tools: ${missing_tools[*]}"
        echo "   Please install them before running this script."
        return 1
    fi

    # Check for optional but recommended tools
    for tool in rsync numfmt; do
        if ! command -v "${tool}" >/dev/null 2>&1; then
            optional_tools+=("${tool}")
        fi
    done

    if [ ${#optional_tools[@]} -gt 0 ]; then
        echo "⚠️  WARNING: Optional tools not found: ${optional_tools[*]}"
        echo "   The script will use fallback methods, but these tools are recommended:"
        echo "     - rsync: For faster directory synchronization"
        echo "     - numfmt: For better file size formatting"
    fi

    # Check Python build tools
    if ! python3 -m pip show build >/dev/null 2>&1 && ! command -v uv >/dev/null 2>&1; then
        echo "⚠️  WARNING: Neither 'build' package nor 'uv' found."
        echo "   The script will attempt to install 'build' automatically, but you may need:"
        echo "     pip install build scikit-build-core wheel"
    fi

    return 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)
            BANANA_PI_USER="$2"
            shift 2
            ;;
        --host)
            BANANA_PI_HOST="$2"
            shift 2
            ;;
        --yes|-y)
            SKIP_CONFIRM=true
            shift
            ;;
        --skip-clang-build)
            SKIP_CLANG_BUILD=1
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --skip-transfer)
            SKIP_TRANSFER=1
            shift
            ;;
        --skip-install)
            SKIP_INSTALL=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "This script automates the complete workflow:"
            echo "  0. Check/Build Clang 19 RISC-V toolchain (if needed)"
            echo "  1. Build sglang-kernel wheel for RISC-V"
            echo "  2. Transfer wheel to Banana Pi"
            echo "  3. Install wheel on Banana Pi"
            echo "  4. Run tests (optional)"
            echo ""
            echo "Options:"
            echo "  --user USER           Banana Pi username (default: $BANANA_PI_USER)"
            echo "  --host HOST           Banana Pi host/IP (default: $BANANA_PI_HOST)"
            echo "  --yes, -y             Skip confirmation prompts"
            echo "  --skip-clang-build    Skip building Clang 19 (use existing)"
            echo "  --skip-build          Skip building wheel"
            echo "  --skip-transfer       Skip transferring wheel to Banana Pi"
            echo "  --skip-install        Skip installing wheel on Banana Pi"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  BANANA_PI_USER        Banana Pi username"
            echo "  BANANA_PI_HOST        Banana Pi host/IP"
            echo "  SKIP_CLANG_BUILD      Skip building Clang 19"
            echo "  SKIP_BUILD            Skip building wheel"
            echo "  SKIP_TRANSFER         Skip transferring wheel"
            echo "  SKIP_INSTALL          Skip installing wheel"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Interactive configuration (if not skipped)
if [ "$SKIP_CONFIRM" = false ]; then
    read -p "Banana Pi Username [$BANANA_PI_USER]: " input_user
    BANANA_PI_USER="${input_user:-$BANANA_PI_USER}"

    read -p "Banana Pi Host/IP [$BANANA_PI_HOST]: " input_host
    BANANA_PI_HOST="${input_host:-$BANANA_PI_HOST}"

    # Ask for GitHub token if not already set (for private repos)
    if [ -z "${GITHUB_TOKEN:-}" ]; then
        echo ""
        echo "The repository may be private and require authentication for downloading releases."
        echo "GitHub token will be used for downloading PyTorch RISC-V from GitHub Release."
        echo ""
        read -s -p "GitHub Personal Access Token (PAT) [press Enter to skip]: " input_token
        echo ""
        if [ -n "${input_token}" ]; then
            GITHUB_TOKEN="${input_token}"
            echo "✓ GitHub token will be used for authentication"
        else
            echo "⚠️  No GitHub token provided. Download may fail if repository is private."
        fi
        echo ""
    fi
fi

echo "============================================"
echo "SGLang-Kernel RISC-V Complete Build & Deploy"
echo "============================================"
echo ""
echo "Prerequisites (SSH Key Setup):"
echo "  To avoid entering your password multiple times, set up SSH keys:"
echo "    1. Generate key (if needed): ssh-keygen -t ed25519"
echo "    2. Copy key to Banana Pi:    ssh-copy-id -i ~/.ssh/id_ed25519.pub ${BANANA_PI_USER}@${BANANA_PI_HOST}"
echo "    Example: ssh-copy-id -i ~/.ssh/id_ed25519.pub jtchen@140.114.78.64"
echo ""
echo "Target: ${BANANA_PI_USER}@${BANANA_PI_HOST}"
echo ""
echo "Configuration:"
echo "  Host: ${BANANA_PI_HOST}"
echo "  User: ${BANANA_PI_USER}"
if [ -n "${SKIP_CLANG_BUILD}" ]; then
    echo "  ⏭️  Skip Clang build: Yes"
fi
if [ -n "${SKIP_BUILD}" ]; then
    echo "  ⏭️  Skip wheel build: Yes"
fi
if [ -n "${SKIP_TRANSFER}" ]; then
    echo "  ⏭️  Skip transfer: Yes"
fi
if [ -n "${SKIP_INSTALL}" ]; then
    echo "  ⏭️  Skip install: Yes"
fi
echo ""

# Check system prerequisites
echo "Checking system prerequisites..."
if ! check_prerequisites; then
    exit 1
fi
echo "✅ Prerequisites check passed"
echo ""

# Function to build Clang 19
build_clang19() {
    local CLANG19_INSTALL_DIR="${HOME}/tools/clang19-riscv"
    local LLVM_SOURCE_DIR="${HOME}/tools/llvm-project"

    echo "============================================"
    echo "Building Clang 19.1.0 RISC-V Toolchain"
    echo "============================================"
    echo "Install directory: ${CLANG19_INSTALL_DIR}"
    echo "Source directory: ${LLVM_SOURCE_DIR}"
    echo ""
    echo "This will require:"
    echo "  - Disk space: ~20GB"
    echo "  - Build time: ~1-2 hours (depending on CPU)"
    echo "  - Memory: Recommended 8GB+"
    echo ""

    if [ "$SKIP_CONFIRM" = false ]; then
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled"
            return 1
        fi
    fi

    # Create directories
    mkdir -p "${HOME}/tools"
    cd "${HOME}/tools"

    # 1. Clone LLVM source code (using llvmorg-19.1.0 tag)
    if [ ! -d "${LLVM_SOURCE_DIR}" ]; then
        echo "Cloning LLVM project (llvmorg-19.1.0)..."
        git clone --depth 1 --branch llvmorg-19.1.0 https://github.com/llvm/llvm-project.git "${LLVM_SOURCE_DIR}"
    else
        echo "LLVM source already exists, switching to llvmorg-19.1.0 tag..."
        cd "${LLVM_SOURCE_DIR}"
        git fetch --depth 1 origin tag llvmorg-19.1.0
        git checkout llvmorg-19.1.0
    fi

    # 2. Create build directory
    BUILD_DIR="${LLVM_SOURCE_DIR}/build"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    # 3. Configure CMake
    echo "Configuring CMake..."
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${CLANG19_INSTALL_DIR}" \
        -DLLVM_ENABLE_PROJECTS="clang;lld" \
        -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
        -DLLVM_TARGETS_TO_BUILD="RISCV;X86" \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_ENABLE_BACKTRACES=OFF \
        -DLLVM_ENABLE_EH=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        ../llvm

    # 4. Build
    echo "Building Clang 19 (this will take some time)..."
    ninja -j 8

    # 5. Install
    echo "Installing to ${CLANG19_INSTALL_DIR}..."
    ninja install

    echo ""
    echo "============================================"
    echo "Clang 19.1.0 build complete!"
    echo "============================================"
    echo "Installation location: ${CLANG19_INSTALL_DIR}"
    echo ""
    return 0
}

# Function to detect Clang 19
detect_clang19() {
    local CLANG19_INSTALL_DIR="${HOME}/tools/clang19-riscv"
    local CLANG19_BIN="${CLANG19_INSTALL_DIR}/bin/clang"
    local CLANG_VERSION=""
    local MAJOR_VERSION=""

    # Check if Clang 19 is installed at expected location
    if [ -f "${CLANG19_BIN}" ]; then
        CLANG_VERSION=$("${CLANG19_BIN}" --version 2>/dev/null | head -1 || echo "")
        MAJOR_VERSION=$(echo "${CLANG_VERSION}" | grep -oE "clang version ([0-9]+)" | grep -oE "[0-9]+" | head -1 || echo "")

        if [ -n "${MAJOR_VERSION}" ] && [ "${MAJOR_VERSION}" -ge 19 ]; then
            echo "✅ Clang 19 detected: ${CLANG_VERSION}"
            echo "   Location: ${CLANG19_INSTALL_DIR}"
            return 0
        fi
    fi

    # Try to setup environment and check
    if setup_clang19_env >/dev/null 2>&1; then
        if [ -n "${CC}" ] && [ -f "${CC}" ]; then
            CLANG_VERSION=$("${CC}" --version 2>/dev/null | head -1 || echo "")
            MAJOR_VERSION=$(echo "${CLANG_VERSION}" | grep -oE "clang version ([0-9]+)" | grep -oE "[0-9]+" | head -1 || echo "")

            if [ -n "${MAJOR_VERSION}" ] && [ "${MAJOR_VERSION}" -ge 19 ]; then
                echo "✅ Clang 19 detected: ${CLANG_VERSION}"
                echo "   Location: ${CC}"
                return 0
            fi
        fi
    fi

    # Check system Clang (as fallback)
    if command -v clang >/dev/null 2>&1; then
        CLANG_VERSION=$(clang --version 2>/dev/null | head -1 || echo "")
        MAJOR_VERSION=$(echo "${CLANG_VERSION}" | grep -oE "clang version ([0-9]+)" | grep -oE "[0-9]+" | head -1 || echo "")

        if [ -n "${MAJOR_VERSION}" ] && [ "${MAJOR_VERSION}" -ge 19 ]; then
            echo "✅ Clang 19 detected in system: ${CLANG_VERSION}"
            return 0
        fi
    fi

    return 1
}

# Step 0: Check/Build Clang 19
if [ -z "${SKIP_CLANG_BUILD}" ]; then
    echo "============================================"
    echo "Step 0: Checking Clang 19 installation"
    echo "============================================"
    echo ""

    if detect_clang19; then
        echo "✅ Step 0 complete: Clang 19 already available"
        echo ""
    else
        echo "❌ Clang 19 not detected"
        echo ""
        echo "Clang 19 or later is required for RISC-V Vector Extension support."
        echo ""
        echo "Prerequisites for building Clang 19:"
        echo "  - CMake (>=3.25)"
        echo "  - Ninja build system"
        echo "  - Git"
        echo "  - ~20GB disk space"
        echo "  - ~1-2 hours build time"
        echo ""
        echo "The script can automatically build Clang 19, or you can:"
        echo "  - Install Clang 19 manually at ~/tools/clang19-riscv"
        echo "  - Set CLANG19_TOOLCHAIN_DIR to point to your Clang 19 installation"
        echo ""

        if [ "$SKIP_CONFIRM" = false ]; then
            echo "What would you like to do?"
            echo "  1) Build Clang 19 now (recommended, requires cmake/ninja/git)"
            echo "  2) Skip Clang build (you must have Clang 19 installed manually)"
            echo "  3) Exit"
            echo ""
            read -p "Enter choice [1-3] (default: 1): " CLANG_CHOICE
            CLANG_CHOICE="${CLANG_CHOICE:-1}"

            case "${CLANG_CHOICE}" in
                1)
                    if build_clang19; then
                        echo "✅ Step 0 complete: Clang 19 built successfully"
                        echo ""
                    else
                        echo "❌ ERROR: Clang 19 build failed or cancelled"
                        exit 1
                    fi
                    ;;
                2)
                    echo "⚠️  Skipping Clang 19 build"
                    echo "   Please ensure Clang 19 is installed at ~/tools/clang19-riscv"
                    echo "   or set CLANG19_TOOLCHAIN_DIR environment variable."
                    SKIP_CLANG_BUILD=1
                    ;;
                3)
                    echo "Exiting..."
                    exit 0
                    ;;
                *)
                    echo "Invalid choice, exiting..."
                    exit 1
                    ;;
            esac
        else
            # Non-interactive mode: try to build
            echo "Non-interactive mode: attempting to build Clang 19..."
            if build_clang19; then
                echo "✅ Step 0 complete: Clang 19 built successfully"
                echo ""
            else
                echo "❌ ERROR: Clang 19 build failed"
                exit 1
            fi
        fi
    fi
else
    echo "⏭️  Skipping Clang 19 check (SKIP_CLANG_BUILD is set)"
    echo ""
fi

# Function to setup Clang 19 RISC-V environment
setup_clang19_env() {
    local CLANG19_TOOLCHAIN_DIR="${CLANG19_TOOLCHAIN_DIR:-${HOME}/tools/clang19-riscv}"
    local ALT_CLANG_TOOLCHAIN_DIR="${ALT_CLANG_TOOLCHAIN_DIR:-}"
    local GCC_TOOLCHAIN_ROOT="${GCC_TOOLCHAIN_ROOT:-}"

    # Discover toolchain
    if [ -d "${CLANG19_TOOLCHAIN_DIR}" ] && [ -x "${CLANG19_TOOLCHAIN_DIR}/bin/clang" ]; then
        TOOLCHAIN="${CLANG19_TOOLCHAIN_DIR}"
        echo "[INFO] ✅ Using Clang toolchain at ${TOOLCHAIN}"
    elif [ -n "${ALT_CLANG_TOOLCHAIN_DIR}" ] && [ -d "${ALT_CLANG_TOOLCHAIN_DIR}" ] && [ -x "${ALT_CLANG_TOOLCHAIN_DIR}/bin/clang" ]; then
        TOOLCHAIN="${ALT_CLANG_TOOLCHAIN_DIR}"
        echo "[WARNING] Using fallback Clang toolchain at ${TOOLCHAIN}"
    else
        echo "[ERROR] Clang toolchain not found."
        echo "        Please build Clang 19 or set CLANG19_TOOLCHAIN_DIR to a valid path."
        echo "        You can also set ALT_CLANG_TOOLCHAIN_DIR for an alternative location."
        return 1
    fi

    export PATH="${TOOLCHAIN}/bin:${PATH}"

    # Set compiler binaries
    if [ -x "${TOOLCHAIN}/bin/clang" ]; then
        export CC="${TOOLCHAIN}/bin/clang"
        export CXX="${TOOLCHAIN}/bin/clang++"
    elif [ -x "${TOOLCHAIN}/bin/clang-19" ]; then
        export CC="${TOOLCHAIN}/bin/clang-19"
        export CXX="${TOOLCHAIN}/bin/clang++-19"
    else
        echo "[ERROR] clang binary not found under ${TOOLCHAIN}/bin"
        return 1
    fi

    if [ -x "${TOOLCHAIN}/bin/llvm-ar" ]; then
        export AR="${TOOLCHAIN}/bin/llvm-ar"
        export RANLIB="${TOOLCHAIN}/bin/llvm-ranlib"
    else
        export AR="ar"
        export RANLIB="ranlib"
        echo "[WARNING] Falling back to system ar/ranlib"
    fi

    if [ -x "${TOOLCHAIN}/bin/ld.lld" ]; then
        export LD="${TOOLCHAIN}/bin/ld.lld"
        export LDFLAGS_EXTRA="-fuse-ld=lld"
    else
        export LD="ld"
    fi

    # Target and sysroot
    export TARGET_TRIPLE="riscv64-unknown-linux-gnu"

    SYSROOT_CANDIDATES=(
        "${RISCV_SYSROOT:-}"
        "${TOOLCHAIN}/sysroot"
    )

    # Add ALT_CLANG_TOOLCHAIN_DIR sysroot if set
    if [ -n "${ALT_CLANG_TOOLCHAIN_DIR:-}" ] && [ -d "${ALT_CLANG_TOOLCHAIN_DIR}/sysroot" ]; then
        SYSROOT_CANDIDATES+=("${ALT_CLANG_TOOLCHAIN_DIR}/sysroot")
    fi

    for candidate in "${SYSROOT_CANDIDATES[@]}"; do
        if [ -n "${candidate}" ] && [ -d "${candidate}" ]; then
            export RISCV_SYSROOT="${candidate}"
            break
        fi
    done

    if [ -z "${RISCV_SYSROOT:-}" ]; then
        echo "[WARNING] RISC-V sysroot not found. Please set RISCV_SYSROOT manually."
        echo "          Cross-compilation may fail without a proper sysroot."
    fi

    # Compiler and linker flags
    COMMON_FLAGS=(--target="${TARGET_TRIPLE}" -march=rv64gcv -mabi=lp64d)
    if [ -n "${RISCV_SYSROOT:-}" ]; then
        COMMON_FLAGS+=(--sysroot="${RISCV_SYSROOT}")
    fi

    GCC_TOOLCHAIN_ROOT="${GCC_TOOLCHAIN_ROOT:-}"
    if [ -d "${HOME}/riscv-toolchain/riscv" ]; then
        GCC_TOOLCHAIN_ROOT="${HOME}/riscv-toolchain/riscv"
    elif [ -d "${HOME}/riscv-toolchain/install" ]; then
        GCC_TOOLCHAIN_ROOT="${HOME}/riscv-toolchain/install"
    fi

    if [ -n "${GCC_TOOLCHAIN_ROOT}" ]; then
        COMMON_FLAGS+=("--gcc-toolchain=${GCC_TOOLCHAIN_ROOT}")
        echo "[INFO] Using GCC toolchain: --gcc-toolchain=${GCC_TOOLCHAIN_ROOT}"
    else
        echo "[WARNING] RISC-V GCC toolchain not found."
        echo "          Expected at ~/riscv-toolchain/riscv or ~/riscv-toolchain/install"
        echo "          The GCC toolchain provides libstdc++ and runtime libraries."
        echo "          Cross-compilation may fail without it."
    fi

    if [ -d "${HOME}/riscv-toolchain/riscv/lib/gcc/riscv64-unknown-linux-gnu" ]; then
        GCC_VERSION_DIR=$(find "${HOME}/riscv-toolchain/riscv/lib/gcc/riscv64-unknown-linux-gnu" -maxdepth 1 -type d | grep -E "[0-9]+\\.[0-9]+(\\.[0-9]+)?" | sort -V | tail -1)
        if [ -n "${GCC_VERSION_DIR}" ]; then
            COMMON_FLAGS+=("-B${GCC_VERSION_DIR}")
            echo "[INFO] Using GCC lib path: ${GCC_VERSION_DIR}"
        fi
    fi

    export CFLAGS="${COMMON_FLAGS[*]}"
    export CXXFLAGS="${CFLAGS} -std=c++17 -Wno-vla-cxx-extension"

    LDFLAGS_PARTS=("${COMMON_FLAGS[@]}")
    if [ -n "${RISCV_SYSROOT:-}" ]; then
        if [ -d "${RISCV_SYSROOT}/usr/lib64/lp64d" ]; then
            LDFLAGS_PARTS+=("-L${RISCV_SYSROOT}/usr/lib64/lp64d")
            echo "[INFO] Added lp64d library path: ${RISCV_SYSROOT}/usr/lib64/lp64d"
        fi

        if [ -d "${RISCV_SYSROOT}/usr/lib/riscv64-linux-gnu" ]; then
            LDFLAGS_PARTS+=("-L${RISCV_SYSROOT}/usr/lib/riscv64-linux-gnu")
        fi
        if [ -d "${RISCV_SYSROOT}/lib/riscv64-linux-gnu" ]; then
            LDFLAGS_PARTS+=("-L${RISCV_SYSROOT}/lib/riscv64-linux-gnu")
        fi
        if [ -d "${RISCV_SYSROOT}/usr/lib64" ]; then
            LDFLAGS_PARTS+=("-L${RISCV_SYSROOT}/usr/lib64")
        fi
        if [ -d "${RISCV_SYSROOT}/lib64" ]; then
            LDFLAGS_PARTS+=("-L${RISCV_SYSROOT}/lib64")
        fi
    fi

    if [ -n "${LDFLAGS_EXTRA:-}" ]; then
        LDFLAGS_PARTS+=("${LDFLAGS_EXTRA}")
    fi

    export LDFLAGS="${LDFLAGS_PARTS[*]}"

    # Diagnostics
    echo "============================================"
    echo "Clang RISC-V Toolchain Environment"
    echo "============================================"
    echo "Toolchain: ${TOOLCHAIN}"
    echo "CC: ${CC}"
    echo "CXX: ${CXX}"
    echo "Target: ${TARGET_TRIPLE}"
    echo "Sysroot: ${RISCV_SYSROOT:-Not set}"

    if command -v "${CC}" >/dev/null 2>&1; then
        echo ""
        echo "Clang version:"
        "${CC}" --version | head -3
        echo ""
        echo "Target triple:"
        "${CC}" -print-target-triple 2>/dev/null || true
        echo ""
        echo "RISC-V Vector Extension support:"
        "${CC}" -march=rv64gcv -mabi=lp64d -target=${TARGET_TRIPLE} -E -dM - < /dev/null 2>/dev/null | grep -i "__riscv_v" | head -5 || echo "Could not verify RVV support"
    else
        echo "[ERROR] clang binary not runnable."
        return 1
    fi

    echo "============================================"
    echo "Environment loaded successfully!"
    echo "============================================"

    return 0
}

# Function: Download PyTorch RISC-V from GitHub Release
download_pytorch_riscv_from_github() {
    local REPO_OWNER="nthu-pllab"
    local REPO_NAME="pllab-sglang"
    local RELEASE_TAG="${PYTORCH_RELEASE_TAG:-v1.1}"
    local ARCHIVE_FILE="pytorch-riscv.tar.gz"
    local INSTALL_DIR="${BANANA_PI_DIR}/riscv_pytorch"
    local GITHUB_TOKEN_VAL="${GITHUB_TOKEN:-}"
    # Trim whitespace from token if provided
    GITHUB_TOKEN_VAL=$(echo "${GITHUB_TOKEN_VAL}" | tr -d '\n\r' | xargs)
    local DOWNLOAD_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${RELEASE_TAG}/${ARCHIVE_FILE}"
    local TEMP_DIR
    local TEMP_ARCHIVE
    local TEMP_EXTRACT_DIR
    local FILE_TYPE
    local HTTP_STATUS
    local CURL_EXIT
    local WGET_EXIT
    local DOWNLOAD_SUCCESS=false
    local CURL_OUTPUT
    local CURL_ERROR
    local ASSET_ID

    TEMP_DIR=$(mktemp -d)
    TEMP_ARCHIVE="${TEMP_DIR}/${ARCHIVE_FILE}"

    echo "============================================"
    echo "Downloading PyTorch RISC-V from GitHub Release"
    echo "============================================"
    echo "Repository: ${REPO_OWNER}/${REPO_NAME}"
    echo "Release tag: ${RELEASE_TAG}"
    echo "Archive: ${ARCHIVE_FILE}"
    echo "Install directory: ${INSTALL_DIR}"
    echo ""

    # Download archive using GitHub API
    echo "Downloading ${ARCHIVE_FILE}..."

    if command -v curl >/dev/null 2>&1; then
        if [ -n "${GITHUB_TOKEN_VAL}" ]; then
            # Use GitHub token for authentication (for private repos)
            # Trim token again before use
            GITHUB_TOKEN_VAL=$(echo "${GITHUB_TOKEN_VAL}" | tr -d '\n\r' | xargs)

            # Debug: Verify token is set (without showing actual token)
            TOKEN_LEN=${#GITHUB_TOKEN_VAL}
            if [ $TOKEN_LEN -lt 20 ]; then
                echo "⚠️  Warning: Token seems too short (${TOKEN_LEN} chars). Please verify it's correct."
            fi
            echo "Using GitHub token for authentication (token length: ${TOKEN_LEN} chars)..."

            # Query release API to get asset ID (required for private repositories)
            API_TEST_RAW=$(curl -s -H "Authorization: token $GITHUB_TOKEN_VAL" \
                "https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/tags/${RELEASE_TAG}" 2>&1)

            # Filter out curl warnings (lines starting with "curl:") to get clean JSON
            API_TEST=$(echo "$API_TEST_RAW" | grep -v "^curl:" || echo "$API_TEST_RAW")

            # Check for authentication errors in API response (before checking tag_name)
            if echo "$API_TEST" | grep -qE '"message".*"Bad credentials"'; then
                API_ERROR_MSG=$(echo "$API_TEST" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('message', 'Unknown error'))" 2>/dev/null || echo "Unknown error")
                echo "❌ ERROR: GitHub API authentication failed"
                echo "   Response: ${API_ERROR_MSG}"
                echo "   Please check your GitHub token has 'repo' scope"
                echo "   Token length: ${TOKEN_LEN} chars"
                echo "   You can check your token at: https://github.com/settings/tokens"
                rm -rf "${TEMP_DIR}"
                return 1
            elif echo "$API_TEST" | grep -q '"tag_name"'; then
                echo "✓ GitHub API response received successfully"
                # Debug: Check API_TEST length
                API_TEST_LEN=${#API_TEST}
                echo "Debug: API response length: $API_TEST_LEN characters"

                # Get asset ID from API response
                if command -v python3 >/dev/null 2>&1; then
                    # Debug: Show what we're looking for
                    echo "Debug: Looking for asset name: '$ARCHIVE_FILE'"

                    # First, get all asset names for debugging
                    ALL_ASSETS=$(echo "$API_TEST" | python3 -c "import sys, json; data=json.load(sys.stdin); assets=data.get('assets', []); print('\n'.join([a.get('name', '') for a in assets]))" 2>&1)

                    if echo "$ALL_ASSETS" | grep -qE "(Error|Traceback|Exception)"; then
                        echo "⚠️  Error getting assets list: $ALL_ASSETS"
                        echo "Debug: First 500 chars of API_TEST:"
                        echo "$API_TEST" | head -c 500
                        echo ""
                    elif [ -n "$ALL_ASSETS" ]; then
                        echo "Debug: All assets in release:"
                        echo "$ALL_ASSETS" | while read -r asset_name; do
                            if [ -n "$asset_name" ]; then
                                echo "  - '$asset_name'"
                            fi
                        done
                    else
                        echo "⚠️  No assets found in API response"
                    fi

                    # Use exact same command as setup_banana_pi.sh: $API_TEST and $ARCHIVE_FILE
                    ASSET_ID=$(echo "$API_TEST" | python3 -c "import sys, json; data=json.load(sys.stdin); assets=data.get('assets', []); target_asset=[a for a in assets if a['name'] == '$ARCHIVE_FILE']; print(target_asset[0]['id'] if target_asset else '')" 2>/dev/null || echo "")

                    if [ -n "$ASSET_ID" ]; then
                        echo "Debug: Found asset ID: $ASSET_ID"
                        # Use API endpoint for private repository assets
                        DOWNLOAD_URL="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/assets/$ASSET_ID"
                        echo "Using GitHub API endpoint for asset ID: $ASSET_ID"
                        CURL_OUTPUT=$(curl -L -w "\nHTTP_STATUS:%{http_code}" \
                            -H "Authorization: token ${GITHUB_TOKEN_VAL}" \
                            -H "Accept: application/octet-stream" \
                            -o "${TEMP_ARCHIVE}" \
                            "$DOWNLOAD_URL" 2>&1)
                    else
                        echo "⚠️  Asset ID not found for '$ARCHIVE_FILE'"
                        # Get available assets for debugging
                        AVAILABLE_ASSETS=$(echo "$API_TEST" | python3 -c "import sys, json; data=json.load(sys.stdin); assets=data.get('assets', []); available=[a.get('name', '<unknown>') for a in assets]; print(', '.join(available) if available else '(none)')" 2>/dev/null || echo "")
                        if [ -n "$AVAILABLE_ASSETS" ]; then
                            echo "   Available assets in release: $AVAILABLE_ASSETS"
                        fi
                        echo "   Trying browser_download_url with token authentication..."
                        # For private repos, browser_download_url also needs token
                        CURL_OUTPUT=$(curl -L -w "\nHTTP_STATUS:%{http_code}" \
                            -H "Authorization: token ${GITHUB_TOKEN_VAL}" \
                            -H "Accept: application/octet-stream" \
                            -o "${TEMP_ARCHIVE}" \
                            "${DOWNLOAD_URL}" 2>&1)
                    fi
                else
                    echo "⚠️  python3 not found, using browser_download_url"
                    CURL_OUTPUT=$(curl -L -w "\nHTTP_STATUS:%{http_code}" \
                        -H "Authorization: token ${GITHUB_TOKEN_VAL}" \
                        -H "Accept: application/octet-stream" \
                        -o "${TEMP_ARCHIVE}" \
                        "${DOWNLOAD_URL}" 2>&1)
                fi
            else
                # Check if API call failed due to authentication
                if echo "${API_TEST}" | grep -qE '"message".*"Not Found"|"message".*"Bad credentials"'; then
                    echo "❌ ERROR: GitHub API authentication failed"
                    echo "   Response: $(echo "${API_TEST}" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('message', 'Unknown error'))" 2>/dev/null || echo 'Unknown error')"
                    echo "   Please check your GitHub token has 'repo' scope"
                    rm -rf "${TEMP_DIR}"
                    return 1
                else
                    echo "⚠️  GitHub API response does not contain 'tag_name'"
                    echo "   This may indicate the release doesn't exist or API access failed"
                    echo "   Trying browser_download_url with token authentication..."
                    # For private repos, browser_download_url also needs token
                    CURL_OUTPUT=$(curl -L -w "\nHTTP_STATUS:%{http_code}" \
                        -H "Authorization: token ${GITHUB_TOKEN_VAL}" \
                        -H "Accept: application/octet-stream" \
                        -o "${TEMP_ARCHIVE}" \
                        "${DOWNLOAD_URL}" 2>&1)
                fi
            fi
            CURL_EXIT=$?
        else
            echo "⚠️  GITHUB_TOKEN is empty! Attempting download without authentication..."
            CURL_OUTPUT=$(curl -L -w "\nHTTP_STATUS:%{http_code}" \
                -o "${TEMP_ARCHIVE}" \
                "${DOWNLOAD_URL}" 2>&1)
            CURL_EXIT=$?
        fi

        HTTP_STATUS=$(echo "${CURL_OUTPUT}" | grep "HTTP_STATUS:" | cut -d: -f2 | tr -d ' ' || echo "")
        CURL_ERROR=$(echo "${CURL_OUTPUT}" | grep -v "HTTP_STATUS:" || echo "${CURL_OUTPUT}")

        # Check if download was successful based on HTTP status and file existence
        if [ "${HTTP_STATUS}" = "200" ] && [ -f "${TEMP_ARCHIVE}" ] && [ -s "${TEMP_ARCHIVE}" ]; then
            DOWNLOAD_SUCCESS=true
        else
            echo "❌ ERROR: Download failed (HTTP ${HTTP_STATUS:-unknown})"
            if [ "${HTTP_STATUS}" = "404" ]; then
                echo "   File not found. Please verify release '${RELEASE_TAG}' exists and contains '${ARCHIVE_FILE}'"
            elif [ "${HTTP_STATUS}" = "401" ] || [ "${HTTP_STATUS}" = "403" ]; then
                echo "   Authentication failed. Please check your GitHub token"
                if [ -f "${TEMP_ARCHIVE}" ]; then
                    echo "   Downloaded file content (first 200 chars):"
                    head -c 200 "${TEMP_ARCHIVE}" | cat -A
                    echo ""
                fi
            elif [ "${HTTP_STATUS}" = "302" ] || [ "${HTTP_STATUS}" = "301" ]; then
                echo "⚠️  Redirect detected (HTTP ${HTTP_STATUS}). This may indicate authentication issues."
            elif [ -z "${HTTP_STATUS}" ]; then
                echo "   No HTTP status code received. Check network connectivity."
                if [ -f "${TEMP_ARCHIVE}" ]; then
                    echo "   Downloaded file content (first 200 chars):"
                    head -c 200 "${TEMP_ARCHIVE}" | cat -A
                    echo ""
                fi
            fi
            if [ "${CURL_EXIT}" -ne 0 ]; then
                echo "   curl exit code: ${CURL_EXIT}"
                if [ -n "${CURL_ERROR}" ]; then
                    echo "   curl error: ${CURL_ERROR}"
                fi
            fi
            rm -rf "${TEMP_DIR}"
            return 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        local WGET_OUTPUT
        if [ -n "${GITHUB_TOKEN_VAL}" ]; then
            WGET_OUTPUT=$(wget --header="Authorization: token ${GITHUB_TOKEN_VAL}" \
                -O "${TEMP_ARCHIVE}" \
                "${DOWNLOAD_URL}" 2>&1)
        else
            WGET_OUTPUT=$(wget -O "${TEMP_ARCHIVE}" \
                "${DOWNLOAD_URL}" 2>&1)
        fi
        WGET_EXIT=$?

        if [ ${WGET_EXIT} -eq 0 ] && [ -f "${TEMP_ARCHIVE}" ] && [ -s "${TEMP_ARCHIVE}" ]; then
            DOWNLOAD_SUCCESS=true
        else
            echo "❌ ERROR: wget download failed (exit code: ${WGET_EXIT})"
            if echo "${WGET_OUTPUT}" | grep -q "404"; then
                echo "   File not found (404). Possible reasons:"
                echo "     1. Release '${RELEASE_TAG}' does not exist"
                echo "     2. File '${ARCHIVE_FILE}' not found in the release"
                echo "     3. Repository is private and token may not have access"
            elif echo "${WGET_OUTPUT}" | grep -q "401\|403"; then
                echo "   Authentication failed (401/403). Please check your GitHub token"
                echo "   Please check your token at: https://github.com/settings/tokens"
            fi
            rm -rf "${TEMP_DIR}"
            return 1
        fi
    else
        echo "❌ ERROR: Neither curl nor wget found. Please install one of them."
        rm -rf "${TEMP_DIR}"
        return 1
    fi

    if [ "${DOWNLOAD_SUCCESS}" = false ]; then
        rm -rf "${TEMP_DIR}"
        return 1
    fi

    echo "✅ Download complete"
    echo ""

    # Verify archive format
    echo "Verifying archive format..."
    FILE_TYPE=$(file "${TEMP_ARCHIVE}" | cut -d: -f2-)
    if echo "${FILE_TYPE}" | grep -qE "(gzip|tar|compressed|POSIX tar)"; then
        echo "✅ Archive format verified: ${FILE_TYPE}"
    else
        echo "❌ ERROR: Downloaded file is not a valid archive!"
        echo "   File type: ${FILE_TYPE}"
        rm -rf "${TEMP_DIR}"
        return 1
    fi
    echo ""

    # Extract archive to temporary directory first to inspect structure
    echo "Extracting archive..."
    TEMP_EXTRACT_DIR=$(mktemp -d)

    # Check if file is gzip compressed or uncompressed tar
    if echo "${FILE_TYPE}" | grep -qE "gzip|compressed"; then
        if ! tar -xzf "${TEMP_ARCHIVE}" -C "${TEMP_EXTRACT_DIR}" 2>&1; then
            echo "❌ ERROR: Failed to extract gzip-compressed archive"
            rm -rf "${TEMP_DIR}" "${TEMP_EXTRACT_DIR}"
            return 1
        fi
    else
        if ! tar -xf "${TEMP_ARCHIVE}" -C "${TEMP_EXTRACT_DIR}" 2>&1; then
            echo "❌ ERROR: Failed to extract tar archive"
            rm -rf "${TEMP_DIR}" "${TEMP_EXTRACT_DIR}"
            return 1
        fi
    fi

    # Inspect archive structure and move to install directory
    echo "Inspecting archive structure..."
    mkdir -p "${INSTALL_DIR}" || {
        echo "❌ ERROR: Failed to create installation directory ${INSTALL_DIR}"
        rm -rf "${TEMP_DIR}" "${TEMP_EXTRACT_DIR}"
        return 1
    }

    # Check for common archive structures
    # Structure 1: pytorch-riscv/torch/... or riscv_pytorch/torch/...
    if [ -d "${TEMP_EXTRACT_DIR}/pytorch-riscv/torch" ]; then
        echo "Detected structure: pytorch-riscv/torch/"
        cp -r "${TEMP_EXTRACT_DIR}/pytorch-riscv"/* "${INSTALL_DIR}/" || {
            echo "❌ ERROR: Failed to copy files"
            rm -rf "${TEMP_DIR}" "${TEMP_EXTRACT_DIR}"
            return 1
        }
    elif [ -d "${TEMP_EXTRACT_DIR}/riscv_pytorch/torch" ]; then
        echo "Detected structure: riscv_pytorch/torch/"
        cp -r "${TEMP_EXTRACT_DIR}/riscv_pytorch"/* "${INSTALL_DIR}/" || {
            echo "❌ ERROR: Failed to copy files"
            rm -rf "${TEMP_DIR}" "${TEMP_EXTRACT_DIR}"
            return 1
        }
    # Structure 2: torch/... (direct torch directory)
    elif [ -d "${TEMP_EXTRACT_DIR}/torch" ]; then
        echo "Detected structure: torch/"
        cp -r "${TEMP_EXTRACT_DIR}/torch" "${INSTALL_DIR}/" || {
            echo "❌ ERROR: Failed to copy files"
            rm -rf "${TEMP_DIR}" "${TEMP_EXTRACT_DIR}"
            return 1
        }
    # Structure 3: Direct files (lib/, share/, etc.)
    elif [ -f "${TEMP_EXTRACT_DIR}/lib/libtorch.so" ] || [ -f "${TEMP_EXTRACT_DIR}/share/cmake/Torch/TorchConfig.cmake" ]; then
        echo "Detected structure: direct files (lib/, share/)"
        cp -r "${TEMP_EXTRACT_DIR}"/* "${INSTALL_DIR}/" || {
            echo "❌ ERROR: Failed to copy files"
            rm -rf "${TEMP_DIR}" "${TEMP_EXTRACT_DIR}"
            return 1
        }
    else
        # Unknown structure, try to copy everything
        echo "⚠️  Unknown archive structure, attempting to copy all files..."
        cp -r "${TEMP_EXTRACT_DIR}"/* "${INSTALL_DIR}/" 2>/dev/null || {
            echo "❌ ERROR: Failed to copy files"
            echo "   Archive contents:"
            ls -la "${TEMP_EXTRACT_DIR}" | head -10
            rm -rf "${TEMP_DIR}" "${TEMP_EXTRACT_DIR}"
            return 1
        }
    fi

    echo "✅ Extraction complete"
    echo ""

    # Clean up temporary directories
    rm -rf "${TEMP_DIR}" "${TEMP_EXTRACT_DIR}"

    # Verify installation
    echo "Verifying PyTorch RISC-V installation..."
    if [ -f "${INSTALL_DIR}/torch/lib/libtorch.so" ] && [ -f "${INSTALL_DIR}/torch/share/cmake/Torch/TorchConfig.cmake" ]; then
        echo "✅ PyTorch RISC-V installed successfully to ${INSTALL_DIR}"
        echo "   Structure: ${INSTALL_DIR}/torch/"
        echo ""
        return 0
    elif [ -f "${INSTALL_DIR}/share/cmake/Torch/TorchConfig.cmake" ] && [ -f "${INSTALL_DIR}/lib/libtorch.so" ]; then
        echo "✅ PyTorch RISC-V installed successfully to ${INSTALL_DIR}"
        echo "   Structure: ${INSTALL_DIR}/ (direct)"
        echo ""
        return 0
    else
        echo "❌ ERROR: PyTorch RISC-V installation verification failed."
        echo "   Expected files (libtorch.so, TorchConfig.cmake) not found in ${INSTALL_DIR}."
        echo ""
        echo "   Archive structure may be different. Please check:"
        echo "     ls -la ${INSTALL_DIR}"
        echo "     find ${INSTALL_DIR} -name 'libtorch.so' -o -name 'TorchConfig.cmake'"
        return 1
    fi
}

# Helper: locate PyTorch cross-compilation prefix (TorchConfig.cmake)
detect_pytorch_cross_prefix() {
    local check_path
    # Priority: banana_pi/riscv_pytorch first, then environment variables, then other locations
    local candidates=(
        "${BANANA_PI_DIR}/riscv_pytorch"
        "${TORCH_PY_PREFIX:-}"
        "${PYTORCH_CROSS_PREFIX:-}"
        "${HOME}/riscv_pytorch"
        "${HOME}/pytorch-riscv-build/pytorch"
        "${HOME}/pytorch-riscv-build/install"
    )

    for check_path in "${candidates[@]}"; do
        [ -n "${check_path}" ] || continue
        if [ -f "${check_path}/share/cmake/Torch/TorchConfig.cmake" ]; then
            TORCH_PY_PREFIX="${check_path}"
            PYTORCH_CROSS_PREFIX="${check_path}"
            export TORCH_PY_PREFIX PYTORCH_CROSS_PREFIX
            return 0
        fi
        if [ -f "${check_path}/torch/share/cmake/Torch/TorchConfig.cmake" ]; then
            TORCH_PY_PREFIX="${check_path}/torch"
            PYTORCH_CROSS_PREFIX="${check_path}/torch"
            export TORCH_PY_PREFIX PYTORCH_CROSS_PREFIX
            return 0
        fi
    done

    return 1
}

# Helper: locate sgl-kernel directory
detect_sgl_kernel_dir() {
    if [ -n "${SGL_KERNEL_DIR}" ] && [ -d "${SGL_KERNEL_DIR}" ]; then
        return 0
    fi

    local candidates=(
        "${SCRIPT_DIR}/../sgl-kernel"
        "${SCRIPT_DIR}/../../sgl-kernel"
        "${HOME}/sgl-kernel"
    )

    # Add common project directory names if they exist
    for project_dir in "pllab-sglang" "sglang" "sglang-kernel"; do
        if [ -d "${HOME}/${project_dir}/sgl-kernel" ]; then
            candidates+=("${HOME}/${project_dir}/sgl-kernel")
        fi
    done

    for candidate in "${candidates[@]}"; do
        if [ -d "${candidate}" ]; then
            SGL_KERNEL_DIR="$(cd "${candidate}" && pwd)"
            export SGL_KERNEL_DIR
            return 0
        fi
    done

    return 1
}

# Step 1: Build wheel
if [ -z "${SKIP_BUILD}" ]; then
    echo "============================================"
    echo "Step 1: Building RISC-V wheel"
    echo "============================================"
    echo ""

    # Setup Clang 19 environment (integrated, no external script needed)
    echo "Setting up Clang 19 RISC-V environment..."
    if ! setup_clang19_env; then
        echo "❌ ERROR: Failed to setup Clang 19 environment."
        echo "   Please ensure Clang 19 is installed at ~/tools/clang19-riscv or set CLANG19_TOOLCHAIN_DIR."
        exit 1
    fi
    echo ""

    # Verify Clang 19
    if [ ! -f "${CC}" ]; then
        echo "❌ ERROR: Clang compiler not found at: ${CC}"
        exit 1
    fi

    CLANG_VERSION=$("${CC}" --version 2>/dev/null | head -1)
    MAJOR_VERSION=$(echo "${CLANG_VERSION}" | grep -oE "clang version ([0-9]+)" | grep -oE "[0-9]+" | head -1)
    if [ -z "${MAJOR_VERSION}" ] || [ "${MAJOR_VERSION}" -lt 19 ]; then
        echo "❌ ERROR: Clang 19 or later is required. Current version: ${MAJOR_VERSION}"
        echo "   Please build Clang 19 first or ensure it's properly configured."
        exit 1
    fi

    echo "Using: ${CLANG_VERSION}"
    echo ""

    # Clear environment variables if they point to old locations (to prioritize banana_pi/riscv_pytorch)
    if [ -n "${TORCH_PY_PREFIX:-}" ] && [ "${TORCH_PY_PREFIX}" != "${BANANA_PI_DIR}/riscv_pytorch" ] && [ "${TORCH_PY_PREFIX}" != "${BANANA_PI_DIR}/riscv_pytorch/torch" ]; then
        if [ ! -f "${TORCH_PY_PREFIX}/share/cmake/Torch/TorchConfig.cmake" ] && [ ! -f "${TORCH_PY_PREFIX}/torch/share/cmake/Torch/TorchConfig.cmake" ]; then
            # Old path doesn't exist, clear it to allow detection of banana_pi path
            unset TORCH_PY_PREFIX
            unset PYTORCH_CROSS_PREFIX
        fi
    fi

    # Ensure PyTorch cross-compilation prefix is available
    if detect_pytorch_cross_prefix; then
        echo "Using PyTorch cross prefix: ${TORCH_PY_PREFIX}"
    else
        echo "❌ PyTorch RISC-V cross-compilation package not found."
        echo ""
        echo "PyTorch RISC-V cross-compilation package is required for building sgl-kernel."
        echo ""
        echo "Searched locations:"
        BANANA_PI_PYTORCH_DIR="$(cd "${BANANA_PI_DIR}/riscv_pytorch" 2>/dev/null && pwd || echo "${BANANA_PI_DIR}/riscv_pytorch")"
        echo "  - ${BANANA_PI_PYTORCH_DIR} (banana_pi directory, priority)"
        echo "  - ${HOME}/riscv_pytorch"
        echo "  - ${HOME}/pytorch-riscv-build/pytorch"
        echo "  - ${HOME}/pytorch-riscv-build/install"
        echo "  - TORCH_PY_PREFIX environment variable (if set)"
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "How to obtain PyTorch RISC-V package:"
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        echo "Option 1: Download from GitHub Release (recommended)"
        echo "  The script can automatically download PyTorch RISC-V from:"
        echo "    https://github.com/nthu-pllab/pllab-sglang/releases/download/v1.1/pytorch-riscv.tar.gz"
        echo ""
        echo "Option 2: Manual installation"
        echo "  If you have a PyTorch RISC-V package archive locally:"
        echo "    mkdir -p ${BANANA_PI_DIR}/riscv_pytorch"
        echo "    tar -xzf /path/to/pytorch-riscv.tar.gz -C ${BANANA_PI_DIR}/riscv_pytorch"
        echo "    # Then re-run this script"
        echo ""
        echo "Option 3: Set custom path"
        echo "  If your PyTorch RISC-V package is elsewhere:"
        echo "    export TORCH_PY_PREFIX=/path/to/pytorch/riscv/package"
        echo "    ./build_and_deploy_sgl-kernel.sh"
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo ""

        if [ "$SKIP_CONFIRM" = false ]; then
            echo "What would you like to do?"
            echo "  1) Download from GitHub Release automatically (recommended)"
            echo "  2) Exit and install manually"
            echo "  3) Continue anyway (will fail at build step)"
            echo ""
            read -p "Enter choice [1-3] (default: 1): " PYTORCH_CHOICE
            PYTORCH_CHOICE="${PYTORCH_CHOICE:-1}"

            case "${PYTORCH_CHOICE}" in
                1)
                    echo ""
                    echo "Downloading PyTorch RISC-V from GitHub Release..."
                    if download_pytorch_riscv_from_github; then
                        # Re-detect after download
                        if detect_pytorch_cross_prefix; then
                            echo "✅ PyTorch RISC-V installed and detected: ${TORCH_PY_PREFIX}"
                        else
                            echo "❌ ERROR: PyTorch RISC-V was downloaded but not detected correctly."
                            echo "   Please check the installation at ${BANANA_PI_DIR}/riscv_pytorch"
                            exit 1
                        fi
                    else
                        echo "❌ ERROR: Failed to download PyTorch RISC-V from GitHub Release."
                        echo ""
                        echo "You can try:"
                        echo "  1. Check your network connection"
                        echo "  2. Verify the release exists: https://github.com/nthu-pllab/pllab-sglang/releases"
                        echo "  3. Set GITHUB_TOKEN if repository is private"
                        echo "  4. Download manually and use Option 2"
                        exit 1
                    fi
                    ;;
                2)
                    echo "Exiting. Please install PyTorch RISC-V package and try again."
                    exit 1
                    ;;
                3)
                    echo "⚠️  Continuing without PyTorch (build will likely fail)"
                    echo "   You can set TORCH_PY_PREFIX later and re-run with --skip-clang-build"
                    ;;
                *)
                    echo "Invalid choice, exiting..."
                    exit 1
                    ;;
            esac
        else
            # Non-interactive mode: try to download automatically
            echo "Non-interactive mode: attempting to download PyTorch RISC-V from GitHub Release..."
            if download_pytorch_riscv_from_github; then
                # Re-detect after download
                if detect_pytorch_cross_prefix; then
                    echo "✅ PyTorch RISC-V installed and detected: ${TORCH_PY_PREFIX}"
                else
                    echo "❌ ERROR: PyTorch RISC-V was downloaded but not detected correctly."
                    echo "   Please check the installation at ${BANANA_PI_DIR}/riscv_pytorch"
                    exit 1
                fi
            else
                echo "❌ ERROR: Failed to download PyTorch RISC-V from GitHub Release."
                echo "   Please install manually or check network connection."
                exit 1
            fi
        fi
    fi

    # Get sgl-kernel directory
    if ! detect_sgl_kernel_dir; then
        echo "❌ ERROR: Unable to locate sgl-kernel directory. Set SGL_KERNEL_DIR env var."
        exit 1
    fi

    cd "${SGL_KERNEL_DIR}"
    echo "Working directory: $(pwd)"
    echo ""

    # Clean previous builds
    echo "Cleaning previous builds..."
    rm -rf build/ dist/ *.egg-info
    find . -maxdepth 3 -name "*.so" -type f -delete 2>/dev/null || true
    find . -maxdepth 3 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "✅ Clean complete"
    echo ""

    # Set environment variables for cross-compilation
    export CMAKE_SYSTEM_PROCESSOR="riscv64"
    export CMAKE_TOOLCHAIN_FILE="${SGL_KERNEL_DIR}/cmake/riscv-toolchain.cmake"

    if [ -n "${TOOLCHAIN}" ]; then
        export TOOLCHAIN_BIN="${TOOLCHAIN}/bin"
    fi

    if [ -n "${RISCV_SYSROOT}" ]; then
        export SYSROOT="${RISCV_SYSROOT}"
    fi

    # Test compile a simple RISC-V program
    echo "Testing RISC-V cross-compilation..."
    cat > /tmp/test_riscv.c << 'EOF'
int main() { return 0; }
EOF

    if "${CC}" ${CFLAGS} /tmp/test_riscv.c -o /tmp/test_riscv 2>/dev/null; then
        if file /tmp/test_riscv | grep -qi "riscv\|RISC-V"; then
            echo "✅ RISC-V cross-compilation test passed"
            rm -f /tmp/test_riscv /tmp/test_riscv.c
        else
            echo "❌ ERROR: Compiled binary is not RISC-V"
            rm -f /tmp/test_riscv /tmp/test_riscv.c
            exit 1
        fi
    else
        echo "❌ ERROR: RISC-V cross-compilation test failed"
        rm -f /tmp/test_riscv /tmp/test_riscv.c
        exit 1
    fi
    echo ""

    # Configure additional CMake args for cross-compilation (disable CUDA/ROCM auto-detection)
    EXTRA_CMAKE_ARGS="-DSGL_ENABLE_CUDA=OFF;-DSGL_ENABLE_ROCM=OFF;-DSGL_ENABLE_VULKAN=OFF;-DCMAKE_CUDA_COMPILER=;-DCUDA_TOOLKIT_ROOT_DIR=;-DUSE_CUDA=OFF;-DUSE_NCCL=OFF;-DUSE_ROCM=OFF"
    if [ -n "${CMAKE_TOOLCHAIN_FILE:-}" ]; then
        EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS};-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
    fi
    if [ -n "${SKBUILD_CMAKE_ARGS:-}" ]; then
        export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS};${EXTRA_CMAKE_ARGS}"
    else
        export SKBUILD_CMAKE_ARGS="${EXTRA_CMAKE_ARGS}"
    fi
    export SGL_ENABLE_CUDA=OFF
    export SGL_ENABLE_ROCM=OFF
    export SGL_ENABLE_VULKAN=OFF

    # Switch to RVV CMakeLists.txt for RISC-V build
    CPU_CMAKE_DIR="${SGL_KERNEL_DIR}/csrc/cpu"
    RVV_CMAKE="${CPU_CMAKE_DIR}/RVVCMakeLists.txt"
    ORIG_CMAKE="${CPU_CMAKE_DIR}/CMakeLists.txt"
    BACKUP_CMAKE="${CPU_CMAKE_DIR}/CMakeLists.txt.orig"

    if [ -f "${RVV_CMAKE}" ]; then
        echo "Switching to RVV CMakeLists.txt for RISC-V build..."
        # Backup original CMakeLists.txt if not already backed up
        if [ ! -f "${BACKUP_CMAKE}" ]; then
            cp "${ORIG_CMAKE}" "${BACKUP_CMAKE}"
        fi
        # Use RVV CMakeLists.txt
        cp "${RVV_CMAKE}" "${ORIG_CMAKE}"
        echo "✅ Using RVVCMakeLists.txt for RVV kernel build"

        # Set cleanup trap to restore original CMakeLists.txt
        trap 'if [ -f "${BACKUP_CMAKE}" ]; then cp "${BACKUP_CMAKE}" "${ORIG_CMAKE}"; rm -f "${BACKUP_CMAKE}"; echo "Restored original CMakeLists.txt"; fi' EXIT
    else
        echo "⚠️  RVVCMakeLists.txt not found at ${RVV_CMAKE}"
        echo "   Using existing CMakeLists.txt (must have RVV support)"
    fi

    # Build wheel
    echo "Building sglang-kernel wheel for RISC-V..."
    export _PYTHON_HOST_PLATFORM="linux-riscv64"

    CMAKE_SOURCE_SETTING="--config-setting=cmake.source-dir=csrc/cpu"
    if command -v uv >/dev/null 2>&1; then
        BUILD_CMD="uv build --wheel -Cbuild-dir=build . --no-build-isolation ${CMAKE_SOURCE_SETTING}"
    elif python3 -m pip show build >/dev/null 2>&1; then
        BUILD_CMD="python3 -m build --wheel ${CMAKE_SOURCE_SETTING}"
    else
        pip install build scikit-build-core wheel
        BUILD_CMD="python3 -m build --wheel ${CMAKE_SOURCE_SETTING}"
    fi

    if ! ${BUILD_CMD} 2>&1 | tee build_riscv_wheel.log; then
        echo "❌ ERROR: Wheel build failed. See build_riscv_wheel.log for details."
        exit 1
    fi

    echo ""
    echo "✅ Step 1 complete: Wheel built successfully"
    echo ""
else
    echo "⏭️  Skipping build step (SKIP_BUILD is set)"
    echo ""
fi

# Find wheel file
if ! detect_sgl_kernel_dir; then
    echo "❌ ERROR: Unable to locate sgl-kernel directory. Set SGL_KERNEL_DIR env var."
    exit 1
fi
WHEEL_FILE=$(find "${SGL_KERNEL_DIR}/dist" -name "sgl_kernel-*linux_riscv64.whl" -o -name "sgl_kernel-*.whl" 2>/dev/null | head -1)

if [ -z "${WHEEL_FILE}" ] || [ ! -f "${WHEEL_FILE}" ]; then
    echo "❌ ERROR: Wheel file not found"
    echo "   Searched in: ${SGL_KERNEL_DIR}/dist"
    exit 1
fi

WHEEL_BASENAME=$(basename "${WHEEL_FILE}")
WHEEL_SIZE=$(stat -c%s "${WHEEL_FILE}" 2>/dev/null || stat -f%z "${WHEEL_FILE}" 2>/dev/null || echo "0")
echo "Wheel file: ${WHEEL_BASENAME} ($(numfmt --to=iec-i --suffix=B ${WHEEL_SIZE} 2>/dev/null || echo "${WHEEL_SIZE} bytes"))"
echo ""

# Step 2: Transfer wheel to Banana Pi
if [ -z "${SKIP_TRANSFER}" ]; then
    echo "============================================"
    echo "Step 2: Transferring wheel to Banana Pi"
    echo "============================================"
    echo ""

    echo "Testing SSH connection..."
    if ! ssh_cmd -o ConnectTimeout=5 -o BatchMode=yes "${BANANA_PI_USER}@${BANANA_PI_HOST}" "echo 'SSH connection successful'" 2>/dev/null; then
        echo "⚠️  WARNING: SSH key authentication may not be set up"
        echo "   You may be prompted for password"
    fi

    echo "Transferring ${WHEEL_BASENAME} to ${BANANA_PI_USER}@${BANANA_PI_HOST}..."
    scp_cmd "${WHEEL_FILE}" "${BANANA_PI_USER}@${BANANA_PI_HOST}:~/${WHEEL_BASENAME}" || {
        echo "❌ ERROR: Failed to transfer wheel"
        exit 1
    }

    echo "Syncing banana_pi tools to ${BANANA_PI_USER}@${BANANA_PI_HOST}:${REMOTE_BANANA_PI_DIR}..."
    echo "Note: Excluding riscv_pytorch/ (only needed for cross-compilation on x86 host)"
    ssh_cmd "${BANANA_PI_USER}@${BANANA_PI_HOST}" "mkdir -p ${REMOTE_BANANA_PI_DIR}" || {
        echo "❌ ERROR: Failed to create remote directory ${REMOTE_BANANA_PI_DIR}"
        exit 1
    }

    if command -v rsync >/dev/null 2>&1; then
        # Exclude riscv_pytorch/ - it's only needed for cross-compilation, not on Banana Pi
        if ! rsync -a --delete --exclude='riscv_pytorch' -e "${SSH_BIN}" "${BANANA_PI_DIR}/" "${BANANA_PI_USER}@${BANANA_PI_HOST}:${REMOTE_BANANA_PI_DIR}/"; then
            rsync -a --delete --exclude='riscv_pytorch' -e "env LD_LIBRARY_PATH= ${SSH_BIN}" "${BANANA_PI_DIR}/" "${BANANA_PI_USER}@${BANANA_PI_HOST}:${REMOTE_BANANA_PI_DIR}/" || {
                echo "❌ ERROR: Failed to sync banana_pi directory via rsync"
                exit 1
            }
        fi
    else
        # For tar, we need to exclude riscv_pytorch directory
        if ! tar -C "${BANANA_PI_DIR}" --exclude='riscv_pytorch' -cf - . | "${SSH_BIN}" "${BANANA_PI_USER}@${BANANA_PI_HOST}" "tar -C ${REMOTE_BANANA_PI_DIR} -xf -"; then
            tar -C "${BANANA_PI_DIR}" --exclude='riscv_pytorch' -cf - . | env LD_LIBRARY_PATH= "${SSH_BIN}" "${BANANA_PI_USER}@${BANANA_PI_HOST}" "tar -C ${REMOTE_BANANA_PI_DIR} -xf -" || {
                echo "❌ ERROR: Failed to sync banana_pi directory via tar"
                exit 1
            }
        fi
    fi

    echo "✅ Step 2 complete: Wheel transferred successfully"
    echo ""
else
    echo "⏭️  Skipping transfer step (SKIP_TRANSFER is set)"
    echo ""
fi

# Step 3: Install wheel on Banana Pi
if [ -z "${SKIP_INSTALL}" ]; then
    echo "============================================"
    echo "Step 3: Installing wheel on Banana Pi"
    echo "============================================"
    echo ""

    ssh_cmd "${BANANA_PI_USER}@${BANANA_PI_HOST}" << 'ENDSSH'
set -e

# Activate virtual environment if exists
if [ -d ~/.local_riscv_env/workspace/venv_sglang ]; then
    source ~/.local_riscv_env/workspace/venv_sglang/bin/activate
    echo "✓ Activated virtual environment"
fi

# Set up OpenMP library paths for RISC-V
# These are required for the custom-built OpenMP runtime
if [ -f ~/.local/lib/libomp.so ]; then
    export LD_PRELOAD=~/.local/lib/libomp.so
    export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH
    echo "✓ Set up OpenMP library paths"
else
    echo "⚠️  Warning: ~/.local/lib/libomp.so not found, OpenMP may not work"
fi

# Find wheel file
    WHEEL_FILE=$(find "$HOME" -maxdepth 1 -name 'sgl_kernel-*.whl' -print 2>/dev/null | sort | tail -1)
    if [ -z "${WHEEL_FILE}" ] || [ ! -f "${WHEEL_FILE}" ]; then
        WHEEL_FILE=~/sgl_kernel.whl
    fi

    if [ ! -f "${WHEEL_FILE}" ]; then
        echo "❌ ERROR: Wheel file not found on Banana Pi"
        exit 1
    fi

    echo "Installing wheel: ${WHEEL_FILE}"

    # Clean up any broken installation
    echo "Cleaning up old installation..."

    # Try to uninstall old version first (ignore errors)
    pip uninstall -y sgl-kernel 2>/dev/null || true

    # Manually remove package directories to avoid version parsing issues
    PYTHON_SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || echo "")
    if [ -n "${PYTHON_SITE}" ] && [ -d "${PYTHON_SITE}/sgl_kernel" ]; then
        echo "Removing ${PYTHON_SITE}/sgl_kernel..."
        rm -rf "${PYTHON_SITE}/sgl_kernel" 2>/dev/null || true
    fi

    # Also check in user site-packages
    PYTHON_USER_SITE=$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || echo "")
    if [ -n "${PYTHON_USER_SITE}" ] && [ -d "${PYTHON_USER_SITE}/sgl_kernel" ]; then
        echo "Removing ${PYTHON_USER_SITE}/sgl_kernel..."
        rm -rf "${PYTHON_USER_SITE}/sgl_kernel" 2>/dev/null || true
    fi

    # Remove any .dist-info directories for sgl-kernel
    find "${PYTHON_SITE}" "${PYTHON_USER_SITE}" -name "sgl_kernel*.dist-info" -type d 2>/dev/null | while read dist_info; do
        echo "Removing ${dist_info}..."
        rm -rf "${dist_info}" 2>/dev/null || true
    done

    # Install new wheel with --ignore-installed to skip all checks
    echo "Installing new wheel..."
    pip install --ignore-installed --no-deps "${WHEEL_FILE}" || {
    echo "❌ ERROR: Failed to install wheel"
    exit 1
}

echo "✓ sgl-kernel installed successfully"
echo ""

# Verify installation
echo "Verifying installation..."
echo "LD_PRELOAD=$LD_PRELOAD"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
python3 -c "import sgl_kernel; print('✓ sgl_kernel imported successfully')" || {
    echo "❌ ERROR: Failed to import sgl_kernel"
    exit 1
}

python3 -c "import torch; print('✓ PyTorch ops available:', hasattr(torch.ops.sgl_kernel, 'decode_attention_cpu'))" || {
    echo "⚠️  WARNING: decode_attention_cpu not available"
}

echo ""
ENDSSH

    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Installation failed on Banana Pi"
        exit 1
    fi

    echo "✅ Step 3 complete: Wheel installed successfully"
    echo ""
else
    echo "⏭️  Skipping install step (SKIP_INSTALL is set)"
    echo ""
fi


echo "============================================"
echo "Deployment Complete!"
echo "============================================"
echo ""
echo "Summary:"
echo "  Wheel file: ${WHEEL_BASENAME}"
echo "  Target: ${BANANA_PI_USER}@${BANANA_PI_HOST}"
echo "  Status: ✅ Installed and verified"
echo ""
