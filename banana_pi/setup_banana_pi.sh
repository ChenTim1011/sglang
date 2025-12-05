#!/bin/bash
# SGLang RISC-V Setup Script for Banana Pi
# This script helps set up SGLang on Banana Pi for running test_tinyllama_rvv.py
#
# Features:
# - Clone sglang repo (from pllab-sglang/riscv_backend branch)
# - Install Python dependencies
# - Install wheels (from GitHub Releases)
# - Install libomp library (from GitHub Releases)
# - Configure environment
# - Run tests
#
# Usage: ./setup_banana_pi.sh [--user USER] [--host HOST] [--yes] [--skip-wheels] [--skip-test] [--wheels-tag TAG]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
BANANA_PI_USER="${BANANA_PI_USER:-jtchen}"
BANANA_PI_HOST="${BANANA_PI_HOST:-140.114.78.64}"
SKIP_CONFIRM=false
SKIP_WHEELS=false
SKIP_TEST=true
SGLANG_REPO="https://github.com/nthu-pllab/pllab-sglang.git"
SGLANG_BRANCH="riscv_backend"
WORKSPACE_DIR="$HOME/.local_riscv_env/workspace"
PROJECT_DIR="$WORKSPACE_DIR/sglang"
# GitHub Releases configuration for wheels
REPO_OWNER="nthu-pllab"
REPO_NAME="pllab-sglang"
WHEELS_RELEASE_TAG="${WHEELS_RELEASE_TAG:-v1.0}"  # Release tag for wheels
GITHUB_TOKEN="${GITHUB_TOKEN:-}"  # GitHub token for private repo access
# Use system SSH and SCP
SSH_CMD="/usr/bin/ssh"
SCP_CMD="/usr/bin/scp"

# Force use system OpenSSL libraries to avoid conda OpenSSL version mismatch
# Remove conda lib paths from LD_LIBRARY_PATH to prevent conda OpenSSL from being used
if [ -n "$CONDA_PREFIX" ]; then
    # Remove conda lib paths from LD_LIBRARY_PATH
    if [ -n "$LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "$CONDA_PREFIX" | tr '\n' ':' | sed 's/:$//')
    fi
fi

# Find and prioritize system OpenSSL libraries
SYSTEM_SSL_LIB=""
for lib_path in /usr/lib/x86_64-linux-gnu /usr/lib64 /lib/x86_64-linux-gnu /lib64; do
    if [ -f "$lib_path/libssl.so.3" ] || [ -f "$lib_path/libssl.so.1.1" ]; then
        SYSTEM_SSL_LIB="$lib_path"
        break
    fi
done

# Prepend system SSL library to LD_LIBRARY_PATH to ensure system OpenSSL is used
if [ -n "$SYSTEM_SSL_LIB" ]; then
    export LD_LIBRARY_PATH="$SYSTEM_SSL_LIB:${LD_LIBRARY_PATH}"
fi

# Unset conda's OpenSSL-related environment variables if they exist
unset OPENSSL_CONF 2>/dev/null || true

# Parse arguments
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
        --skip-wheels)
            SKIP_WHEELS=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --wheels-tag)
            WHEELS_RELEASE_TAG="$2"
            shift 2
            ;;
        --github-token)
            GITHUB_TOKEN="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --user USER      Banana Pi username (default: jtchen)"
            echo "  --host HOST      Banana Pi host/IP (default: 140.114.78.64)"
            echo "  --yes, -y        Skip confirmation prompts"
            echo "  --skip-wheels    Skip wheel installation"
            echo "  --skip-test      Skip running test after setup"
            echo "  --wheels-tag TAG  Release tag for wheels (default: v1.0)"
            echo "  --github-token TOKEN  GitHub token for private repo access (or set GITHUB_TOKEN env var, or will be prompted)"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  BANANA_PI_USER   Banana Pi username"
            echo "  BANANA_PI_HOST   Banana Pi host/IP"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Logging functions
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

# Display usage information (always show)
echo ""
echo "============================================================"
echo "  SGLang RISC-V Setup for Banana Pi"
echo "============================================================"
echo ""
echo "This script will:"
echo "  1. Clone/update sglang repository (pllab-sglang/riscv_backend)"
echo "  2. Download wheels and libomp from GitHub Releases"
echo "  3. Install Python dependencies"
echo "  4. Configure environment"
echo ""
echo "Options:"
echo "  --user USER         Banana Pi username (default: $BANANA_PI_USER)"
echo "  --host HOST         Banana Pi host/IP (default: $BANANA_PI_HOST)"
echo "  --yes, -y           Skip confirmation prompts"
echo "  --skip-wheels       Skip wheel installation"
echo "  --skip-test         Skip running test after setup"
echo "  --wheels-tag TAG    Release tag for wheels (default: v1.0)"
echo "  --github-token TOKEN  GitHub token for private repo (or will be prompted)"
echo "  --help, -h          Show detailed help"
echo ""

# Interactive configuration
if [ "$SKIP_CONFIRM" = false ]; then
    read -p "Banana Pi Username [$BANANA_PI_USER]: " input_user
    BANANA_PI_USER="${input_user:-$BANANA_PI_USER}"

    read -p "Banana Pi Host/IP [$BANANA_PI_HOST]: " input_host
    BANANA_PI_HOST="${input_host:-$BANANA_PI_HOST}"

    # Ask for GitHub token if not already set
    if [ -z "$GITHUB_TOKEN" ]; then
        echo ""
        log_info "The repository is private and requires authentication."
        log_info "GitHub token will be used for:"
        log_info "  1. Downloading Release files"
        log_info "  2. Git operations (clone/fetch/pull) if needed"
        echo ""
        read -s -p "GitHub Personal Access Token (PAT): " input_token
        echo ""
        GITHUB_TOKEN="${input_token}"
        if [ -n "$GITHUB_TOKEN" ]; then
            log_info "✓ GitHub token will be used for authentication"
        else
            log_warn "No GitHub token provided."
            log_warn "  - Release downloads may fail"
            log_warn "  - Git operations will prompt for username/password"
            log_warn "  - Password prompt: Use your GitHub PAT as the password"
        fi
    else
        log_info "✓ GitHub token already provided (will be used for Git operations and Release downloads)"
    fi
    echo ""
fi

log_info "Target: $BANANA_PI_USER@$BANANA_PI_HOST"
log_info "Workspace: $WORKSPACE_DIR"
log_info "Project: $PROJECT_DIR"
echo ""

# Verify SSH command exists
if [ ! -x "$SSH_CMD" ]; then
    log_error "System SSH not found: $SSH_CMD"
    log_info "Please install OpenSSH: sudo apt-get install openssh-client"
    exit 1
fi

log_info "Using SSH: $SSH_CMD"
log_info "Using SCP: $SCP_CMD"

# Check SSH connection (with proper OpenSSL environment)
log_step "Checking SSH connection..."
set +e
SSH_CONNECTION_TEST=$(timeout 10 "$SSH_CMD" -o ConnectTimeout=5 -o BatchMode=yes "$BANANA_PI_USER@$BANANA_PI_HOST" exit 2>&1)
SSH_CONNECTION_EXIT=$?
set -e

if [ $SSH_CONNECTION_EXIT -eq 0 ]; then
    log_info "✓ SSH connection OK"
elif [ $SSH_CONNECTION_EXIT -eq 124 ]; then
    log_error "SSH connection timeout (10 seconds)"
    log_info "Please check:"
    log_info "  1. Host/IP is correct: $BANANA_PI_HOST"
    log_info "  2. SSH service is running on Banana Pi"
    log_info "  3. Network connectivity"
    exit 1
elif [ $SSH_CONNECTION_EXIT -eq 255 ]; then
    log_warn "Cannot connect via SSH without password"
    log_info "You may be asked for password during setup"
else
    log_warn "SSH connection failed (exit code: $SSH_CONNECTION_EXIT)"
    if [ -n "$SSH_CONNECTION_TEST" ]; then
        # Filter out OpenSSL warnings but show other errors
        FILTERED_TEST=$(echo "$SSH_CONNECTION_TEST" | grep -v "OpenSSL version mismatch" || true)
        if [ -n "$FILTERED_TEST" ]; then
            log_info "Error details:"
            echo "$FILTERED_TEST" | while IFS= read -r line; do
                echo "  $line"
            done
        fi
    fi
    log_info "You may be asked for password during setup"
fi
echo ""

# Remote setup script
REMOTE_SETUP_SCRIPT=$(cat << 'REMOTE_SCRIPT_EOF'
#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

WORKSPACE_DIR="$HOME/.local_riscv_env/workspace"
PROJECT_DIR="$WORKSPACE_DIR/sglang"
SGLANG_REPO="https://github.com/nthu-pllab/pllab-sglang.git"
SGLANG_BRANCH="riscv_backend"
WHEEL_BUILDER_URL="https://gitlab.com/api/v4/projects/riseproject%2Fpython%2Fwheel_builder/packages/pypi/simple"

# Step 1: Create workspace directory
log_step "Creating workspace directory..."
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"
log_info "✓ Workspace directory created"

# Step 2: Clone or update sglang repo
log_step "Setting up sglang repository..."

if [ -d "$PROJECT_DIR/.git" ]; then
    # Repository exists - ask user what to do
    log_info "Repository already exists at: $PROJECT_DIR"
    echo ""
    echo "What would you like to do?"
    echo "  1) Update existing repository (git pull)"
    echo "  2) Remove and re-clone (fresh start)"
    echo "  3) Skip repository setup (use existing)"
    echo ""

    if [ "$SKIP_CONFIRM" = "true" ]; then
        # Non-interactive mode: default to update
        REPO_ACTION="1"
        log_info "Non-interactive mode: updating existing repository"
    else
        read -p "Enter choice [1-3] (default: 1): " REPO_ACTION
        REPO_ACTION="${REPO_ACTION:-1}"
    fi

    case "$REPO_ACTION" in
        1)
            log_info "Updating existing repository..."
            cd "$PROJECT_DIR"
            # Check and update remote URL if needed
            CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
            if [ "$CURRENT_REMOTE" != "$SGLANG_REPO" ]; then
                log_info "Updating remote URL to: $SGLANG_REPO"
                git remote set-url origin "$SGLANG_REPO" 2>/dev/null || git remote add origin "$SGLANG_REPO"
            fi

            # Configure Git to use token if available (for private repos)
            if [ -n "$GITHUB_TOKEN" ]; then
                # Use token in URL for authentication (only for this session)
                GIT_CREDENTIAL_URL=$(echo "$SGLANG_REPO" | sed "s|https://|https://${GITHUB_TOKEN}@|")
                git remote set-url origin "$GIT_CREDENTIAL_URL" 2>/dev/null
            fi

            # Stash local changes before switching branches
            if ! git diff-index --quiet HEAD -- 2>/dev/null; then
                log_info "Stashing local changes before updating..."
                git stash push -m "Auto-stash before branch switch $(date +%Y%m%d_%H%M%S)" 2>/dev/null || log_warn "Could not stash changes"
            fi

            # Handle untracked files that might conflict with incoming changes
            # Check if there are untracked files that would be overwritten
            git fetch origin
            CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
            if [ "$CURRENT_BRANCH" != "$SGLANG_BRANCH" ]; then
                log_info "Switching to branch: $SGLANG_BRANCH"
                git checkout "$SGLANG_BRANCH" 2>/dev/null || git checkout -b "$SGLANG_BRANCH" origin/"$SGLANG_BRANCH" 2>/dev/null || {
                    log_warn "Could not switch to branch, trying to reset..."
                    git reset --hard origin/"$SGLANG_BRANCH" 2>/dev/null || log_warn "Could not reset branch"
                }
            fi

            # Try to pull and handle conflicts
            set +e
            PULL_OUTPUT=$(git pull origin "$SGLANG_BRANCH" 2>&1)
            PULL_EXIT=$?
            set -e

            # Check for different types of conflicts
            if echo "$PULL_OUTPUT" | grep -q "untracked working tree files would be overwritten"; then
                # Untracked files conflict - stop and ask user
                log_error "Untracked files detected that would be overwritten by merge!"
                echo ""
                echo "The following untracked files would be overwritten:"
                CONFLICTING_FILES=$(echo "$PULL_OUTPUT" | grep "would be overwritten by merge" | sed 's/.*: *//' | sort -u)
                echo "$CONFLICTING_FILES" | while read -r file; do
                    if [ -n "$file" ]; then
                        echo "  - $file"
                    fi
                done
                echo ""
                echo "How would you like to resolve this?"
                echo "  1) Remove conflicting untracked files and pull (use remote version)"
                echo "  2) Keep local files and skip update"
                echo "  3) Manually handle files (script will exit, you handle manually)"
                echo "  4) Continue anyway (may cause errors)"
                echo ""

                if [ "$SKIP_CONFIRM" = "true" ]; then
                    # Non-interactive mode: default to remove conflicting files
                    UNTRACKED_CHOICE="1"
                    log_info "Non-interactive mode: removing conflicting untracked files"
                else
                    read -p "Enter choice [1-4] (default: 1): " UNTRACKED_CHOICE
                    UNTRACKED_CHOICE="${UNTRACKED_CHOICE:-1}"
                fi

                case "$UNTRACKED_CHOICE" in
                    1)
                        log_info "Removing conflicting untracked files..."
                        echo "$CONFLICTING_FILES" | while read -r file; do
                            if [ -n "$file" ] && ([ -f "$file" ] || [ -d "$file" ]); then
                                log_info "  Removing: $file"
                                rm -rf "$file" 2>/dev/null || log_warn "    Could not remove: $file"
                            fi
                        done
                        # Retry pull after removing conflicting files
                        git pull origin "$SGLANG_BRANCH" || log_warn "Could not pull after cleanup, continuing..."
                        ;;
                    2)
                        log_info "Keeping local files and skipping update..."
                        log_warn "Using existing local version (not updated)"
                        ;;
                    3)
                        log_info "Exiting script for manual file handling"
                        echo ""
                        log_info "To handle conflicting untracked files and push to remote, follow these steps:"
                        echo ""
                        log_info "1. Navigate to the project directory:"
                        echo "   cd $PROJECT_DIR"
                        echo ""
                        log_info "2. Handle the conflicting files (choose one):"
                        echo "   Option A - Remove conflicting files:"
                        echo "     rm <conflicting-file1> <conflicting-file2> ..."
                        echo ""
                        echo "   Option B - Move/rename conflicting files:"
                        echo "     mv <conflicting-file> <new-location-or-name>"
                        echo ""
                        echo "   Option C - Add files to git (if you want to keep them):"
                        echo "     git add <file1> <file2> ..."
                        echo "     git commit -m 'Add local files'"
                        echo ""
                        log_info "3. After handling files, pull from remote:"
                        echo "   git pull origin $SGLANG_BRANCH"
                        echo ""
                        log_info "4. If you added/modified files and want to push to remote:"
                        echo "   git push origin $SGLANG_BRANCH"
                        echo ""
                        log_info "5. Re-run this setup script:"
                        echo "   ./setup_banana_pi.sh"
                        echo ""
                        exit 1
                        ;;
                    4)
                        log_warn "Continuing with conflicting untracked files (may cause errors)"
                        ;;
                    *)
                        log_warn "Invalid choice, removing conflicting untracked files"
                        echo "$CONFLICTING_FILES" | while read -r file; do
                            if [ -n "$file" ] && ([ -f "$file" ] || [ -d "$file" ]); then
                                log_info "  Removing: $file"
                                rm -rf "$file" 2>/dev/null || log_warn "    Could not remove: $file"
                            fi
                        done
                        git pull origin "$SGLANG_BRANCH" || log_warn "Could not pull after cleanup, continuing..."
                        ;;
                esac
            elif echo "$PULL_OUTPUT" | grep -qE "(needs merge|unmerged files|unresolved conflict|Pulling is not possible)"; then
                # Merge conflict detected - stop and ask user
                log_error "Git merge conflict detected!"
                echo ""
                echo "The following files have merge conflicts:"
                # Get unmerged files using multiple methods
                UNMERGED_FILES=$(git diff --name-only --diff-filter=U 2>/dev/null || git status --short | grep -E "^UU|^AA|^DD" | awk '{print $2}' || echo "")
                if [ -n "$UNMERGED_FILES" ]; then
                    echo "$UNMERGED_FILES" | while read -r file; do
                        if [ -n "$file" ]; then
                            echo "  - $file"
                        fi
                    done
                else
                    # Fallback: show git status
                    git status --short | grep -E "^UU|^AA|^DD" | while read -r line; do
                        echo "  $line"
                    done
                fi
                echo ""
                echo "How would you like to resolve this?"
                echo "  1) Abort merge and use remote version (discard local changes)"
                echo "  2) Abort merge and keep local version (skip update)"
                echo "  3) Manually resolve conflicts (script will exit, you resolve manually)"
                echo "  4) Continue anyway (may cause errors)"
                echo ""

                if [ "$SKIP_CONFIRM" = "true" ]; then
                    # Non-interactive mode: default to abort and use remote
                    CONFLICT_CHOICE="1"
                    log_info "Non-interactive mode: aborting merge and using remote version"
                else
                    read -p "Enter choice [1-4] (default: 1): " CONFLICT_CHOICE
                    CONFLICT_CHOICE="${CONFLICT_CHOICE:-1}"
                fi

                case "$CONFLICT_CHOICE" in
                    1)
                        log_info "Aborting merge and using remote version..."
                        git merge --abort 2>/dev/null || true
                        git reset --hard origin/"$SGLANG_BRANCH" 2>/dev/null || {
                            log_error "Failed to reset to remote version"
                            exit 1
                        }
                        log_info "✓ Repository reset to remote version"
                        ;;
                    2)
                        log_info "Aborting merge and keeping local version..."
                        git merge --abort 2>/dev/null || true
                        log_warn "Using existing local version (not updated)"
                        ;;
                    3)
                        log_info "Exiting script for manual conflict resolution"
                        echo ""
                        log_info "To resolve conflicts and push to remote, follow these steps:"
                        echo ""
                        log_info "1. Navigate to the project directory:"
                        echo "   cd $PROJECT_DIR"
                        echo ""
                        log_info "2. Resolve conflicts in the files (edit files to fix conflicts):"
                        echo "   - Edit the conflicted files and resolve merge markers"
                        echo "   - Or use: git checkout --ours <file>  (use local version)"
                        echo "   - Or use: git checkout --theirs <file> (use remote version)"
                        echo ""
                        log_info "3. Stage the resolved files:"
                        echo "   git add <resolved-file1> <resolved-file2> ..."
                        echo "   # Or stage all resolved files:"
                        echo "   git add -u"
                        echo ""
                        log_info "4. Commit the merge:"
                        echo "   git commit -m 'Resolve merge conflicts'"
                        echo ""
                        log_info "5. Push to remote (if you have push access):"
                        echo "   git push origin $SGLANG_BRANCH"
                        echo ""
                        log_info "6. Re-run this setup script:"
                        echo "   ./setup_banana_pi.sh"
                        echo ""
                        exit 1
                        ;;
                    4)
                        log_warn "Continuing with unresolved conflicts (may cause errors)"
                        ;;
                    *)
                        log_warn "Invalid choice, aborting merge and using remote version"
                        git merge --abort 2>/dev/null || true
                        git reset --hard origin/"$SGLANG_BRANCH" 2>/dev/null || {
                            log_error "Failed to reset to remote version"
                            exit 1
                        }
                        ;;
                esac
            elif [ $PULL_EXIT -ne 0 ]; then
                # Other pull errors
                log_warn "Could not pull (exit code: $PULL_EXIT)"
                log_info "Pull output:"
                echo "$PULL_OUTPUT" | head -10
                log_warn "Continuing with existing repository state..."
            fi

            # Restore remote URL without token (for security)
            if [ -n "$GITHUB_TOKEN" ]; then
                git remote set-url origin "$SGLANG_REPO" 2>/dev/null
            fi

            log_info "✓ Repository updated"
            ;;
        2)
        log_info "Removing existing repository and cloning fresh copy..."
        rm -rf "$PROJECT_DIR"
        # Use token in URL for authentication if available (for private repos)
        CLONE_URL="$SGLANG_REPO"
        if [ -n "$GITHUB_TOKEN" ]; then
            CLONE_URL=$(echo "$SGLANG_REPO" | sed "s|https://|https://${GITHUB_TOKEN}@|")
        fi
        git clone -b "$SGLANG_BRANCH" "$CLONE_URL" "$PROJECT_DIR" || {
            log_error "Failed to clone repository"
            if [ -z "$GITHUB_TOKEN" ]; then
                log_info "💡 Tip: For private repositories, you may need to provide a GitHub token."
                log_info "   Password prompt: Use your GitHub Personal Access Token (PAT) as the password."
            fi
            exit 1
        }
        # Restore remote URL without token (for security)
        if [ -n "$GITHUB_TOKEN" ]; then
            cd "$PROJECT_DIR"
            git remote set-url origin "$SGLANG_REPO" 2>/dev/null
        fi
        log_info "✓ Repository cloned"
            ;;
        3)
            log_info "Skipping repository setup, using existing repository"
            ;;
        *)
            log_warn "Invalid choice, defaulting to update"
            cd "$PROJECT_DIR"
            # Check and update remote URL if needed
            CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
            if [ "$CURRENT_REMOTE" != "$SGLANG_REPO" ]; then
                log_info "Updating remote URL to: $SGLANG_REPO"
                git remote set-url origin "$SGLANG_REPO" 2>/dev/null || git remote add origin "$SGLANG_REPO"
            fi

            # Configure Git to use token if available (for private repos)
            if [ -n "$GITHUB_TOKEN" ]; then
                GIT_CREDENTIAL_URL=$(echo "$SGLANG_REPO" | sed "s|https://|https://${GITHUB_TOKEN}@|")
                git remote set-url origin "$GIT_CREDENTIAL_URL" 2>/dev/null
            fi

            # Stash local changes before switching branches
            if ! git diff-index --quiet HEAD -- 2>/dev/null; then
                log_info "Stashing local changes before updating..."
                git stash push -m "Auto-stash before branch switch $(date +%Y%m%d_%H%M%S)" 2>/dev/null || log_warn "Could not stash changes"
            fi

            git fetch origin
            CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
            if [ "$CURRENT_BRANCH" != "$SGLANG_BRANCH" ]; then
                log_info "Switching to branch: $SGLANG_BRANCH"
                git checkout "$SGLANG_BRANCH" 2>/dev/null || git checkout -b "$SGLANG_BRANCH" origin/"$SGLANG_BRANCH" 2>/dev/null || {
                    log_warn "Could not switch to branch, trying to reset..."
                    git reset --hard origin/"$SGLANG_BRANCH" 2>/dev/null || log_warn "Could not reset branch"
                }
            fi

            # Check for untracked files that would conflict with pull
            set +e
            PULL_OUTPUT=$(git pull origin "$SGLANG_BRANCH" 2>&1)
            PULL_EXIT=$?
            echo "$PULL_OUTPUT" | grep -q "untracked working tree files would be overwritten"
            UNTRACKED_CONFLICT=$?
            set -e

            if [ $UNTRACKED_CONFLICT -eq 0 ]; then
                log_warn "Untracked files detected that would be overwritten by merge"
                log_info "Removing conflicting untracked files..."
                # Extract conflicting file paths from git pull output
                CONFLICTING_FILES=$(echo "$PULL_OUTPUT" | grep "would be overwritten by merge" | sed 's/.*: *//' | sort -u)
                for file in $CONFLICTING_FILES; do
                    if [ -n "$file" ] && ([ -f "$file" ] || [ -d "$file" ]); then
                        log_info "  Removing: $file"
                        rm -rf "$file" 2>/dev/null || log_warn "    Could not remove: $file"
                    fi
                done
                # Retry pull after removing conflicting files
                git pull origin "$SGLANG_BRANCH" || log_warn "Could not pull after cleanup, continuing..."
            elif [ $PULL_EXIT -ne 0 ]; then
                log_warn "Could not pull (exit code: $PULL_EXIT), continuing..."
            fi

            # Restore remote URL without token (for security)
            if [ -n "$GITHUB_TOKEN" ]; then
                git remote set-url origin "$SGLANG_REPO" 2>/dev/null
            fi

            log_info "✓ Repository updated"
            ;;
    esac
else
    # Repository doesn't exist - ask user if they want to clone
    log_info "Repository does not exist at: $PROJECT_DIR"
    echo ""

    if [ "$SKIP_CONFIRM" = "true" ]; then
        # Non-interactive mode: default to clone
        CLONE_REPO="y"
        log_info "Non-interactive mode: cloning repository"
    else
        read -p "Do you want to clone the repository? [Y/n]: " CLONE_REPO
        CLONE_REPO="${CLONE_REPO:-y}"
    fi

    if [[ "$CLONE_REPO" =~ ^[Yy] ]]; then
        log_info "Cloning repository..."
        # Use token in URL for authentication if available (for private repos)
        CLONE_URL="$SGLANG_REPO"
        if [ -n "$GITHUB_TOKEN" ]; then
            CLONE_URL=$(echo "$SGLANG_REPO" | sed "s|https://|https://${GITHUB_TOKEN}@|")
        fi
        git clone -b "$SGLANG_BRANCH" "$CLONE_URL" "$PROJECT_DIR" || {
            log_error "Failed to clone repository"
            if [ -z "$GITHUB_TOKEN" ]; then
                log_info "💡 Tip: For private repositories, you may need to provide a GitHub token."
                log_info "   Password prompt: Use your GitHub Personal Access Token (PAT) as the password."
            fi
            exit 1
        }
        # Restore remote URL without token (for security)
        if [ -n "$GITHUB_TOKEN" ]; then
            cd "$PROJECT_DIR"
            git remote set-url origin "$SGLANG_REPO" 2>/dev/null
        fi
        log_info "✓ Repository cloned"
    else
        log_warn "Skipping repository clone"
        log_info "You will need to set up the repository manually"
        log_info "Expected location: $PROJECT_DIR"
    fi
fi

# Verify repository exists
if [ ! -d "$PROJECT_DIR/.git" ]; then
    log_error "Repository not found at $PROJECT_DIR"
    log_info "Please clone the repository manually or run this script again"
    exit 1
fi

log_info "✓ Repository ready"

# Step 3: Create virtual environment
log_step "Setting up Python virtual environment..."
VENV_DIR="$WORKSPACE_DIR/venv_sglang"
if [ ! -d "$VENV_DIR" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" || {
        log_error "Failed to create virtual environment"
        exit 1
    }
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
log_info "✓ Virtual environment ready"

# Upgrade pip
log_info "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Step 4: Install wheels (if not skipped)
INSTALLED_WHEELS=()
FAILED_WHEELS=()
LIBOMP_INSTALLED=false

if [ "$SKIP_WHEELS" != "true" ]; then
    log_step "Installing wheels from GitHub Releases..."
    cd "$PROJECT_DIR"

    # GitHub Releases configuration
    REPO_OWNER="nthu-pllab"
    REPO_NAME="pllab-sglang"
    RELEASE_TAG="${WHEELS_RELEASE_TAG:-v1.0}"
    # Trim whitespace from token if provided
    GITHUB_TOKEN=$(echo "${GITHUB_TOKEN:-}" | tr -d '\n\r' | xargs)
    WHEELS_DIR="$WORKSPACE_DIR/wheels"

    # Debug: Verify token is set (without showing actual token)
    if [ -n "$GITHUB_TOKEN" ]; then
        TOKEN_LEN=${#GITHUB_TOKEN}
        log_info "GITHUB_TOKEN is set (length: $TOKEN_LEN chars)"
        if [ $TOKEN_LEN -lt 20 ]; then
            log_warn "Warning: Token seems too short. Please verify it's correct."
        fi
    else
        log_warn "GITHUB_TOKEN is not set!"
    fi

    # Create temporary directory for wheels
    mkdir -p "$WHEELS_DIR"
    cd "$WHEELS_DIR"

    log_info "Downloading riscv_wheels_and_libomp.tar.gz from GitHub Releases (tag: $RELEASE_TAG)..."

    # Download the combined archive
    ARCHIVE_FILE="riscv_wheels_and_libomp.tar.gz"
    DOWNLOAD_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${RELEASE_TAG}/${ARCHIVE_FILE}"

    # Log the download URL for debugging
    log_info "Download URL: $DOWNLOAD_URL"

    # Try to download with better error reporting
    # For private repos, use GitHub token if available
    DOWNLOAD_SUCCESS=false
    if [ -n "$GITHUB_TOKEN" ]; then
        log_info "Using GitHub token for authentication..."
    fi

    if command -v curl >/dev/null 2>&1; then
        log_info "Using curl to download..."
        # Use -f to fail on HTTP errors, but capture stderr for error messages
        # Use -w to capture HTTP status code
        if [ -n "$GITHUB_TOKEN" ]; then
            # Trim token to remove any whitespace and newlines
            GITHUB_TOKEN=$(echo "$GITHUB_TOKEN" | tr -d '\n\r' | xargs)
            # Query release API to get asset ID (required for private repositories)
            API_TEST=$(curl -s -H "Authorization: token $GITHUB_TOKEN" "https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/tags/${RELEASE_TAG}" 2>&1)

            if echo "$API_TEST" | grep -q '"tag_name"'; then
                # Get asset ID from API response
                if command -v python3 >/dev/null 2>&1; then
                    ASSET_ID=$(echo "$API_TEST" | python3 -c "import sys, json; data=json.load(sys.stdin); assets=data.get('assets', []); target_asset=[a for a in assets if a['name'] == '$ARCHIVE_FILE']; print(target_asset[0]['id'] if target_asset else '')" 2>/dev/null || echo "")

                    if [ -n "$ASSET_ID" ]; then
                        # Use API endpoint for private repository assets
                        DOWNLOAD_URL="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/assets/${ASSET_ID}"
                        CURL_OUTPUT=$(curl -L -w "\nHTTP_STATUS:%{http_code}" -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/octet-stream" -o "$ARCHIVE_FILE" "$DOWNLOAD_URL" 2>&1)
                    else
                        log_warn "Asset ID not found, using browser_download_url"
                        CURL_OUTPUT=$(curl -L -w "\nHTTP_STATUS:%{http_code}" -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/octet-stream" -o "$ARCHIVE_FILE" "$DOWNLOAD_URL" 2>&1)
                    fi
                else
                    log_warn "python3 not found, using browser_download_url"
                    CURL_OUTPUT=$(curl -L -w "\nHTTP_STATUS:%{http_code}" -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/octet-stream" -o "$ARCHIVE_FILE" "$DOWNLOAD_URL" 2>&1)
                fi
            else
                log_warn "⚠ Token may not have access to release API, trying browser_download_url"
                CURL_OUTPUT=$(curl -L -w "\nHTTP_STATUS:%{http_code}" -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/octet-stream" -o "$ARCHIVE_FILE" "$DOWNLOAD_URL" 2>&1)
            fi
            CURL_EXIT=$?
        else
            log_warn "GITHUB_TOKEN is empty! Attempting download without authentication..."
            CURL_OUTPUT=$(curl -L -w "\nHTTP_STATUS:%{http_code}" -o "$ARCHIVE_FILE" "$DOWNLOAD_URL" 2>&1)
            CURL_EXIT=$?
        fi

        HTTP_STATUS=$(echo "$CURL_OUTPUT" | grep "HTTP_STATUS:" | cut -d: -f2 | tr -d ' ' || echo "")
        CURL_ERROR=$(echo "$CURL_OUTPUT" | grep -v "HTTP_STATUS:" || echo "$CURL_OUTPUT")

        # Check if download was successful based on HTTP status and file existence
        if [ "$HTTP_STATUS" = "200" ] && [ -f "$ARCHIVE_FILE" ] && [ -s "$ARCHIVE_FILE" ]; then
            DOWNLOAD_SUCCESS=true
        else
            log_error "Download failed (HTTP $HTTP_STATUS)"
            if [ "$HTTP_STATUS" = "404" ]; then
                log_error "File not found. Please verify release '$RELEASE_TAG' exists and contains '$ARCHIVE_FILE'"
            elif [ "$HTTP_STATUS" = "401" ] || [ "$HTTP_STATUS" = "403" ]; then
                log_error "Authentication failed. Please check your GitHub token"
                if [ -f "$ARCHIVE_FILE" ]; then
                    log_info "Downloaded file content (first 200 chars):"
                    head -c 200 "$ARCHIVE_FILE" | cat -A
                    echo ""
                fi
            elif [ "$HTTP_STATUS" = "302" ] || [ "$HTTP_STATUS" = "301" ]; then
                log_warn "Redirect detected (HTTP $HTTP_STATUS). This may indicate authentication issues."
            elif [ -z "$HTTP_STATUS" ]; then
                log_error "No HTTP status code received. Check network connectivity."
                if [ -f "$ARCHIVE_FILE" ]; then
                    log_info "Downloaded file content (first 200 chars):"
                    head -c 200 "$ARCHIVE_FILE" | cat -A
                    echo ""
                fi
            fi
            if [ "$CURL_EXIT" -ne 0 ]; then
                log_error "curl exit code: $CURL_EXIT"
                if [ -n "$CURL_ERROR" ]; then
                    log_info "curl error: $CURL_ERROR"
                fi
            fi
        fi
    elif command -v wget >/dev/null 2>&1; then
        log_info "Using wget to download..."
        if [ -n "$GITHUB_TOKEN" ]; then
            WGET_OUTPUT=$(wget --header="Authorization: token $GITHUB_TOKEN" -O "$ARCHIVE_FILE" "$DOWNLOAD_URL" 2>&1)
        else
            WGET_OUTPUT=$(wget -O "$ARCHIVE_FILE" "$DOWNLOAD_URL" 2>&1)
        fi
        WGET_EXIT=$?
        if [ $WGET_EXIT -eq 0 ] && [ -f "$ARCHIVE_FILE" ] && [ -s "$ARCHIVE_FILE" ]; then
            DOWNLOAD_SUCCESS=true
        else
            log_error "wget download failed (exit code: $WGET_EXIT)"
            if echo "$WGET_OUTPUT" | grep -q "404"; then
                log_error "File not found (404). Possible reasons:"
                log_info "  1. Release '$RELEASE_TAG' does not exist"
                log_info "  2. File '$ARCHIVE_FILE' not found in the release"
                log_info "  3. Repository is private and token may not have access"
            elif echo "$WGET_OUTPUT" | grep -q "401\|403"; then
                log_error "Authentication failed (401/403). Possible reasons:"
                log_info "  1. GitHub token is invalid or expired"
                log_info "  2. Token does not have 'repo' scope for private repository"
                log_info "  3. Token does not have access to this repository"
                log_info ""
                log_info "Please check your token at: https://github.com/settings/tokens"
            elif echo "$WGET_OUTPUT" | grep -q "422"; then
                log_error "Unprocessable Entity (422). Release may not exist or file name is incorrect."
            else
                log_warn "Error details:"
                echo "$WGET_OUTPUT" | head -10
            fi
        fi
    else
        log_error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    if [ "$DOWNLOAD_SUCCESS" = true ] && [ -f "$ARCHIVE_FILE" ]; then
        log_info "  ✓ Downloaded $ARCHIVE_FILE"

        # Verify downloaded file is actually a tar archive (gzip or uncompressed)
        log_info "Verifying archive format..."
        FILE_TYPE=$(file "$ARCHIVE_FILE" | cut -d: -f2-)
        if echo "$FILE_TYPE" | grep -qE "(gzip|tar|compressed|POSIX tar)"; then
            log_info "  ✓ Archive format verified: $FILE_TYPE"
        else
            log_error "Downloaded file is not a valid archive!"
            log_info "File type: $FILE_TYPE"
            log_info "First 200 characters of downloaded file:"
            head -c 200 "$ARCHIVE_FILE" | cat -A
            echo ""
            log_info "This usually means:"
            log_info "  1. Download returned an HTML error page instead of the archive"
            log_info "  2. GitHub token may not have access to the release"
            log_info "  3. Release or file may not exist"
            if [ -n "$HTTP_STATUS" ] && [ "$HTTP_STATUS" != "200" ]; then
                log_error "HTTP Status: $HTTP_STATUS"
            fi
            rm -f "$ARCHIVE_FILE"
            exit 1
        fi

        # Extract archive - support both .tar and .tar.gz formats
        log_info "Extracting archive..."
        # Check if file is gzip compressed or uncompressed tar
        if echo "$FILE_TYPE" | grep -qE "gzip|compressed"; then
            # File is gzip compressed, use -z flag
            log_info "  Detected gzip-compressed tar archive"
            if ! tar -xzf "$ARCHIVE_FILE" -C "$WHEELS_DIR" 2>&1; then
                log_error "Failed to extract gzip-compressed archive"
                log_info "Please check:"
                log_info "  1. Archive file exists and is readable"
                log_info "  2. Destination directory is writable: $WHEELS_DIR"
                log_info "  3. Archive file is not corrupted"
                exit 1
            fi
        else
            # File is uncompressed tar, use -xf (no -z flag)
            log_info "  Detected uncompressed tar archive"
            if ! tar -xf "$ARCHIVE_FILE" -C "$WHEELS_DIR" 2>&1; then
                log_error "Failed to extract tar archive"
                log_info "Please check:"
                log_info "  1. Archive file exists and is readable"
                log_info "  2. Destination directory is writable: $WHEELS_DIR"
                log_info "  3. Archive file is not corrupted"
                exit 1
            fi
        fi
        log_info "  ✓ Extracted archive"

        # Install all wheels (check if already installed first)
        log_info "Installing Python wheels..."
        for wheel_file in "$WHEELS_DIR"/riscv_wheels_and_libomp/*.whl; do
            if [ -f "$wheel_file" ]; then
                wheel_name=$(basename "$wheel_file")
                # Extract package name from wheel filename (e.g., torch-2.8.0... -> torch)
                package_name=$(echo "$wheel_name" | sed -E 's/-[0-9].*//' | sed 's/_/-/g')

                # Check if package is already installed
                if python -c "import ${package_name//-/_}" 2>/dev/null || \
                   pip show "$package_name" >/dev/null 2>&1; then
                    log_info "  ⊙ $wheel_name already installed, skipping"
                    INSTALLED_WHEELS+=("$wheel_name (already installed)")
                else
                    log_info "  Installing $wheel_name..."
                    if pip install "$wheel_file" --quiet 2>&1; then
                        log_info "  ✓ Installed $wheel_name"
                        INSTALLED_WHEELS+=("$wheel_name")
                    else
                        log_warn "  ✗ Failed to install $wheel_name"
                        FAILED_WHEELS+=("$wheel_name")
                    fi
                fi
            fi
        done

        # Setup libomp (check if already installed)
        # Search for libomp_riscv.tar.gz in riscv_wheels_and_libomp subdirectory
        LIBOMP_FILE=""
        LIBOMP_DIR="$HOME/.local/lib"

        # Try direct path first
        if [ -f "$WHEELS_DIR/riscv_wheels_and_libomp/libomp_riscv.tar.gz" ]; then
            LIBOMP_FILE="$WHEELS_DIR/riscv_wheels_and_libomp/libomp_riscv.tar.gz"
        else
            # Search in subdirectories
            LIBOMP_FILE=$(find "$WHEELS_DIR/riscv_wheels_and_libomp" -name "libomp_riscv.tar.gz" -type f 2>/dev/null | head -1)
        fi

        if [ -n "$LIBOMP_FILE" ] && [ -f "$LIBOMP_FILE" ]; then
            if [ -f "$LIBOMP_DIR/libomp.so" ]; then
                log_info "libomp already installed at $LIBOMP_DIR/libomp.so, skipping"
                LIBOMP_INSTALLED=true
            else
                log_info "Setting up OpenMP library..."
                log_info "  Found libomp archive at: $LIBOMP_FILE"
                mkdir -p "$LIBOMP_DIR"
                # Extract to temporary directory first to handle archive structure
                # Archive structure may be: libomp_riscv/lib/libomp.so or lib/libomp.so
                # We need: $LIBOMP_DIR/libomp.so
                TEMP_EXTRACT_DIR=$(mktemp -d)

                # Check file type to determine extraction method
                LIBOMP_FILE_TYPE=$(file "$LIBOMP_FILE" | cut -d: -f2-)
                if echo "$LIBOMP_FILE_TYPE" | grep -qE "gzip|compressed"; then
                    log_info "  Extracting gzip-compressed libomp archive..."
                    EXTRACT_CMD="tar -xzf"
                else
                    log_info "  Extracting uncompressed libomp archive..."
                    EXTRACT_CMD="tar -xf"
                fi

                if $EXTRACT_CMD "$LIBOMP_FILE" -C "$TEMP_EXTRACT_DIR" 2>&1; then
                    # Look for libomp.so in the extracted structure
                    # Try common paths: libomp_riscv/lib/libomp.so or lib/libomp.so
                    FOUND_LIBOMP=""
                    for possible_path in "$TEMP_EXTRACT_DIR"/*/lib/libomp.so "$TEMP_EXTRACT_DIR"/lib/libomp.so "$TEMP_EXTRACT_DIR"/*/libomp.so "$TEMP_EXTRACT_DIR"/libomp.so; do
                        if [ -f "$possible_path" ]; then
                            FOUND_LIBOMP="$possible_path"
                            break
                        fi
                    done

                    if [ -n "$FOUND_LIBOMP" ]; then
                        # Copy libomp.so to the target directory
                        if cp "$FOUND_LIBOMP" "$LIBOMP_DIR/libomp.so" 2>/dev/null; then
                            rm -rf "$TEMP_EXTRACT_DIR"

                            if [ -f "$LIBOMP_DIR/libomp.so" ]; then
                                if ! grep -q "LD_PRELOAD.*libomp.so" ~/.bashrc 2>/dev/null; then
                                    echo "" >> ~/.bashrc
                                    echo "# OpenMP library configuration (for sgl-kernel)" >> ~/.bashrc
                                    echo "export LD_PRELOAD=\"$LIBOMP_DIR/libomp.so\${LD_PRELOAD:+:\$LD_PRELOAD}\"" >> ~/.bashrc
                                    echo "export LD_LIBRARY_PATH=\"$LIBOMP_DIR:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
                                fi
                                log_info "  ✓ libomp configured at $LIBOMP_DIR/libomp.so"
                                LIBOMP_INSTALLED=true
                            else
                                log_warn "  ✗ libomp.so not found at expected location after copy"
                                LIBOMP_INSTALLED=false
                            fi
                        else
                            log_warn "  ✗ Failed to copy libomp.so to $LIBOMP_DIR"
                            rm -rf "$TEMP_EXTRACT_DIR"
                            LIBOMP_INSTALLED=false
                        fi
                    else
                        log_warn "  ✗ libomp.so not found in archive structure"
                        log_info "    Archive contents:"
                        find "$TEMP_EXTRACT_DIR" -type f -name "*.so" 2>/dev/null | head -5 | while read -r file; do
                            log_info "      $file"
                        done
                        rm -rf "$TEMP_EXTRACT_DIR"
                        LIBOMP_INSTALLED=false
                    fi
                else
                    log_warn "  ✗ Failed to extract libomp archive"
                    log_info "    Extraction command: $EXTRACT_CMD"
                    log_info "    File type: $LIBOMP_FILE_TYPE"
                    LIBOMP_INSTALLED=false
                fi
            fi
        else
            log_warn "libomp_riscv.tar.gz not found in archive"
            log_info "  Searched in: $WHEELS_DIR/riscv_wheels_and_libomp"
            log_info "  Available files in riscv_wheels_and_libomp:"
            find "$WHEELS_DIR/riscv_wheels_and_libomp" -maxdepth 2 -type f \( -name "*libomp*" -o -name "*.tar*" \) 2>/dev/null | head -10 | while read -r file; do
                log_info "    $file"
            done || log_info "    (none found)"
        fi
    else
        log_error "Failed to download $ARCHIVE_FILE from GitHub Releases"
        log_info "URL: $DOWNLOAD_URL"
        log_info ""
        log_info "Possible reasons:"
        log_info "  1. Release tag '$RELEASE_TAG' does not exist"
        log_info "  2. File '$ARCHIVE_FILE' not found in the release"
        log_info "  3. Network connectivity issues"
        log_info ""
        log_info "Please check:"
        log_info "  https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/tag/${RELEASE_TAG}"
        log_info ""
        log_info "You can manually download the file and place it at: $WHEELS_DIR/$ARCHIVE_FILE"
        exit 1
    fi

    # Clean up
    cd "$PROJECT_DIR"
    rm -rf "$WHEELS_DIR"
else
    log_info "Skipping wheel installation"
fi

# Step 5: Install Python dependencies
log_step "Installing Python dependencies..."
cd "$PROJECT_DIR"

# Install from requirements.txt if exists
if [ -f "banana_pi/test_tinyllama_rvv/requirements.txt" ]; then
    log_info "Installing from requirements.txt..."
    pip install -r "banana_pi/test_tinyllama_rvv/requirements.txt" --quiet || log_warn "Some packages failed to install"
fi

# Install common packages that might be missing
log_info "Installing common packages..."

# Packages available in wheel_builder (install from wheel_builder first for faster installation)
WHEEL_BUILDER_PACKAGES=(
    "numpy"
    "scipy"
    "pandas"
    "pillow"
    "matplotlib"
    "tokenizers"
    "tiktoken"
    "psutil"
    "uvloop"
    "pyyaml"
    "aiohttp"
    "orjson"
    "msgspec"
    "sentencepiece"
    "setproctitle"
    "xxhash"
    "openai-harmony"
    "pydantic-core"
)

# Other core dependencies (install from PyPI)
OTHER_CORE_DEPS=(
    "tqdm"
    "packaging"
    "pybase64"
    "fastapi"
    "uvicorn"
    "python-multipart"
    "pydantic>=2.0"
    "compressed-tensors"
    "gguf"
    "huggingface_hub"
    "einops"
    "prometheus-client>=0.20.0"
    "requests"
    "interegular"
    "jsonschema"
    "partial_json_parser"
    "openai"
    "dill"
    "multiprocess"
    "torchao"
    "xgrammar"
    "pytest"
    "pyzmq"
    "ipython"
)

# Track installation results
INSTALLED_PACKAGES=()
FAILED_PACKAGES=()
SKIPPED_PACKAGES=()

# Install packages from wheel_builder first (faster, pre-compiled)
log_info "Installing packages from wheel_builder (pre-compiled wheels)..."
for package in "${WHEEL_BUILDER_PACKAGES[@]}"; do
    # Extract base package name (remove version constraints)
    base_package="${package%%[>=<]*}"
    # Handle special cases
    if [ "$package" = "pyyaml" ]; then
        import_name="yaml"
    elif [ "$package" = "openai-harmony" ]; then
        import_name="openai_harmony"
    elif [ "$package" = "pydantic-core" ]; then
        import_name="pydantic_core"
    else
        import_name="$base_package"
    fi

    # Check if already installed
    if python -c "import $import_name" 2>/dev/null; then
        log_info "  ⊙ $package already installed, skipping"
        SKIPPED_PACKAGES+=("$package")
    else
        log_info "Installing $package from wheel_builder..."
        set +e
        pip install "$package" --index-url "$WHEEL_BUILDER_URL" --quiet 2>/dev/null
        INSTALL_EXIT=$?
        set -e

        if [ $INSTALL_EXIT -eq 0 ] || python -c "import $import_name" 2>/dev/null; then
            log_info "  ✓ Installed $package"
            INSTALLED_PACKAGES+=("$package")
        else
            log_warn "  ✗ Failed from wheel_builder, trying PyPI..."
            set +e
            pip install "$package" --quiet 2>/dev/null
            INSTALL_EXIT=$?
            set -e

            if [ $INSTALL_EXIT -eq 0 ] || python -c "import $import_name" 2>/dev/null; then
                log_info "  ✓ Installed $package from PyPI"
                INSTALLED_PACKAGES+=("$package")
            else
                log_warn "  ✗ Failed to install $package"
                FAILED_PACKAGES+=("$package")
            fi
        fi
    fi
done

# Install other core dependencies from PyPI
log_info "Installing other core dependencies from PyPI..."
for package in "${OTHER_CORE_DEPS[@]}"; do
    # Extract base package name for import check
    base_package="${package%%[>=<]*}"
    # Handle special cases for import names
    case "$base_package" in
        "prometheus-client")
            import_name="prometheus_client"
            ;;
        "python-multipart")
            import_name="multipart"
            ;;
        "huggingface_hub")
            import_name="huggingface_hub"
            ;;
        "partial_json_parser")
            import_name="partial_json_parser"
            ;;
        "torchao")
            import_name="torchao"
            ;;
        "pybase64")
            import_name="pybase64"
            ;;
        "compressed-tensors")
            import_name="compressed_tensors"
            ;;
        "xgrammar")
            import_name="xgrammar"
            ;;
        "pyzmq")
            import_name="zmq"
            ;;
        "ipython")
            import_name="IPython"
            ;;
        *)
            import_name="$base_package"
            ;;
    esac

    # Check if already installed
    if python -c "import $import_name" 2>/dev/null; then
        log_info "  ⊙ $package already installed, skipping"
        SKIPPED_PACKAGES+=("$package")
    else
        log_info "Installing $package..."
        set +e
        # Capture error output for debugging
        INSTALL_ERROR=$(pip install "$package" 2>&1)
        INSTALL_EXIT=$?
        set -e

        # Check if package is already satisfied (pip may return non-zero but package is installed)
        # Match pattern: "Requirement already satisfied" followed by package name (with or without version)
        # Also check if package can be imported (more reliable check)
        if echo "$INSTALL_ERROR" | grep -qiE "Requirement already satisfied.*$base_package" || \
           python -c "import $import_name" 2>/dev/null; then
            # Double-check by trying to import
            if python -c "import $import_name" 2>/dev/null; then
                log_info "  ⊙ $package already installed, skipping"
                SKIPPED_PACKAGES+=("$package")
            else
                # pip says it's installed but import fails
                # This is OK - some packages don't need to be imported or have different import names
                # Just check if pip shows it's installed
                if echo "$INSTALL_ERROR" | grep -qiE "Requirement already satisfied.*$base_package"; then
                    log_info "  ⊙ $package already installed, skipping"
                    SKIPPED_PACKAGES+=("$package")
                else
                    # Can't import and pip doesn't say it's installed - might be an issue
                    log_warn "  ⚠ $package installed but cannot be imported"
                    FAILED_PACKAGES+=("$package")
                fi
            fi
        elif [ $INSTALL_EXIT -eq 0 ]; then
            # Installation succeeded
            if python -c "import $import_name" 2>/dev/null; then
                log_info "  ✓ Installed $package"
                INSTALLED_PACKAGES+=("$package")
            else
                # Installation succeeded but can't import - this is OK for some packages
                # Check if pip actually installed it successfully
                if pip show "$base_package" >/dev/null 2>&1; then
                    log_info "  ✓ Installed $package (import check skipped)"
                    INSTALLED_PACKAGES+=("$package")
                else
                    log_warn "  ⚠ $package installation may have failed"
                    FAILED_PACKAGES+=("$package")
                fi
            fi
        elif python -c "import $import_name" 2>/dev/null; then
            # Installation failed but package can be imported (might be installed from elsewhere)
            log_info "  ⊙ $package is available (import check passed)"
            SKIPPED_PACKAGES+=("$package")
        else
            log_warn "  ✗ Failed to install $package"
            # Show error details for debugging (especially for RISC-V platform)
            if echo "$INSTALL_ERROR" | grep -qE "(No matching distribution|Could not find|not available)"; then
                log_info "    Reason: Package may not have wheels for RISC-V architecture"
                log_info "    This is expected for some packages on RISC-V platform"
            elif echo "$INSTALL_ERROR" | grep -qE "(error|Error|ERROR)"; then
                log_info "    Error details (first 5 lines):"
                echo "$INSTALL_ERROR" | head -5 | while read -r line; do
                    log_info "      $line"
                done
            fi
            FAILED_PACKAGES+=("$package")
        fi
    fi
done

# Install sglang in development mode using --no-deps (to skip GPU-specific packages)
SGLANG_INSTALL_LOG="$WORKSPACE_DIR/sglang_install.log"
log_info "Installing sglang (logs: $SGLANG_INSTALL_LOG)..."
log_info "  Using --no-deps to skip GPU-specific packages (decord2, cuda-python, etc.)"
set +e
# SGLang's pyproject.toml is in the python/ directory
if [ -f "$PROJECT_DIR/python/pyproject.toml" ]; then
    pip install --no-deps -e "$PROJECT_DIR/python" 2>&1 | tee "$SGLANG_INSTALL_LOG"
    SGLANG_INSTALL_EXIT=${PIPESTATUS[0]}
else
    # Fallback: try root directory
    pip install --no-deps -e "$PROJECT_DIR" 2>&1 | tee "$SGLANG_INSTALL_LOG"
    SGLANG_INSTALL_EXIT=${PIPESTATUS[0]}
fi
set -e

if [ $SGLANG_INSTALL_EXIT -eq 0 ]; then
    log_info "  ✓ Installed sglang (without dependencies)"
    log_info "  Note: Dependencies should be installed in previous steps or can be installed manually as needed"
    INSTALLED_PACKAGES+=("sglang")
else
    log_warn "  ✗ Failed to install sglang in development mode"
    if [ -f "$SGLANG_INSTALL_LOG" ]; then
        log_warn "  └─ See last 40 log lines for details:"
        tail -n 40 "$SGLANG_INSTALL_LOG"
    fi
    FAILED_PACKAGES+=("sglang")
fi

# Install triton stub as a proper package so subprocess can import it
# This creates a real 'triton' package in site-packages that can be imported directly
log_info "Setting up triton stub package for subprocess import..."
TRITON_STUB_SOURCE="$PROJECT_DIR/banana_pi/test_tinyllama_rvv/triton_stub.py"
if [ -f "$TRITON_STUB_SOURCE" ]; then
    # Get site-packages directory
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || python -c "import site; print(site.USER_SITE)" 2>/dev/null || echo "")
    if [ -n "$SITE_PACKAGES" ] && [ -d "$SITE_PACKAGES" ]; then
        # Create triton package directory
        TRITON_PKG_DIR="$SITE_PACKAGES/triton"
        mkdir -p "$TRITON_PKG_DIR"

        # Copy triton_stub.py content directly to triton/__init__.py
        # This ensures that 'import triton' works directly in subprocess
        TRITON_INIT="$TRITON_PKG_DIR/__init__.py"
        if cp "$TRITON_STUB_SOURCE" "$TRITON_INIT" 2>/dev/null; then
            log_info "  ✓ Installed triton stub package to $TRITON_PKG_DIR"
            log_info "  Note: 'import triton' will now work in subprocess (e.g., sglang.launch_server)"

            # Also copy triton_stub.py to site-packages for backward compatibility
            TRITON_STUB_TARGET="$SITE_PACKAGES/triton_stub.py"
            if cp "$TRITON_STUB_SOURCE" "$TRITON_STUB_TARGET" 2>/dev/null; then
                log_info "  ✓ Also installed triton_stub.py to $TRITON_STUB_TARGET (for backward compatibility)"
            fi
        else
            log_warn "  ✗ Failed to copy triton_stub to triton package directory"
        fi
    else
        log_warn "  ⚠ Could not find site-packages directory"
    fi
else
    log_warn "  ⚠ triton_stub.py not found at $TRITON_STUB_SOURCE"
fi

# Install vllm stub as a proper package so subprocess can import it
# This creates a real 'vllm' package in site-packages that can be imported directly
log_info "Setting up vllm stub package for subprocess import..."
VLLM_STUB_SOURCE="$PROJECT_DIR/banana_pi/test_tinyllama_rvv/vllm_stub.py"
if [ -f "$VLLM_STUB_SOURCE" ]; then
    # Get site-packages directory (reuse from triton stub setup)
    if [ -n "$SITE_PACKAGES" ] && [ -d "$SITE_PACKAGES" ]; then
        # Create vllm package directory structure
        VLLM_PKG_DIR="$SITE_PACKAGES/vllm"
        mkdir -p "$VLLM_PKG_DIR"
        mkdir -p "$VLLM_PKG_DIR/model_executor/layers"
        mkdir -p "$VLLM_PKG_DIR/distributed"

        # Copy vllm_stub.py content directly to vllm/__init__.py
        # This ensures that 'import vllm' works directly in subprocess
        VLLM_INIT="$VLLM_PKG_DIR/__init__.py"
        if cp "$VLLM_STUB_SOURCE" "$VLLM_INIT" 2>/dev/null; then
            log_info "  ✓ Installed vllm stub package to $VLLM_PKG_DIR"
            log_info "  Note: 'import vllm' will now work in subprocess (e.g., sglang.launch_server)"

            # Also copy vllm_stub.py to site-packages for backward compatibility
            VLLM_STUB_TARGET="$SITE_PACKAGES/vllm_stub.py"
            if cp "$VLLM_STUB_SOURCE" "$VLLM_STUB_TARGET" 2>/dev/null; then
                log_info "  ✓ Also installed vllm_stub.py to $VLLM_STUB_TARGET (for backward compatibility)"
            fi
        else
            log_warn "  ✗ Failed to copy vllm_stub to vllm package directory"
        fi
    else
        log_warn "  ⚠ Could not find site-packages directory"
    fi
else
    log_warn "  ⚠ vllm_stub.py not found at $VLLM_STUB_SOURCE"
fi

# Display installation summary
echo ""
log_step "Installation Summary"
echo "============================================================"
if [ ${#INSTALLED_WHEELS[@]} -gt 0 ]; then
    log_info "Successfully installed/verified wheels (${#INSTALLED_WHEELS[@]}):"
    for wheel in "${INSTALLED_WHEELS[@]}"; do
        echo "  ✓ $wheel"
    done
fi

if [ "$LIBOMP_INSTALLED" = true ]; then
    log_info "OpenMP library: ✓ Installed/configured"
else
    log_warn "OpenMP library: ✗ Not installed"
fi

if [ ${#INSTALLED_PACKAGES[@]} -gt 0 ]; then
    log_info "Successfully installed packages (${#INSTALLED_PACKAGES[@]}):"
    for pkg in "${INSTALLED_PACKAGES[@]}"; do
        echo "  ✓ $pkg"
    done
fi

if [ ${#SKIPPED_PACKAGES[@]} -gt 0 ]; then
    log_info "Skipped (already installed) (${#SKIPPED_PACKAGES[@]}):"
    for pkg in "${SKIPPED_PACKAGES[@]}"; do
        echo "  ⊙ $pkg"
    done
fi

if [ ${#FAILED_WHEELS[@]} -gt 0 ] || [ ${#FAILED_PACKAGES[@]} -gt 0 ]; then
    log_warn "Failed installations:"
    for wheel in "${FAILED_WHEELS[@]}"; do
        echo "  ✗ $wheel"
    done
    for pkg in "${FAILED_PACKAGES[@]}"; do
        echo "  ✗ $pkg"
    done
    echo ""
    log_info "You may need to install these manually or check for compatibility issues"
fi
echo "============================================================"
echo ""

# Step 6: Configure OpenMP library
log_step "Configuring OpenMP library..."
LIBOMP_DIR="$HOME/.local/lib"
LIBOMP_SO="$LIBOMP_DIR/libomp.so"

if [ -f "$LIBOMP_SO" ]; then
    # Add to .bashrc if not already there
    if ! grep -q "LD_PRELOAD.*libomp.so" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# OpenMP library configuration (for sgl-kernel)" >> ~/.bashrc
        echo "export LD_PRELOAD=\"$LIBOMP_SO\${LD_PRELOAD:+:\$LD_PRELOAD}\"" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"$LIBOMP_DIR:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
    fi
    log_info "✓ OpenMP library configured"
else
    log_warn "libomp.so not found at $LIBOMP_SO"
    log_info "You may need to install OpenMP library manually"
fi

# Step 7: Verify setup
log_step "Verifying setup..."
cd "$PROJECT_DIR"

VERIFY_ERRORS=0

# Check if test file exists
TEST_FILE="banana_pi/test_tinyllama_rvv/test_tinyllama_rvv.py"
if [ -f "$TEST_FILE" ]; then
    log_info "✓ Test file found: $TEST_FILE"
else
    log_warn "Test file not found: $TEST_FILE"
    log_warn "  └─ Pull latest changes or re-run this script with option 1 'update' or 2 're-clone'"
    VERIFY_ERRORS=1
fi

# Check if config file exists
CONFIG_FILE="banana_pi/test_tinyllama_rvv/config_rvv.yaml"
if [ -f "$CONFIG_FILE" ]; then
    log_info "✓ Config file found: $CONFIG_FILE"
else
    log_warn "Config file not found: $CONFIG_FILE"
    log_warn "  └─ Ensure repository includes banana_pi/test_tinyllama_rvv assets"
    VERIFY_ERRORS=1
fi

if [ $VERIFY_ERRORS -eq 0 ]; then
    log_info "✓ Setup verification passed"
else
    log_warn "Setup verification completed with warnings. Please address the missing files above."
fi

echo ""
log_info "Setup completed successfully!"
echo ""
log_info "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source $VENV_DIR/bin/activate"
echo ""
echo "  2. Run the test:"
echo "     cd $PROJECT_DIR"
echo "     python $TEST_FILE"
echo ""
REMOTE_SCRIPT_EOF
)

# Transfer and execute setup script
log_step "Transferring setup script to Banana Pi..."
log_info "You may be prompted for SSH password..."

# Write script to temporary file
TEMP_SCRIPT=$(mktemp)
echo "$REMOTE_SETUP_SCRIPT" > "$TEMP_SCRIPT"
trap "rm -f '$TEMP_SCRIPT'" EXIT

# Generate unique remote script filename to avoid permission conflicts
# Use PID and timestamp to ensure uniqueness
REMOTE_SCRIPT_NAME="setup_sglang_$$_$(date +%s).sh"
REMOTE_SCRIPT_PATH="/tmp/$REMOTE_SCRIPT_NAME"

# Try to clean up old script files (if they exist and we have permission)
log_info "Cleaning up old setup scripts (if any)..."
set +e
"$SSH_CMD" -t "$BANANA_PI_USER@$BANANA_PI_HOST" "rm -f /tmp/setup_sglang*.sh 2>/dev/null || true"
set -e

# Use scp to transfer script (allows password input)
# Don't use timeout or output redirection here to allow interactive password input
set +e
log_info "Transferring script file..."
echo ""  # Add blank line before password prompt
"$SCP_CMD" "$TEMP_SCRIPT" "$BANANA_PI_USER@$BANANA_PI_HOST:$REMOTE_SCRIPT_PATH"
SCP_EXIT_CODE=$?
set -e

if [ $SCP_EXIT_CODE -ne 0 ]; then
    log_error "Failed to transfer setup script (exit code: $SCP_EXIT_CODE)"
    log_info "Please check:"
    log_info "  1. SSH password is correct"
    log_info "  2. Network connectivity"
    log_info "  3. SSH service is running on Banana Pi"
    log_info "  4. /tmp directory is writable"
    exit 1
fi

log_info "✓ Script transferred successfully to $REMOTE_SCRIPT_PATH"

# Make script executable
set +e
log_info "Making script executable..."
"$SSH_CMD" -t "$BANANA_PI_USER@$BANANA_PI_HOST" "chmod +x $REMOTE_SCRIPT_PATH"
SSH_EXIT_CODE=$?
set -e

if [ $SSH_EXIT_CODE -ne 0 ]; then
    log_warn "Failed to make script executable (exit code: $SSH_EXIT_CODE)"
    log_info "Continuing anyway..."
fi

log_step "Running setup on Banana Pi..."
log_info "This may take several minutes..."
log_info "You may be prompted for SSH password if not using key-based authentication..."
# Use -t to force pseudo-terminal allocation for password input and interactive prompts
# Don't capture output in variable to allow real-time display and password input
set +e
# Ensure stderr is also displayed by redirecting it to stdout
# Also ensure cleanup happens even if script fails
# Use a more robust approach to avoid quote issues
"$SSH_CMD" -t "$BANANA_PI_USER@$BANANA_PI_HOST" \
    "export SKIP_WHEELS='$SKIP_WHEELS' && \
     export SKIP_CONFIRM='$SKIP_CONFIRM' && \
     export WHEELS_RELEASE_TAG='$WHEELS_RELEASE_TAG' && \
     export GITHUB_TOKEN='$GITHUB_TOKEN' && \
     bash $REMOTE_SCRIPT_PATH 2>&1; \
     EXIT_CODE=\$?; \
     rm -f $REMOTE_SCRIPT_PATH 2>/dev/null || true; \
     exit \$EXIT_CODE"
SSH_EXIT_CODE=$?
set -e

# Cleanup remote script (in case the above cleanup didn't work)
log_info "Cleaning up remote script..."
set +e
"$SSH_CMD" -t "$BANANA_PI_USER@$BANANA_PI_HOST" "rm -f $REMOTE_SCRIPT_PATH 2>/dev/null || true"
set -e

if [ $SSH_EXIT_CODE -ne 0 ]; then
    log_error "Failed to run setup script on Banana Pi (exit code: $SSH_EXIT_CODE)"
    log_info "Please check the output above for error details"
    exit 1
fi

# Test execution is disabled by default
# Users can run the test manually after setup completes

echo ""
log_info "Setup completed!"
log_info "To run tests manually, SSH to Banana Pi and run:"
echo "  cd ~/.local_riscv_env/workspace/sglang"
echo "  source ~/.local_riscv_env/workspace/venv_sglang/bin/activate"
echo "  python banana_pi/test_tinyllama_rvv/test_tinyllama_rvv.py"
echo ""
