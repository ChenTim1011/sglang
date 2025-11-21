#!/usr/bin/env python3
"""Test RISC-V environment setup"""

import sys
import os

# Import stubs FIRST, before any other imports that might try to import these modules
# This ensures triton and PIL are available in sys.modules before SGLang tries to import them
_script_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure default libomp environment variables are exported early
_default_libomp = os.path.expanduser("~/.local/lib/libomp.so")
_default_lib_dir = os.path.expanduser("~/.local/lib")

if os.path.exists(_default_libomp):
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if _default_libomp not in ld_preload.split(":"):
        os.environ["LD_PRELOAD"] = (
            f"{_default_libomp}{':' + ld_preload if ld_preload else ''}"
        )

if os.path.isdir(_default_lib_dir):
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _default_lib_dir not in [p for p in ld_library_path.split(":") if p]:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{_default_lib_dir}{':' + ld_library_path if ld_library_path else ''}"
        )

# Load triton_stub
_triton_stub_path = os.path.join(_script_dir, 'triton_stub.py')
if os.path.exists(_triton_stub_path):
    # Add script directory to path
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    try:
        import triton_stub  # This will register triton in sys.modules
        print("✓ triton_stub loaded (triton will be available for imports)")
    except ImportError:
        print("⚠ Could not import triton_stub (triton may not be available)")
else:
    print("⚠ triton_stub.py not found (triton may not be available)")

# Check if PIL is available (from wheel_builder or PyPI)
try:
    from PIL import Image
    print(f"✓ PIL (Pillow) is available: {Image.__version__}")
except ImportError:
    print("⚠ PIL (Pillow) is not available")
    print("  💡 Install Pillow: pip install Pillow --index-url https://gitlab.com/api/v4/projects/riseproject%2Fpython%2Fwheel_builder/packages/pypi/simple")

# Auto-configure libomp environment variables if not set


def setup_libomp_env():
    """Automatically set up LD_PRELOAD and LD_LIBRARY_PATH for libomp"""
    libomp_paths = [
        os.path.expanduser('~/.local/lib/libomp.so'),
        '/usr/local/lib/libomp.so',
        '/usr/lib/riscv64-linux-gnu/libomp.so',
    ]

    # Find libomp.so
    libomp_so = None
    for path in libomp_paths:
        if os.path.exists(path):
            libomp_so = path
            break

    if libomp_so:
        libomp_dir = os.path.dirname(libomp_so)

        # Set LD_PRELOAD if not set or doesn't include libomp
        ld_preload = os.environ.get('LD_PRELOAD', '')
        if 'libomp.so' not in ld_preload:
            os.environ['LD_PRELOAD'] = f"{libomp_so}{':' + ld_preload if ld_preload else ''}"

        # Set LD_LIBRARY_PATH if not set or doesn't include libomp directory
        ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
        if libomp_dir not in ld_library_path.split(':'):
            os.environ['LD_LIBRARY_PATH'] = f"{libomp_dir}{':' + ld_library_path if ld_library_path else ''}"

        return True
    return False


# Auto-setup before any imports that might need libomp
setup_libomp_env()


def check_virtual_env():
    """Check if virtual environment is activated"""
    venv_path = os.environ.get('VIRTUAL_ENV', '')
    if not venv_path:
        print("⚠️  WARNING: Virtual environment not activated!")
        print("   Please activate venv_sglang first:")
        print("   source ~/.local_riscv_env/workspace/venv_sglang/bin/activate")
        print("")
        return False
    else:
        print(f"✓ Virtual environment: {venv_path}")
        return True


def test_imports():
    """Test all required imports"""
    print("Testing imports...")

    # Ensure libomp environment is set up before importing sgl_kernel
    libomp_setup = setup_libomp_env()
    if libomp_setup:
        libomp_so = os.environ.get('LD_PRELOAD', '').split(
            ':')[0] if 'libomp.so' in os.environ.get('LD_PRELOAD', '') else None
        if libomp_so:
            print(f"  ✓ Auto-configured libomp: {libomp_so}")

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  Device: {torch.device('cpu')}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        print("  💡 Solution:")
        print("     1. Activate virtual environment:")
        print("        source ~/.local_riscv_env/workspace/venv_sglang/bin/activate")
        print(
            "     2. PyTorch wheel should be installed automatically by setup_banana_pi.sh")
        print("     3. If missing, download from GitHub Releases:")
        print("        https://github.com/nthu-pllab/pllab-sglang/releases/tag/v1.0")
        print("     4. Install manually: pip install torch-*.whl")
        return False

    try:
        import sgl_kernel
        print("✓ sgl-kernel: installed")
    except ImportError as e:
        print(f"✗ sgl-kernel: {e}")
        if 'libomp' in str(e).lower() or 'undefined symbol' in str(e).lower():
            print("  💡 This might be a libomp issue. Check:")
            print("     - libomp should be installed automatically by setup_banana_pi.sh")
            print("     - LD_PRELOAD: echo $LD_PRELOAD")
            print("     - LD_LIBRARY_PATH: echo $LD_LIBRARY_PATH")
            print("     - libomp.so exists: ls -la ~/.local/lib/libomp.so")
            print("     - If missing, download from GitHub Releases:")
            print("       https://github.com/nthu-pllab/pllab-sglang/releases/tag/v1.0")
            print("     - If missing, run: source ~/.bashrc")
            print("     - If missing, run: export LD_PRELOAD=~/.local/lib/libomp.so")
            print(
                "     - If missing, run: export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH")
        return False

    # Check triton (may not be available on RISC-V, but stub should be loaded)
    try:
        import triton
        version = getattr(triton, '__version__', 'unknown')
        if 'stub' in str(version).lower():
            print(f"✓ triton: {version} (stub module - RISC-V compatible)")
        else:
            print(f"✓ triton: {version} (real installation)")
    except ImportError:
        print("⚠ triton: not available (neither real nor stub)")
        print("  💡 triton_stub.py should have been loaded at script start")
        print("  💡 Check if triton_stub.py exists in the script directory")

    try:
        import sglang
        print(f"✓ SGLang: {sglang.__version__}")
    except ImportError as e:
        print(f"✗ SGLang: {e}")
        if 'triton' in str(e).lower():
            print(
                "  💡 SGLang is trying to import triton, which is not available on RISC-V")
            print("  💡 Solution: triton_stub.py should handle this automatically")
            print("     Check if triton_stub.py exists in the script directory")
            print("     It should be loaded at the start of this script")
        return False

    try:
        import requests
        print(f"✓ requests: {requests.__version__}")
    except ImportError as e:
        print(f"✗ requests: {e}")
        return False

    try:
        import psutil
        print(f"✓ psutil: {psutil.__version__}")
    except ImportError as e:
        print(f"✗ psutil: {e}")
        return False

    try:
        import yaml
        print("✓ pyyaml: installed")
    except ImportError as e:
        print(f"✗ pyyaml: {e}")
        return False

    # Check optional wheel builder packages
    print("\nChecking optional packages (from wheel_builder):")

    optional_packages = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('pandas', 'Pandas'),
    ]

    for module_name, display_name in optional_packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name}: {version}")
        except ImportError:
            print(f"⚠ {display_name}: not installed (optional)")
            print(
                f"  💡 To install: pip install {module_name} --index-url https://gitlab.com/api/v4/projects/riseproject%2Fpython%2Fwheel_builder/packages/pypi/simple")
            print(f"     Or run setup_banana_pi.sh to install all dependencies")

    return True


def test_config():
    """Test configuration file"""
    print("\nTesting configuration...")
    try:
        import yaml
        with open('config_riscv.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✓ config_riscv.yaml: valid")
        print(f"  Model: {config.get('model-path', 'N/A')}")
        print(f"  Device: {config.get('device', 'N/A')}")
        return True
    except Exception as e:
        print(f"✗ config_riscv.yaml: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("RISC-V Environment Test")
    print("=" * 60)
    print("")

    # Auto-setup libomp environment (runs automatically on import, but show status)
    libomp_configured = setup_libomp_env()
    if libomp_configured:
        ld_preload = os.environ.get('LD_PRELOAD', '')
        ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
        if 'libomp.so' in ld_preload:
            print("✓ libomp environment auto-configured")
            print(f"  LD_PRELOAD: {ld_preload.split(':')[0]}")
            print(
                f"  LD_LIBRARY_PATH: {ld_library_path.split(':')[0] if ld_library_path else 'N/A'}")
        print("")

    # Check virtual environment first
    venv_ok = check_virtual_env()
    print("")

    imports_ok = test_imports()
    config_ok = test_config()

    print("\n" + "=" * 60)
    if venv_ok and imports_ok and config_ok:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        if not venv_ok:
            print("\n💡 Most common issue: Virtual environment not activated")
            print("   Run: source ~/.local_riscv_env/workspace/venv_sglang/bin/activate")
        if not libomp_configured:
            print("\n💡 libomp not found. To set up:")
            print("   1. libomp should be installed automatically by setup_banana_pi.sh")
            print("   2. If missing, download from GitHub Releases:")
            print("      https://github.com/nthu-pllab/pllab-sglang/releases/tag/v1.0")
            print("   3. Extract to ~/.local/lib/:")
            print("      tar -xzf libomp_riscv.tar.gz -C ~/.local/lib/")
            print(
                "   4. Add to ~/.bashrc (should be done automatically by setup_banana_pi.sh):")
            print("      export LD_PRELOAD=~/.local/lib/libomp.so")
            print("      export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH")
            print("   5. Reload: source ~/.bashrc")
        sys.exit(1)
