#!/usr/bin/env python3
"""
RVV Environment Manager
Merged functionality of setup_stubs.py and test_environment.py.

This script handles:
1. Installation of stubs (triton, vllm, torchvision) to site-packages.
2. Verification of the environment (imports, libomp, config).
"""

import argparse
import importlib.util
import os
import shutil
import site
import sys

# ==========================================
# Stub Installation Logic
# ==========================================


def install_stub(stub_name, target_pkg_name, stub_file_path):
    print(f"Installing {stub_name} stub...")

    # Check if stub file exists
    if not os.path.exists(stub_file_path):
        print(f"  ❌ Error: Stub file not found: {stub_file_path}")
        return False

    # Find site-packages
    site_packages = site.getsitepackages()[0]
    target_dir = os.path.join(site_packages, target_pkg_name)

    print(f"  Target: {target_dir}")

    # Create target directory
    os.makedirs(target_dir, exist_ok=True)

    # Install __init__.py
    target_init = os.path.join(target_dir, "__init__.py")

    # Write import logic that converts the stub file into a package
    with open(target_init, "w") as f:
        f.write(f"# Auto-generated stub for {target_pkg_name}\n")
        f.write(f"import sys\n")
        f.write(f"import os\n")
        f.write(f"import importlib.util\n")

        # Read the raw stub content and embed it directly to avoid path issues
        with open(stub_file_path, "r") as stub_f:
            content = stub_f.read()

        f.write(content)
        f.write("\n")

    # Special handling for submodules required by import chains
    if target_pkg_name == "triton":
        # Create triton/language/extra/libdevice.py and necessary __init__s
        extra_dir = os.path.join(target_dir, "language", "extra")
        os.makedirs(extra_dir, exist_ok=True)

        os.makedirs(os.path.join(target_dir, "language"), exist_ok=True)

        # Determine paths to touch
        touch_paths = [
            os.path.join(target_dir, "language", "__init__.py"),
            os.path.join(target_dir, "language", "extra", "__init__.py"),
        ]

        for p in touch_paths:
            with open(p, "w") as f:
                pass

        # Create libdevice.py
        with open(os.path.join(extra_dir, "libdevice.py"), "w") as f:
            f.write("def flush_denormals(): pass\n")

    if target_pkg_name == "torchvision":
        # Create torchvision/transforms/functional.py with InterpolationMode
        transforms_dir = os.path.join(target_dir, "transforms")
        os.makedirs(transforms_dir, exist_ok=True)

        with open(os.path.join(transforms_dir, "__init__.py"), "w") as f:
            pass

        with open(os.path.join(transforms_dir, "functional.py"), "w") as f:
            f.write("class InterpolationMode:\n")
            f.write("    BICUBIC = 'bicubic'\n")
            f.write("    NEAREST = 'nearest'\n")
            f.write("    BILINEAR = 'bilinear'\n")
            f.write("    BOX = 'box'\n")
            f.write("    HAMMING = 'hamming'\n")
            f.write("    LANCZOS = 'lanczos'\n")

    print(f"  ✓ Installed {target_pkg_name}")
    return True


def run_stub_installation():
    print("=" * 60)
    print("Step 1: Installing Stubs")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    stubs_dir = os.path.join(script_dir, "stubs")

    stubs = [
        ("Triton", "triton", os.path.join(stubs_dir, "triton_stub.py")),
        ("vLLM", "vllm", os.path.join(stubs_dir, "vllm_stub.py")),
        ("Torchvision", "torchvision", os.path.join(stubs_dir, "torchvision_stub.py")),
    ]

    success = True
    for name, pkg, path in stubs:
        if not install_stub(name, pkg, path):
            success = False

    if success:
        print("\n✅ All stubs installed successfully!")
        return True
    else:
        print("\n❌ Some stubs failed to install.")
        return False


# ==========================================
# Environment Verification Logic
# ==========================================


def setup_libomp_env():
    """Automatically set up LD_PRELOAD and LD_LIBRARY_PATH for libomp"""
    libomp_paths = [
        os.path.expanduser("~/.local/lib/libomp.so"),
        "/usr/local/lib/libomp.so",
        "/usr/lib/riscv64-linux-gnu/libomp.so",
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
        ld_preload = os.environ.get("LD_PRELOAD", "")
        if "libomp.so" not in ld_preload:
            os.environ["LD_PRELOAD"] = (
                f"{libomp_so}{':' + ld_preload if ld_preload else ''}"
            )

        # Set LD_LIBRARY_PATH if not set or doesn't include libomp directory
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        if libomp_dir not in ld_library_path.split(":"):
            os.environ["LD_LIBRARY_PATH"] = (
                f"{libomp_dir}{':' + ld_library_path if ld_library_path else ''}"
            )
        return True
    return False


def check_virtual_env():
    """Check if virtual environment is activated"""
    venv_path = os.environ.get("VIRTUAL_ENV", "")
    if not venv_path:
        print("WARNING: Virtual environment not activated!")
        print("   Please activate venv_sglang first:")
        print("   source ~/.local_riscv_env/workspace/venv_sglang/bin/activate")
        return False
    else:
        print(f"✓ Virtual environment: {venv_path}")
        return True


def test_imports():
    """Test all required imports"""
    print("\nTesting imports...")

    # Ensure libomp environment is set up before importing sgl_kernel
    libomp_setup = setup_libomp_env()
    if libomp_setup:
        libomp_so = (
            os.environ.get("LD_PRELOAD", "").split(":")[0]
            if "libomp.so" in os.environ.get("LD_PRELOAD", "")
            else None
        )
        if libomp_so:
            print(f"Auto-configured libomp: {libomp_so}")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"Device: {torch.device('cpu')}")
    except ImportError as e:
        print(f"PyTorch: {e}")
        return False

    try:
        import sgl_kernel

        print("sgl-kernel: installed")
    except ImportError as e:
        print(f"sgl-kernel: {e}")
        return False

    # Check triton
    try:
        import triton

        version = getattr(triton, "__version__", "unknown")
        if "stub" in str(version).lower():
            print(f"triton: {version} (stub module - RISC-V compatible)")
        else:
            print(f"triton: {version} (real installation)")
    except ImportError:
        print("triton: not available (neither real nor stub)")

    # Skip sglang import check to avoid triggering pip build dependency installation
    # SGLang will be imported during actual benchmark execution
    try:
        import importlib.util

        sglang_spec = importlib.util.find_spec("sglang")
        if sglang_spec is not None:
            print("SGLang: installed (import check skipped to avoid pip rebuild)")
        else:
            print("SGLang: not found in Python path")
            return False
    except Exception as e:
        print(f"SGLang: {e}")
        return False

    try:
        import requests

        print(f"requests: {requests.__version__}")
    except ImportError as e:
        print(f"requests: {e}")
        return False

    try:
        import psutil

        print(f"psutil: {psutil.__version__}")
    except ImportError as e:
        print(f"psutil: {e}")
        return False

    try:
        import yaml

        print("pyyaml: installed")
    except ImportError as e:
        print(f"pyyaml: {e}")
        return False

    # Check optional wheel builder packages
    print("\nChecking optional packages (from wheel_builder):")
    optional_packages = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
    ]

    for module_name, display_name in optional_packages:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{display_name}: {version}")
        except ImportError:
            print(f"{display_name}: not installed (optional)")

    return True


def run_verification():
    print("\n" + "=" * 60)
    print("Step 2: verifying Environment")
    print("=" * 60)

    # Auto-setup libomp environment (runs automatically on import, but show status)
    libomp_configured = setup_libomp_env()
    if libomp_configured:
        ld_preload = os.environ.get("LD_PRELOAD", "")
        if "libomp.so" in ld_preload:
            print("libomp environment auto-configured")

    venv_ok = check_virtual_env()
    imports_ok = test_imports()

    print("\n" + "-" * 60)
    if venv_ok and imports_ok:
        print("✅ All verification steps passed!")
        return True
    else:
        print("❌ Some verification steps failed.")
        return False


# ==========================================
# Main
# ==========================================


def main():
    parser = argparse.ArgumentParser(description="RVV Environment Manager")
    parser.add_argument(
        "--action",
        choices=["install", "verify", "all"],
        default="all",
        help="Action to perform: install stubs, verify environment, or both",
    )
    args = parser.parse_args()

    success = True

    # Always check environment first if we are going to modify it
    if args.action in ["install", "all"]:
        if not check_virtual_env():
            print("❌ Aborting installation: Virtual environment not active.")
            print(
                "   Please activate it first using: source ~/.local_riscv_env/workspace/venv_sglang/bin/activate"
            )
            sys.exit(1)

    if args.action in ["install", "all"]:
        if not run_stub_installation():
            success = False

    if args.action in ["verify", "all"]:
        # check_virtual_env is already run above for "all" case, but "verify" might need it too
        if args.action == "verify":
            check_virtual_env()

        if not run_verification():
            success = False

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
