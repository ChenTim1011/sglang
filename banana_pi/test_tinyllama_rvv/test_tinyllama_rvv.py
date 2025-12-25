#!/usr/bin/env python3
"""
TinyLlama Interactive Chat Script with Server Management
"""

import ctypes
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

import psutil
import requests

# Import stubs FIRST, before any other imports that might try to import these modules
# This ensures triton, vllm, and PIL are available in sys.modules before SGLang tries to import them
_script_dir = os.path.dirname(os.path.abspath(__file__))

# Add script directory to path
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Load triton_stub
_triton_stub_path = os.path.join(_script_dir, "triton_stub.py")
if os.path.exists(_triton_stub_path):
    try:
        import triton_stub  # This will register triton in sys.modules

        # Silent import - we'll check later if needed
    except ImportError:
        pass  # Will handle later if triton is actually needed

# Load vllm_stub
_vllm_stub_path = os.path.join(_script_dir, "vllm_stub.py")
if os.path.exists(_vllm_stub_path):
    try:
        import vllm_stub  # This will register vllm in sys.modules

        # Silent import - we'll check later if needed
    except ImportError:
        pass  # Will handle later if vllm is actually needed


def wait_for_service(max_wait=3600):
    """Wait for service to start (deprecated - use launch_server's built-in waiting)"""
    print("⏳ Waiting for SGLang service to start...")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://127.0.0.1:30000/health", timeout=2)
            if response.status_code == 200:
                print("✅ Service is ready!")
                return True
        except:
            pass
        time.sleep(5)
        elapsed = int(time.time() - start_time)
        if elapsed % 30 == 0:  # Print every 30 seconds
            print(f"   Waiting... ({elapsed}s elapsed)")

    print("❌ Service startup timeout")
    return False


def check_server_running():
    """Check if SGLang server is running"""
    try:
        response = requests.get("http://127.0.0.1:30000/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_cpu_threads_bind_from_config(config_path):
    """Extract cpu-omp-threads-bind from YAML config when possible"""
    value = None
    if not os.path.exists(config_path):
        return None

    # Try to parse via PyYAML if available
    try:
        import yaml  # type: ignore

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            value = data.get("cpu-omp-threads-bind") or data.get("cpu_omp_threads_bind")
            if isinstance(value, (int, float)):
                value = str(value)
    except ModuleNotFoundError:
        pass
    except Exception:
        # Fall back to manual parsing below
        pass

    if value:
        return value.strip()

    # Fallback: parse manually to avoid YAML dependency
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("cpu-omp-threads-bind"):
                    _, rhs = line.split(":", 1)
                    candidate = rhs.strip().strip('"').strip("'")
                    if candidate:
                        return candidate
    except Exception:
        pass

    return None


def find_sglang_processes():
    """Find running SGLang processes"""
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["name"] and "python" in proc.info["name"].lower():
                cmdline = " ".join(proc.info["cmdline"]) if proc.info["cmdline"] else ""
                if "sglang" in cmdline and "launch_server" in cmdline:
                    processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


def check_disk_space():
    """Check available disk space"""
    try:
        result = subprocess.run(
            ["df", "-h", os.path.expanduser("~")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.decode("utf-8", errors="ignore").strip().split("\n")
            if len(lines) >= 2:
                # Parse df output
                parts = lines[1].split()
                if len(parts) >= 4:
                    filesystem = parts[0]
                    total = parts[1]
                    used = parts[2]
                    available = parts[3]
                    use_percent = parts[4] if len(parts) >= 5 else "N/A"
                    mount_point = parts[5] if len(parts) >= 6 else "N/A"

                    print(f"  📦 Disk space: {available} available / {total} total")
                    print(f"     Used: {used} ({use_percent})")

                    # Check if disk space is low (< 1GB)
                    try:
                        # Parse available space (e.g., "5.2G" -> 5.2)
                        avail_str = available.upper().replace("G", "").replace("M", "")
                        avail_value = float(avail_str)
                        if "M" in available.upper():
                            # Less than 1GB if in MB
                            if avail_value < 1024:
                                print("  ⚠ Warning: Low disk space (< 1GB)")
                                return False
                        elif "G" in available.upper():
                            if avail_value < 1.0:
                                print("  ⚠ Warning: Low disk space (< 1GB)")
                                return False
                    except:
                        pass

                    return True
    except FileNotFoundError:
        print("  ⚠ 'df' command not found")
    except:
        pass
    return True  # Don't fail if we can't check


def check_libomp():
    """Check if libomp is available and configured correctly"""
    print("🔍 Checking libomp configuration...")

    issues = []
    warnings = []

    # Check LD_PRELOAD
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if ld_preload:
        libomp_in_preload = "libomp.so" in ld_preload
        if libomp_in_preload:
            print(f"  ✓ LD_PRELOAD is set: {ld_preload}")
        else:
            warnings.append("LD_PRELOAD is set but doesn't include libomp.so")
    else:
        warnings.append("LD_PRELOAD is not set (libomp may not be loaded)")

    # Check LD_LIBRARY_PATH
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    libomp_paths = [
        os.path.expanduser("~/.local/lib"),
        "/usr/local/lib",
        "/usr/lib/riscv64-linux-gnu",
    ]

    found_libomp = False
    libomp_location = None

    # Check common locations
    for path in libomp_paths:
        libomp_so = os.path.join(path, "libomp.so")
        if os.path.exists(libomp_so):
            found_libomp = True
            libomp_location = libomp_so
            print(f"  ✓ Found libomp.so: {libomp_location}")
            break

    # Also check LD_LIBRARY_PATH
    if ld_library_path:
        for path in ld_library_path.split(":"):
            if path:
                libomp_so = os.path.join(path, "libomp.so")
                if os.path.exists(libomp_so):
                    found_libomp = True
                    libomp_location = libomp_so
                    print(f"  ✓ Found libomp.so in LD_LIBRARY_PATH: {libomp_location}")
                    break

    if not found_libomp:
        # Try to find using find command (if available)
        try:
            result = subprocess.run(
                ["find", os.path.expanduser("~"), "/usr/local", "/usr/lib"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            # This is too broad, let's use a more targeted approach
        except:
            pass

        issues.append("libomp.so not found in common locations")
        print("  ⚠ libomp.so not found")
    else:
        # Try to verify libomp has required symbols
        try:
            result = subprocess.run(
                ["nm", "-D", libomp_location],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            if result.returncode == 0:
                output = result.stdout.decode("utf-8", errors="ignore")
                if "__kmpc_for_static_init_8" in output:
                    print("  ✓ libomp contains required symbols for sgl-kernel")
                else:
                    warnings.append("libomp may not contain required symbols")
        except FileNotFoundError:
            warnings.append("'nm' command not found, cannot verify libomp symbols")
        except:
            warnings.append("Could not verify libomp symbols")

    # Check if libomp can be loaded
    try:
        if libomp_location:
            lib = ctypes.CDLL(libomp_location)
            print("  ✓ libomp can be loaded successfully")
        elif found_libomp:
            # Try loading from LD_LIBRARY_PATH
            try:
                lib = ctypes.CDLL("libomp.so")
                print("  ✓ libomp can be loaded from system paths")
            except OSError:
                issues.append("libomp found but cannot be loaded")
                print("  ⚠ libomp found but cannot be loaded")
    except OSError as e:
        issues.append(f"libomp cannot be loaded: {e}")
        print(f"  ⚠ libomp cannot be loaded: {e}")
    except Exception as e:
        issues.append(f"Unexpected error when loading libomp: {e}")
        print(f"  ⚠ Unexpected error when loading libomp: {e}")

    # Print warnings and issues
    if warnings:
        print("\n  ⚠ Warnings:")
        for warning in warnings:
            print(f"     - {warning}")

    if issues:
        print("\n  ❌ Issues found:")
        for issue in issues:
            print(f"     - {issue}")
        print("\n  💡 Solutions:")
        print("     1. libomp should be installed automatically by setup_banana_pi.sh")
        print("     2. If missing, download from GitHub Releases:")
        print("        https://github.com/nthu-pllab/pllab-sglang/releases/tag/v1.0")
        print("     3. Extract to ~/.local/lib/:")
        print("        tar -xzf libomp_riscv.tar.gz -C ~/.local/lib/")
        print("     4. Set LD_PRELOAD and LD_LIBRARY_PATH:")
        print("        export LD_PRELOAD=~/.local/lib/libomp.so")
        print("        export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH")
        print("     5. Add to ~/.bashrc for permanent setting")
        print("        echo 'export LD_PRELOAD=~/.local/lib/libomp.so' >> ~/.bashrc")
        print(
            "        echo 'export LD_LIBRARY_PATH=~/.local/lib:\\$LD_LIBRARY_PATH' >> ~/.bashrc"
        )
        print("        source ~/.bashrc")
        return False

    if warnings:
        print("\n  ⚠ Some warnings, but may still work")
        return True

    print("  ✅ libomp configuration looks good!")
    return True


def launch_server():
    """Launch SGLang server if not running"""
    if check_server_running():
        print("✅ SGLang server is already running!")
        return True

    print("🚀 Launching SGLang server...")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change to the script directory
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")

    # Generate config file dynamically (same as benchmark_rvv_backends.py)
    try:
        from benchmark_rvv_backends import create_config_file

        config_file = create_config_file("rvv", script_dir)
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            print(f"   Expected in: {script_dir}")
            return False
    except ImportError:
        print("❌ Failed to import create_config_file from benchmark_rvv_backends")
        print("   Please ensure benchmark_rvv_backends.py is in the same directory")
        return False

    print(f"📄 Using config file: {os.path.basename(config_file)}")

    # Create log file for server output
    log_file = os.path.join(script_dir, "sglang_server.log")
    print(f"📝 Server logs will be written to: {log_file}")

    try:
        # Start server in background with logging
        with open(log_file, "w") as log:
            log.write(
                f"SGLang Server Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            log.write(f"Working directory: {script_dir}\n")
            log.write(f"Config file: {config_file}\n")
            log.write("=" * 60 + "\n\n")
            log.flush()

        # Try to use triton stub if available
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        if script_dir not in pythonpath.split(":"):
            env["PYTHONPATH"] = f"{script_dir}{':' + pythonpath if pythonpath else ''}"

        # Disable torch.compile on RISC-V (Inductor C++ compilation fails with -march=native)
        # PyTorch Inductor tries to compile C++ code with -march=native, but RISC-V g++ doesn't support it
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["SGLANG_ENABLE_TORCH_COMPILE"] = "0"

        # Ensure CPU binding is configured for TP workers
        if not env.get("SGLANG_CPU_OMP_THREADS_BIND"):
            config_bind = get_cpu_threads_bind_from_config(config_file)
            if config_bind:
                env["SGLANG_CPU_OMP_THREADS_BIND"] = config_bind
                print(f"  ✓ Applying cpu-omp-threads-bind from config: {config_bind}")
            else:
                cpu_count = os.cpu_count() or 1
                if cpu_count <= 1:
                    default_bind = "0"
                else:
                    default_bind = f"0-{cpu_count - 1}"
                env["SGLANG_CPU_OMP_THREADS_BIND"] = default_bind
                print(
                    "  ⚠ cpu-omp-threads-bind not found in config; "
                    f"defaulting to cores {default_bind}"
                )
        else:
            print(
                f"  ✓ Using existing SGLANG_CPU_OMP_THREADS_BIND="
                f"{env['SGLANG_CPU_OMP_THREADS_BIND']}"
            )

        # Check if triton is available (real or stub)
        # IMPORTANT: Even if triton is available in current process (via stub),
        # we need to ensure it's loaded in the subprocess too
        triton_stub_path = os.path.join(script_dir, "triton_stub.py")
        use_wrapper = False

        try:
            import triton

            version = getattr(triton, "__version__", "unknown")
            is_stub = "stub" in str(version).lower() or getattr(
                triton, "_is_stub", False
            )
            if is_stub:
                print("  ✓ triton stub is loaded in main process (RISC-V compatible)")
                # Even if stub is loaded in main process, subprocess needs it too
                if os.path.exists(triton_stub_path):
                    use_wrapper = True
            else:
                print(f"  ✓ triton is installed (version: {version})")
                # Real triton installed, no need for wrapper
        except ImportError:
            # triton not available in main process, check for stub
            if os.path.exists(triton_stub_path):
                print("  ⚠ triton not available, will use triton_stub.py in subprocess")
                use_wrapper = True
            else:
                print("  ⚠ triton not available and triton_stub.py not found")
                print("  ⚠ Server may fail to start due to missing triton")
                print("  💡 Create triton_stub.py in the script directory")

        # Determine Python executable to use (same as current process)
        python_exe = sys.executable
        print(f"  ✓ Using Python: {python_exe}")

        # Create wrapper script if needed (for triton stub only)
        if use_wrapper:
            wrapper_script = os.path.join(script_dir, "launch_server_wrapper.py")
            with open(wrapper_script, "w") as f:
                f.write("#!/usr/bin/env python3\n")
                f.write(
                    "# Wrapper script to load triton_stub and ensure riscv backend is registered\n"
                )
                f.write("import sys\n")
                f.write("import os\n")
                f.write("import importlib\n")
                f.write("import traceback\n")
                script_dir_real = os.path.realpath(script_dir)
                f.write(f'sys.path.insert(0, r"{script_dir_real}")\n')
                f.write("\n")
                f.write(
                    "# Step 1: Load triton_stub and vllm_stub FIRST, before any other imports\n"
                )
                f.write(
                    "# Try importing from venv site-packages first, then from script directory\n"
                )
                f.write("\n")
                f.write("# Load triton_stub\n")
                f.write("triton_stub_loaded = False\n")
                f.write("try:\n")
                f.write(
                    "    # Try importing from venv (if installed via setup script)\n"
                )
                f.write(
                    "    import triton_stub  # This will register triton in sys.modules\n"
                )
                f.write('    print("✓ triton_stub loaded from venv site-packages")\n')
                f.write("    triton_stub_loaded = True\n")
                f.write("except ImportError:\n")
                f.write("    # Try importing from script directory\n")
                f.write(
                    f'    triton_stub_path = os.path.join(r"{script_dir_real}", "triton_stub.py")\n'
                )
                f.write("    if os.path.exists(triton_stub_path):\n")
                f.write("        try:\n")
                f.write("            import importlib.util\n")
                f.write(
                    '            spec = importlib.util.spec_from_file_location("triton_stub", triton_stub_path)\n'
                )
                f.write(
                    "            triton_stub = importlib.util.module_from_spec(spec)\n"
                )
                f.write("            spec.loader.exec_module(triton_stub)\n")
                f.write(
                    '            print("✓ triton_stub loaded from script directory")\n'
                )
                f.write("            triton_stub_loaded = True\n")
                f.write("        except Exception as e:\n")
                f.write(
                    '            print(f"⚠ Failed to load triton_stub from script directory: {e}")\n'
                )
                f.write("            traceback.print_exc()\n")
                f.write("    else:\n")
                f.write(
                    '        print("⚠ triton_stub.py not found in script directory")\n'
                )
                f.write('        print(f"  Expected at: {triton_stub_path}")\n')
                f.write("\n")
                f.write("if not triton_stub_loaded:\n")
                f.write(
                    '    print("⚠ Warning: triton_stub not loaded, server may fail to start")\n'
                )
                f.write("\n")
                f.write("# Load vllm_stub\n")
                f.write("vllm_stub_loaded = False\n")
                f.write("try:\n")
                f.write(
                    "    # Try importing from venv (if installed via setup script)\n"
                )
                f.write(
                    "    import vllm_stub  # This will register vllm in sys.modules\n"
                )
                f.write('    print("✓ vllm_stub loaded from venv site-packages")\n')
                f.write("    vllm_stub_loaded = True\n")
                f.write("except ImportError:\n")
                f.write("    # Try importing from script directory\n")
                f.write(
                    f'    vllm_stub_path = os.path.join(r"{script_dir_real}", "vllm_stub.py")\n'
                )
                f.write("    if os.path.exists(vllm_stub_path):\n")
                f.write("        try:\n")
                f.write("            import importlib.util\n")
                f.write(
                    '            spec = importlib.util.spec_from_file_location("vllm_stub", vllm_stub_path)\n'
                )
                f.write(
                    "            vllm_stub = importlib.util.module_from_spec(spec)\n"
                )
                f.write("            spec.loader.exec_module(vllm_stub)\n")
                f.write(
                    '            print("✓ vllm_stub loaded from script directory")\n'
                )
                f.write("            vllm_stub_loaded = True\n")
                f.write("        except Exception as e:\n")
                f.write(
                    '            print(f"⚠ Failed to load vllm_stub from script directory: {e}")\n'
                )
                f.write("            traceback.print_exc()\n")
                f.write("    else:\n")
                f.write(
                    '        print("⚠ vllm_stub.py not found in script directory")\n'
                )
                f.write('        print(f"  Expected at: {vllm_stub_path}")\n')
                f.write("\n")
                f.write("if not vllm_stub_loaded:\n")
                f.write(
                    '    print("⚠ Warning: vllm_stub not loaded, server may fail to start")\n'
                )
                f.write("\n")
                f.write("# Step 2: Force import and registration of riscv backend\n")
                f.write(
                    "# This ensures the backend is registered even if __pycache__ is stale\n"
                )
                f.write("try:\n")
                f.write("    # Import rvv_backend to trigger registration\n")
                f.write("    from sglang.srt.layers.attention import rvv_backend\n")
                f.write('    print("✓ rvv_backend module imported")\n')
                f.write(
                    "    # Force reload attention_registry to ensure registration\n"
                )
                f.write(
                    "    from sglang.srt.layers.attention import attention_registry\n"
                )
                f.write("    importlib.reload(attention_registry)\n")
                f.write("    # Verify registration\n")
                f.write('    if "rvv" in attention_registry.ATTENTION_BACKENDS:\n')
                f.write('        print("✓ riscv backend is registered")\n')
                f.write("    else:\n")
                f.write(
                    '        print("⚠ riscv backend NOT registered after import!")\n'
                )
                f.write(
                    '        print(f"Available backends: {list(attention_registry.ATTENTION_BACKENDS.keys())}")\n'
                )
                f.write("except Exception as e:\n")
                f.write('    print(f"⚠ Error importing rvv_backend: {e}")\n')
                f.write("    traceback.print_exc()\n")
                f.write("    # Continue anyway, may still work\n")
                f.write("\n")
                f.write("# Step 3: Launch server with improved error handling\n")
                f.write("import subprocess\n")
                f.write(f'config_file = r"{config_file}"\n')
                f.write('print("=" * 60)\n')
                f.write('print("Launching SGLang server...")\n')
                f.write('print("=" * 60)\n')
                f.write("try:\n")
                f.write("    # Use subprocess.run to capture output and errors\n")
                f.write("    result = subprocess.run(\n")
                f.write(
                    '        [sys.executable, "-m", "sglang.launch_server", "--config", config_file],\n'
                )
                f.write("        capture_output=True,\n")
                f.write("        text=True,\n")
                f.write("        check=False\n")
                f.write("    )\n")
                f.write("    # Print output to stdout/stderr\n")
                f.write("    if result.stdout:\n")
                f.write("        print(result.stdout)\n")
                f.write("    if result.stderr:\n")
                f.write("        print(result.stderr, file=sys.stderr)\n")
                f.write("    if result.returncode != 0:\n")
                f.write(
                    '        print(f"\\n❌ Server exited with code {result.returncode}")\n'
                )
                f.write('        print("\\nFull error output:")\n')
                f.write("        print(result.stderr)\n")
                f.write("    sys.exit(result.returncode)\n")
                f.write("except Exception as e:\n")
                f.write('    print(f"\\n❌ Exception when launching server: {e}")\n')
                f.write("    traceback.print_exc()\n")
                f.write("    sys.exit(1)\n")
            os.chmod(wrapper_script, 0o755)
            server_cmd = [python_exe, wrapper_script]
            print(
                "  ✓ Created wrapper script to load triton_stub and ensure riscv backend"
            )
        else:
            # Use standard command (real triton installed or no stub available)
            # Still need to ensure rvv backend is registered
            # Create a minimal wrapper to force registration
            wrapper_script = os.path.join(script_dir, "launch_server_wrapper.py")
            with open(wrapper_script, "w") as f:
                f.write("#!/usr/bin/env python3\n")
                f.write("# Wrapper script to ensure riscv backend is registered\n")
                f.write("import sys\n")
                f.write("import os\n")
                f.write("import importlib\n")
                f.write("import traceback\n")
                script_dir_real = os.path.realpath(script_dir)
                f.write(f'sys.path.insert(0, r"{script_dir_real}")\n')
                f.write("\n")
                f.write("# Step 1: Load vllm_stub FIRST, before any other imports\n")
                f.write(
                    "# Try importing from venv site-packages first, then from script directory\n"
                )
                f.write("vllm_stub_loaded = False\n")
                f.write("try:\n")
                f.write(
                    "    # Try importing from venv (if installed via setup script)\n"
                )
                f.write(
                    "    import vllm_stub  # This will register vllm in sys.modules\n"
                )
                f.write('    print("✓ vllm_stub loaded from venv site-packages")\n')
                f.write("    vllm_stub_loaded = True\n")
                f.write("except ImportError:\n")
                f.write("    # Try importing from script directory\n")
                f.write(
                    f'    vllm_stub_path = os.path.join(r"{script_dir_real}", "vllm_stub.py")\n'
                )
                f.write("    if os.path.exists(vllm_stub_path):\n")
                f.write("        try:\n")
                f.write("            import importlib.util\n")
                f.write(
                    '            spec = importlib.util.spec_from_file_location("vllm_stub", vllm_stub_path)\n'
                )
                f.write(
                    "            vllm_stub = importlib.util.module_from_spec(spec)\n"
                )
                f.write("            spec.loader.exec_module(vllm_stub)\n")
                f.write(
                    '            print("✓ vllm_stub loaded from script directory")\n'
                )
                f.write("            vllm_stub_loaded = True\n")
                f.write("        except Exception as e:\n")
                f.write(
                    '            print(f"⚠ Failed to load vllm_stub from script directory: {e}")\n'
                )
                f.write("            traceback.print_exc()\n")
                f.write("    else:\n")
                f.write(
                    '        print("⚠ vllm_stub.py not found in script directory")\n'
                )
                f.write('        print(f"  Expected at: {vllm_stub_path}")\n')
                f.write("\n")
                f.write("if not vllm_stub_loaded:\n")
                f.write(
                    '    print("⚠ Warning: vllm_stub not loaded, server may fail to start")\n'
                )
                f.write("\n")
                f.write("# Step 2: Force import and registration of riscv backend\n")
                f.write("try:\n")
                f.write("    from sglang.srt.layers.attention import rvv_backend\n")
                f.write('    print("✓ rvv_backend module imported")\n')
                f.write(
                    "    from sglang.srt.layers.attention import attention_registry\n"
                )
                f.write("    importlib.reload(attention_registry)\n")
                f.write('    if "rvv" in attention_registry.ATTENTION_BACKENDS:\n')
                f.write('        print("✓ riscv backend is registered")\n')
                f.write("    else:\n")
                f.write('        print("⚠ riscv backend NOT registered!")\n')
                f.write(
                    '        print(f"Available: {list(attention_registry.ATTENTION_BACKENDS.keys())}")\n'
                )
                f.write("except Exception as e:\n")
                f.write('    print(f"⚠ Error: {e}")\n')
                f.write("    traceback.print_exc()\n")
                f.write("\n")
                f.write("# Launch server with improved error handling\n")
                f.write("import subprocess\n")
                f.write(f'config_file = r"{config_file}"\n')
                f.write('print("=" * 60)\n')
                f.write('print("Launching SGLang server...")\n')
                f.write('print("=" * 60)\n')
                f.write("try:\n")
                f.write("    # Use subprocess.run to capture output and errors\n")
                f.write("    result = subprocess.run(\n")
                f.write(
                    '        [sys.executable, "-m", "sglang.launch_server", "--config", config_file],\n'
                )
                f.write("        capture_output=True,\n")
                f.write("        text=True,\n")
                f.write("        check=False\n")
                f.write("    )\n")
                f.write("    # Print output to stdout/stderr\n")
                f.write("    if result.stdout:\n")
                f.write("        print(result.stdout)\n")
                f.write("    if result.stderr:\n")
                f.write("        print(result.stderr, file=sys.stderr)\n")
                f.write("    if result.returncode != 0:\n")
                f.write(
                    '        print(f"\\n❌ Server exited with code {result.returncode}")\n'
                )
                f.write('        print("\\nFull error output:")\n')
                f.write("        print(result.stderr)\n")
                f.write("    sys.exit(result.returncode)\n")
                f.write("except Exception as e:\n")
                f.write('    print(f"\\n❌ Exception when launching server: {e}")\n')
                f.write("    traceback.print_exc()\n")
                f.write("    sys.exit(1)\n")
            os.chmod(wrapper_script, 0o755)
            server_cmd = [python_exe, wrapper_script]
            print("  ✓ Created wrapper script to ensure riscv backend registration")

        # Start process with output redirected to log file
        log_f = open(log_file, "a")
        try:
            process = subprocess.Popen(
                server_cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=script_dir,
                env=env,
            )
        finally:
            log_f.close()

        print(f"📡 Server process started with PID: {process.pid}")
        print(f"💡 Monitor progress: tail -f {log_file}")
        print("")

        # Check if process is still running after a short delay
        time.sleep(2)
        if process.poll() is not None:
            # Process exited immediately, read error from log
            print("❌ Server process exited immediately!")
            print(f"📄 Last 20 lines of log file:")
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    for line in lines[-20:]:
                        print(f"   {line.rstrip()}")
            except (OSError, IOError):
                pass
            return False

        # Wait for service to start with progress updates
        print(
            "⏳ Waiting for server to start (this may take 5-15 minutes on RISC-V)..."
        )
        print("   Progress updates every 30 seconds:")

        start_time = time.time()
        check_interval = 30  # Check every 30 seconds
        last_check = start_time

        while time.time() - start_time < 3600:  # Max 1 hour
            # Check if process is still running
            if process.poll() is not None:
                print(
                    f"\n❌ Server process exited unexpectedly (exit code: {process.returncode})"
                )
                print(f"📄 Last 30 lines of log file:")
                try:
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        for line in lines[-30:]:
                            print(f"   {line.rstrip()}")
                except Exception:
                    pass
                return False

            # Check for errors in log file
            try:
                with open(log_file, "r") as f:
                    log_content = f.read()
                    # Check for common error patterns
                    error_keywords = [
                        "Traceback (most recent call last)",
                        "Error:",
                        "Exception:",
                        "Failed to",
                        "ImportError",
                        "ModuleNotFoundError",
                        "AttributeError",
                        "ValueError",
                        "RuntimeError",
                        "OSError",
                        "CUDA error",  # Even on CPU, some code might check CUDA
                    ]
                    for keyword in error_keywords:
                        if keyword in log_content:
                            # Find the error context (last occurrence)
                            lines = log_content.split("\n")
                            error_line_idx = None
                            for i in range(len(lines) - 1, -1, -1):
                                if keyword in lines[i]:
                                    error_line_idx = i
                                    break

                            if error_line_idx is not None:
                                print(f"\n❌ Error detected in log: '{keyword}'")
                                print(
                                    f"📄 Showing error context (last 20 lines around error):"
                                )
                                # Show context around error (10 lines before, 10 lines after)
                                start_idx = max(0, error_line_idx - 10)
                                end_idx = min(len(lines), error_line_idx + 11)
                                for i in range(start_idx, end_idx):
                                    marker = ">>> " if i == error_line_idx else "    "
                                    print(f"{marker}{lines[i]}")
                                return False
            except Exception as e:
                # If we can't read log, continue (don't fail on log read errors)
                pass

            # Check server health
            if check_server_running():
                elapsed = int(time.time() - start_time)
                print(f"\n✅ Server launched successfully! (took {elapsed} seconds)")
                return True

            # Print progress every 30 seconds
            elapsed = int(time.time() - start_time)
            elapsed_since_last_check = int(time.time() - last_check)
            if elapsed_since_last_check >= check_interval:
                print(f"   ⏱️  Still starting... ({elapsed}s elapsed)")
                # Show last few lines of log (more context)
                try:
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            # Show last 3 lines for better context
                            last_lines = [l.strip() for l in lines[-3:] if l.strip()]
                            if last_lines:
                                print(f"   📝 Latest log entries:")
                                for line in last_lines:
                                    print(f"      {line[:100]}")
                except (FileNotFoundError, IOError):
                    pass
                last_check = time.time()

            time.sleep(5)

        print(f"\n❌ Server startup timeout (exceeded 1 hour)")
        print(f"📄 Check full log: {log_file}")
        return False

    except Exception as e:
        print(f"❌ Failed to launch server: {e}")
        import traceback

        traceback.print_exc()
        return False


def shutdown_server():
    """Shutdown SGLang server"""
    print("🛑 Shutting down SGLang server...")

    processes = find_sglang_processes()

    if not processes:
        print("ℹ️ No SGLang server processes found")
        return True

    for proc in processes:
        try:
            print(f"🔄 Terminating process PID: {proc.pid}")
            proc.terminate()

            # Wait for graceful shutdown
            try:
                proc.wait(timeout=10)
                print(f"✅ Process {proc.pid} terminated gracefully")
            except psutil.TimeoutExpired:
                print(f"⚠️ Process {proc.pid} didn't terminate, forcing kill...")
                proc.kill()
                proc.wait()
                print(f"✅ Process {proc.pid} killed")

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"⚠️ Could not terminate process {proc.pid}: {e}")

    # Verify shutdown
    time.sleep(2)
    if not check_server_running():
        print("✅ Server shutdown completed!")
        return True
    else:
        print("⚠️ Server may still be running")
        return False


def check_server_health():
    """Check server health status"""
    try:
        response = requests.get("http://127.0.0.1:30000/health", timeout=3)
        return response.status_code == 200
    except:
        return False


def interactive_chat():
    """Interactive chat mode"""
    print("\n💬 Entering Interactive Chat Mode")
    print("Type 'quit' or 'exit' to quit")
    print("Type 'shutdown' to shutdown server")
    print("Type 'restart' to restart server")
    print("Type 'health' to check server health")
    print("Type 'help' to see help")
    print("-" * 50)

    request_count = 0
    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "shutdown":
                print("\n🛑 Shutting down server...")
                shutdown_server()
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "restart":
                print("\n🔄 Restarting server...")
                shutdown_server()
                time.sleep(2)
                if launch_server():
                    print("✅ Server restarted successfully!")
                else:
                    print("❌ Failed to restart server")
                continue
            elif user_input.lower() == "health":
                if check_server_health():
                    print("✅ Server is healthy")
                else:
                    print("❌ Server is not responding")
                continue
            elif user_input.lower() == "help":
                print("\n📖 Help:")
                print("  - Type questions directly to chat")
                print("  - 'shutdown': Shutdown server and exit")
                print("  - 'restart': Restart server")
                print("  - 'health': Check server health")
                print("  - 'quit' or 'exit': Quit chat only")
                print("  - 'help': Show this help")
                continue
            elif not user_input:
                continue

            # Check server health every 10 requests
            request_count += 1
            if request_count % 10 == 0:
                if not check_server_health():
                    print("⚠️ Server appears unresponsive, attempting restart...")
                    shutdown_server()
                    time.sleep(2)
                    if not launch_server():
                        print("❌ Failed to restart server, exiting...")
                        break

            # Send chat request
            url = "http://127.0.0.1:30000/v1/chat/completions"
            data = {
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Keep responses very short and concise.",
                    },
                    {"role": "user", "content": user_input},
                ],
                "max_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.7,
                "stop": ["\n", "User:", "Human:", "Assistant:", "."],
                "stream": False,
            }

            print("🤖 AI: ", end="", flush=True)

            # Dynamic timeout adjustment
            timeout = 1800 if request_count < 5 else 1800
            response = requests.post(url, json=data, timeout=timeout)

            if response.status_code == 200:
                result = response.json()
                ai_response = result["choices"][0]["message"]["content"]
                print(ai_response)
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                try:
                    error_details = response.json()
                    print(f"Details: {error_details}")
                except:
                    print(f"Response text: {response.text}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except requests.exceptions.Timeout:
            print("⏰ Request timed out. Server may be overloaded or unresponsive.")
            print("💡 Attempting to restart server...")
            shutdown_server()
            time.sleep(3)
            if launch_server():
                print("✅ Server restarted, please try again")
            else:
                print("❌ Failed to restart server")
        except requests.exceptions.ConnectionError:
            print("🔌 Connection error. Server may not be running.")
            print("💡 Attempting to restart server...")
            if launch_server():
                print("✅ Server restarted, please try again")
            else:
                print("❌ Failed to restart server")
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            print(f"💡 Error type: {type(e).__name__}")


def main():
    """Main function"""
    print("🚀 TinyLlama Interactive Chat with Server Management")
    print("=" * 60)
    print("")

    # Check dependencies and system resources
    print("Step 1: Checking dependencies and system resources...")

    # Check disk space
    print("\n📦 Checking disk space...")
    disk_ok = check_disk_space()
    print("")

    # Check libomp
    libomp_ok = check_libomp()
    print("")

    if not libomp_ok:
        print("⚠️  libomp configuration issues detected!")
        print("   The server may fail to start or run incorrectly.")
        print("   Please fix libomp configuration before continuing.")
        print("")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response not in ["y", "yes"]:
            print("Exiting. Please fix libomp configuration first.")
            return
        print("")

    # Check if server is running, if not launch it
    print("Step 2: Checking server status...")
    if not check_server_running():
        print("📡 SGLang server not running, launching...")
        if not launch_server():
            print("\n❌ Failed to start server")
            print("Please check your configuration and try again")
            print("\n💡 Troubleshooting tips:")
            print("   1. Check server log: tail -f sglang_server.log")
            print("   2. Verify libomp: check_libomp() output above")
            print("   3. Check memory: free -h")
            print("   4. Check disk space: df -h")
            print("   5. Check missing Python modules:")
            print("      - Run setup_banana_pi.sh to install all dependencies")
            print("      - Or manually: pip install <module_name>")
            print(
                "      - For wheel_builder packages: pip install <module> --index-url https://gitlab.com/api/v4/projects/riseproject%2Fpython%2Fwheel_builder/packages/pypi/simple"
            )
            print("   6. Verify environment setup:")
            print("      - Run: python test_environment.py")
            print("      - Check virtual environment is activated")
            print("      - Check wheels are installed from GitHub Releases")
            return
    else:
        print("✅ SGLang server is already running!")
    print("")

    # Start interactive chat
    print("Step 3: Starting interactive chat...")
    interactive_chat()


if __name__ == "__main__":
    main()
