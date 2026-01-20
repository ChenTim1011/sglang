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


def wait_for_service(max_wait=3600):
    """Wait for service to start (deprecated - use launch_server's built-in waiting)"""
    print(" Waiting for SGLang service to start...")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://127.0.0.1:30000/health", timeout=2)
            if response.status_code == 200:
                print(" Service is ready!")
                return True
        except:
            pass
        time.sleep(5)
        elapsed = int(time.time() - start_time)
        if elapsed % 30 == 0:  # Print every 30 seconds
            print(f"   Waiting... ({elapsed}s elapsed)")

    print(" Service startup timeout")
    return False


def check_server_running():
    """Check if SGLang server is running"""
    try:
        response = requests.get("http://127.0.0.1:30000/health", timeout=2)
        return response.status_code == 200
    except:
        return False


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

                    print(f"   Disk space: {available} available / {total} total")
                    print(f"     Used: {used} ({use_percent})")

                    # Check if disk space is low (< 1GB)
                    try:
                        # Parse available space (e.g., "5.2G" -> 5.2)
                        avail_str = available.upper().replace("G", "").replace("M", "")
                        avail_value = float(avail_str)
                        if "M" in available.upper():
                            # Less than 1GB if in MB
                            if avail_value < 1024:
                                print("   Warning: Low disk space (< 1GB)")
                                return False
                        elif "G" in available.upper():
                            if avail_value < 1.0:
                                print("   Warning: Low disk space (< 1GB)")
                                return False
                    except:
                        pass

                    return True
    except FileNotFoundError:
        print("   'df' command not found")
    except:
        pass
    return True  # Don't fail if we can't check


def check_libomp():
    """Check if libomp is available and configured correctly"""
    print(" Checking libomp configuration...")

    issues = []
    warnings = []

    # Check LD_PRELOAD
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if ld_preload:
        libomp_in_preload = "libomp.so" in ld_preload
        if libomp_in_preload:
            print(f"   LD_PRELOAD is set: {ld_preload}")
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
            print(f"   Found libomp.so: {libomp_location}")
            break

    # Also check LD_LIBRARY_PATH
    if ld_library_path:
        for path in ld_library_path.split(":"):
            if path:
                libomp_so = os.path.join(path, "libomp.so")
                if os.path.exists(libomp_so):
                    found_libomp = True
                    libomp_location = libomp_so
                    print(f"   Found libomp.so in LD_LIBRARY_PATH: {libomp_location}")
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
        print("   libomp.so not found")
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
                    print("   libomp contains required symbols for sgl-kernel")
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
            print("   libomp can be loaded successfully")
        elif found_libomp:
            # Try loading from LD_LIBRARY_PATH
            try:
                lib = ctypes.CDLL("libomp.so")
                print("   libomp can be loaded from system paths")
            except OSError:
                issues.append("libomp found but cannot be loaded")
                print("   libomp found but cannot be loaded")
    except OSError as e:
        issues.append(f"libomp cannot be loaded: {e}")
        print(f"   libomp cannot be loaded: {e}")
    except Exception as e:
        issues.append(f"Unexpected error when loading libomp: {e}")
        print(f"   Unexpected error when loading libomp: {e}")

    # Print warnings and issues
    if warnings:
        print("\n   Warnings:")
        for warning in warnings:
            print(f"     - {warning}")

    if issues:
        print("\n   Issues found:")
        for issue in issues:
            print(f"     - {issue}")
        print("\n   Solutions:")
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
        print("\n   Some warnings, but may still work")
        return True

    print("   libomp configuration looks good!")
    return True


def launch_server():
    """Launch SGLang server if not running"""
    if check_server_running():
        print(" SGLang server is already running!")
        return True

    print(" Launching SGLang server...")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change to the script directory
    os.chdir(script_dir)
    print(f" Working directory: {script_dir}")

    # Create log file for server output
    log_file = os.path.join(script_dir, "sglang_server.log")
    print(f" Server logs will be written to: {log_file}")

    try:
        # Start server in background with logging
        with open(log_file, "w") as log:
            log.write(
                f"SGLang Server Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            log.write(f"Working directory: {script_dir}\n")
            log.write("=" * 60 + "\n\n")
            log.flush()

        # Prepare environment
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        if script_dir not in pythonpath.split(":"):
            env["PYTHONPATH"] = f"{script_dir}{':' + pythonpath if pythonpath else ''}"

        # Ensure toolchain python path is included if running from source
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        python_path = os.path.join(repo_root, "python")
        if os.path.exists(python_path) and python_path not in env.get(
            "PYTHONPATH", ""
        ).split(":"):
            env["PYTHONPATH"] = f"{python_path}:{env.get('PYTHONPATH', '')}"

        # SGLang Settings for RVV
        env["SGLANG_USE_CPU_ENGINE"] = "1"
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["SGLANG_ENABLE_TORCH_COMPILE"] = "0"

        # Ensure CPU binding is configured
        if not env.get("SGLANG_CPU_OMP_THREADS_BIND"):
            env["SGLANG_CPU_OMP_THREADS_BIND"] = "all"

        # Use the shared launch_server_rvv.py script
        launcher_script = os.path.join(script_dir, "launch_server_rvv.py")
        if not os.path.exists(launcher_script):
            print(f" Error: Launcher script not found at {launcher_script}")
            return False

        server_cmd = [
            sys.executable,
            launcher_script,
            "--model-path",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--host",
            "127.0.0.1",
            "--port",
            "30000",
            "--attention-backend",
            "rvv",
            "--mem-fraction-static",
            "0.5",
            "--max-prefill-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--chunked-prefill-size",
            "256",
            "--cpu-offload-gb",
            "0",
            "--stream-interval",
            "1",
            "--dtype",
            "float16",
            "--kv-cache-dtype",
            "auto",
            "--tp",
            "1",
        ]

        # Start server
        with open(log_file, "a") as log_f:
            # Expand command for easier debugging
            log_f.write(f"Command: {' '.join(server_cmd)}\n")
            process = subprocess.Popen(
                server_cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=script_dir,
            )

        print(f"   Server PID: {process.pid}")

        # Wait for server to start
        start_time = time.time()
        while time.time() - start_time < 300:  # 5 minutes timeout
            if check_server_running():
                print(" Server started successfully!")
                return True

            if process.poll() is not None:
                print(f" Server process exited with code: {process.returncode}")
                print(f"   Check logs at {log_file}")
                return False

            time.sleep(2)

        print(" Server startup timeout")
        return False

    except Exception as e:
        print(f" Failed to launch server: {e}")
        return False


def shutdown_server():
    """Shutdown SGLang server"""
    print(" Shutting down SGLang server...")

    processes = find_sglang_processes()

    if not processes:
        print(" No SGLang server processes found")
        return True

    for proc in processes:
        try:
            print(f" Terminating process PID: {proc.pid}")
            proc.terminate()

            # Wait for graceful shutdown
            try:
                proc.wait(timeout=10)
                print(f" Process {proc.pid} terminated gracefully")
            except psutil.TimeoutExpired:
                print(f" Process {proc.pid} didn't terminate, forcing kill...")
                proc.kill()
                proc.wait()
                print(f" Process {proc.pid} killed")

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f" Could not terminate process {proc.pid}: {e}")

    # Verify shutdown
    time.sleep(2)
    if not check_server_running():
        print(" Server shutdown completed!")
        return True
    else:
        print(" Server may still be running")
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
    print("\n Entering Interactive Chat Mode")
    print("Type 'quit' or 'exit' to quit")
    print("Type 'shutdown' to shutdown server")
    print("Type 'restart' to restart server")
    print("Type 'health' to check server health")
    print("Type 'help' to see help")
    print("-" * 50)

    request_count = 0
    while True:
        try:
            user_input = input("\n You: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print(" Goodbye!")
                break
            elif user_input.lower() == "shutdown":
                print("\n Shutting down server...")
                shutdown_server()
                print(" Goodbye!")
                break
            elif user_input.lower() == "restart":
                print("\n Restarting server...")
                shutdown_server()
                time.sleep(2)
                if launch_server():
                    print(" Server restarted successfully!")
                else:
                    print(" Failed to restart server")
                continue
            elif user_input.lower() == "health":
                if check_server_health():
                    print(" Server is healthy")
                else:
                    print(" Server is not responding")
                continue
            elif user_input.lower() == "help":
                print("\n Help:")
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
                    print(" Server appears unresponsive, attempting restart...")
                    shutdown_server()
                    time.sleep(2)
                    if not launch_server():
                        print(" Failed to restart server, exiting...")
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

            print(" AI: ", end="", flush=True)

            # Dynamic timeout adjustment
            timeout = 1800 if request_count < 5 else 1800
            response = requests.post(url, json=data, timeout=timeout)

            if response.status_code == 200:
                result = response.json()
                ai_response = result["choices"][0]["message"]["content"]
                print(ai_response)
            else:
                print(f" HTTP Error: {response.status_code}")
                try:
                    error_details = response.json()
                    print(f"Details: {error_details}")
                except:
                    print(f"Response text: {response.text}")

        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
        except requests.exceptions.Timeout:
            print(" Request timed out. Server may be overloaded or unresponsive.")
            print(" Attempting to restart server...")
            shutdown_server()
            time.sleep(3)
            if launch_server():
                print(" Server restarted, please try again")
            else:
                print(" Failed to restart server")
        except requests.exceptions.ConnectionError:
            print(" Connection error. Server may not be running.")
            print(" Attempting to restart server...")
            if launch_server():
                print(" Server restarted, please try again")
            else:
                print(" Failed to restart server")
        except Exception as e:
            print(f" Error occurred: {e}")
            print(f" Error type: {type(e).__name__}")


def main():
    """Main function"""
    print(" TinyLlama Interactive Chat with Server Management")
    print("=" * 60)
    print("")

    # Check dependencies and system resources
    print("Step 1: Checking dependencies and system resources...")

    # Check disk space
    print("\n Checking disk space...")
    disk_ok = check_disk_space()
    print("")

    # Check libomp
    libomp_ok = check_libomp()
    print("")

    if not libomp_ok:
        print("  libomp configuration issues detected!")
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
        print(" SGLang server not running, launching...")
        if not launch_server():
            print("\n Failed to start server")
            print("Please check your configuration and try again")
            print("\n Troubleshooting tips:")
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
        print(" SGLang server is already running!")
    print("")

    # Start interactive chat
    print("Step 3: Starting interactive chat...")
    interactive_chat()


if __name__ == "__main__":
    main()
