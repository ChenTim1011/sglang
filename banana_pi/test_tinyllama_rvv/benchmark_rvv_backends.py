#!/usr/bin/env python3
"""
End-to-End Model Benchmark Script for RVV vs torch_native Backend Comparison

This script measures:
- TTFT (Time To First Token): Prefill latency
- Token generation throughput: tokens/second
- End-to-end latency: Total generation time

Usage:
    # Show all available options
    python benchmark_backends.py --help

    # Run all tests with default settings (uses existing server if running)
    python benchmark_backends.py

    # Run with custom parameters
    python benchmark_backends.py --warmup 2 --num-runs 5 --max-tokens 50

    # Run only torch_native backend
    python benchmark_backends.py --backend torch_native

    # Run only rvv backend
    python benchmark_backends.py --backend rvv

    # Restart server for each backend test
    python benchmark_backends.py --restart

    # Save results to JSON file
    python benchmark_backends.py --output results.json
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Add script directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Try to import requests
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Error: requests library not found. Please install it: pip install requests")
    sys.exit(1)

# Try to import numpy for better statistics
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available, using basic statistics")

# Try to import psutil for process management
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Error: psutil library not found. Please install it: pip install psutil")
    sys.exit(1)


# Memory Monitoring with Child Process Support
class BenchmarkMemoryMonitor(threading.Thread):
    def __init__(self, output_file="benchmark_memory.csv", interval=0.2):
        super().__init__()
        self.output_file = output_file
        self.interval = interval
        self.running = True
        self.stop_event = threading.Event()
        self.current_phase = "Init"
        self.token_count = 0  # To track decoding progress
        self.peak_rss_mb = 0  # Track peak globally per session or reset if needed

        # Write header
        with open(self.output_file, "w") as f:
            f.write(
                "timestamp,elapsed,phase,token_count,total_rss_mb,parent_rss_mb,children_rss_mb,num_processes\n"
            )

    def set_phase(self, phase, token_count=0):
        self.current_phase = phase
        self.token_count = token_count
        # Log immediately on phase change
        self.log_memory()

    def update_token_count(self, count):
        self.token_count = count

    def get_total_memory(self):
        try:
            parent = psutil.Process()  # Current process
            children = parent.children(recursive=True)

            parent_mem = parent.memory_info().rss
            children_mem = sum(p.memory_info().rss for p in children)

            return parent_mem, children_mem, len(children) + 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0, 0, 0

    def log_memory(self):
        try:
            parent_mem, children_mem, num_procs = self.get_total_memory()
            total_mb = (parent_mem + children_mem) / (1024 * 1024)
            parent_mb = parent_mem / (1024 * 1024)
            child_mb = children_mem / (1024 * 1024)

            self.peak_rss_mb = max(self.peak_rss_mb, total_mb)

            elapsed = time.time() - self.start_time
            timestamp = time.strftime("%H:%M:%S")

            with open(self.output_file, "a") as f:
                f.write(
                    f"{timestamp},{elapsed:.2f},{self.current_phase},{self.token_count},{total_mb:.2f},{parent_mb:.2f},{child_mb:.2f},{num_procs}\n"
                )

            return total_mb
        except Exception as e:
            # Avoid crashing the benchmark just for logging
            return 0

    def run(self):
        self.start_time = time.time()
        while not self.stop_event.is_set():
            self.log_memory()
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()


@dataclass
class BenchmarkResult:
    """Holds benchmark results for a single run"""

    backend: str
    prompt: str
    ttft_ms: float  # Time to first token in milliseconds (prefill latency)
    total_time_ms: float  # Total generation time in milliseconds
    tokens_generated: int
    throughput_tps: float  # Tokens per second (excluding TTFT)
    decode_latency_ms: float = 0.0  # Average decode latency per token
    end_to_end_latency_ms: float = 0.0  # Total latency / tokens
    peak_memory_mb: float = 0.0  # Peak memory usage during this run
    success: bool = True
    error_msg: Optional[str] = None


@dataclass
class AccuracyResult:
    """Holds accuracy test results"""

    backend: str
    kv_cache_dtype: str
    perplexity: float
    avg_latency_ms: float
    success: bool = True
    error_msg: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Holds aggregated benchmark results"""

    backend: str
    num_runs: int
    avg_ttft_ms: float
    min_ttft_ms: float
    max_ttft_ms: float
    std_ttft_ms: float = 0.0
    avg_throughput_tps: float = 0.0
    min_throughput_tps: float = 0.0
    max_throughput_tps: float = 0.0
    std_throughput_tps: float = 0.0
    avg_total_time_ms: float = 0.0
    avg_decode_latency_ms: float = 0.0
    avg_tokens_generated: float = 0.0
    avg_peak_memory_mb: float = 0.0
    success_rate: float = 0.0
    results: List[BenchmarkResult] = field(default_factory=list)
    accuracy_result: Optional[AccuracyResult] = None


def find_sglang_processes():
    """Find running SGLang processes"""
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["name"] and "python" in proc.info["name"].lower():
                cmdline = proc.info.get("cmdline") or []
                cmdline_str = " ".join(cmdline)
                if "sglang" in cmdline_str.lower() or "launch_server" in cmdline_str:
                    processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


def shutdown_server():
    """Shutdown SGLang server"""
    print("Shutting down existing SGLang server...")
    processes = find_sglang_processes()

    if not processes:
        print("   No SGLang server processes found")
        return True

    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=10)
            print(f"   Terminated process {proc.pid}")
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                proc.kill()
                print(f"   Killed process {proc.pid}")
            except psutil.NoSuchProcess:
                pass

    time.sleep(2)
    return True


def check_server_running(host="127.0.0.1", port=30000, timeout=600):
    """
    Check if SGLang server is running by trying a simple GET request.
    Uses longer timeout (600s) since server may be under load.
    """
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.Timeout:
        # Health endpoint might be slow under load; this is OK, server likely running
        return True
    except:
        return False


def launch_server(
    config_path: str, log_file: str, timeout: int = 1800, backend: Optional[str] = None
) -> bool:
    """Launch SGLang server with specified config"""
    print(f"Launching server with config: {config_path}")
    print(f"Log file: {log_file}")

    # Prepare environment
    env = os.environ.copy()
    script_dir = os.path.dirname(config_path)
    pythonpath = env.get("PYTHONPATH", "")
    if script_dir not in pythonpath.split(":"):
        env["PYTHONPATH"] = f"{script_dir}{':' + pythonpath if pythonpath else ''}"

    # Force SGLang to recognize this as a CPU environment to avoid looking for vLLM GPU kernels
    env["SGLANG_USE_CPU_ENGINE"] = "1"

    # For RVV we disable torch.compile (Inductor issues on riscv toolchains).
    # For torch_native keep defaults (may allow JIT/compile to proceed).
    if backend == "rvv":
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["SGLANG_ENABLE_TORCH_COMPILE"] = "0"
    elif backend == "torch_native":
        # Disable RVV GEMM for torch_native to get true baseline comparison
        # Without this, RVV GEMM would still be used even with torch_native attention
        env["SGLANG_DISABLE_RVV_GEMM"] = "1"
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["SGLANG_ENABLE_TORCH_COMPILE"] = "0"

    # Set OpenMP library paths for RVV (required for sgl-kernel)
    # Respect user's LD_PRELOAD if set, otherwise try to find it
    if "LD_PRELOAD" in env:
        print(f"   Using existing LD_PRELOAD: {env['LD_PRELOAD']}")
    else:
        home_dir = os.path.expanduser("~")
        omp_lib_path = os.path.join(home_dir, ".local", "lib", "libomp.so")
        if os.path.exists(omp_lib_path):
            env["LD_PRELOAD"] = omp_lib_path
            local_lib = os.path.join(home_dir, ".local", "lib")
            existing_ld_path = env.get("LD_LIBRARY_PATH", "")
            if local_lib not in existing_ld_path:
                env["LD_LIBRARY_PATH"] = (
                    f"{local_lib}:{existing_ld_path}" if existing_ld_path else local_lib
                )
            print(f"   OpenMP: LD_PRELOAD={omp_lib_path} (Auto-detected)")
        else:
            print(
                "Warning: libomp.so not found in ~/.local/lib and LD_PRELOAD not set. RVV kernels may hang."
            )


def launch_server(
    log_file: str,
    backend: str,
    model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    port: int = 30000,
    host: str = "127.0.0.1",
    timeout: int = 1800,
    mem_fraction_static: float = 0.5,
    max_prefill: int = 512,
    max_total: int = 1024,
    chunked_prefill: int = 256,
    disable_radix: bool = False,
    kv_cache_dtype: str = "auto",
    quantization: Optional[str] = None,
) -> bool:
    """Launch SGLang server with specified parameters"""
    print(f"Launching server for backend: {backend}")
    print(f"Log file: {log_file}")

    # Prepare environment
    env = os.environ.copy()

    # Ensure toolchain python path is included if running from source
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    python_path = os.path.join(repo_root, "python")
    if os.path.exists(python_path) and python_path not in env.get(
        "PYTHONPATH", ""
    ).split(":"):
        env["PYTHONPATH"] = f"{python_path}:{env.get('PYTHONPATH', '')}"

    # Force SGLang to recognize this as a CPU environment
    env["SGLANG_USE_CPU_ENGINE"] = "1"

    # Backend-specific environment variables
    if backend == "rvv":
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["SGLANG_ENABLE_TORCH_COMPILE"] = "0"
    elif backend == "torch_native":
        env["SGLANG_DISABLE_RVV_GEMM"] = "1"
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["SGLANG_ENABLE_TORCH_COMPILE"] = "0"

    # Set OpenMP library paths for RVV (required for sgl-kernel)
    home_dir = os.path.expanduser("~")
    omp_lib_path = os.path.join(home_dir, ".local", "lib", "libomp.so")
    if os.path.exists(omp_lib_path):
        env["LD_PRELOAD"] = omp_lib_path
        local_lib = os.path.join(home_dir, ".local", "lib")
        existing_ld_path = env.get("LD_LIBRARY_PATH", "")
        if local_lib not in existing_ld_path:
            env["LD_LIBRARY_PATH"] = (
                f"{local_lib}:{existing_ld_path}" if existing_ld_path else local_lib
            )
        print(f"   OpenMP: LD_PRELOAD={omp_lib_path} (Auto-detected)")

    # Set CPU threads binding if not set
    if not env.get("SGLANG_CPU_OMP_THREADS_BIND"):
        env["SGLANG_CPU_OMP_THREADS_BIND"] = "all"

    # Use the shared launch_server_rvv.py script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    launcher_script = os.path.join(script_dir, "launch_server_rvv.py")
    if not os.path.exists(launcher_script):
        print(f"Error: Launcher script not found at {launcher_script}")
        return False

    # Construct server arguments
    server_cmd = [
        sys.executable,
        launcher_script,
        "--model-path",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--attention-backend",
        backend,
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--max-prefill-tokens",
        str(max_prefill),
        "--max-total-tokens",
        str(max_total),
        "--chunked-prefill-size",
        str(chunked_prefill),
        "--cpu-offload-gb",
        "0",
        "--stream-interval",
        "1",
        "--dtype",
        "bfloat16" if kv_cache_dtype == "bfloat16" else "float16",
        "--kv-cache-dtype",
        kv_cache_dtype if kv_cache_dtype != "fp16" else "auto",
        "--tp",
        "1",  # tensor-parallel-size
    ]

    if disable_radix:
        server_cmd.append("--disable-radix-cache")

    if quantization:
        server_cmd.extend(["--quantization", quantization])

    if env.get("SGLANG_CPU_OMP_THREADS_BIND") == "all":
        # SGLang might need explicit bind args if not env var driven alone
        pass

    # Start server
    with open(log_file, "w") as log_f:
        log_f.write(f"Starting server at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"Cmd: {' '.join(server_cmd)}\n")
        log_f.write("=" * 60 + "\n\n")
        log_f.flush()

        process = subprocess.Popen(
            server_cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=script_dir,
        )

    print(f"   Server PID: {process.pid}")

    # Wait for server to start. Tail the log periodically so user can see progress.
    start_time = time.time()
    last_tail = 0
    tail_interval = 30
    tail_lines = 200

    def tail_log(path, n=200):
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                end = f.tell()
                size = 1024
                data = b""
                while len(data.splitlines()) <= n and end > 0:
                    read_size = min(size, end)
                    f.seek(end - read_size)
                    chunk = f.read(read_size)
                    data = chunk + data
                    end -= read_size
                lines = data.splitlines()[-n:]
                return b"\n".join(lines).decode("utf-8", errors="replace")
        except Exception:
            return ""

    while time.time() - start_time < timeout:
        # If process exited early, show logs and fail fast
        if process.poll() is not None:
            print(f"Server process exited with code: {process.returncode}")
            print("Last log snippet:")
            print(tail_log(log_file, tail_lines))
            return False

        # Check if server is responding (may take time under load, so use longer timeout)
        elapsed = int(time.time() - start_time)
        try:
            # Try a simple request with generous timeout to verify server is responding
            response = requests.get(
                f"http://127.0.0.1:30000/health",
                timeout=200,  # Generous timeout for slow CPU
            )
            if response.status_code == 200:
                print(f"Server started in {elapsed:.1f}s")
                return True
        except requests.exceptions.Timeout:
            try:
                resp = requests.post(
                    "http://127.0.0.1:30000/v1/chat/completions",
                    json={
                        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1,
                    },
                    timeout=200,
                )
                if resp.status_code in [200, 400, 422, 500]:
                    print(
                        f"Server started in {elapsed:.1f}s (verified via chat endpoint)"
                    )
                    return True
            except:
                pass
        except:
            pass

        if time.time() - last_tail >= tail_interval:
            last_tail = time.time()
            print(
                f"   Waiting... ({elapsed}s) - showing last {tail_lines} lines of log:"
            )
            print("--- Log tail start ---")
            print(tail_log(log_file, tail_lines))
            print("--- Log tail end ---")

        time.sleep(5)

    print(f"Server startup timeout ({timeout}s)")
    print("Final log snippet:")
    print(tail_log(log_file, tail_lines))
    return False


def run_single_benchmark(
    prompt: str,
    backend: str,
    max_tokens: int = 50,
    host: str = "127.0.0.1",
    port: int = 30000,
    timeout: int = 1800,
    mem_monitor: Optional[BenchmarkMemoryMonitor] = None,
    phase_prefix: str = "",
) -> BenchmarkResult:
    """Run a single benchmark request and measure timing"""

    url = f"http://{host}:{port}/v1/chat/completions"

    data = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True,  # Use streaming to measure TTFT
    }

    if mem_monitor:
        mem_monitor.set_phase(f"{phase_prefix}Prefill", 0)
        start_peak = (
            mem_monitor.peak_rss_mb
        )  # Capture peak before request starts if needed, but usually we care about peak during request

    try:
        start_time = time.perf_counter()
        ttft = None
        tokens_generated = 0

        response = requests.post(url, json=data, timeout=timeout, stream=True)

        if response.status_code != 200:
            return BenchmarkResult(
                backend=backend,
                prompt=prompt,
                ttft_ms=0,
                total_time_ms=0,
                tokens_generated=0,
                throughput_tps=0,
                success=False,
                error_msg=f"HTTP {response.status_code}: {response.text[:200]}",
            )

        # Process streaming response
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                if ttft is None:
                                    ttft = (time.perf_counter() - start_time) * 1000
                                    # First token received: Prefill done, Decode starts
                                    if mem_monitor:
                                        mem_monitor.set_phase(
                                            f"{phase_prefix}Decode", 1
                                        )

                                # Count tokens (approximate by splitting)
                                tokens_generated += 1
                                if mem_monitor:
                                    mem_monitor.update_token_count(tokens_generated)

                    except json.JSONDecodeError:
                        pass

        total_time = (time.perf_counter() - start_time) * 1000

        if ttft is None:
            ttft = total_time

        # Calculate throughput (tokens generated after first token)
        decode_time = total_time - ttft
        if decode_time > 0 and tokens_generated > 1:
            throughput = (tokens_generated - 1) / (decode_time / 1000)
            decode_latency = decode_time / (tokens_generated - 1)
        else:
            throughput = 0
            decode_latency = 0

        # End-to-end latency per token
        e2e_latency = total_time / tokens_generated if tokens_generated > 0 else 0

        peak_mem = mem_monitor.peak_rss_mb if mem_monitor else 0.0

        return BenchmarkResult(
            backend=backend,
            prompt=prompt,
            ttft_ms=ttft,
            total_time_ms=total_time,
            tokens_generated=tokens_generated,
            throughput_tps=throughput,
            decode_latency_ms=decode_latency,
            end_to_end_latency_ms=e2e_latency,
            peak_memory_mb=peak_mem,
            success=True,
        )

    except requests.exceptions.Timeout:
        return BenchmarkResult(
            backend=backend,
            prompt=prompt,
            ttft_ms=0,
            total_time_ms=0,
            tokens_generated=0,
            throughput_tps=0,
            success=False,
            error_msg="Request timeout",
        )
    except Exception as e:
        return BenchmarkResult(
            backend=backend,
            prompt=prompt,
            ttft_ms=0,
            total_time_ms=0,
            tokens_generated=0,
            throughput_tps=0,
            success=False,
            error_msg=str(e),
        )


def run_accuracy_test(
    backend: str,
    kv_cache_dtype: str,
    host: str = "127.0.0.1",
    port: int = 30000,
    timeout: int = 1800,
) -> AccuracyResult:
    """
    Run Perplexity (PPL) test using a sliding window approach with a known text.
    Uses wikitext-2 like approach with a single long prompt.
    """
    print(f"\nRunning Accuracy Test (PPL) for {backend} ({kv_cache_dtype})...")

    # Use a longer text for PPL calculation
    # Taken from Wikitext-2 validation set (first few lines)
    text = (
        "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . "
        "The development team wanted to clear up potential ambiguities with the numbering of the series , so the game was "
        "original conceived as a spin-off . While the game retained the standard turn-based tactical role-playing battle system , "
        "it also introduced new additions to the series ' gameplay ."
    )

    url = f"http://{host}:{port}/v1/completions"  # Use completions endpoint for PPL

    # We can't easily get true PPL from the API without logprobs support which might be limited
    # Instead, we'll measure the time to process this prompt (prefill) and
    # check if we can get logprobs to calculate perplexity.

    # SGLang local server supports echo=True and logprobs=1
    data = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": text,
        "max_tokens": 0,  # Only prefill
        "temperature": 0.0,
        "echo": True,
        "logprobs": 1,
    }

    try:
        start_time = time.perf_counter()
        response = requests.post(url, json=data, timeout=timeout)
        latency = (time.perf_counter() - start_time) * 1000

        if response.status_code != 200:
            return AccuracyResult(
                backend=backend,
                kv_cache_dtype=kv_cache_dtype,
                perplexity=0.0,
                avg_latency_ms=0.0,
                success=False,
                error_msg=f"HTTP {response.status_code}: {response.text[:200]}",
            )

        result = response.json()

        # Calculate Perplexity from logprobs
        # PPL = exp(-1/N * sum(log_prob))
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "logprobs" in choice and choice["logprobs"]:
                token_logprobs = choice["logprobs"].get("token_logprobs", [])
                # Filter out None values (first token usually has None logprob)
                valid_logprobs = [lp for lp in token_logprobs if lp is not None]

                if valid_logprobs:
                    avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
                    ppl = (
                        np.exp(-avg_logprob) if HAS_NUMPY else 2.71828 ** (-avg_logprob)
                    )
                else:
                    ppl = 0.0
            else:
                ppl = 0.0  # Logprobs not returned
        else:
            ppl = 0.0

        print(f"   Perplexity: {ppl:.4f}")
        print(f"   Latency: {latency:.2f} ms")

        return AccuracyResult(
            backend=backend,
            kv_cache_dtype=kv_cache_dtype,
            perplexity=ppl,
            avg_latency_ms=latency,
            success=True,
        )

    except Exception as e:
        return AccuracyResult(
            backend=backend,
            kv_cache_dtype=kv_cache_dtype,
            perplexity=0.0,
            avg_latency_ms=0.0,
            success=False,
            error_msg=str(e),
        )


def run_benchmark_suite(
    backend: str,
    prompts: List[str],
    warmup_runs: int = 2,
    num_runs: int = 5,
    max_tokens: int = 50,
    test_accuracy: bool = False,
    kv_cache_dtype: str = "auto",
    mem_monitor: Optional[BenchmarkMemoryMonitor] = None,
) -> BenchmarkSummary:
    """Run a complete benchmark suite for a backend"""

    print(f"\n{'='*60}")
    print(f"Benchmarking {backend.upper()} Backend")
    print(f"{'='*60}")

    # Run Accuracy Test first if requested
    accuracy_res = None
    if test_accuracy:
        if mem_monitor:
            mem_monitor.set_phase(f"Testing_{backend}_Accuracy")
        accuracy_res = run_accuracy_test(backend, kv_cache_dtype)
        if not accuracy_res.success:
            print(f"Accuracy test failed: {accuracy_res.error_msg}")

    results = []

    # Warmup runs
    if warmup_runs > 0:
        print(f"\nWarmup ({warmup_runs} runs)...")
        if mem_monitor:
            mem_monitor.set_phase(f"Testing_{backend}_Warmup")
        for i in range(warmup_runs):
            warmup_prompt = "Hello"
            result = run_single_benchmark(
                warmup_prompt,
                backend,
                max_tokens=10,
                mem_monitor=mem_monitor,
                phase_prefix=f"Warmup_{i}_",
            )
            status = "✓" if result.success else "✗"
            print(f"   Warmup {i+1}: {status} (tokens={result.tokens_generated})")

    # Actual benchmark runs
    print(f"\nBenchmark runs ({num_runs} runs per prompt, max_tokens={max_tokens})...")
    for prompt_idx, prompt in enumerate(prompts):
        print(
            f'\n   Prompt {prompt_idx + 1}: "{prompt[:50]}..."'
            if len(prompt) > 50
            else f'\n   Prompt {prompt_idx + 1}: "{prompt}"'
        )

        for run_idx in range(num_runs):
            result = run_single_benchmark(
                prompt,
                backend,
                max_tokens=max_tokens,
                mem_monitor=mem_monitor,
                phase_prefix=f"Prompt{prompt_idx+1}_Run{run_idx}_",
            )
            results.append(result)

            if result.success:
                print(
                    f"      Run {run_idx + 1}: TTFT={result.ttft_ms:.1f}ms, "
                    f"Tokens={result.tokens_generated}, "
                    f"Decode={result.decode_latency_ms:.1f}ms/tok, "
                    f"Throughput={result.throughput_tps:.2f} tok/s"
                )
            else:
                print(f"      Run {run_idx + 1}: FAILED - {result.error_msg}")

    # Calculate summary statistics
    successful_results = [r for r in results if r.success]
    if len(successful_results) == 0:
        return BenchmarkSummary(
            backend=backend,
            num_runs=len(results),
            avg_ttft_ms=0,
            min_ttft_ms=0,
            max_ttft_ms=0,
            std_ttft_ms=0,
            avg_throughput_tps=0,
            min_throughput_tps=0,
            max_throughput_tps=0,
            std_throughput_tps=0,
            avg_total_time_ms=0,
            success_rate=0.0,
            results=results,
        )

    # User requested to exclude the first run from the average calculation if possible
    stats_results = (
        successful_results[1:] if len(successful_results) > 1 else successful_results
    )

    ttfts = [r.ttft_ms for r in stats_results]
    throughputs = [r.throughput_tps for r in stats_results if r.throughput_tps > 0]
    total_times = [r.total_time_ms for r in stats_results]
    decode_latencies = [
        r.decode_latency_ms for r in stats_results if r.decode_latency_ms > 0
    ]
    tokens_list = [r.tokens_generated for r in stats_results]
    peak_mems = [r.peak_memory_mb for r in stats_results]

    # Calculate statistics with numpy if available
    if HAS_NUMPY:
        avg_ttft = np.mean(ttfts) if ttfts else 0
        min_ttft = np.min(ttfts) if ttfts else 0
        max_ttft = np.max(ttfts) if ttfts else 0
        std_ttft = np.std(ttfts) if len(ttfts) > 1 else 0
        avg_throughput = np.mean(throughputs) if throughputs else 0
        min_throughput = np.min(throughputs) if throughputs else 0
        max_throughput = np.max(throughputs) if throughputs else 0
        std_throughput = np.std(throughputs) if len(throughputs) > 1 else 0
        avg_total_time = np.mean(total_times) if total_times else 0
        avg_decode_latency = np.mean(decode_latencies) if decode_latencies else 0
        avg_tokens = np.mean(tokens_list) if tokens_list else 0
        avg_peak_mem = (
            np.max(peak_mems) if peak_mems else 0
        )  # Use Max of peaks as the conservative metric
    else:
        # Fallback to basic statistics
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
        min_ttft = min(ttfts) if ttfts else 0
        max_ttft = max(ttfts) if ttfts else 0
        std_ttft = 0
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
        min_throughput = min(throughputs) if throughputs else 0
        max_throughput = max(throughputs) if throughputs else 0
        std_throughput = 0
        avg_total_time = sum(total_times) / len(total_times) if total_times else 0
        avg_decode_latency = (
            sum(decode_latencies) / len(decode_latencies) if decode_latencies else 0
        )
        avg_tokens = sum(tokens_list) / len(tokens_list) if tokens_list else 0
        avg_peak_mem = max(peak_mems) if peak_mems else 0

    return BenchmarkSummary(
        backend=backend,
        num_runs=len(results),
        avg_ttft_ms=avg_ttft,
        min_ttft_ms=min_ttft,
        max_ttft_ms=max_ttft,
        std_ttft_ms=std_ttft,
        avg_throughput_tps=avg_throughput,
        min_throughput_tps=min_throughput,
        max_throughput_tps=max_throughput,
        std_throughput_tps=std_throughput,
        avg_total_time_ms=avg_total_time,
        avg_decode_latency_ms=avg_decode_latency,
        avg_tokens_generated=avg_tokens,
        avg_peak_memory_mb=avg_peak_mem,
        success_rate=len(successful_results) / len(results) * 100,
        results=results,
        accuracy_result=accuracy_res,
    )


def print_summary_table(summaries: List[BenchmarkSummary]):
    """Print a comparison table of benchmark results"""

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    # Accuracy / PPL Table
    print(
        f"\n{'Backend':<15} {'KV Type':<15} {'Perplexity':<12} {'PPL Latency':<12} {'Success':<10}"
    )
    print("-" * 100)
    for summary in summaries:
        if summary.accuracy_result:
            print(
                f"{summary.backend:<15} "
                f"{summary.accuracy_result.kv_cache_dtype:<15} "
                f"{summary.accuracy_result.perplexity:>10.4f}   "
                f"{summary.accuracy_result.avg_latency_ms:>9.1f} ms  "
                f"{str(summary.accuracy_result.success):<10}"
            )
        else:
            print(
                f"{summary.backend:<15} {'N/A':<15} {'N/A':<12} {'N/A':<12} {'N/A':<10}"
            )

    print("-" * 100)

    # Header - more metrics
    print(
        f"\n{'Backend':<15} {'Avg TTFT':<12} {'Avg Decode':<14} {'Avg TPS':<12} {'Avg Tokens':<12} {'Avg Peak Mem':<14} {'Success':<10}"
    )
    print("-" * 115)

    for summary in summaries:
        print(
            f"{summary.backend:<15} "
            f"{summary.avg_ttft_ms:>8.1f} ms  "
            f"{summary.avg_decode_latency_ms:>10.1f} ms/t "
            f"{summary.avg_throughput_tps:>8.2f} t/s "
            f"{summary.avg_tokens_generated:>8.1f}     "
            f"{summary.avg_peak_memory_mb:>10.2f} MB "
            f"{summary.success_rate:>8.1f}%"
        )

    print("-" * 115)

    # Detailed table
    print(
        f"\n{'Backend':<15} {'Min TTFT':<12} {'Max TTFT':<12} {'Min TPS':<12} {'Max TPS':<12} {'Avg Total':<12}"
    )
    print("-" * 100)

    for summary in summaries:
        print(
            f"{summary.backend:<15} "
            f"{summary.min_ttft_ms:>8.1f} ms  "
            f"{summary.max_ttft_ms:>8.1f} ms  "
            f"{summary.min_throughput_tps:>8.2f} t/s "
            f"{summary.max_throughput_tps:>8.2f} t/s "
            f"{summary.avg_total_time_ms:>8.1f} ms"
        )

    print("-" * 100)

    # Comparison if we have both backends
    if len(summaries) == 2:
        s1, s2 = summaries[0], summaries[1]

        # Only compare if both are successful enough
        if s1.success_rate > 0 and s2.success_rate > 0:
            print("\nCOMPARISON (RISC-V vs torch_native):")

            # Determine which is which
            if s1.backend == "rvv":
                riscv, torch = s1, s2
            else:
                riscv, torch = s2, s1

            if torch.avg_ttft_ms > 0:
                ttft_speedup = (
                    torch.avg_ttft_ms / riscv.avg_ttft_ms
                    if riscv.avg_ttft_ms > 0
                    else 0
                )
                print(
                    f"   TTFT Speedup: {ttft_speedup:.2f}x {'(RISC-V faster)' if ttft_speedup > 1 else '(torch_native faster)'}"
                )

            if torch.avg_throughput_tps > 0:
                tps_speedup = (
                    riscv.avg_throughput_tps / torch.avg_throughput_tps
                    if torch.avg_throughput_tps > 0
                    else 0
                )
                print(
                    f"   Throughput Speedup: {tps_speedup:.2f}x {'(RISC-V faster)' if tps_speedup > 1 else '(torch_native faster)'}"
                )

            if torch.avg_decode_latency_ms > 0 and riscv.avg_decode_latency_ms > 0:
                decode_speedup = (
                    torch.avg_decode_latency_ms / riscv.avg_decode_latency_ms
                )
                print(
                    f"   Decode Latency Speedup: {decode_speedup:.2f}x {'(RISC-V faster)' if decode_speedup > 1 else '(torch_native faster)'}"
                )

            # PPL Comparison
            if riscv.accuracy_result and torch.accuracy_result:
                ppl_diff = (
                    riscv.accuracy_result.perplexity - torch.accuracy_result.perplexity
                )
                print(
                    f"   Perplexity Diff: {ppl_diff:.4f} (RISC-V: {riscv.accuracy_result.perplexity:.4f}, Torch: {torch.accuracy_result.perplexity:.4f})"
                )


def save_results_json(summaries: List[BenchmarkSummary], output_file: str):
    """Save benchmark results to JSON file"""
    results = []
    for summary in summaries:
        acc_res = None
        if summary.accuracy_result:
            acc_res = {
                "kv_cache_dtype": summary.accuracy_result.kv_cache_dtype,
                "perplexity": summary.accuracy_result.perplexity,
                "latency_ms": summary.accuracy_result.avg_latency_ms,
                "success": summary.accuracy_result.success,
                "error": summary.accuracy_result.error_msg,
            }

        results.append(
            {
                "backend": summary.backend,
                "num_runs": summary.num_runs,
                "avg_ttft_ms": summary.avg_ttft_ms,
                "min_ttft_ms": summary.min_ttft_ms,
                "max_ttft_ms": summary.max_ttft_ms,
                "avg_throughput_tps": summary.avg_throughput_tps,
                "min_throughput_tps": summary.min_throughput_tps,
                "max_throughput_tps": summary.max_throughput_tps,
                "avg_total_time_ms": summary.avg_total_time_ms,
                "success_rate": summary.success_rate,
                "accuracy": acc_res,
                "individual_results": [
                    {
                        "prompt": r.prompt[:50],
                        "ttft_ms": r.ttft_ms,
                        "total_time_ms": r.total_time_ms,
                        "tokens_generated": r.tokens_generated,
                        "throughput_tps": r.throughput_tps,
                        "success": r.success,
                        "error_msg": r.error_msg,
                    }
                    for r in summary.results
                ],
            }
        )

    with open(output_file, "w") as f:
        json.dump(
            {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results},
            f,
            indent=2,
        )

    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RVV vs torch_native attention backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full comparison benchmark
  python benchmark_backends.py

  # Run only RISC-V backend
  python benchmark_backends.py --backend rvv

  # Custom parameters
  python benchmark_backends.py --warmup 3 --num-runs 10 --max-tokens 100

  # Restart server for each backend (useful for comparing different backends)
  python benchmark_backends.py --restart

  # Save results to JSON
  python benchmark_backends.py --output results.json
""",
    )

    parser.add_argument(
        "--backend",
        choices=["rvv", "torch_native", "both"],
        default="both",
        help="Which backend(s) to benchmark",
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Number of warmup runs before measurement"
    )
    parser.add_argument(
        "--num-runs", type=int, default=6, help="Number of measurement runs per prompt"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Server startup timeout in seconds"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file for results"
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        help="KV cache dtype (auto, float16, int8)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Model quantization method (e.g., w8a8_int8, fp8)",
    )
    parser.add_argument(
        "--test-accuracy",
        action="store_true",
        help="Run accuracy (perplexity) test in addition to latency benchmarks",
    )
    parser.add_argument(
        "--memory-log",
        default="benchmark_ram_usage.csv",
        help="CSV file to log memory usage",
    )

    # Internal arguments for server dispatch
    parser.add_argument("--exec-server", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Dispatch to server mode if requested
    if args.exec_server:
        if not args.config:
            print("Error: --config required for --exec-server")
            sys.exit(1)
        run_server_process(args.config)
        return

    # Start Memory Monitor
    print(f"Starting memory monitor (log: {args.memory_log})...")
    mem_monitor = BenchmarkMemoryMonitor(output_file=args.memory_log, interval=0.1)
    mem_monitor.start()

    try:
        mem_monitor.set_phase("Startup")
        # Test prompts of varying complexity
        prompts = [
            "What is 2+2?",
            "Explain what a CPU does in one sentence.",
            "Write a short poem about computers.",
        ]

        script_dir = os.path.dirname(os.path.abspath(__file__))

        backends_to_test = []
        if args.backend == "both":
            backends_to_test = ["rvv", "torch_native"]
        else:
            backends_to_test = [args.backend]

        summaries = []

        print("SGLang Backend Benchmark")
        print("=" * 60)
        print(f"Backends: {', '.join(backends_to_test)}")
        print(f"Warmup runs: {args.warmup}")
        print(f"Measurement runs: {args.num_runs}")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Prompts: {len(prompts)}")
        print(f"KV Cache: {args.kv_cache_dtype}")
        print(f"Quantization: {args.quantization}")
        print(f"Test Accuracy: {args.test_accuracy}")
        print("=" * 60)

        for backend in backends_to_test:
            mem_monitor.set_phase(f"Testing_{backend}")
            print(f"\n\n{'#'*60}")
            print(f"# Testing {backend.upper()} Backend")
            print(f"{'#'*60}")

            shutdown_server()
            time.sleep(3)

            log_file = os.path.join(script_dir, f"benchmark_{backend}.log")

            if not launch_server(
                log_file=log_file,
                backend=backend,
                timeout=args.timeout,
                kv_cache_dtype=args.kv_cache_dtype,
                quantization=args.quantization,
            ):
                print(f"Failed to start server for {backend} backend")
                continue

            # Give server a moment to stabilize
            time.sleep(5)

            # Run benchmarks
            summary = run_benchmark_suite(
                backend=backend,
                prompts=prompts,
                warmup_runs=args.warmup,
                num_runs=args.num_runs,
                max_tokens=args.max_tokens,
                test_accuracy=args.test_accuracy,
                kv_cache_dtype=args.kv_cache_dtype,
                mem_monitor=mem_monitor,
            )
            summaries.append(summary)

            # Shutdown after run
            shutdown_server()
            time.sleep(2)

    finally:
        mem_monitor.stop()
        mem_monitor.join()

    # Print comparison
    if summaries:
        print_summary_table(summaries)

        if args.output:
            save_results_json(summaries, args.output)
    else:
        print("\nNo successful benchmarks completed")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
