#!/usr/bin/env python3
"""
Benchmark Script for RVV vs torch_native Backend Comparison

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
    avg_throughput_tps: float
    min_throughput_tps: float
    max_throughput_tps: float
    std_throughput_tps: float = 0.0
    avg_total_time_ms: float
    avg_decode_latency_ms: float = 0.0
    avg_tokens_generated: float = 0.0
    success_rate: float = 0.0
    results: List[BenchmarkResult] = field(default_factory=list)


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
    print("🛑 Shutting down existing SGLang server...")
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


def check_server_running(host="127.0.0.1", port=30000, timeout=10):
    """
    Check if SGLang server is running by trying a simple GET request.
    Uses longer timeout (10s) since server may be under load.
    """
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.Timeout:
        # Health endpoint might be slow under load; this is OK, server likely running
        return True
    except:
        return False


def create_config_file(backend: str, config_dir: str) -> str:
    """Create a config file for the specified backend.

    This function writes a reduced-resource benchmark config (optimized to
    lower startup memory/CPU load) to the `config_dir` and returns the path
    to the generated YAML file.
    """
    # Reduced-resource settings for Banana Pi (16GB RAM)
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    mem_fraction_static = 0.5
    max_prefill = 32
    max_total = 64
    chunked_prefill = 32
    cpu_offload = 0
    stream_interval = 1
    enable_metrics = False

    config_content = f"""# SGLang Benchmark Configuration - {backend.upper()} Backend
model-path: {model_path}
device: cpu
host: 127.0.0.1
port: 30000

tensor-parallel-size: 1

# Attention Backend Configuration
attention-backend: {backend}
prefill-attention-backend: {backend}
decode-attention-backend: {backend}

model-impl: transformers
dtype: float16
kv-cache-dtype: auto

# Memory Management
mem-fraction-static: {mem_fraction_static}
max-running-requests: 1
max-prefill-tokens: {max_prefill}
max-total-tokens: {max_total}
chunked-prefill-size: {chunked_prefill}
cpu-offload-gb: {cpu_offload}

stream-interval: {stream_interval}
num-continuous-decode-steps: 1

enable-metrics: {str(enable_metrics).lower()}
log-requests: false
"""
    config_path = os.path.join(config_dir, f"config_benchmark_{backend}.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path


def launch_server(
    config_path: str, log_file: str, timeout: int = 1800, backend: Optional[str] = None
) -> bool:
    """Launch SGLang server with specified config"""
    print(f"🚀 Launching server with config: {config_path}")
    print(f"📝 Log file: {log_file}")

    # Prepare environment
    env = os.environ.copy()
    script_dir = os.path.dirname(config_path)
    pythonpath = env.get("PYTHONPATH", "")
    if script_dir not in pythonpath.split(":"):
        env["PYTHONPATH"] = f"{script_dir}{':' + pythonpath if pythonpath else ''}"

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
        print(f"   OpenMP: LD_PRELOAD={omp_lib_path}")

    # Set CPU threads binding if not set
    if not env.get("SGLANG_CPU_OMP_THREADS_BIND"):
        env["SGLANG_CPU_OMP_THREADS_BIND"] = "all"

    # Build wrapper script that loads stubs and launches server properly
    wrapper_script = os.path.join(script_dir, "_benchmark_launcher.py")
    with open(wrapper_script, "w") as f:
        f.write(
            f"""#!/usr/bin/env python3
import sys
import os
import importlib
import traceback

# Add stubs to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Step 1: Load vllm_stub FIRST
vllm_stub_loaded = False
try:
    import vllm_stub
    print("✓ vllm_stub loaded")
    vllm_stub_loaded = True
except ImportError:
    vllm_stub_path = os.path.join(script_dir, "vllm_stub.py")
    if os.path.exists(vllm_stub_path):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("vllm_stub", vllm_stub_path)
            vllm_stub = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vllm_stub)
            print("✓ vllm_stub loaded from script directory")
            vllm_stub_loaded = True
        except Exception as e:
            print(f"⚠ Failed to load vllm_stub: {{e}}")

# Step 2: Load triton_stub
try:
    import triton_stub
    print("✓ triton_stub loaded")
except ImportError:
    triton_stub_path = os.path.join(script_dir, "triton_stub.py")
    if os.path.exists(triton_stub_path):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("triton_stub", triton_stub_path)
            triton_stub = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(triton_stub)
            print("✓ triton_stub loaded from script directory")
        except Exception as e:
            print(f"⚠ Failed to load triton_stub: {{e}}")

# Step 3: Force import and registration of rvv backend
try:
    from sglang.srt.layers.attention import rvv_backend
    print("✓ rvv_backend module imported")
    from sglang.srt.layers.attention import attention_registry
    importlib.reload(attention_registry)
    if "rvv" in attention_registry.ATTENTION_BACKENDS:
        print("✓ rvv backend is registered")
    else:
        print("⚠ rvv backend NOT registered!")
        print(f"Available: {{list(attention_registry.ATTENTION_BACKENDS.keys())}}")
except Exception as e:
    print(f"⚠ Error registering rvv backend: {{e}}")
    traceback.print_exc()

# Step 4: Launch server using sglang.launch_server module
import subprocess
config_file = r"{config_path}"
print("=" * 60)
print("Launching SGLang server...")
print("=" * 60)
try:
    result = subprocess.run(
        [sys.executable, "-m", "sglang.launch_server", "--config", config_file],
        check=False
    )
    sys.exit(result.returncode)
except Exception as e:
    print(f"\\n❌ Exception when launching server: {{e}}")
    traceback.print_exc()
    sys.exit(1)
"""
        )
    os.chmod(wrapper_script, 0o755)

    # Start server
    with open(log_file, "w") as log_f:
        log_f.write(f"Starting server at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"Config: {config_path}\n")
        log_f.write("=" * 60 + "\n\n")
        log_f.flush()

        process = subprocess.Popen(
            [sys.executable, wrapper_script],
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
            print(f"❌ Server process exited with code: {process.returncode}")
            print("--- Last log snippet ---")
            print(tail_log(log_file, tail_lines))
            return False

        # Check if server is responding (may take time under load, so use longer timeout)
        elapsed = int(time.time() - start_time)
        try:
            # Try a simple request with generous timeout to verify server is responding
            response = requests.get(
                f"http://127.0.0.1:30000/health",
                timeout=15,  # Generous timeout for slow CPU
            )
            if response.status_code == 200:
                print(f"✅ Server started in {elapsed:.1f}s")
                return True
        except requests.exceptions.Timeout:
            # Timeout on health check might mean server is running but slow
            # Try to send an actual request to verify (this proves server is live)
            try:
                resp = requests.post(
                    "http://127.0.0.1:30000/v1/chat/completions",
                    json={
                        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1,
                    },
                    timeout=15,
                )
                # If we got any response (even an error), server is up
                if resp.status_code in [200, 400, 422, 500]:
                    print(
                        f"✅ Server started in {elapsed:.1f}s (verified via chat endpoint)"
                    )
                    return True
            except:
                pass
        except:
            pass

        # Periodically show tail of log to help debugging long startups
        if time.time() - last_tail >= tail_interval:
            last_tail = time.time()
            print(
                f"   Waiting... ({elapsed}s) - showing last {tail_lines} lines of log:"
            )
            print("--- Log tail start ---")
            print(tail_log(log_file, tail_lines))
            print("--- Log tail end ---")

        time.sleep(5)

    print(f"❌ Server startup timeout ({timeout}s)")
    print("--- Final log snippet ---")
    print(tail_log(log_file, tail_lines))
    return False


def run_single_benchmark(
    prompt: str,
    backend: str,
    max_tokens: int = 50,
    host: str = "127.0.0.1",
    port: int = 30000,
    timeout: int = 1800,
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
                                # Count tokens (approximate by splitting)
                                tokens_generated += 1
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

        return BenchmarkResult(
            backend=backend,
            prompt=prompt,
            ttft_ms=ttft,
            total_time_ms=total_time,
            tokens_generated=tokens_generated,
            throughput_tps=throughput,
            decode_latency_ms=decode_latency,
            end_to_end_latency_ms=e2e_latency,
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


def run_benchmark_suite(
    backend: str,
    prompts: List[str],
    warmup_runs: int = 2,
    num_runs: int = 5,
    max_tokens: int = 50,
) -> BenchmarkSummary:
    """Run a complete benchmark suite for a backend"""

    print(f"\n{'='*60}")
    print(f"📊 Benchmarking {backend.upper()} Backend")
    print(f"{'='*60}")

    results = []

    # Warmup runs
    if warmup_runs > 0:
        print(f"\n🔥 Warmup ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            warmup_prompt = "Hello"
            result = run_single_benchmark(warmup_prompt, backend, max_tokens=10)
            status = "✓" if result.success else "✗"
            print(f"   Warmup {i+1}: {status} (tokens={result.tokens_generated})")

    # Actual benchmark runs
    print(
        f"\n📈 Benchmark runs ({num_runs} runs per prompt, max_tokens={max_tokens})..."
    )
    for prompt_idx, prompt in enumerate(prompts):
        print(
            f'\n   Prompt {prompt_idx + 1}: "{prompt[:50]}..."'
            if len(prompt) > 50
            else f'\n   Prompt {prompt_idx + 1}: "{prompt}"'
        )

        for run_idx in range(num_runs):
            result = run_single_benchmark(prompt, backend, max_tokens=max_tokens)
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
        success_rate=len(successful_results) / len(results) * 100,
        results=results,
    )


def print_summary_table(summaries: List[BenchmarkSummary]):
    """Print a comparison table of benchmark results"""

    print("\n" + "=" * 100)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    # Header - more metrics
    print(
        f"\n{'Backend':<15} {'Avg TTFT':<12} {'Avg Decode':<14} {'Avg TPS':<12} {'Avg Tokens':<12} {'Success':<10}"
    )
    print("-" * 100)

    for summary in summaries:
        print(
            f"{summary.backend:<15} "
            f"{summary.avg_ttft_ms:>8.1f} ms  "
            f"{summary.avg_decode_latency_ms:>10.1f} ms/t "
            f"{summary.avg_throughput_tps:>8.2f} t/s "
            f"{summary.avg_tokens_generated:>8.1f}     "
            f"{summary.success_rate:>8.1f}%"
        )

    print("-" * 100)

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

        print("\n📈 COMPARISON (RISC-V vs torch_native):")

        # Determine which is which
        if s1.backend == "rvv":
            riscv, torch = s1, s2
        else:
            riscv, torch = s2, s1

        if torch.avg_ttft_ms > 0:
            ttft_speedup = (
                torch.avg_ttft_ms / riscv.avg_ttft_ms if riscv.avg_ttft_ms > 0 else 0
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
            decode_speedup = torch.avg_decode_latency_ms / riscv.avg_decode_latency_ms
            print(
                f"   Decode Latency Speedup: {decode_speedup:.2f}x {'(RISC-V faster)' if decode_speedup > 1 else '(torch_native faster)'}"
            )


def save_results_json(summaries: List[BenchmarkSummary], output_file: str):
    """Save benchmark results to JSON file"""
    results = []
    for summary in summaries:
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

    print(f"\n📄 Results saved to: {output_file}")


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

    args = parser.parse_args()

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

    print("🚀 SGLang Backend Benchmark")
    print("=" * 60)
    print(f"   Backends: {', '.join(backends_to_test)}")
    print(f"   Warmup runs: {args.warmup}")
    print(f"   Measurement runs: {args.num_runs}")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Prompts: {len(prompts)}")
    print("=" * 60)

    for backend in backends_to_test:
        print(f"\n\n{'#'*60}")
        print(f"# Testing {backend.upper()} Backend")
        print(f"{'#'*60}")

        shutdown_server()
        time.sleep(3)

        # Create config and launch server
        config_path = create_config_file(backend, script_dir)
        log_file = os.path.join(script_dir, f"benchmark_{backend}.log")

        if not launch_server(
            config_path, log_file, timeout=args.timeout, backend=backend
        ):
            print(f"❌ Failed to start server for {backend} backend")
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
        )
        summaries.append(summary)

        # Shutdown after run
        shutdown_server()
        time.sleep(2)

    # Print comparison
    if summaries:
        print_summary_table(summaries)

        if args.output:
            save_results_json(summaries, args.output)
    else:
        print("\n❌ No successful benchmarks completed")

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
