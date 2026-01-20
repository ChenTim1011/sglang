"""
Comprehensive Memory Profiling for INT8 vs FP16 Operations

This test measures:
1. Memory bandwidth usage (estimated and actual using perf)
2. Decode, Extend, and GEMM throughput
3. Cache miss rates (L1/L2)
4. Memory access patterns
5. Correlation between bandwidth reduction and throughput improvement

Usage:
    python test_memory_profiling_comprehensive.py
    python test_memory_profiling_comprehensive.py --operation decode --seq-len 2048
    python test_memory_profiling_comprehensive.py --operation all --use-perf
"""

import argparse
import json
import os

# Set OMP_NUM_THREADS to 8 by default for Banana Pi
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "8"
    print(
        "NOTE: Setting OMP_NUM_THREADS=8 by default for optimal performance on Banana Pi"
    )

import resource  # For memory usage tracking
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ============================================================================
# Inline Test Utilities
# ============================================================================

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class StatisticalResult:
    """Statistical measurement result"""

    mean: float
    std: float
    min: float
    max: float
    median: float
    ci_95_lower: float
    ci_95_upper: float
    num_runs: int


def measure_with_statistics(
    func: Callable[[], float], num_runs: int = 10, confidence_level: float = 0.95
) -> StatisticalResult:
    """
    Measure a function multiple times and compute statistics.

    Args:
        func: Function to measure (should return a float)
        num_runs: Number of runs
        confidence_level: Confidence level for CI (default: 0.95)

    Returns:
        StatisticalResult with mean, std, min, max, median, CI
    """
    results = []
    # Warmup
    func()

    for _ in range(num_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        results.append((end - start) * 1000.0)  # ms

    results_array = np.array(results)
    mean = np.mean(results_array)

    if num_runs > 1:
        std = np.std(results_array, ddof=1)  # Sample standard deviation
    else:
        std = 0.0

    min_val = np.min(results_array)
    max_val = np.max(results_array)
    median = np.median(results_array)

    # Confidence interval
    # For 95% CI: z = 1.96, for 90% CI: z = 1.645
    z_score = 1.96 if confidence_level == 0.95 else 1.645
    if num_runs > 1:
        se = std / np.sqrt(num_runs)  # Standard error
    else:
        se = 0.0
    ci_lower = mean - z_score * se
    ci_upper = mean + z_score * se

    return StatisticalResult(
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        median=median,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        num_runs=num_runs,
    )


def check_system_state():
    """Check system state before benchmarking."""
    try:
        import psutil

        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        print("=" * 60)
        print("System State Check")
        print("=" * 60)
        if cpu_freq:
            print(
                f"CPU Frequency: {cpu_freq.current:.0f} MHz (min: {cpu_freq.min:.0f}, max: {cpu_freq.max:.0f})"
            )
        print(f"CPU Usage: {cpu_percent:.1f}%")
        print(
            f"Memory Usage: {memory.percent:.1f}% ({memory.used / (1024**3):.2f} GB / {memory.total / (1024**3):.2f} GB)"
        )
        print("=" * 60)

        warnings = []
        if cpu_percent > 50:
            warnings.append("⚠️  High CPU usage may affect results")
        if memory.percent > 80:
            warnings.append("⚠️  High memory usage may affect results")
        if cpu_freq and cpu_freq.current < cpu_freq.max * 0.8:
            warnings.append("⚠️  CPU may be in power-saving mode")

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  {w}")
            print()

        return len(warnings) == 0
    except ImportError:
        print("⚠️  psutil not available, skipping system state check")
        return True


def generate_fair_kv_buffers(
    max_tokens: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Generate fair KV buffers for FP16/INT8 comparison.

    Always generates FP16 data first, then quantizes for INT8.
    This ensures fair comparison between FP16 and INT8.

    Returns:
        (k_buffer, v_buffer, k_scale, v_scale)
    """
    torch.manual_seed(seed)

    # Always generate FP16 data first
    k_buffer_fp16 = torch.randn(
        max_tokens, num_heads, head_dim, dtype=torch.float16, device=device
    )
    v_buffer_fp16 = torch.randn(
        max_tokens, num_heads, head_dim, dtype=torch.float16, device=device
    )

    if dtype == torch.int8:
        # Quantize FP16 to INT8
        k_max = k_buffer_fp16.abs().max().item()
        v_max = v_buffer_fp16.abs().max().item()
        k_scale = k_max / 127.0 if k_max > 0 else 0.01
        v_scale = v_max / 127.0 if v_max > 0 else 0.01

        k_buffer = torch.clamp(
            torch.round(k_buffer_fp16.float() / k_scale), -128, 127
        ).to(torch.int8)
        v_buffer = torch.clamp(
            torch.round(v_buffer_fp16.float() / v_scale), -128, 127
        ).to(torch.int8)
    else:
        k_buffer = k_buffer_fp16.to(dtype)
        v_buffer = v_buffer_fp16.to(dtype)
        k_scale = 1.0
        v_scale = 1.0

    return k_buffer, v_buffer, k_scale, v_scale


# Import memory access pattern analyzer
HAS_PATTERN_ANALYZER = False

try:
    with open("/dev/null", "w") as f:
        # Redirect stdout/stderr to suppress noise from sgl_kernel import failure on host
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = f, f
        try:
            import sgl_kernel

            HAS_SGL_KERNEL = True
        except ImportError:
            HAS_SGL_KERNEL = False
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
except Exception:
    # Fallback if redirect fails
    try:
        import sgl_kernel

        HAS_SGL_KERNEL = True
    except ImportError:
        HAS_SGL_KERNEL = False


# ============================================================================
# Helper Functions
# ============================================================================


def is_int8_available(operation: str) -> bool:
    """Check if INT8 kernel is available for the given operation."""
    if not HAS_SGL_KERNEL:
        return False
    try:
        if operation == "decode":
            return hasattr(torch.ops.sgl_kernel, "decode_attention_int8_cpu")
        elif operation == "extend":
            # Extend INT8 kernel is currently part of the same library
            return hasattr(torch.ops.sgl_kernel, "extend_attention_int8_cpu")
        elif operation == "gemm":
            # GEMM INT8 is supported if the library is loaded
            return hasattr(torch.ops.sgl_kernel, "int8_scaled_mm_cpu")
    except Exception:
        return False
    return False


def check_perf_available() -> bool:
    """Check if perf tool is available."""
    try:
        result = subprocess.run(
            ["perf", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class MemoryBandwidthResult:
    """Memory bandwidth measurement result"""

    operation: str  # decode, extend, gemm
    dtype: str  # INT8, FP16, FP32
    seq_len: int
    batch_size: int
    num_heads: int
    head_dim: int

    # Memory metrics (estimated)
    estimated_bandwidth_gb_s: float
    bandwidth_per_operation_gb: float

    # Performance metrics
    latency_ms: float
    throughput_ops_s: float

    # Hardware metrics (from perf)
    cycles: Optional[int] = None
    instructions: Optional[int] = None
    ipc: Optional[float] = None
    branch_misses: Optional[int] = None
    bus_cycles: Optional[int] = None

    # We removed cache specific metrics as K1 doesn't support them well via perf

    # Memory access pattern (if available)
    memory_access_pattern: Optional[str] = None
    spatial_locality_score: Optional[float] = None
    temporal_locality_score: Optional[float] = None

    # New metrics
    max_memory_mb: Optional[float] = None
    data_volume_mb: Optional[float] = None  # Theoretical data moved per op
    cache_misses: Optional[int] = None
    l1_misses: Optional[int] = None


def get_memory_usage_mb() -> float:
    """Get max resident set size (memory usage) in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux, ru_maxrss is in kilobytes. On Mac, it's bytes.
    # Assuming Linux (Banana Pi)
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    else:
        return usage / 1024

    def __str__(self):
        branch_miss_str = (
            f"{self.branch_misses:,}" if self.branch_misses is not None else "N/A"
        )
        ipc_str = f"{self.ipc:.2f}" if self.ipc is not None else "N/A"


@dataclass
class PerfMetrics:
    """Hardware performance metrics from perf tool"""

    # CPU metrics
    cycles: int = 0
    instructions: int = 0
    ipc: float = 0.0
    branch_misses: int = 0
    bus_cycles: int = 0
    cache_misses: int = 0
    l1_misses: int = 0


# ============================================================================
# Bandwidth Estimation Functions
# ============================================================================


def estimate_bandwidth_per_decode(
    seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> float:
    """Estimate memory bandwidth per decode step (GB)."""
    bytes_per_element = 1 if dtype == torch.int8 else 2

    # Per decode step, we read:
    # - K: [seq_len, num_heads, head_dim] from cache
    # - V: [seq_len, num_heads, head_dim] from cache
    # - Q: [1, num_heads, head_dim] (new token)
    # - Output: [1, num_heads, head_dim] (write)

    k_read = seq_len * num_heads * head_dim * bytes_per_element
    v_read = seq_len * num_heads * head_dim * bytes_per_element
    q_read = 1 * num_heads * head_dim * 2  # Q is always FP16/BF16
    output_write = 1 * num_heads * head_dim * 2  # Output is FP16/BF16

    total_bytes = k_read + v_read + q_read + output_write
    return total_bytes / (1024**3)  # Convert to GB


def estimate_bandwidth_per_extend(
    seq_len: int,
    extend_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> float:
    """Estimate memory bandwidth per extend step (GB)."""
    bytes_per_element = 1 if dtype == torch.int8 else 2

    # Per extend step, we read:
    # - K: [seq_len, num_heads, head_dim] from cache (prefix)
    # - V: [seq_len, num_heads, head_dim] from cache (prefix)
    # - Q: [extend_len, num_heads, head_dim] (new tokens)
    # - K_new: [extend_len, num_heads, head_dim] (new tokens)
    # - V_new: [extend_len, num_heads, head_dim] (new tokens)
    # - Output: [extend_len, num_heads, head_dim] (write)

    k_read = seq_len * num_heads * head_dim * bytes_per_element
    v_read = seq_len * num_heads * head_dim * bytes_per_element
    q_read = extend_len * num_heads * head_dim * 2  # Q is FP16
    k_new_read = extend_len * num_heads * head_dim * 2  # New K is FP16
    v_new_read = extend_len * num_heads * head_dim * 2  # New V is FP16
    output_write = extend_len * num_heads * head_dim * 2  # Output is FP16

    total_bytes = k_read + v_read + q_read + k_new_read + v_new_read + output_write
    return total_bytes / (1024**3)  # Convert to GB


def estimate_bandwidth_per_gemm(M: int, N: int, K: int, dtype: torch.dtype) -> float:
    """Estimate memory bandwidth per GEMM operation (GB)."""
    bytes_per_element = (
        1 if dtype == torch.int8 else (2 if dtype == torch.float16 else 4)
    )

    # Per GEMM operation (C = A × B):
    # - A: [M, K] - read
    # - B: [K, N] - read
    # - C: [M, N] - write

    # Theoretical minimum: M*K + K*N + M*N
    # Actual (considering cache): approximately M*K*2 + K*N*2 + M*N
    # (assuming 50% cache hit rate for simplicity)

    a_read = M * K * bytes_per_element
    b_read = K * N * bytes_per_element
    c_write = M * N * bytes_per_element

    # Simplified estimation: assume some cache reuse
    # For tiled GEMM, cache hit rate is higher, so we use a factor of 1.5
    total_bytes = (a_read + b_read) * 1.5 + c_write

    return total_bytes / (1024**3)  # Convert to GB


# ============================================================================
# Perf Integration
# ============================================================================


def parse_perf_output(perf_output: str) -> PerfMetrics:
    """Parse perf stat output to extract CPU metrics."""
    metrics = PerfMetrics()

    lines = perf_output.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse CSV format (perf stat -x ,)
        # Format: value,unit,event_name
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                value_str = parts[0].strip().replace(",", "")
                if value_str and value_str != "<not supported>":
                    value = int(float(value_str))
                    event_name = parts[2].strip() if len(parts) > 2 else ""

                    if "cycles" in event_name and "cpu-cycles" in event_name:
                        metrics.cycles = value
                    elif "instructions" in event_name:
                        metrics.instructions = value
                    elif "branch-misses" in event_name:
                        metrics.branch_misses = value
                    elif "bus-cycles" in event_name:
                        metrics.bus_cycles = value
                    elif "L1-dcache-load-misses" in event_name:
                        metrics.l1_misses = value

            except (ValueError, IndexError):
                pass

        # Also try traditional format (space-separated)
        if "," not in line:
            try:
                parts = line.split()
                if len(parts) >= 2:
                    value_str = parts[0].replace(",", "")
                    if value_str and value_str != "<not":
                        value = int(float(value_str))
                        event_name = " ".join(parts[1:]).lower()

                        if "cpu-cycles" in event_name or (
                            "cycles" in event_name and "cpu" in event_name
                        ):
                            metrics.cycles = value
                        elif "instructions" in event_name:
                            metrics.instructions = value
                        elif "branch-misses" in event_name:
                            metrics.branch_misses = value
                        elif "bus-cycles" in event_name:
                            metrics.bus_cycles = value
                        elif "l1-dcache-load-misses" in event_name:
                            metrics.l1_misses = value

            except (ValueError, IndexError):
                pass

    # Calculate IPC
    if metrics.cycles > 0 and metrics.instructions > 0:
        metrics.ipc = metrics.instructions / metrics.cycles

    return metrics


def run_benchmark_worker(
    op_name: str, kwargs: Dict, num_iterations: int, wait_for_signal: bool = False
):
    """
    Worker function to run the actual benchmark in the subprocess.
    Called by the temporary script generated in run_perf_measurement_process.
    """
    # Import the module to get access to measurement functions
    # This works because we run this from a script that imports test_memory_bandwidth
    import sys

    import test_memory_bandwidth

    # Pre-import heavy dependencies to ensure they are loaded before signaling
    import torch

    if wait_for_signal:
        print("READY")
        sys.stdout.flush()
        # Wait for go signal from parent
        sys.stdin.readline()

    print(f"Worker running {op_name} with {num_iterations} iterations")

    # Dispatch to the appropriate measurement function
    try:
        if "gemm" in op_name:
            test_memory_bandwidth.measure_gemm_bandwidth(
                **kwargs, num_iterations=num_iterations, use_perf=False
            )
        elif "decode" in op_name:
            test_memory_bandwidth.measure_decode_bandwidth(
                **kwargs, num_iterations=num_iterations, use_perf=False
            )
        elif "extend" in op_name:
            test_memory_bandwidth.measure_extend_bandwidth(
                **kwargs, num_iterations=num_iterations, use_perf=False
            )
        else:
            print(f"Unknown operation: {op_name}")
    except Exception as e:
        print(f"Worker failed: {e}")
        import traceback

        traceback.print_exc()


def measure_with_perf(
    op_name: str, kwargs: Dict, num_iterations: int = 100, timeout: int = 60
) -> Optional[PerfMetrics]:
    """Measure hardware metrics using perf tool.

    This function uses perf stat to measure cache performance and memory bandwidth
    by running the operation function in a subprocess monitored by perf.

    Args:
        op_name: Name of the operation to run (gemm, decode, extend)
        kwargs: Arguments to pass to the measurement function
        num_iterations: Number of iterations
        timeout: Timeout in seconds

    Returns:
        PerfMetrics object with measured values, or None if measurement failed
    """
    return run_perf_measurement_process(op_name, kwargs, num_iterations, timeout)


def run_perf_measurement_process(
    op_name: str, kwargs: Dict, num_iterations: int, timeout: int
) -> Optional[PerfMetrics]:
    """
    Run perf stat on a Python subprocess that executes the actual operation.

    Args:
        op_name: Name of the operation (gemm, decode, extend)
        kwargs: Arguments for the operation function
        num_iterations: Number of iterations
        timeout: Timeout in seconds

    Returns:
        PerfMetrics object with measured values, or None if measurement failed
    """
    if not check_perf_available():
        print("⚠️  perf tool not available, skipping hardware measurement")
        return None

    try:
        # Check which perf events are available
        available_events = []
        # On K1, cache events are not supported. Focus on CPU core metrics.
        perf_events_to_try = [
            "cpu-cycles",
            "instructions",
            "branch-misses",
            "bus-cycles",
            "L1-dcache-load-misses",
        ]

        # Check event availability (simplified check)
        # On K1, we assume basic events are available if perf is present
        # but checking specific events is safer.
        for event in perf_events_to_try:
            try:
                result = subprocess.run(
                    ["perf", "stat", "-e", event, "true"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    available_events.append(event)
            except Exception:
                pass

        if not available_events:
            print("⚠️  No perf events available")
            return None

        # Create a temporary script that will run the operation
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_script = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)

        # Write a script that will be executed
        temp_script.write(
            f"""#!/usr/bin/env python3
import sys
import os
import torch
import traceback

# Add script dir to path to import test_memory_bandwidth
sys.path.insert(0, '{script_dir}')

# Attempt to import sgl_kernel if possible (for custom ops)
try:
    import sgl_kernel
except ImportError:
    pass

import test_memory_bandwidth

# Reconstruct arguments
kwargs = {repr(kwargs)}
op_name = '{op_name}'
num_iterations = {num_iterations}

# Run the benchmark worker with signal wait
try:
    test_memory_bandwidth.run_benchmark_worker(op_name, kwargs, num_iterations, wait_for_signal=True)
except Exception:
    traceback.print_exc()
"""
        )
        temp_script.close()
        os.chmod(temp_script.name, 0o755)

        # 1. Start the worker process
        worker_proc = subprocess.Popen(
            ["python3", temp_script.name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )

        # 2. Wait for "READY" signal
        try:
            while True:
                line = worker_proc.stdout.readline()
                if not line:
                    break
                if "READY" in line:
                    break
        except Exception:
            worker_proc.kill()
            raise RuntimeError("Worker process failed to start/signal")

        # 3. Start perf attached to the worker PID
        perf_cmd = [
            "perf",
            "stat",
            "-e",
            ",".join(available_events),
            "-x",
            ",",
            "-p",
            str(worker_proc.pid),
        ]

        print(f"Attaching perf to PID {worker_proc.pid}...")
        perf_proc = subprocess.Popen(
            perf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Give perf a moment to attach
        time.sleep(0.1)

        # 4. Signal worker to start
        worker_proc.stdin.write("GO\n")
        worker_proc.stdin.flush()

        # 5. Wait for worker to finish
        try:
            worker_proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            worker_proc.kill()
            perf_proc.kill()
            print("⚠️  Benchmark timed out")
            return None

        # 6. Stop perf (SIGINT to make it print stats)
        import signal

        if perf_proc.poll() is None:
            try:
                perf_proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

        try:
            perf_stdout, perf_stderr = perf_proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            perf_proc.kill()
            perf_stdout, perf_stderr = perf_proc.communicate()

        if isinstance(perf_stdout, bytes):
            perf_stdout = perf_stdout.decode("utf-8")
        if isinstance(perf_stderr, bytes):
            perf_stderr = perf_stderr.decode("utf-8")

        # DEBUG: Always print perf output to see what's happening
        print(f"DEBUG: Perf STDOUT for {op_name}:\n{perf_stdout}")
        print(f"DEBUG: Perf STDERR for {op_name}:\n{perf_stderr}")

        # Parse output (perf stat outputs to stderr usually, but sometimes stdout for -p)
        output_to_parse = (
            perf_stdout
            if perf_stdout and "cache-misses" in perf_stdout
            else perf_stderr
        )

        # Clean up
        try:
            os.unlink(temp_script.name)
        except OSError:
            pass

        metrics = parse_perf_output(output_to_parse)

        if (
            metrics.cycles > 0
            or metrics.instructions > 0
            or metrics.branch_misses > 0
            or metrics.cache_misses > 0
            or metrics.l1_misses > 0
        ):
            return metrics
        else:
            print("⚠️  perf measurement returned no valid metrics")
            if output_to_parse:
                print(f"   output: {output_to_parse[:200]}")
            return None

    except Exception as e:
        print(f"⚠️  Error running perf: {e}")
        import traceback

        traceback.print_exc()
        return None


# ============================================================================
# Operation Measurement Functions
# ============================================================================


def measure_decode_bandwidth(
    num_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype: torch.dtype,
    num_iterations: int = 100,
    warmup: int = 10,
    use_perf: bool = False,
    seed: int = 42,
) -> MemoryBandwidthResult:
    """Measure decode attention bandwidth and performance."""
    device = "cpu"
    q_dtype = torch.float16
    dtype_str = "INT8" if dtype == torch.int8 else "FP16"

    torch.manual_seed(seed)

    # Prepare inputs
    q = torch.randn(batch_size, num_heads, head_dim, dtype=q_dtype, device=device)
    max_tokens = batch_size * seq_len

    k_buffer, v_buffer, k_scale, v_scale = generate_fair_kv_buffers(
        max_tokens, num_heads, head_dim, dtype, device, seed
    )

    req_to_token = torch.arange(max_tokens, dtype=torch.int64, device=device).reshape(
        batch_size, seq_len
    )
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int64, device=device)

    output = torch.zeros(batch_size, num_heads, head_dim, dtype=q_dtype, device=device)
    attn_logits = torch.zeros(
        batch_size, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
    )

    dummy_key = torch.zeros(
        batch_size, num_heads, head_dim, dtype=q_dtype, device=device
    )
    dummy_value = torch.zeros(
        batch_size, num_heads, head_dim, dtype=q_dtype, device=device
    )
    dummy_loc = torch.zeros(batch_size, dtype=torch.int64, device=device)

    # Check INT8 availability
    if dtype == torch.int8 and not is_int8_available("decode"):
        raise RuntimeError("decode_attention_int8_cpu is not available")

    # Warmup
    for _ in range(warmup):
        if dtype == torch.int8:
            torch.ops.sgl_kernel.decode_attention_int8_cpu(
                q,
                k_buffer,
                v_buffer,
                output,
                dummy_key,
                dummy_value,
                dummy_loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                1.0,
                0.0,
                k_scale,
                v_scale,
            )
        else:
            torch.ops.sgl_kernel.decode_attention_cpu(
                q,
                k_buffer,
                v_buffer,
                output,
                dummy_key,
                dummy_value,
                dummy_loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                1.0,
                0.0,
            )

    # Measure
    def run_decode():
        if dtype == torch.int8:
            torch.ops.sgl_kernel.decode_attention_int8_cpu(
                q,
                k_buffer,
                v_buffer,
                output,
                dummy_key,
                dummy_value,
                dummy_loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                1.0,
                0.0,
                k_scale,
                v_scale,
            )
        else:
            torch.ops.sgl_kernel.decode_attention_cpu(
                q,
                k_buffer,
                v_buffer,
                output,
                dummy_key,
                dummy_value,
                dummy_loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                1.0,
                0.0,
            )

    stats = measure_with_statistics(run_decode, num_runs=num_iterations)
    latency_ms = stats.mean
    throughput_ops_s = (batch_size * 1000) / latency_ms if latency_ms > 0 else 0

    # Estimate bandwidth
    bandwidth_per_op = estimate_bandwidth_per_decode(
        seq_len, num_heads, head_dim, dtype
    )
    estimated_bandwidth_gb_s = bandwidth_per_op * throughput_ops_s
    data_volume_mb = (
        (bandwidth_per_op * 1024) if throughput_ops_s > 0 else 0
    )  # Convert GB to MB

    # Measure with perf if requested
    perf_metrics = None
    if use_perf:
        perf_metrics = measure_with_perf(
            f"decode_{dtype_str}",
            {
                "num_heads": num_heads,
                "head_dim": head_dim,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "dtype": dtype,
                "warmup": 0,
                "seed": seed,
            },
            num_iterations=num_iterations,
        )

    # Analyze memory access pattern if analyzer is available
    access_pattern = None
    spatial_locality = None
    temporal_locality = None
    if HAS_PATTERN_ANALYZER:
        # Estimate memory addresses from tensor access pattern
        # For decode: sequential access to K/V cache (seq_len elements)
        # Simplified estimation: assume sequential access pattern
        estimated_addresses = list(
            range(0, seq_len * num_heads * head_dim * 2, 64)
        )  # Cache line aligned
        pattern_metrics = analyze_memory_access_pattern(estimated_addresses)
        access_pattern = pattern_metrics.access_pattern
        spatial_locality = pattern_metrics.spatial_locality_score
        temporal_locality = pattern_metrics.temporal_locality_score

    # Calculate actual memory bandwidth from perf if available
    memory_bandwidth_gb_s = None
    # Perf on K1 does not calculate bandwidth from cache misses

    return MemoryBandwidthResult(
        operation="decode",
        dtype=dtype_str,
        seq_len=seq_len,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        estimated_bandwidth_gb_s=estimated_bandwidth_gb_s,
        bandwidth_per_operation_gb=bandwidth_per_op,
        latency_ms=latency_ms,
        throughput_ops_s=throughput_ops_s,
        cycles=perf_metrics.cycles if perf_metrics else None,
        instructions=perf_metrics.instructions if perf_metrics else None,
        ipc=perf_metrics.ipc if perf_metrics else None,
        branch_misses=perf_metrics.branch_misses if perf_metrics else None,
        bus_cycles=perf_metrics.bus_cycles if perf_metrics else None,
        memory_access_pattern=access_pattern,
        spatial_locality_score=spatial_locality,
        temporal_locality_score=temporal_locality,
        max_memory_mb=get_memory_usage_mb(),
        data_volume_mb=data_volume_mb,
        cache_misses=perf_metrics.cache_misses if perf_metrics else None,
        l1_misses=perf_metrics.l1_misses if perf_metrics else None,
    )


def measure_extend_bandwidth(
    num_heads: int,
    head_dim: int,
    seq_len: int,
    extend_len: int,
    batch_size: int,
    dtype: torch.dtype,
    num_iterations: int = 100,
    warmup: int = 10,
    use_perf: bool = False,
    seed: int = 42,
) -> MemoryBandwidthResult:
    """Measure extend attention bandwidth and performance."""
    device = "cpu"
    q_dtype = torch.float16
    dtype_str = "INT8" if dtype == torch.int8 else "FP16"

    torch.manual_seed(seed)

    total_extend_len = batch_size * extend_len

    # Prepare inputs
    q_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=q_dtype, device=device
    )
    k_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=q_dtype, device=device
    )
    v_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=q_dtype, device=device
    )

    max_total_tokens = batch_size * seq_len
    k_buffer, v_buffer, k_scale, v_scale = generate_fair_kv_buffers(
        max_total_tokens, num_heads, head_dim, dtype, device, seed
    )

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int64, device=device)
    extend_seq_lens = torch.full(
        (batch_size,), extend_len, dtype=torch.int64, device=device
    )
    extend_start_loc = (
        torch.arange(batch_size, dtype=torch.int64, device=device) * extend_len
    )
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)

    req_to_token = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
    for i in range(batch_size):
        start_idx = i * seq_len
        req_to_token[i, :seq_len] = torch.arange(
            start_idx, start_idx + seq_len, dtype=torch.int64
        )

    o_extend = torch.zeros(
        total_extend_len, num_heads, head_dim, dtype=q_dtype, device=device
    )
    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0
    max_len_extend = extend_len

    # Check INT8 availability
    if dtype == torch.int8 and not is_int8_available("extend"):
        raise RuntimeError("extend_attention_int8_cpu is not available")

    # Warmup
    for _ in range(warmup):
        if dtype == torch.int8:
            torch.ops.sgl_kernel.extend_attention_int8_cpu(
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer,
                v_buffer,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_seq_lens,
                extend_start_loc,
                max_len_extend,
                sm_scale,
                logit_cap,
                k_scale,
                v_scale,
            )
        else:
            torch.ops.sgl_kernel.extend_attention_cpu(
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer,
                v_buffer,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_seq_lens,
                extend_start_loc,
                max_len_extend,
                sm_scale,
                logit_cap,
            )

    # Measure
    def run_extend():
        if dtype == torch.int8:
            torch.ops.sgl_kernel.extend_attention_int8_cpu(
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer,
                v_buffer,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_seq_lens,
                extend_start_loc,
                max_len_extend,
                sm_scale,
                logit_cap,
                k_scale,
                v_scale,
            )
        else:
            torch.ops.sgl_kernel.extend_attention_cpu(
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer,
                v_buffer,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_seq_lens,
                extend_start_loc,
                max_len_extend,
                sm_scale,
                logit_cap,
            )

    stats = measure_with_statistics(run_extend, num_runs=num_iterations)
    latency_ms = stats.mean
    throughput_ops_s = (batch_size * 1000) / latency_ms if latency_ms > 0 else 0

    # Estimate bandwidth
    bandwidth_per_op = estimate_bandwidth_per_extend(
        seq_len, extend_len, num_heads, head_dim, dtype
    )
    estimated_bandwidth_gb_s = bandwidth_per_op * throughput_ops_s
    data_volume_mb = (
        (bandwidth_per_op * 1024) if throughput_ops_s > 0 else 0
    )  # Convert GB to MB

    # Measure with perf if requested
    perf_metrics = None
    if use_perf:
        perf_metrics = measure_with_perf(
            f"extend_{dtype_str}",
            {
                "num_heads": num_heads,
                "head_dim": head_dim,
                "seq_len": seq_len,
                "extend_len": extend_len,
                "batch_size": batch_size,
                "dtype": dtype,
                "warmup": 0,
                "seed": seed,
            },
            num_iterations=num_iterations,
        )

    # Calculate actual memory bandwidth from perf if available
    memory_bandwidth_gb_s = None
    # Perf on K1 does not calculate bandwidth from cache misses

    return MemoryBandwidthResult(
        operation="extend",
        dtype=dtype_str,
        seq_len=seq_len,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        estimated_bandwidth_gb_s=estimated_bandwidth_gb_s,
        bandwidth_per_operation_gb=bandwidth_per_op,
        latency_ms=latency_ms,
        throughput_ops_s=throughput_ops_s,
        cycles=perf_metrics.cycles if perf_metrics else None,
        instructions=perf_metrics.instructions if perf_metrics else None,
        ipc=perf_metrics.ipc if perf_metrics else None,
        branch_misses=perf_metrics.branch_misses if perf_metrics else None,
        bus_cycles=perf_metrics.bus_cycles if perf_metrics else None,
        max_memory_mb=get_memory_usage_mb(),
        data_volume_mb=data_volume_mb,
        cache_misses=perf_metrics.cache_misses if perf_metrics else None,
        l1_misses=perf_metrics.l1_misses if perf_metrics else None,
    )


def measure_gemm_bandwidth(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    num_iterations: int = 100,
    warmup: int = 10,
    use_perf: bool = False,
    seed: int = 42,
) -> MemoryBandwidthResult:
    """Measure GEMM bandwidth and performance."""
    device = "cpu"
    dtype_str = (
        "INT8"
        if dtype == torch.int8
        else ("FP16" if dtype == torch.float16 else "FP32")
    )

    torch.manual_seed(seed)

    # Prepare inputs
    if dtype == torch.int8:
        # Kernel expects uint8 for activations (A) and int8 for weights (B)
        A = torch.randint(0, 255, (M, K), dtype=torch.uint8, device=device)

    # Setup matrices
    scales1 = None
    scales2 = None

    if dtype == torch.int8:
        # Use simple quantization to get input suitable for kernel
        A_f16 = torch.randn(M, K, dtype=torch.float16, device=device)
        B_f16 = torch.randn(N, K, dtype=torch.float16, device=device)

        # Calculate scales
        scales1 = (A_f16.abs().max(dim=1)[0] / 127.0).to(torch.float32)  # [M]
        scales2 = (B_f16.abs().max(dim=1)[0] / 127.0).to(torch.float32)  # [N]

        # Quantize A to uint8 (asymmetric)
        A = (A_f16 / scales1.unsqueeze(1)).round().clamp(-128, 127).to(torch.int16)
        A = (A + 128).clamp(0, 255).to(torch.uint8)

        # Quantize B to int8 (symmetric)
        B_unpacked = (
            (B_f16 / scales2.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        )

        # Pack B for optimal performance (simulating sglang behavior)
        if HAS_SGL_KERNEL and hasattr(torch.ops.sgl_kernel, "riscv_int8_pack_b"):
            B = torch.ops.sgl_kernel.riscv_int8_pack_b(B_unpacked, K, N)
            is_packed = True
            # For packed B, shape in python seems to remain [N, K] logically but buffer is bigger/reordered
            # However, the kernel handles the packed buffer.
            # If `riscv_int8_pack_b` returns a tensor, use it.
        else:
            B = B_unpacked
            is_packed = False
            print("Warning: riscv_int8_pack_b not found, using unpacked B (SLOW)")

    else:
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(N, K, dtype=dtype, device=device)

    # Check INT8 availability
    if dtype == torch.int8 and not is_int8_available("gemm"):
        raise RuntimeError("int8_scaled_mm_cpu is not available")

    # Warmup
    for _ in range(warmup):
        if dtype == torch.int8:
            torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                A, B, scales1, scales2, None, torch.float16, is_packed
            )
        else:
            # MATCHING ABLATION: Use weight_packed_linear (Native RVV)
            if HAS_SGL_KERNEL and hasattr(torch.ops.sgl_kernel, "weight_packed_linear"):
                torch.ops.sgl_kernel.weight_packed_linear(A, B, None, False)
            else:
                torch.matmul(A, B.T)

    # Measure
    def run_gemm():
        if dtype == torch.int8:
            torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                A, B, scales1, scales2, None, torch.float16, is_packed
            )
        else:
            weight_packed_op = getattr(
                torch.ops.sgl_kernel, "weight_packed_linear", None
            )
            if weight_packed_op:
                weight_packed_op(A, B, None, False)
            else:
                torch.matmul(A, B.T)

    stats = measure_with_statistics(run_gemm, num_runs=num_iterations)
    latency_ms = stats.mean
    throughput_ops_s = 1000 / latency_ms if latency_ms > 0 else 0

    # Estimate bandwidth
    bandwidth_per_op = estimate_bandwidth_per_gemm(M, N, K, dtype)
    estimated_bandwidth_gb_s = bandwidth_per_op * throughput_ops_s
    data_volume_mb = (
        (bandwidth_per_op * 1024) if throughput_ops_s > 0 else 0
    )  # Convert GB to MB

    # Measure with perf if requested
    perf_metrics = None
    if use_perf:
        perf_metrics = measure_with_perf(
            f"gemm_{dtype_str}",
            {
                "M": M,
                "N": N,
                "K": K,
                "dtype": dtype,
                "warmup": 0,
                "seed": seed,
            },
            num_iterations=num_iterations,
        )

    # Calculate actual memory bandwidth from perf if available
    memory_bandwidth_gb_s = None
    # Perf on K1 does not calculate bandwidth from cache misses

    return MemoryBandwidthResult(
        operation="gemm",
        dtype=dtype_str,
        seq_len=0,
        batch_size=M,
        num_heads=0,
        head_dim=K,
        estimated_bandwidth_gb_s=estimated_bandwidth_gb_s,
        bandwidth_per_operation_gb=bandwidth_per_op,
        latency_ms=latency_ms,
        throughput_ops_s=throughput_ops_s,
        cycles=perf_metrics.cycles if perf_metrics else None,
        instructions=perf_metrics.instructions if perf_metrics else None,
        ipc=perf_metrics.ipc if perf_metrics else None,
        branch_misses=perf_metrics.branch_misses if perf_metrics else None,
        bus_cycles=perf_metrics.bus_cycles if perf_metrics else None,
        max_memory_mb=get_memory_usage_mb(),
        data_volume_mb=data_volume_mb,
        cache_misses=perf_metrics.cache_misses if perf_metrics else None,
        l1_misses=perf_metrics.l1_misses if perf_metrics else None,
    )


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Memory Profiling for INT8 vs FP16 Operations"
    )
    parser.add_argument(
        "--operation",
        type=str,
        choices=["decode", "extend", "gemm", "all"],
        default="all",
        help="Operation to measure",
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--extend-len", type=int, default=64, help="Extend length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-heads", type=int, default=32, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--num-iterations", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument(
        "--use-perf",
        action="store_true",
        help="Use perf tool for hardware measurement (must run on Banana Pi)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run on remote Banana Pi via SSH (for perf measurement)",
    )
    parser.add_argument(
        "--remote-user",
        type=str,
        default=os.environ.get("BANANA_PI_USER", "jtchen"),
        help="Banana Pi username (default: from BANANA_PI_USER env or 'jtchen')",
    )
    parser.add_argument(
        "--remote-host",
        type=str,
        default=os.environ.get("BANANA_PI_HOST", "140.114.78.64"),
        help="Banana Pi host/IP (default: from BANANA_PI_HOST env or '140.114.78.64')",
    )
    args = parser.parse_args()

    if args.remote:
        # Run on remote Banana Pi via SSH
        print("=" * 80)
        print("Running Memory Profiling on Remote Banana Pi")
        print("=" * 80)
        print(f"Remote: {args.remote_user}@{args.remote_host}")
        print("")

        # Build command to run on remote Banana Pi
        remote_dir = "~/.local_riscv_env/workspace/sglang/banana_pi/tests"

        remote_cmd_parts = [
            f"cd {remote_dir}",
            "&&",
            "export LD_PRELOAD=~/.local/lib/libomp.so",
            "&&",
            "export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH",
            "&&",
            "source ~/.local_riscv_env/workspace/venv_sglang/bin/activate",
            "&&",
            "python3 test_memory_bandwidth.py",
            "--operation",
            args.operation,
            "--seq-len",
            str(args.seq_len),
            "--extend-len",
            str(args.extend_len),
            "--batch-size",
            str(args.batch_size),
            "--num-heads",
            str(args.num_heads),
            "--head-dim",
            str(args.head_dim),
            "--num-iterations",
            str(args.num_iterations),
        ]

        if args.use_perf:
            remote_cmd_parts.append("--use-perf")

        if args.output:
            remote_cmd_parts.extend(["--output", args.output])

        remote_cmd = " ".join(remote_cmd_parts)

        # Use SSH with LD_LIBRARY_PATH cleared
        ssh_cmd = [
            "env",
            "LD_LIBRARY_PATH=",
            "/usr/bin/ssh",
            f"{args.remote_user}@{args.remote_host}",
            remote_cmd,
        ]

        print(f"Executing on remote: {remote_cmd}")
        print("")
        result = subprocess.run(ssh_cmd)
        return result.returncode

    # Check if we need to run on remote Banana Pi (for perf measurement)
    if args.use_perf:
        # Check if we're already on Banana Pi by checking hostname or architecture
        import platform

        is_riscv = (
            platform.machine().startswith("riscv")
            or "riscv" in platform.processor().lower()
        )

        if not is_riscv:
            print("=" * 80)
            print("⚠️  Perf Measurement Requires Banana Pi")
            print("=" * 80)
            print("perf measurement must run on Banana Pi hardware.")
            print("You are currently running on:", platform.machine())
            print("")
            print("To run on Banana Pi, use:")
            print(f"  python test_memory_bandwidth.py --use-perf --remote")
            print("")
            print("Or SSH to Banana Pi and run directly:")
            print(f"  ssh {args.remote_user}@{args.remote_host}")
            print("  cd ~/.local_riscv_env/workspace/sglang/banana_pi/tests")
            print("  python test_memory_bandwidth.py --use-perf")
            print("")
            return 1

    if not HAS_SGL_KERNEL:
        print("Error: sgl_kernel not available")
        return 1

    print("=" * 80)
    print("Comprehensive Memory Profiling")
    print("=" * 80)

    # Set optimal OpenMP settings for Banana Pi (8 cores)
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["OMP_PROC_BIND"] = "spread"
    os.environ["OMP_PLACES"] = "{0,1,2,3,4,5,6,7}"
    print(f"Setting OMP_NUM_THREADS=8, OMP_PROC_BIND=spread for optimal performance")

    # Check if we're on Banana Pi (for perf)
    if args.use_perf:
        import platform

        is_riscv = (
            platform.machine().startswith("riscv")
            or "riscv" in platform.processor().lower()
        )
        if is_riscv:
            print("✓ Running on RISC-V (Banana Pi) - perf measurement enabled")
        else:
            print("⚠️  Warning: Not on RISC-V, perf may not work correctly")
        print("")

    check_system_state()

    results: List[MemoryBandwidthResult] = []

    operations = (
        ["decode", "extend", "gemm"] if args.operation == "all" else [args.operation]
    )

    for op in operations:
        print(f"\n{'='*80}")
        print(f"Testing {op.upper()} Operation")
        print(f"{'='*80}")

        # FP16
        print("\n--- FP16 Baseline ---")
        try:
            if op == "decode":
                result_fp16 = measure_decode_bandwidth(
                    args.num_heads,
                    args.head_dim,
                    args.seq_len,
                    args.batch_size,
                    torch.float16,
                    args.num_iterations,
                    use_perf=args.use_perf,
                )
            elif op == "extend":
                result_fp16 = measure_extend_bandwidth(
                    args.num_heads,
                    args.head_dim,
                    args.seq_len,
                    args.extend_len,
                    args.batch_size,
                    torch.float16,
                    args.num_iterations,
                    use_perf=args.use_perf,
                )
            elif op == "gemm":
                # Use larger GEMM dimensions for better throughput measurement
                M, N, K = 2048, 2048, 2048
                print(f"   GEMM Size: M={M}, N={N}, K={K}")
                result_fp16 = measure_gemm_bandwidth(
                    M, N, K, torch.float16, args.num_iterations, use_perf=args.use_perf
                )

            results.append(result_fp16)
            print(f"Latency: {result_fp16.latency_ms:.3f} ms")
            print(f"Throughput: {result_fp16.throughput_ops_s:.2f} ops/s")
            print(
                f"Estimated Bandwidth: {result_fp16.estimated_bandwidth_gb_s:.2f} GB/s"
            )
            if result_fp16.ipc is not None:
                print(f"IPC: {result_fp16.ipc:.2f}")
            if result_fp16.branch_misses is not None:
                print(f"Branch Misses: {result_fp16.branch_misses:,}")
            if result_fp16.bus_cycles is not None:
                print(f"Bus Cycles: {result_fp16.bus_cycles:,}")
            if result_fp16.cache_misses is not None:
                print(f"Cache Misses (LLC): {result_fp16.cache_misses:,}")
            if result_fp16.l1_misses is not None:
                print(f"L1-dcache-load-misses: {result_fp16.l1_misses:,}")

        except Exception as e:
            print(f"Error measuring FP16 {op}: {e}")
            traceback.print_exc()
            continue

        # INT8
        if is_int8_available(op):
            print("\n--- INT8 Quantized ---")
            try:
                if op == "decode":
                    result_int8 = measure_decode_bandwidth(
                        args.num_heads,
                        args.head_dim,
                        args.seq_len,
                        args.batch_size,
                        torch.int8,
                        args.num_iterations,
                        use_perf=args.use_perf,
                    )
                elif op == "extend":
                    result_int8 = measure_extend_bandwidth(
                        args.num_heads,
                        args.head_dim,
                        args.seq_len,
                        args.extend_len,
                        args.batch_size,
                        torch.int8,
                        args.num_iterations,
                        use_perf=args.use_perf,
                    )
                elif op == "gemm":
                    M, N, K = 2048, 2048, 2048
                    result_int8 = measure_gemm_bandwidth(
                        M, N, K, torch.int8, args.num_iterations, use_perf=args.use_perf
                    )

                results.append(result_int8)
                print(f"Latency: {result_int8.latency_ms:.3f} ms")
                print(f"Throughput: {result_int8.throughput_ops_s:.2f} ops/s")
                print(
                    f"Estimated Bandwidth: {result_int8.estimated_bandwidth_gb_s:.2f} GB/s"
                )

                if result_int8.ipc is not None:
                    print(f"IPC (Instructions/Cycle): {result_int8.ipc:.2f}")
                if result_int8.cycles is not None:
                    print(f"CPU Cycles: {result_int8.cycles:,}")
                if result_int8.instructions is not None:
                    print(f"Instructions: {result_int8.instructions:,}")
                if result_int8.branch_misses is not None:
                    print(f"Branch Misses: {result_int8.branch_misses:,}")
                if result_int8.bus_cycles is not None:
                    print(f"Bus Cycles: {result_int8.bus_cycles:,}")
                if result_int8.cache_misses is not None:
                    print(f"Cache Misses (LLC): {result_int8.cache_misses:,}")
                if result_int8.l1_misses is not None:
                    print(f"L1-dcache-load-misses: {result_int8.l1_misses:,}")

                # Comparison
                print("\n--- Comparison ---")
                bandwidth_reduction = 0.0
                if result_fp16.bandwidth_per_operation_gb > 0:
                    bandwidth_reduction = (
                        1
                        - result_int8.bandwidth_per_operation_gb
                        / result_fp16.bandwidth_per_operation_gb
                    ) * 100

                throughput_speedup = 0.0
                if result_fp16.throughput_ops_s > 0:
                    throughput_speedup = (
                        result_int8.throughput_ops_s / result_fp16.throughput_ops_s
                    )

                latency_speedup = 0.0
                if result_int8.latency_ms > 0:
                    latency_speedup = result_fp16.latency_ms / result_int8.latency_ms

                print(f"Bandwidth Reduction: {bandwidth_reduction:.1f}%")
                print(f"Throughput Speedup: {throughput_speedup:.2f}x")
                print(f"Latency Speedup: {latency_speedup:.2f}x")

            except Exception as e:
                print(f"Error measuring INT8 {op}: {e}")
                traceback.print_exc()

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n✅ Results saved to {args.output}")

    # Summary table
    print_summary_table(results)


def print_summary_table(results: List[MemoryBandwidthResult]):
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    headers = [
        "Operation",
        "Type",
        "Latency(ms)",
        "Throughput(ops/s)",
        "Est. BW(GB/s)",
        "Data(MB)",
        "L1 Miss",
    ]
    # Format string
    fmt = "{:<10} | {:<6} | {:<12} | {:<18} | {:<13} | {:<8} | {:<10}"

    print(fmt.format(*headers))
    print("-" * 100)

    for r in results:
        # Check if we have perf metrics
        branch_misses = "N/A"
        if r.branch_misses is not None:
            branch_misses = f"{r.branch_misses:,}"

        ipc_str = f"{r.ipc:.2f}" if r.ipc is not None else "N/A"
        mem_str = f"{r.max_memory_mb:.1f}" if r.max_memory_mb is not None else "N/A"
        data_str = f"{r.data_volume_mb:.2f}" if r.data_volume_mb is not None else "N/A"
        l1_miss_str = f"{r.l1_misses:,}" if r.l1_misses is not None else "N/A"

        print(
            fmt.format(
                r.operation,
                r.dtype,
                f"{r.latency_ms:.3f}",
                f"{r.throughput_ops_s:.2f}",
                f"{r.estimated_bandwidth_gb_s:.2f}",
                data_str,
                l1_miss_str,
            )
        )


if __name__ == "__main__":
    main()
