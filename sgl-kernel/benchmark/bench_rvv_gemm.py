"""
Benchmark for RVV GEMM kernel (gemm_rvv.cpp).

This script benchmarks the RVV optimized GEMM implementation
against PyTorch's native implementation for Linear layer operations.

Usage:
    # Run full benchmark
    python benchmark/bench_rvv_gemm.py

    # Quick benchmark (CI mode)
    CI=true python benchmark/bench_rvv_gemm.py

    # On RISC-V hardware (e.g., Banana Pi):
    cd ~/.local_riscv_env/workspace/sglang/sgl-kernel
    python benchmark/bench_rvv_gemm.py
"""

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

# ============================================================================
# Configuration & Environment Setup
# ============================================================================

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def setup_triton_stub():
    """
    Setup triton_stub for RISC-V environments where full Triton is not available.
    """
    try:
        import triton

        return
    except ImportError:
        pass

    # Check multiple possible locations for triton_stub.py
    possible_stub_paths = [
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "banana_pi",
            "test_tinyllama_rvv",
            "triton_stub.py",
        ),
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "banana_pi",
            "test_tinyllama_rvv",
            "triton_stub.py",
        ),
    ]

    triton_stub_path = None
    for path in possible_stub_paths:
        if os.path.exists(path):
            triton_stub_path = path
            break

    if triton_stub_path:
        stub_namespace = {
            "__file__": triton_stub_path,
            "__name__": "triton_stub",
            "__package__": "",
        }
        stub_namespace.update(sys.modules)
        with open(triton_stub_path, "r") as f:
            exec(compile(f.read(), triton_stub_path, "exec"), stub_namespace)


# Setup environment
setup_triton_stub()


def is_riscv_platform() -> bool:
    """Check if running on RISC-V platform."""
    machine = platform.machine().lower()
    return machine in ("riscv64", "riscv32", "riscv")


# Try to import sgl_kernel
try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False


# ============================================================================
# Benchmark Data Classes
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""

    M: int  # Batch size / sequence length
    N: int  # Output features
    K: int  # Input features
    description: str


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    config: BenchmarkConfig
    pytorch_ms: float
    pytorch_std: float
    rvv_ms: float | None
    rvv_std: float | None
    speedup: float | None
    correct: bool | None


# ============================================================================
# Benchmark Configurations
# ============================================================================

# TinyLlama dimensions: hidden=2048, intermediate=5632
TINYLLAMA_CONFIGS = [
    # Decode (single token)
    BenchmarkConfig(1, 2048, 2048, "TinyLlama QKV decode"),
    BenchmarkConfig(1, 5632, 2048, "TinyLlama FFN up decode"),
    BenchmarkConfig(1, 2048, 5632, "TinyLlama FFN down decode"),
]

# Batch decode configurations
BATCH_DECODE_CONFIGS = [
    BenchmarkConfig(4, 2048, 2048, "TinyLlama QKV batch=4"),
    BenchmarkConfig(8, 2048, 2048, "TinyLlama QKV batch=8"),
    BenchmarkConfig(16, 2048, 2048, "TinyLlama QKV batch=16"),
    BenchmarkConfig(32, 2048, 2048, "TinyLlama QKV batch=32"),
]

# Prefill configurations
PREFILL_CONFIGS = [
    BenchmarkConfig(32, 2048, 2048, "Prefill seq=32"),
    BenchmarkConfig(64, 2048, 2048, "Prefill seq=64"),
    BenchmarkConfig(128, 2048, 2048, "Prefill seq=128"),
    BenchmarkConfig(256, 2048, 2048, "Prefill seq=256"),
]

# Small sizes for quick testing
SMALL_CONFIGS = [
    BenchmarkConfig(1, 64, 64, "Small 64x64"),
    BenchmarkConfig(4, 128, 128, "Small 128x128"),
    BenchmarkConfig(8, 256, 256, "Small 256x256"),
]

# CI uses simplified configurations
if IS_CI:
    ALL_CONFIGS = SMALL_CONFIGS[:2] + TINYLLAMA_CONFIGS[:1]
else:
    ALL_CONFIGS = (
        SMALL_CONFIGS + TINYLLAMA_CONFIGS + BATCH_DECODE_CONFIGS + PREFILL_CONFIGS
    )


# ============================================================================
# Benchmark Functions
# ============================================================================


def benchmark_function(
    fn: Callable, warmup: int = 3, repeat: int = 10
) -> tuple[float, float]:
    """
    Benchmark a function and return mean and std time in milliseconds.

    Args:
        fn: Function to benchmark (no arguments)
        warmup: Number of warmup iterations
        repeat: Number of timed iterations

    Returns:
        Tuple of (mean_ms, std_ms)
    """
    # Warmup
    for _ in range(warmup):
        fn()

    # Timed runs
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_ms = sum(times) / len(times)
    variance = sum((t - mean_ms) ** 2 for t in times) / len(times)
    std_ms = variance**0.5

    return mean_ms, std_ms


def run_benchmark(
    config: BenchmarkConfig, warmup: int = 3, repeat: int = 10
) -> BenchmarkResult:
    """
    Run benchmark for a single configuration.

    Args:
        config: Benchmark configuration
        warmup: Number of warmup iterations
        repeat: Number of timed iterations

    Returns:
        BenchmarkResult with timing data
    """
    M, N, K = config.M, config.N, config.K

    # Create input tensors
    # RVV kernel uses AT_DISPATCH_REDUCED_FLOATING_TYPES (float16/bfloat16) for x/weight
    # but bias must be float32 (API requirement)
    x_rvv = torch.randn(M, K, dtype=torch.float16)
    weight_rvv = torch.randn(N, K, dtype=torch.float16)
    bias = torch.randn(N, dtype=torch.float32)

    # Create float32 versions for PyTorch baseline
    x_ref = x_rvv.float()
    weight_ref = weight_rvv.float()

    # Benchmark PyTorch (using float32 for fair comparison with optimized code)
    def pytorch_fn():
        return torch.mm(x_ref, weight_ref.t()) + bias

    pytorch_ms, pytorch_std = benchmark_function(pytorch_fn, warmup, repeat)
    y_ref = pytorch_fn()

    # Benchmark RVV if available
    rvv_ms = None
    rvv_std = None
    speedup = None
    correct = None

    if HAS_SGL_KERNEL:
        try:

            def rvv_fn():
                return sgl_kernel.weight_packed_linear(x_rvv, weight_rvv, bias, False)

            rvv_ms, rvv_std = benchmark_function(rvv_fn, warmup, repeat)
            y_rvv = rvv_fn()

            # Check correctness (convert to float32 for comparison)
            correct = torch.allclose(y_ref, y_rvv.float(), atol=0.5, rtol=0.1)

            # Calculate speedup
            if rvv_ms > 0:
                speedup = pytorch_ms / rvv_ms
        except Exception as e:
            print(f"    RVV Error: {e}")

    return BenchmarkResult(
        config=config,
        pytorch_ms=pytorch_ms,
        pytorch_std=pytorch_std,
        rvv_ms=rvv_ms,
        rvv_std=rvv_std,
        speedup=speedup,
        correct=correct,
    )


def print_result(result: BenchmarkResult):
    """Print a single benchmark result in a formatted way."""
    config = result.config

    print(
        f"  {config.description:<35} [{config.M:4}x{config.K:4}] @ [{config.N:4}x{config.K:4}]"
    )
    print(f"    PyTorch: {result.pytorch_ms:8.3f} ± {result.pytorch_std:.3f} ms")

    if result.rvv_ms is not None:
        status = "✓" if result.correct else "✗"
        speedup_str = f"{result.speedup:.2f}x" if result.speedup else "N/A"
        print(
            f"    RVV:     {result.rvv_ms:8.3f} ± {result.rvv_std:.3f} ms  "
            f"[{status}] speedup: {speedup_str}"
        )
    else:
        print(f"    RVV:     N/A (sgl_kernel not available)")


def print_summary(results: list[BenchmarkResult]):
    """Print summary statistics."""
    rvv_results = [r for r in results if r.rvv_ms is not None]

    if not rvv_results:
        print("\nNo RVV results available for summary.")
        return

    correct_count = sum(1 for r in rvv_results if r.correct)
    total_count = len(rvv_results)

    speedups = [r.speedup for r in rvv_results if r.speedup is not None]
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0
    min_speedup = min(speedups) if speedups else 0
    max_speedup = max(speedups) if speedups else 0

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Correctness: {correct_count}/{total_count} passed")
    print(f"  Speedup:")
    print(f"    Average: {avg_speedup:.2f}x")
    print(f"    Min:     {min_speedup:.2f}x")
    print(f"    Max:     {max_speedup:.2f}x")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Benchmark RVV GEMM kernel")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Timed iterations")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark (CI mode)"
    )
    args = parser.parse_args()

    # Override configs for quick mode
    configs = SMALL_CONFIGS if args.quick else ALL_CONFIGS

    if args.quick:
        args.warmup = 2
        args.repeat = 10

    print("=" * 70)
    print("RVV GEMM Kernel Benchmark")
    print("=" * 70)
    print(f"Platform: {platform.machine()}")
    print(f"RISC-V: {is_riscv_platform()}")
    print(f"sgl_kernel available: {HAS_SGL_KERNEL}")
    print(f"CI mode: {IS_CI}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
    print()

    results = []

    for config in configs:
        result = run_benchmark(config, args.warmup, args.repeat)
        results.append(result)
        print_result(result)
        print()

    print_summary(results)


if __name__ == "__main__":
    main()
