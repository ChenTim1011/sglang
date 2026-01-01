"""
Test Utilities for Statistical Analysis and System State Checking

This module provides utilities for:
1. Statistical analysis (mean, std, CI, min, max)
2. System state checking
3. Fair data generation for FP16/INT8 comparison
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch


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
    for _ in range(num_runs):
        result = func()
        results.append(result)

    results_array = np.array(results)
    mean = np.mean(results_array)
    std = np.std(results_array, ddof=1)  # Sample standard deviation
    min_val = np.min(results_array)
    max_val = np.max(results_array)
    median = np.median(results_array)

    # Confidence interval
    # For 95% CI: z = 1.96, for 90% CI: z = 1.645
    z_score = 1.96 if confidence_level == 0.95 else 1.645
    se = std / np.sqrt(num_runs)  # Standard error
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


def print_statistics(stats: StatisticalResult, label: str = ""):
    """Print statistical results in a readable format."""
    if label:
        print(f"\n{label}:")
    print(f"  Mean:   {stats.mean:.6f}")
    print(f"  Std:    {stats.std:.6f}")
    print(f"  Min:    {stats.min:.6f}")
    print(f"  Max:    {stats.max:.6f}")
    print(f"  Median: {stats.median:.6f}")
    print(f"  95% CI: [{stats.ci_95_lower:.6f}, {stats.ci_95_upper:.6f}]")
    print(f"  Runs:   {stats.num_runs}")


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
        k_buffer = k_buffer_fp16
        v_buffer = v_buffer_fp16
        k_scale = 1.0
        v_scale = 1.0

    return k_buffer, v_buffer, k_scale, v_scale


def compare_tensors_fair(
    tensor_fp16: torch.Tensor, tensor_int8: torch.Tensor
) -> Dict[str, float]:
    """
    Compare FP16 and INT8 tensors fairly.

    Returns:
        Dictionary with cosine_similarity, mse, max_error, relative_error
    """
    tensor_fp16_f = tensor_fp16.float()
    tensor_int8_f = tensor_int8.float()

    # Flatten for comparison
    flat_fp16 = tensor_fp16_f.flatten()
    flat_int8 = tensor_int8_f.flatten()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_fp16.unsqueeze(0), flat_int8.unsqueeze(0), dim=1
    ).item()

    # MSE
    mse = torch.mean((flat_fp16 - flat_int8) ** 2).item()

    # Max absolute error
    max_err = torch.max(torch.abs(flat_fp16 - flat_int8)).item()

    # Relative error (mean)
    # Avoid division by zero
    abs_fp16 = torch.abs(flat_fp16)
    relative_err = torch.mean(
        torch.where(
            abs_fp16 > 1e-8,
            torch.abs(flat_fp16 - flat_int8) / abs_fp16,
            torch.zeros_like(flat_fp16),
        )
    ).item()

    return {
        "cosine_similarity": cos_sim,
        "mse": mse,
        "max_error": max_err,
        "relative_error": relative_err,
    }
