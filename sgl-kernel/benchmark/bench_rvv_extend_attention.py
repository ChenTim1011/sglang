"""
Benchmark script for RVV extend attention kernel.

This script benchmarks the RVV optimized extend attention implementation
against PyTorch's native backend through SGLang's attention backend system.

Usage:
    python3 bench_rvv_extend_attention.py [--num-iterations N]
"""

import argparse
import os
import sys
import time

import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

try:
    from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    HAS_SGLANG = True
except ImportError:
    HAS_SGLANG = False
    print("SGLang not found")


class MockRunner:
    def __init__(self, num_heads, head_dim):
        self.device = "cpu"
        self.model_config = argparse.Namespace(num_attention_heads=num_heads)
        self.tp_size = 1
        self.token_to_kv_pool = MockTokenToKVPool(num_heads, head_dim)


class MockTokenToKVPool:
    def __init__(self, num_heads, head_dim, max_tokens=10000):
        self.max_tokens = max_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.k_buffer = torch.randn(max_tokens, num_heads, head_dim)
        self.v_buffer = torch.randn(max_tokens, num_heads, head_dim)

    def get_key_buffer(self, layer_id):
        return self.k_buffer

    def get_value_buffer(self, layer_id):
        return self.v_buffer

    def set_kv_buffer(self, layer, loc, k, v):
        # Handle both layer object and layer_id
        if hasattr(loc, "__len__"):
            self.k_buffer[loc] = k
            self.v_buffer[loc] = v


class MockLayer:
    def __init__(self, num_heads, head_dim):
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_heads
        self.tp_v_head_num = num_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = head_dim
        self.layer_id = 0
        self.scaling = 1.0 / (head_dim**0.5)
        self.logit_cap = 50.0
        self.is_cross_attention = False
        self.attn_type = None


class MockForwardMode:
    def is_decode_or_idle(self):
        return False

    def is_extend(self):
        return True


class MockForwardBatch:
    def __init__(self, num_reqs, seq_len, extend_len, num_heads, head_dim):
        self.batch_size = num_reqs
        self.req_pool_indices = torch.arange(num_reqs)
        self.seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int64)
        self.extend_seq_lens = torch.full((num_reqs,), extend_len, dtype=torch.int64)
        self.extend_prefix_lens = self.seq_lens - self.extend_seq_lens
        self.extend_start_loc = torch.arange(num_reqs) * extend_len
        max_tokens = num_reqs * seq_len * 2  # Ensure enough space
        self.req_to_token_pool = MockReqToTokenPool(num_reqs, seq_len, max_tokens)
        self.token_to_kv_pool = MockTokenToKVPool(num_heads, head_dim, max_tokens)
        self.forward_mode = MockForwardMode()
        self.out_cache_loc = torch.arange(num_reqs * extend_len)


class MockReqToTokenPool:
    def __init__(self, num_reqs, seq_len, max_tokens):
        self.req_to_token = torch.zeros(num_reqs, seq_len, dtype=torch.int64)
        for i in range(num_reqs):
            # Ensure indices are within max_tokens
            self.req_to_token[i] = torch.arange(seq_len) % max_tokens


def benchmark(
    backend_name, num_reqs, seq_len, extend_len, num_heads, head_dim, mode="extend"
):
    """
    Benchmark attention kernel.

    Args:
        mode: "extend" (has prefix in cache) or "prefill" (no prefix, extend_len == seq_len)
    """
    runner = MockRunner(num_heads, head_dim)
    layer = MockLayer(num_heads, head_dim)

    if backend_name == "rvv":
        backend = RVVAttnBackend(runner)
    else:
        backend = TorchNativeAttnBackend(runner)

    # For prefill mode, extend_len == seq_len (no prefix)
    actual_extend_len = seq_len if mode == "prefill" else extend_len
    actual_seq_len = seq_len

    batch = MockForwardBatch(
        num_reqs, actual_seq_len, actual_extend_len, num_heads, head_dim
    )
    backend.init_forward_metadata(batch)

    # Data
    total_extend_len = num_reqs * actual_extend_len
    q = torch.randn(total_extend_len, num_heads, head_dim)
    k = torch.randn(total_extend_len, num_heads, head_dim)
    v = torch.randn(total_extend_len, num_heads, head_dim)

    # Warmup
    for _ in range(5):
        backend.forward_extend(q, k, v, layer, batch)

    start = time.time()
    iters = 20
    for _ in range(iters):
        backend.forward_extend(q, k, v, layer, batch)
    end = time.time()

    avg_time = (end - start) / iters
    print(f"{backend_name}: {avg_time*1000:.2f} ms")
    return avg_time


if __name__ == "__main__":
    if not HAS_SGLANG:
        sys.exit(1)

    num_reqs = 1
    num_heads = 8
    head_dim = 64

    # Test 1: Extend mode (has prefix in cache)
    print("=" * 60)
    print("Test 1: EXTEND mode (seq_len=128, extend_len=32, prefix_len=96)")
    print("=" * 60)
    seq_len = 128
    extend_len = 32
    print(f"Parameters: reqs={num_reqs}, seq_len={seq_len}, extend_len={extend_len}")

    t_riscv_ext = benchmark(
        "riscv", num_reqs, seq_len, extend_len, num_heads, head_dim, mode="extend"
    )
    t_torch_ext = benchmark(
        "torch_native",
        num_reqs,
        seq_len,
        extend_len,
        num_heads,
        head_dim,
        mode="extend",
    )
    print(f"Extend Speedup: {t_torch_ext/t_riscv_ext:.2f}x")

    # Test 2: Prefill mode (no prefix, all new tokens)
    print()
    print("=" * 60)
    print("Test 2: PREFILL mode (seq_len=128, extend_len=128, prefix_len=0)")
    print("=" * 60)
    seq_len = 128
    print(
        f"Parameters: reqs={num_reqs}, seq_len={seq_len}, extend_len={seq_len} (prefill)"
    )

    t_riscv_pre = benchmark(
        "riscv", num_reqs, seq_len, seq_len, num_heads, head_dim, mode="prefill"
    )
    t_torch_pre = benchmark(
        "torch_native", num_reqs, seq_len, seq_len, num_heads, head_dim, mode="prefill"
    )
    print(f"Prefill Speedup: {t_torch_pre/t_riscv_pre:.2f}x")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Extend mode  - RISC-V: {t_riscv_ext*1000:.2f}ms, torch_native: {t_torch_ext*1000:.2f}ms, Speedup: {t_torch_ext/t_riscv_ext:.2f}x"
    )
    print(
        f"Prefill mode - RISC-V: {t_riscv_pre*1000:.2f}ms, torch_native: {t_torch_pre*1000:.2f}ms, Speedup: {t_torch_pre/t_riscv_pre:.2f}x"
    )
