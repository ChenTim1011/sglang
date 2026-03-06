import argparse
import os
import platform
import time
from dataclasses import dataclass
from typing import Callable
from unittest.mock import Mock

import torch

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def benchmark_function(
    fn: Callable, warmup: int = 5, repeat: int = 20
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()

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


def print_benchmark_result(
    description: str,
    params: str,
    rvv_ms: float,
    torch_ms: float,
    speedup: float,
    correct: bool = None,
    throughput_rvv: float = None,
    rvv_std: float | None = None,
    torch_std: float | None = None,
):
    print(f"  {description:<35} {params}")

    # PyTorch Line
    if torch_ms is not None:
        torch_line = f"    PyTorch: {torch_ms:8.3f}"
        if torch_std is not None:
            torch_line += f" ± {torch_std:.3f}"
        torch_line += " ms"
        print(torch_line)

    # RVV Line
    if rvv_ms is not None:
        rvv_line = f"    RVV:     {rvv_ms:8.3f}"
        if rvv_std is not None:
            rvv_line += f" ± {rvv_std:.3f}"
        rvv_line += " ms"

        if speedup:
            rvv_line += f"  speedup: {speedup:.2f}x"

        if throughput_rvv is not None:
            rvv_line += f" ({throughput_rvv:.1f} req/s)"

        if correct is not None:
            status = "✓" if correct else "✗"
            rvv_line += f" [{status}]"
        print(rvv_line)
    else:
        print("    RVV:     N/A")
    print()


def print_benchmark_summary(results):
    valid_results = [r for r in results if getattr(r, "speedup", None) is not None]
    if not valid_results:
        print("No RVV results available for summary.")
        return

    speedups = [r.speedup for r in valid_results]
    avg_speedup = sum(speedups) / len(speedups)
    min_speedup = min(speedups)
    max_speedup = max(speedups)

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total Benchmarks: {len(valid_results)}")
    print(f"  Speedup:")
    print(f"    Average: {avg_speedup:.2f}x")
    print(f"    Min:     {min_speedup:.2f}x")
    print(f"    Max:     {max_speedup:.2f}x")
    print()


def create_decode_mock_runner(num_heads, head_dim, v_head_dim, dtype=torch.float16):
    mock_runner = Mock()
    mock_runner.device = torch.device("cpu")
    mock_runner.model_config = Mock()
    mock_runner.model_config.num_attention_heads = num_heads
    mock_runner.model_config.num_hidden_layers = 1  # benchmark uses layer_id=0 only
    mock_runner.tp_size = 1
    mock_runner.kv_cache_dtype = "int8" if dtype == torch.int8 else "auto"

    mock_runner.req_to_token_pool = Mock()
    mock_runner.req_to_token_pool.size = 256

    mock_runner.token_to_kv_pool = Mock()
    mock_runner.token_to_kv_pool.full_attention_layer_id_mapping = {0: 0}

    def create_buffer(num_tokens, heads, hdim, dtype):
        if dtype == torch.int8:
            return (torch.randn(num_tokens, heads, hdim, dtype=torch.float32) * 50).to(
                torch.int8
            )
        else:
            return torch.randn(num_tokens, heads, hdim, dtype=dtype)

    mock_runner.token_to_kv_pool.get_key_buffer = Mock(
        return_value=create_buffer(10000, num_heads, head_dim, dtype)
    )
    mock_runner.token_to_kv_pool.get_value_buffer = Mock(
        return_value=create_buffer(10000, num_heads, v_head_dim, dtype)
    )

    return mock_runner


def create_decode_mock_layer(num_heads, head_dim, v_head_dim):
    mock_layer = Mock()
    mock_layer.tp_q_head_num = num_heads
    mock_layer.qk_head_dim = head_dim
    mock_layer.v_head_dim = v_head_dim
    mock_layer.layer_id = 0
    mock_layer.scaling = 1.0 / (head_dim**0.5)
    mock_layer.logit_cap = 50.0
    mock_layer.is_cross_attention = False
    mock_layer.k_scale_float = 1.0
    mock_layer.v_scale_float = 1.0
    # Pre-set cached floats so Mock.__getattr__ doesn't return a child Mock object,
    # which would fool _ensure_cached_scales into returning early with a non-float.
    mock_layer._cached_k_scale_float = 1.0
    mock_layer._cached_v_scale_float = 1.0
    return mock_layer


def create_decode_mock_forward_batch(
    num_requests, num_heads, head_dim, v_head_dim, max_seq_len, dtype=torch.float16
):
    mock_batch = Mock()
    mock_batch.batch_size = num_requests
    mock_batch.out_cache_loc = torch.zeros(num_requests, dtype=torch.int64)
    mock_batch.encoder_out_cache_loc = None
    mock_batch.seq_lens = torch.ones(num_requests, dtype=torch.int64) * max_seq_len
    mock_batch.req_pool_indices = torch.arange(num_requests, dtype=torch.int64)

    mock_batch.req_to_token_pool = Mock()
    mock_batch.req_to_token_pool.req_to_token = torch.zeros(
        num_requests, max_seq_len, dtype=torch.int64
    )

    mock_batch.token_to_kv_pool = Mock()

    def create_buffer(num_tokens, heads, hdim, dtype):
        if dtype == torch.int8:
            return (torch.randn(num_tokens, heads, hdim, dtype=torch.float32) * 50).to(
                torch.int8
            )
        else:
            return torch.randn(num_tokens, heads, hdim, dtype=dtype)

    mock_batch.token_to_kv_pool.get_key_buffer = Mock(
        return_value=create_buffer(
            max_seq_len * num_requests + 100, num_heads, head_dim, dtype
        )
    )
    mock_batch.token_to_kv_pool.get_value_buffer = Mock(
        return_value=create_buffer(
            max_seq_len * num_requests + 100, num_heads, v_head_dim, dtype
        )
    )
    return mock_batch


class ExtendMockRunner:
    def __init__(self, num_heads, head_dim, kv_dtype=torch.float16):
        self.device = "cpu"
        self.model_config = argparse.Namespace(
            num_attention_heads=num_heads,
            num_hidden_layers=1,  # benchmark uses layer_id=0 only
        )
        self.tp_size = 1
        self.req_to_token_pool = Mock()
        self.req_to_token_pool.size = 256
        self.token_to_kv_pool = ExtendMockTokenToKVPool(
            num_heads, head_dim, dtype=kv_dtype
        )
        self.kv_cache_dtype = "int8" if kv_dtype == torch.int8 else "auto"


class ExtendMockTokenToKVPool:
    def __init__(self, num_heads, head_dim, max_tokens=10000, dtype=torch.float16):
        self.max_tokens = max_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.full_attention_layer_id_mapping = {0: 0}

        if dtype == torch.int8:
            self.k_buffer = (
                torch.randn(max_tokens, num_heads, head_dim, dtype=torch.float32) * 50
            ).to(torch.int8)
            self.v_buffer = (
                torch.randn(max_tokens, num_heads, head_dim, dtype=torch.float32) * 50
            ).to(torch.int8)
        else:
            self.k_buffer = torch.randn(max_tokens, num_heads, head_dim, dtype=dtype)
            self.v_buffer = torch.randn(max_tokens, num_heads, head_dim, dtype=dtype)

    def get_key_buffer(self, layer_id):
        return self.k_buffer

    def get_value_buffer(self, layer_id):
        return self.v_buffer

    def set_kv_buffer(self, layer, loc, k, v):
        if hasattr(loc, "__len__"):
            self.k_buffer[loc] = k
            self.v_buffer[loc] = v


class ExtendMockLayer:
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
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0


class ExtendMockForwardMode:
    def is_decode_or_idle(self):
        return False

    def is_extend(self):
        return True


class ExtendMockReqToTokenPool:
    def __init__(self, num_reqs, seq_len, max_tokens):
        self.req_to_token = torch.zeros(num_reqs, seq_len, dtype=torch.int64)
        for i in range(num_reqs):
            self.req_to_token[i] = torch.arange(seq_len) % max_tokens


class ExtendMockForwardBatch:
    def __init__(
        self, num_reqs, seq_len, extend_len, num_heads, head_dim, kv_dtype=torch.float16
    ):
        self.batch_size = num_reqs
        self.req_pool_indices = torch.arange(num_reqs)
        self.seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int64)
        self.extend_seq_lens = torch.full((num_reqs,), extend_len, dtype=torch.int64)
        self.extend_prefix_lens = self.seq_lens - self.extend_seq_lens
        self.extend_start_loc = torch.arange(num_reqs) * extend_len
        max_tokens = num_reqs * seq_len * 2
        self.req_to_token_pool = ExtendMockReqToTokenPool(num_reqs, seq_len, max_tokens)
        self.token_to_kv_pool = ExtendMockTokenToKVPool(
            num_heads, head_dim, max_tokens, dtype=kv_dtype
        )
        self.forward_mode = ExtendMockForwardMode()
        self.out_cache_loc = torch.arange(num_reqs * extend_len)
