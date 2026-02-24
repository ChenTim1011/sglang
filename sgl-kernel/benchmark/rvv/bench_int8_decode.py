# Benchmark RVV vs torch_native int8 decode

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass

import torch
from utils import (
    IS_CI,
    create_decode_mock_forward_batch,
    create_decode_mock_layer,
    create_decode_mock_runner,
    print_benchmark_result,
)

try:
    from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    HAS_SGLANG = True
except ImportError:
    HAS_SGLANG = False
    print("SGLang not found")
    sys.exit(1)


@dataclass
class BenchmarkConfig:
    num_requests: int
    num_heads: int
    head_dim: int
    seq_len: int
    description: str
    kv_dtype: torch.dtype = torch.int8


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    rvv_ms: float
    torch_ms: float
    speedup: float
    throughput_rvv: float


INT8_CONFIGS = [
    BenchmarkConfig(1, 8, 64, 128, "INT8 Small Batch (BS=1)"),
    BenchmarkConfig(4, 8, 64, 128, "INT8 Medium Batch (BS=4)"),
    BenchmarkConfig(32, 8, 64, 128, "INT8 Large Batch (BS=32)"),
    BenchmarkConfig(1, 32, 64, 1024, "INT8 Decode (BS=1)"),
    BenchmarkConfig(8, 32, 64, 1024, "INT8 Decode (BS=8)"),
]

CI_CONFIGS = [
    BenchmarkConfig(1, 4, 32, 32, "CI INT8 Quick Test"),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = INT8_CONFIGS


class RVVAttnBackendInt8(RVVAttnBackend):
    def __init__(self, model_runner):
        super().__init__(model_runner)
        self.k_scale = 0.01
        self.v_scale = 0.01

        if hasattr(torch.ops.sgl_kernel, "decode_attention_int8_cpu"):
            self.decode_attention_fwd_int8 = (
                torch.ops.sgl_kernel.decode_attention_int8_cpu
            )
        else:
            self.decode_attention_fwd_int8 = None
            print("Warning: decode_attention_int8_cpu not available")

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        if k_buffer.dtype == torch.int8 and self.decode_attention_fwd_int8:
            if q.dtype not in (torch.float16, torch.bfloat16):
                q = q.half()

            if k.dtype != torch.int8:
                k = (k / self.k_scale).round().clamp(-128, 127).to(torch.int8)
                v = (v / self.v_scale).round().clamp(-128, 127).to(torch.int8)

            if len(self.forward_metadata) == 4:
                attn_logits, _, k_scale_tensor, v_scale_tensor = self.forward_metadata
            else:
                attn_logits, _ = self.forward_metadata
            q_reshaped = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

            if layer.qk_head_dim != layer.v_head_dim:
                o = torch.empty(
                    (q_reshaped.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
            else:
                o = torch.empty_like(q_reshaped)

            v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            self.decode_attention_fwd_int8(
                q_reshaped.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k_buffer,
                v_buffer,
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                k,
                v,
                forward_batch.out_cache_loc,
                attn_logits,
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                layer.scaling,
                layer.logit_cap,
                self.k_scale,
                self.v_scale,
            )
            return o
        else:
            return super().forward_decode(q, k, v, layer, forward_batch, save_kv_cache)


def run_single_backend(backend_name, config, num_iterations=20, warmup=5):
    v_head_dim = config.head_dim

    runner_dtype = torch.float16
    if backend_name == "rvv" and config.kv_dtype == torch.int8:
        runner_dtype = torch.int8

    mock_runner = create_decode_mock_runner(
        config.num_heads, config.head_dim, v_head_dim, runner_dtype
    )
    mock_layer = create_decode_mock_layer(config.num_heads, config.head_dim, v_head_dim)

    if backend_name == "rvv":
        if config.kv_dtype == torch.int8:
            backend = RVVAttnBackendInt8(mock_runner)
        else:
            backend = RVVAttnBackend(mock_runner)
    elif backend_name == "torch_native":
        backend = TorchNativeAttnBackend(mock_runner)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    dtype = torch.float16
    q = torch.randn(config.num_requests, config.num_heads, config.head_dim, dtype=dtype)
    k = torch.randn(config.num_requests, config.num_heads, config.head_dim, dtype=dtype)
    v = torch.randn(config.num_requests, config.num_heads, v_head_dim, dtype=dtype)

    forward_batch = create_decode_mock_forward_batch(
        config.num_requests,
        config.num_heads,
        config.head_dim,
        v_head_dim,
        config.seq_len,
        runner_dtype,
    )

    backend.init_forward_metadata(forward_batch)

    for _ in range(warmup):
        try:
            backend.forward_decode(
                q, k, v, mock_layer, forward_batch, save_kv_cache=True
            )
        except Exception as e:
            if backend_name == "rvv":
                print(f"Warning: RVV warmup failed: {e}")
                return 0.001
            raise

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        backend.forward_decode(q, k, v, mock_layer, forward_batch, save_kv_cache=True)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_iterations
    return avg_time


def run_benchmark(config: BenchmarkConfig, quick=False) -> BenchmarkResult:
    iterations = 3 if quick else 10
    warmup = 1 if quick else 3

    rvv_time = run_single_backend("rvv", config, iterations, warmup)

    torch_time = run_single_backend("torch_native", config, iterations, warmup)

    speedup = torch_time / rvv_time
    throughput = config.num_requests / rvv_time

    return BenchmarkResult(
        config=config,
        rvv_ms=rvv_time * 1000,
        torch_ms=torch_time * 1000,
        speedup=speedup,
        throughput_rvv=throughput,
    )


def print_result(result: BenchmarkResult):
    c = result.config
    params = f"BS={c.num_requests}, H={c.num_heads}, D={c.head_dim}, INT8"
    print_benchmark_result(
        c.description,
        params,
        result.rvv_ms,
        result.torch_ms,
        result.speedup,
        throughput_rvv=result.throughput_rvv,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RVV Decode Attention (INT8)"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.quick:
        configs = CI_CONFIGS

    print("=" * 60)
    print("RVV Decode Attention Benchmark (INT8)")
    print("=" * 60)
    print(f"Platform: {platform.machine()}")
    print(f"CI Mode: {IS_CI}")
    print(f"Quick Mode: {args.quick}")
    print("-" * 60)

    results = []
    for config in configs:
        try:
            result = run_benchmark(config, quick=args.quick)
            results.append(result)
            print_result(result)
        except Exception as e:
            print(f"FAILED {config.description}: {e}")

    if results:
        avg_speedup = sum(r.speedup for r in results) / len(results)
        print("=" * 60)
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Total Benchmarks: {len(results)}")
        print("=" * 60)
    else:
        print("No successful benchmarks completed.")


if __name__ == "__main__":
    main()
