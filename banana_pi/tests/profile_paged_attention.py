import argparse
import os
import random
import sys
import time
from dataclasses import dataclass

import torch


# Setup OpenMP and PyTorch threading before importing sgl_kernel
def setup_threading():
    """Setup OpenMP and PyTorch threading for multi-core execution."""
    import multiprocessing

    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()

    # Set OpenMP environment variables
    os.environ.setdefault("OMP_NUM_THREADS", str(num_cores))
    os.environ.setdefault("OMP_PROC_BIND", "true")
    os.environ.setdefault("OMP_PLACES", "cores")

    # Set PyTorch thread count
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(
        1
    )  # Inter-op parallelism not needed for single operation

    # Setup LD_PRELOAD and LD_LIBRARY_PATH if libomp.so exists
    libomp_path = os.path.expanduser("~/.local/lib/libomp.so")
    if os.path.exists(libomp_path):
        lib_dir = os.path.dirname(libomp_path)
        if "LD_PRELOAD" not in os.environ:
            os.environ["LD_PRELOAD"] = libomp_path
        if (
            "LD_LIBRARY_PATH" not in os.environ
            or lib_dir not in os.environ["LD_LIBRARY_PATH"]
        ):
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = (
                f"{lib_dir}:{current_ld_path}" if current_ld_path else lib_dir
            )

    print(
        f"[INFO] Threading setup: {num_cores} cores, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
        f"PyTorch threads={torch.get_num_threads()}"
    )


# Setup threading first
setup_threading()

# Import test utilities (required, no fallback)
from test_utils import check_system_state, generate_fair_kv_buffers

# Try to import sgl_kernel
try:
    import sgl_kernel

    print("[INFO] sgl_kernel imported successfully.")

    # Verify RVV support
    if hasattr(torch.ops.sgl_kernel, "get_rvv_vlenb"):
        vlen = torch.ops.sgl_kernel.get_rvv_vlenb() * 8
        print(f"[INFO] ✅ RVV Extensions Detected! VLEN = {vlen} bits")
    else:
        print("[WARNING] ❌ RVV Extensions NOT found.")

except ImportError as e:
    print(f"[ERROR] Failed to import sgl_kernel: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Paged Attention Simulation Components
# -----------------------------------------------------------------------------


class MockBlockManager:
    def __init__(self, total_tokens, block_size=16):
        self.total_tokens = total_tokens
        self.block_size = block_size
        self.num_blocks = total_tokens // block_size
        self.free_blocks = list(range(self.num_blocks))
        # Randomize free blocks to simulate fragmentation/non-contiguous allocation
        random.shuffle(self.free_blocks)

    def allocate(self, num_tokens):
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < needed_blocks:
            raise RuntimeError("Out of memory in MockBlockManager")

        allocated = []
        for _ in range(needed_blocks):
            allocated.append(self.free_blocks.pop())
        return allocated


def create_paged_runner(
    num_heads, head_dim, v_head_dim, dtype=torch.float16, total_capacity=100000, seed=42
):
    mock_runner = type("", (), {})()  # Empty object
    mock_runner.device = torch.device("cpu")
    mock_runner.model_config = type("", (), {})()
    mock_runner.model_config.num_attention_heads = num_heads
    mock_runner.kv_cache_dtype = "int8" if dtype == torch.int8 else "auto"
    mock_runner.token_to_kv_pool = type("", (), {})()

    # Generate fair KV buffers (FP16 first, then quantize for INT8)
    k_buffer, _, k_scale, _ = generate_fair_kv_buffers(
        total_capacity, num_heads, head_dim, dtype, "cpu", seed
    )
    # For v_head_dim, generate separately (use different seed offset to ensure different data)
    _, v_buffer, _, v_scale = generate_fair_kv_buffers(
        total_capacity, num_heads, v_head_dim, dtype, "cpu", seed + 1000
    )

    mock_runner.token_to_kv_pool.get_key_buffer = lambda layer_id: k_buffer
    mock_runner.token_to_kv_pool.get_value_buffer = lambda layer_id: v_buffer

    return mock_runner


def create_paged_batch(
    num_requests, num_heads, head_dim, seq_len, block_manager, block_size=16
):
    mock_batch = type("", (), {})()
    mock_batch.batch_size = num_requests
    mock_batch.out_cache_loc = torch.zeros(num_requests, dtype=torch.int64)
    mock_batch.seq_lens = torch.ones(num_requests, dtype=torch.int64) * seq_len
    mock_batch.req_pool_indices = torch.arange(num_requests, dtype=torch.int64)

    # Construct req_to_token mapping based on allocated blocks
    # Shape: [num_requests, max_seq_len]
    req_to_token = torch.zeros(num_requests, seq_len, dtype=torch.int64)

    for r in range(num_requests):
        blocks = block_manager.allocate(seq_len)
        token_indices = []
        for b_idx in blocks:
            # Each block covers 'block_size' tokens
            # The physical indices are [b_idx*block_size ... (b_idx+1)*block_size - 1]
            start_idx = b_idx * block_size
            token_indices.extend(range(start_idx, start_idx + block_size))

        # Trim to exact seq_len
        token_indices = token_indices[:seq_len]
        req_to_token[r] = torch.tensor(token_indices, dtype=torch.int64)

    mock_batch.req_to_token_pool = type("", (), {})()
    mock_batch.req_to_token_pool.req_to_token = req_to_token

    # We need seq_lens to match
    mock_batch.seq_lens = torch.full((num_requests,), seq_len, dtype=torch.int64)

    return mock_batch


# -----------------------------------------------------------------------------
# RVV Backend (Same as before)
# -----------------------------------------------------------------------------
class RVVAttnBackend:
    def __init__(self, model_runner):
        self.decode_attention_fwd = torch.ops.sgl_kernel.decode_attention_cpu
        self.model_runner = model_runner
        self.num_heads = model_runner.model_config.num_attention_heads
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.forward_metadata = None

    def init_forward_metadata(self, forward_batch):
        pass  # Lazy init

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if self.forward_metadata is None:
            batch_size = q.size(0)
            num_heads = q.size(1)
            head_dim = q.size(2)
            self.forward_metadata = (
                torch.empty(
                    (batch_size, num_heads, 1, head_dim + 1),
                    dtype=torch.float32,
                    device="cpu",
                ),
                None,
            )
        return self._rvv_decode(q, k, v, layer, forward_batch, save_kv_cache)

    def _rvv_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        k_buffer = self.model_runner.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = self.model_runner.token_to_kv_pool.get_value_buffer(layer.layer_id)
        attn_logits, _ = self.forward_metadata
        q_reshaped = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o = torch.empty_like(q_reshaped)

        self.decode_attention_fwd(
            q_reshaped,
            k_buffer,
            v_buffer,
            o,
            k,
            v,
            forward_batch.out_cache_loc,
            attn_logits,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            layer.scaling,
            layer.logit_cap,
        )
        return o


class RVVAttnBackendInt8(RVVAttnBackend):
    def __init__(self, model_runner):
        super().__init__(model_runner)
        self.k_scale = 0.01
        self.v_scale = 0.01
        self.decode_attention_fwd_int8 = torch.ops.sgl_kernel.decode_attention_int8_cpu

    def _rvv_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        k_buffer = self.model_runner.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = self.model_runner.token_to_kv_pool.get_value_buffer(layer.layer_id)
        attn_logits, _ = self.forward_metadata
        q_reshaped = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o = torch.empty_like(q_reshaped)

        self.decode_attention_fwd_int8(
            q_reshaped,
            k_buffer,
            v_buffer,
            o,
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


# -----------------------------------------------------------------------------
# Profiling Logic
# -----------------------------------------------------------------------------
def run_paged_profile(head_dim=128):
    # Check system state
    check_system_state()

    # Test specific sensitive lengths around 1024
    seq_lens = [512, 1023, 1024, 1025, 2048]
    num_heads = 32
    batch_size = 1

    print(f"\nRunning Paged Attention (Block Layout) Profile")
    print(f"Config: H={num_heads}, D={head_dim}, BS={batch_size}, BlockSize=16")
    print("-" * 60)
    print(f"{'SeqLen':<8} | {'FP16 (ms)':<10} | {'INT8 (ms)':<10} | {'Speedup':<8}")
    print("-" * 60)

    # Reuse block manager to keep fragmentation high?
    # Or fresh for each seq_len? Fresh is cleaner for measurement.

    for seq_len in seq_lens:
        # Create Block Manager
        total_capacity = 50000  # Enough tokens
        block_mgr = MockBlockManager(total_capacity, block_size=16)

        # --- FP16 ---
        runner_fp16 = create_paged_runner(
            num_heads, head_dim, head_dim, torch.float16, total_capacity, seed=42
        )
        layer_fp16 = type(
            "",
            (),
            {
                "layer_id": 0,
                "tp_q_head_num": num_heads,
                "qk_head_dim": head_dim,
                "scaling": 1.0,
                "logit_cap": 50.0,
            },
        )()
        backend_fp16 = RVVAttnBackend(runner_fp16)
        batch_fp16 = create_paged_batch(
            batch_size, num_heads, head_dim, seq_len, block_mgr
        )

        # --- INT8 ---
        # Reset block mgr for fairness (same layout)
        random.seed(42)  # Ensure same random blocks are picked
        block_mgr = MockBlockManager(total_capacity, block_size=16)

        runner_int8 = create_paged_runner(
            num_heads, head_dim, head_dim, torch.int8, total_capacity, seed=42
        )
        layer_int8 = type(
            "",
            (),
            {
                "layer_id": 0,
                "tp_q_head_num": num_heads,
                "qk_head_dim": head_dim,
                "scaling": 1.0,
                "logit_cap": 50.0,
            },
        )()
        backend_int8 = RVVAttnBackendInt8(runner_int8)
        batch_int8 = create_paged_batch(
            batch_size, num_heads, head_dim, seq_len, block_mgr
        )

        # Inputs
        q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float16)

        # Warmup
        for _ in range(20):  # Increased warmup
            backend_fp16.forward_decode(q, k, v, layer_fp16, batch_fp16)
            backend_int8.forward_decode(q, k, v, layer_int8, batch_int8)

        # Measure with statistics
        from test_utils import measure_with_statistics

        def measure_fp16():
            start = time.perf_counter()
            backend_fp16.forward_decode(q, k, v, layer_fp16, batch_fp16)
            return (time.perf_counter() - start) * 1000

        def measure_int8():
            start = time.perf_counter()
            backend_int8.forward_decode(q, k, v, layer_int8, batch_int8)
            return (time.perf_counter() - start) * 1000

        stats_fp16 = measure_with_statistics(measure_fp16, num_runs=50)
        stats_int8 = measure_with_statistics(measure_int8, num_runs=50)
        dt_fp16 = stats_fp16.mean
        dt_int8 = stats_int8.mean

        print(
            f"{seq_len:<8} | {dt_fp16:<10.3f} | {dt_int8:<10.3f} | {dt_fp16/dt_int8:<8.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    run_paged_profile(args.head_dim)
