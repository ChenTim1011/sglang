"""Shared utilities used by RVV benchmark scripts."""

import argparse
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional
from unittest.mock import Mock
from urllib.parse import urlparse

import psutil
import torch

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

try:
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server
except ImportError:
    print("sglang not found")
    raise

# ----- model / URL constants -----

BF16_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
W8A8_MODEL = "RedHatAI/Qwen2.5-1.5B-quantized.w8a8"
BASE_URL = DEFAULT_URL_FOR_TEST

# ----- prompts -----

_SHORT_PROMPT = "The quick fox: "  # ~5 tokens
_CTX_128 = ("the quick brown fox jumps over the lazy dog " * 19).strip()  # ~128 tokens
_CTX_512 = (_CTX_128 + " ") * 4  # ~512 tokens (for capacity / concurrent tests)

# ----- KV architecture constants (Qwen2.5-1.5B-Instruct) -----
# Source: HuggingFace config — num_hidden_layers=28, num_key_value_heads=2,
#   hidden_size=1536, num_attention_heads=12, head_dim=128
_KV_N_LAYERS = 28
_KV_N_KV_HEADS = 2  # GQA: 2 KV heads (not 12 Q heads)
_KV_HEAD_DIM = 128  # 1536 / 12 = 128

_E2E_WEIGHT_BYTES_BF16 = 3_000_000_000  # ~3 GB BF16 weights (1.5B params)

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

# ----- BenchConfig -----


@dataclass
class BenchConfig:
    prompt: str
    max_tokens: int
    batch_size: int
    desc: str


def benchmark_function(
    fn: Callable, warmup: int = 5, repeat: int = 20
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()

    times_ms = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

    mean_ms = sum(times_ms) / len(times_ms)
    variance = sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)
    return mean_ms, variance**0.5


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

    if torch_ms is not None:
        torch_line = f"    PyTorch: {torch_ms:8.3f}"
        if torch_std is not None:
            torch_line += f" ± {torch_std:.3f}"
        torch_line += " ms"
        print(torch_line)

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
    print("  Speedup:")
    print(f"    Average: {avg_speedup:.2f}x")
    print(f"    Min:     {min_speedup:.2f}x")
    print(f"    Max:     {max_speedup:.2f}x")
    print()


def _create_random_kv_buffer(
    num_tokens: int, heads: int, hdim: int, dtype: torch.dtype
) -> torch.Tensor:
    if dtype == torch.int8:
        return (torch.randn(num_tokens, heads, hdim, dtype=torch.float32) * 50).to(
            torch.int8
        )
    return torch.randn(num_tokens, heads, hdim, dtype=dtype)


def create_decode_mock_runner(num_heads, head_dim, v_head_dim, dtype=torch.float16):
    mock_runner = Mock()
    mock_runner.device = torch.device("cpu")
    mock_runner.model_config = Mock()
    mock_runner.model_config.num_attention_heads = num_heads
    mock_runner.model_config.num_hidden_layers = 1
    mock_runner.tp_size = 1
    mock_runner.kv_cache_dtype = "int8" if dtype == torch.int8 else "auto"

    mock_runner.req_to_token_pool = Mock()
    mock_runner.req_to_token_pool.size = 256

    mock_runner.token_to_kv_pool = Mock()
    mock_runner.token_to_kv_pool.full_attention_layer_id_mapping = {0: 0}
    mock_runner.token_to_kv_pool.get_key_buffer = Mock(
        return_value=_create_random_kv_buffer(10000, num_heads, head_dim, dtype)
    )
    mock_runner.token_to_kv_pool.get_value_buffer = Mock(
        return_value=_create_random_kv_buffer(10000, num_heads, v_head_dim, dtype)
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
    mock_batch.seq_lens = torch.full((num_requests,), max_seq_len, dtype=torch.int64)
    mock_batch.req_pool_indices = torch.arange(num_requests, dtype=torch.int64)

    mock_batch.req_to_token_pool = Mock()
    mock_batch.req_to_token_pool.req_to_token = torch.zeros(
        num_requests, max_seq_len, dtype=torch.int64
    )

    mock_batch.token_to_kv_pool = Mock()
    size = max_seq_len * num_requests + 100
    mock_batch.token_to_kv_pool.get_key_buffer = Mock(
        return_value=_create_random_kv_buffer(size, num_heads, head_dim, dtype)
    )
    mock_batch.token_to_kv_pool.get_value_buffer = Mock(
        return_value=_create_random_kv_buffer(size, num_heads, v_head_dim, dtype)
    )
    return mock_batch


class ExtendMockRunner:
    def __init__(self, num_heads, head_dim, kv_dtype=torch.float16):
        self.device = "cpu"
        self.model_config = argparse.Namespace(
            num_attention_heads=num_heads,
            num_hidden_layers=1,
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
        self.k_buffer = _create_random_kv_buffer(max_tokens, num_heads, head_dim, dtype)
        self.v_buffer = _create_random_kv_buffer(max_tokens, num_heads, head_dim, dtype)

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


# ----- server lifecycle constants -----

SERVER_START_TIMEOUT = 1800
POST_KILL_SLEEP = 180

# ----- memory helpers -----


def _process_tree_rss_mb(pid: int) -> float:
    """Return total RSS (MB) of process and all descendants.

    SGLang uses multiprocessing; model + KV cache live in child processes.
    Single-process RSS underreports; this captures the full process tree.
    """
    try:
        proc = psutil.Process(pid)
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return -1.0


class _MemoryTraceMonitor:
    """Record (decode_step, rss_mb) samples during a streaming request.

    Lifecycle:
      1. Call pre_request_sample() BEFORE sending the HTTP request.
      2. Call sample(step) for each SSE chunk received (decode_step 0, 1, 2, ...).
      3. Call analyze() to get the full breakdown.
    """

    def __init__(self, pid: int):
        self.pid = pid
        self._baseline_mb: float = -1.0
        self._decode_samples: List[tuple] = []  # (step, rss_mb)

    def pre_request_sample(self) -> float:
        rss = _process_tree_rss_mb(self.pid)
        self._baseline_mb = max(rss, 0.0)
        return rss

    def sample(self, decode_step: int) -> float:
        rss = _process_tree_rss_mb(self.pid)
        if rss >= 0:
            self._decode_samples.append((decode_step, rss))
        return rss

    def analyze(self) -> dict:
        if not self._decode_samples:
            return {
                "peak_mb": -1,
                "baseline_mb": self._baseline_mb,
                "prefill_rss": -1,
                "prefill_delta_mb": -1,
                "decode_stable": False,
                "stability_std": -1,
                "samples": [],
            }

        all_rss = [self._baseline_mb] + [r for _, r in self._decode_samples]
        peak_mb = max(r for r in all_rss if r >= 0)

        prefill_rss = self._decode_samples[0][1]
        baseline_mb = self._baseline_mb
        prefill_delta_mb = (prefill_rss - baseline_mb) if baseline_mb >= 0 else -1

        if len(self._decode_samples) >= 4:
            half = len(self._decode_samples) // 2
            tail_rss = [r for _, r in self._decode_samples[half:]]
            mean_r = sum(tail_rss) / len(tail_rss)
            std = (sum((x - mean_r) ** 2 for x in tail_rss) / len(tail_rss)) ** 0.5
            decode_stable = std < 0.05 * mean_r or std < 50
        else:
            std = -1
            decode_stable = False

        return {
            "peak_mb": peak_mb,
            "baseline_mb": baseline_mb,
            "prefill_rss": prefill_rss,
            "prefill_delta_mb": prefill_delta_mb,
            "decode_stable": decode_stable,
            "stability_std": std,
            "samples": self._decode_samples,
        }


class _MemoryMonitor:
    """Poll server process RSS in a background thread to capture peak during decode."""

    def __init__(self, pid: int, interval: float = 0.2):
        self.pid = pid
        self.interval = interval
        self.baseline_mb: float = -1.0
        self.peak_mb: float = -1.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self, baseline_mb: float):
        self.baseline_mb = baseline_mb
        self.peak_mb = baseline_mb
        self._thread.start()

    def stop(self) -> float:
        """Stop monitoring and return peak RSS in MB."""
        self._stop.set()
        self._thread.join()
        return self.peak_mb

    def _run(self):
        try:
            while not self._stop.is_set():
                rss = _process_tree_rss_mb(self.pid)
                if rss >= 0 and rss > self.peak_mb:
                    self.peak_mb = rss
                self._stop.wait(self.interval)
        except Exception:
            pass


# ----- KV capacity helpers -----


def _kv_bytes_per_token(kv_dtype_bytes: int) -> int:
    """Physical bytes consumed per token in the KV cache (Qwen2.5-1.5B)."""
    return _KV_N_LAYERS * 2 * _KV_N_KV_HEADS * _KV_HEAD_DIM * kv_dtype_bytes


def _max_tokens_for_memory_budget(kv_memory_mb: int, kv_dtype_bytes: int) -> int:
    """How many KV tokens fit in kv_memory_mb with given dtype."""
    return int(kv_memory_mb * 1024 * 1024 / _kv_bytes_per_token(kv_dtype_bytes))


def _estimate_memory_bandwidth_gbs(
    input_seq_len: int,
    n_decode_steps: int,
    decode_time_s: float,
    kv_dtype_bytes: int = 2,
) -> float:
    """Estimate effective memory bandwidth (GB/s) during decode phase.

    Each decode step reads model weights (dominant) + KV cache entries.
    """
    if decode_time_s <= 0 or n_decode_steps <= 0:
        return -1.0
    avg_seq = input_seq_len + n_decode_steps / 2
    kv_bytes_per_step = (
        _KV_N_LAYERS * 2 * _KV_N_KV_HEADS * _KV_HEAD_DIM * avg_seq * kv_dtype_bytes
    )
    total_bytes_per_step = _E2E_WEIGHT_BYTES_BF16 + kv_bytes_per_step
    latency_per_step_s = decode_time_s / n_decode_steps
    return total_bytes_per_step / latency_per_step_s / (1024**3)


# ----- server lifecycle helpers -----


def _port_in_use(host: str, port: int) -> bool:
    """Return True if something is listening on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(1)
            s.connect((host, port))
            return True
        except (socket.error, OSError):
            return False


def _wait_for_port_free(host: str, port: int, timeout: float = 300) -> bool:
    """Wait until port is free (connection refused). Returns True if free."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if not _port_in_use(host, port):
            return True
        time.sleep(5)
    return False


def _ensure_server_port_ready(
    base_url: str = BASE_URL, timeout: float = 300
) -> tuple[str, int]:
    """Ensure BASE_URL port is free before launching a new server."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 21000
    if _port_in_use(host, port):
        _wait_for_port_free(host, port, timeout=timeout)
    return host, port


def _kill_lingering_server():
    """Kill any leftover sglang serve processes from a previous interrupted run."""
    parsed = urlparse(BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 21000

    killed_any = False
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any("sglang" in c for c in cmdline) and (
                any("serve" in c for c in cmdline)
                or any("launch_server" in c for c in cmdline)
            ):
                print(
                    f"[cleanup] Killing lingering server pid={proc.pid}: "
                    f"{' '.join(cmdline[:4])}",
                    flush=True,
                )
                kill_process_tree(proc.pid)
                killed_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if killed_any:
        time.sleep(5)
        if not _wait_for_port_free(host, port, timeout=60):
            print(f"[cleanup] Warning: port {port} still in use after 60s", flush=True)
        else:
            print(f"[cleanup] Port {port} is free.", flush=True)


def _launch(
    model: str,
    kv_cache_dtype: str = "",
    quant_arg: str = "",
    max_total_tokens: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
):
    """Launch an SGLang server with CPU device and return the process."""
    other_args = [
        "--dtype",
        "bfloat16",
        "--device",
        "cpu",
        "--watchdog-timeout",
        "2400",
    ]
    if kv_cache_dtype:
        other_args += ["--kv-cache-dtype", kv_cache_dtype]
    if quant_arg:
        other_args += ["--quantization", quant_arg]
    if max_total_tokens is not None:
        other_args += ["--max-total-tokens", str(max_total_tokens)]
    if extra_args:
        other_args += extra_args
    return popen_launch_server(
        model, BASE_URL, timeout=SERVER_START_TIMEOUT, other_args=other_args
    )


def _select_quant_modes(
    use_bf16: bool,
    use_w8a8: bool,
    bf16_model: str = BF16_MODEL,
    w8a8_model: str = W8A8_MODEL,
) -> list[tuple[str, str, str, str]]:
    """Return list of quant benchmark modes.

    Tuple layout: (name, model, quant_arg, kv_cache_dtype).
    """
    modes: list[tuple[str, str, str, str]] = []
    if use_bf16:
        modes.extend(
            [
                ("BF16 packed", bf16_model, "", ""),
                ("INT8 KV cache", bf16_model, "", "int8"),
            ]
        )
    if use_w8a8:
        modes.extend(
            [
                ("W8A8", w8a8_model, "w8a8_int8", ""),
                ("W8A8+INT8KV", w8a8_model, "w8a8_int8", "int8"),
            ]
        )
    return modes


__all__ = [
    "BASE_URL",
    "BF16_MODEL",
    "BenchConfig",
    "ExtendMockForwardBatch",
    "ExtendMockLayer",
    "ExtendMockRunner",
    "IS_CI",
    "POST_KILL_SLEEP",
    "SERVER_START_TIMEOUT",
    "W8A8_MODEL",
    "benchmark_function",
    "create_decode_mock_forward_batch",
    "create_decode_mock_layer",
    "create_decode_mock_runner",
    "print_benchmark_result",
    "print_benchmark_summary",
    "_CTX_128",
    "_CTX_512",
    "_SHORT_PROMPT",
    "_ensure_server_port_ready",
    "_estimate_memory_bandwidth_gbs",
    "_kill_lingering_server",
    "_kv_bytes_per_token",
    "_launch",
    "_max_tokens_for_memory_budget",
    "_MemoryMonitor",
    "_MemoryTraceMonitor",
    "_port_in_use",
    "_process_tree_rss_mb",
    "_select_quant_modes",
    "_wait_for_port_free",
]
