"""Benchmark GSM8K accuracy on the RVV backend."""

import argparse
import ast
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass, field

# Support direct execution from project root or this directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from _rvv_bench_utils import IS_CI

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

try:
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server

    HAS_SGLANG = True
except ImportError:
    HAS_SGLANG = False
    print("sglang not found")
    sys.exit(1)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
W8A8_MODEL = "RedHatAI/Qwen2.5-1.5B-quantized.w8a8"
LLAMA32_1B_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
BASE_URL = DEFAULT_URL_FOR_TEST
INVALID = -9999999

# Named presets for model/quantization/KV dtype, with optional eval-size overrides.
GSM8K_EXPERIMENT_CONFIGS = {
    "default": {
        "model": MODEL,
        "quantization": None,
        "kv_int8": False,
        "num_shots": 4,
    },
    "int8_kv": {
        "model": MODEL,
        "quantization": None,
        "kv_int8": True,
        "num_questions": 10,
        "num_shots": 8,
    },
    "w8a8": {
        "model": W8A8_MODEL,
        "quantization": "w8a8_int8",
        "kv_int8": False,
        "num_shots": 4,
    },
    "w8a8_int8_kv": {
        "model": W8A8_MODEL,
        "quantization": "w8a8_int8",
        "kv_int8": True,
    },
    "llama32_1b": {
        "model": LLAMA32_1B_MODEL,
        "quantization": None,
        "kv_int8": False,
        "num_shots": 8,
        "use_chat_template": True,
    },
    "llama32_1b_int8_kv": {
        "model": LLAMA32_1B_MODEL,
        "quantization": None,
        "kv_int8": True,
        "num_shots": 8,
        "use_chat_template": True,
    },
}

# Embedded subset for offline fallback (no HuggingFace download)
GSM8K_EMBEDDED = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72",
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10",
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. #### 5",
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "answer": "Maila read 12 x 2 = <<12*2=24>>24 pages today. So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday. There are 120 - 36 = <<120-36=84>>84 pages left to be read. Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages. #### 42",
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": "He writes each friend 3*2=<<3*2=6>>6 pages a week. So he writes 6*2=<<6*2=12>>12 pages every week. That means he writes 12*52=<<12*52=624>>624 pages a year. #### 624",
    },
]


@dataclass
class BenchmarkConfig:
    num_questions: int
    num_shots: int
    max_new_tokens: int
    description: str
    lines: list = field(default_factory=list)
    use_chat_template: bool = False


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    accuracy: float
    invalid_rate: float
    latency_s: float
    output_throughput: float
    num_correct: int
    num_total: int


def _get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def _get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def _get_few_shot_examples(lines, k):
    return "".join(_get_one_example(lines, i, True) + "\n\n" for i in range(k))


def _wrap_llama3_chat_template(text: str) -> str:
    """Wrap raw few-shot completion text in Llama-3 chat template tokens."""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        + text
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _load_dataset(platinum: bool, data_path: str, num_questions: int, num_shots: int):
    """Load GSM8K data. Returns (lines, source). Need num_questions + num_shots rows."""
    need = num_questions + num_shots
    if platinum:
        try:
            from datasets import load_dataset

            dataset = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
            lines = [
                {"question": item["question"], "answer": item["answer"]}
                for item in dataset
            ]
            return lines[:need], "gsm8k-platinum"
        except Exception as e:
            print(
                f"Warning: Could not load GSM8K Platinum ({e}). Using embedded subset."
            )
            return GSM8K_EMBEDDED[: min(need, len(GSM8K_EMBEDDED))], "embedded"

    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    try:
        from sglang.utils import download_and_cache_file, read_jsonl

        if not os.path.isfile(data_path):
            data_path = download_and_cache_file(url)
        lines = list(read_jsonl(data_path))[:need]
        return lines, "gsm8k"
    except Exception as e:
        print(
            f"Warning: Could not load GSM8K from {data_path} ({e}). Using embedded subset."
        )
        return GSM8K_EMBEDDED[: min(need, len(GSM8K_EMBEDDED))], "embedded"


def _kill_port(port: int) -> None:
    """Kill any process listening on the given TCP port (no external tools needed)."""
    import os
    import signal

    try:
        # /proc/net/tcp6 covers both IPv4-mapped and IPv6; fall back to /proc/net/tcp
        for tcp_file in ("/proc/net/tcp6", "/proc/net/tcp"):
            try:
                with open(tcp_file) as f:
                    lines = f.readlines()[1:]  # skip header
            except FileNotFoundError:
                continue
            hex_port = f"{port:04X}"
            for line in lines:
                parts = line.split()
                # local_address field is "addr:port"; state 0A = LISTEN
                if len(parts) < 10 or parts[3] != "0A":
                    continue
                if parts[1].split(":")[1].upper() == hex_port:
                    inode = parts[9]
                    # Find the PID that owns this inode
                    for pid in os.listdir("/proc"):
                        if not pid.isdigit():
                            continue
                        try:
                            fd_dir = f"/proc/{pid}/fd"
                            for fd in os.listdir(fd_dir):
                                link = os.readlink(f"{fd_dir}/{fd}")
                                if f"socket:[{inode}]" in link:
                                    os.kill(int(pid), signal.SIGKILL)
                        except (PermissionError, FileNotFoundError, ProcessLookupError):
                            pass
    except Exception:
        pass  # best-effort; if it fails the server launch will report the conflict


def run_benchmark(
    config: BenchmarkConfig,
    model: str,
    quantization: str = None,
    kv_int8: bool = False,
):
    """Launch RVV server, run GSM8K questions, kill server, return BenchmarkResult."""
    other_args = [
        "--attention-backend",
        "rvv",
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.5",
        "--max-total-tokens",
        "2048",
        "--device",
        "cpu",
    ]
    if quantization:
        other_args += ["--quantization", quantization]
    if kv_int8:
        other_args += ["--kv-cache-dtype", "int8"]
    # Kill any stale server left from a previous crashed run.
    import re as _re

    _m = _re.search(r":(\d+)", BASE_URL)
    if _m:
        _port = int(_m.group(1))
        _kill_port(_port)

    process = popen_launch_server(
        model,
        BASE_URL,
        timeout=1800,
        other_args=other_args,
    )

    try:
        import sglang as sgl
        from sglang.lang.api import set_default_backend

        set_default_backend(sgl.RuntimeEndpoint(BASE_URL))

        lines = config.lines
        n = config.num_questions
        few_shot_prefix = _get_few_shot_examples(lines, config.num_shots)
        questions = [_get_one_example(lines, i, False) for i in range(n)]
        labels = [_get_answer_value(lines[i]["answer"]) for i in range(n)]

        if config.use_chat_template:
            stop_tokens = ["<|eot_id|>", "<|start_header_id|>", "Question"]
        else:
            stop_tokens = ["Question", "Assistant:", "<|separator|>"]

        @sgl.function
        def few_shot_gsm8k(s, question):
            prompt = few_shot_prefix + question
            if config.use_chat_template:
                prompt = _wrap_llama3_chat_template(prompt)
            s += prompt
            s += sgl.gen(
                "answer",
                max_tokens=config.max_new_tokens,
                stop=stop_tokens,
            )

        arguments = [{"question": q} for q in questions]
        start = time.perf_counter()
        states = few_shot_gsm8k.run_batch(
            arguments, temperature=0, num_threads=1, progress_bar=False
        )
        latency = time.perf_counter() - start

        preds = [_get_answer_value(s["answer"]) for s in states]
        num_correct = sum(p == l for p, l in zip(preds, labels))
        accuracy = num_correct / len(labels)
        invalid_rate = sum(p == INVALID for p in preds) / len(preds)
        num_output_tokens = sum(
            s.get_meta_info("answer")["completion_tokens"] for s in states
        )
        output_throughput = num_output_tokens / latency if latency > 0 else 0

    finally:
        kill_process_tree(process.pid)

    return BenchmarkResult(
        config=config,
        accuracy=accuracy,
        invalid_rate=invalid_rate,
        latency_s=latency,
        output_throughput=output_throughput,
        num_correct=num_correct,
        num_total=len(labels),
    )


def print_result(result: BenchmarkResult):
    c = result.config
    passed = result.accuracy >= 0.20
    status = "PASS" if passed else "FAIL"
    print(f"  {c.description}")
    print(
        f"    Accuracy:   {result.accuracy:.3f} ({result.accuracy:.1%})  [{status}]  {result.num_correct}/{result.num_total} correct"
    )
    print(f"    Invalid:    {result.invalid_rate:.3f} ({result.invalid_rate:.1%})")
    print(f"    Latency:    {result.latency_s:.3f} s")
    print(f"    Throughput: {result.output_throughput:.3f} tok/s")
    print()


def save_result(
    result: BenchmarkResult,
    model: str,
    quantization: str,
    kv_int8: bool,
    output_dir="/tmp",
):
    result_file = os.path.join(output_dir, "rvv_gsm8k_results.json")
    data = {
        "task": "gsm8k-rvv",
        "backend": "rvv",
        "model": model,
        "quantization": quantization or "none",
        "kv_cache_dtype": "int8" if kv_int8 else "bfloat16",
        "num_questions": result.config.num_questions,
        "num_shots": result.config.num_shots,
        "accuracy": round(result.accuracy, 3),
        "invalid_rate": round(result.invalid_rate, 3),
        "latency_seconds": round(result.latency_s, 3),
        "output_throughput_tokens_per_sec": round(result.output_throughput, 3),
    }
    with open(result_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to: {result_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RVV backend accuracy on GSM8K (Banana Pi K1 scale)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark (2q, 0-shot)"
    )
    parser.add_argument("--save", action="store_true", help="Save results to /tmp")
    parser.add_argument(
        "--platinum",
        action="store_true",
        help="Use GSM8K Platinum dataset (HuggingFace). Fallback to embedded if offline.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of questions (default 5 for K1). Official uses 200.",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=1,
        help="Few-shot examples (default 1 for K1). Official uses 5.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens per answer (default 256 for K1).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="test.jsonl",
        help="Path to test.jsonl when not using --platinum.",
    )
    parser.add_argument(
        "--kv-int8",
        action="store_true",
        help="Use INT8 KV cache (RVV decode/extend int8 path). Requires RISC-V with RVV.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["w8a8_int8"],
        default=None,
        help="Weight quantization mode. Use w8a8_int8 for RedHatAI W8A8 checkpoints.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override the model path used for the benchmark.",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=list(GSM8K_EXPERIMENT_CONFIGS),
        default=None,
        help="Named experiment: default, int8_kv, w8a8, w8a8_int8_kv.",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Wrap prompts with model chat template (auto-set for llama32_1b config).",
    )
    args = parser.parse_args()

    if args.quick:
        num_questions, num_shots, max_tokens = 2, 0, 256
    else:
        num_questions = args.num_questions
        num_shots = args.num_shots
        max_tokens = args.max_new_tokens

    run_model = MODEL
    run_quantization = args.quantization
    run_kv_int8 = args.kv_int8
    run_chat_template = args.use_chat_template
    if args.config:
        cfg = GSM8K_EXPERIMENT_CONFIGS[args.config]
        run_model = cfg["model"]
        run_quantization = cfg.get("quantization", run_quantization)
        run_kv_int8 = cfg["kv_int8"]
        run_chat_template = cfg.get("use_chat_template", run_chat_template)
        if "num_questions" in cfg:
            num_questions = cfg["num_questions"]
        if "num_shots" in cfg:
            num_shots = cfg["num_shots"]

    # Explicit CLI flags override named preset fields.
    if args.model_path:
        run_model = args.model_path
    if args.quantization:
        run_quantization = args.quantization
    if args.kv_int8:
        run_kv_int8 = True
    if args.use_chat_template:
        run_chat_template = True

    lines, source = _load_dataset(
        args.platinum, args.data_path, num_questions, num_shots
    )
    num_questions = min(num_questions, len(lines) - num_shots)
    if num_questions <= 0:
        print("Error: Not enough data for num_questions + num_shots.")
        sys.exit(1)
    lines = lines[: num_shots + num_questions]

    model_short = run_model.split("/")[-1]
    if run_quantization:
        model_short = f"{model_short} [{run_quantization}]"
    config = BenchmarkConfig(
        num_questions=num_questions,
        num_shots=num_shots,
        max_new_tokens=max_tokens,
        description=f"{model_short} ({num_questions}q, {num_shots}-shot, {source})",
        lines=lines,
        use_chat_template=run_chat_template,
    )

    print("=" * 60)
    print("RVV GSM8K Accuracy Benchmark")
    print("=" * 60)
    print(f"Platform: {platform.machine()}")
    print(f"Model   : {run_model}")
    print(f"Quant   : {run_quantization or 'none'}")
    print(f"CI Mode : {IS_CI}")
    print(f"Data    : {source} ({num_questions} questions)")
    if run_kv_int8:
        print(
            "KV cache: int8 (RVV decode_attention_int8_cpu / extend_attention_int8_cpu)"
        )
    else:
        print("KV cache: bfloat16 (default)")
    if args.config:
        print(f"Config  : {args.config}")
    print("-" * 60)

    try:
        result = run_benchmark(
            config,
            model=run_model,
            quantization=run_quantization,
            kv_int8=run_kv_int8,
        )
        print_result(result)
        if args.save:
            save_result(
                result,
                model=run_model,
                quantization=run_quantization,
                kv_int8=run_kv_int8,
            )
    except Exception as e:
        print(f"FAILED: {e}")
        raise

    print("=" * 60)


if __name__ == "__main__":
    main()
