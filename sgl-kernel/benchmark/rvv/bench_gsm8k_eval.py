# Benchmark RVV backend accuracy on GSM8K (5-question subset, 1-shot)

import argparse
import ast
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass

from utils import IS_CI

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

try:
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server

    HAS_SGLANG = True
except ImportError:
    HAS_SGLANG = False
    print("sglang not found")
    sys.exit(1)


MODEL = "meta-llama/Llama-3.2-1B-Instruct"
BASE_URL = DEFAULT_URL_FOR_TEST
INVALID = -9999999

# Hardcoded to avoid dataset download overhead on BPI-F3
GSM8K_QUESTIONS = [
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


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    accuracy: float
    invalid_rate: float
    latency_s: float
    output_throughput: float  # tokens/s
    num_correct: int
    num_total: int


STANDARD_CONFIGS = [
    BenchmarkConfig(5, 1, 512, "Llama-3.2-1B-Instruct (5q, 1-shot)"),
]

CI_CONFIGS = [
    BenchmarkConfig(2, 0, 256, "Llama-3.2-1B-Instruct (2q, 0-shot) CI"),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = STANDARD_CONFIGS


def _get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def _format_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def _build_few_shot_prefix(lines, k):
    return "".join(_format_example(lines, i, True) + "\n\n" for i in range(k))


def run_benchmark(config):
    """Launch RVV server, run GSM8K questions, kill server, return BenchmarkResult."""
    process = popen_launch_server(
        MODEL,
        BASE_URL,
        timeout=1800,
        other_args=[
            "--attention-backend",
            "rvv",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.5",
            "--max-total-tokens",
            "2048",
            "--device",
            "cpu",
        ],
    )

    try:
        import sglang as sgl
        from sglang.lang.api import set_default_backend

        set_default_backend(sgl.RuntimeEndpoint(BASE_URL))

        lines = GSM8K_QUESTIONS[: config.num_questions]
        few_shot_prefix = _build_few_shot_prefix(lines, config.num_shots)
        questions = [_format_example(lines, i, False) for i in range(len(lines))]
        labels = [_get_answer_value(lines[i]["answer"]) for i in range(len(lines))]

        @sgl.function
        def few_shot_gsm8k(s, question):
            s += few_shot_prefix + question
            s += sgl.gen(
                "answer",
                max_tokens=config.max_new_tokens,
                stop=["Question", "Assistant:", "<|separator|>"],
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


def print_result(result):
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


def save_result(result, output_dir="/tmp"):
    result_file = os.path.join(output_dir, "rvv_gsm8k_results.json")
    data = {
        "task": "gsm8k-rvv",
        "backend": "rvv",
        "model": MODEL,
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
        description="Benchmark RVV backend accuracy on GSM8K"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--save", action="store_true", help="Save results to /tmp")
    args = parser.parse_args()

    configs = CI_CONFIGS if args.quick else ALL_CONFIGS

    print("=" * 60)
    print("RVV GSM8K Accuracy Benchmark")
    print("=" * 60)
    print(f"Platform: {platform.machine()}")
    print(f"Model   : {MODEL}")
    print(f"CI Mode : {IS_CI}")
    print("-" * 60)

    for config in configs:
        print(f"\nRunning: {config.description}")
        try:
            result = run_benchmark(config)
            print_result(result)
            if args.save:
                save_result(result)
        except Exception as e:
            print(f"FAILED {config.description}: {e}")

    print("=" * 60)


if __name__ == "__main__":
    main()
