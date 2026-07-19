# Reproduce TorchInductor RVV on Banana Pi

This guide reproduces the TorchInductor RVV experiments on Banana Pi from the
`torch_inductor_rvv` branch. It covers the container setup, model download,
one-batch benchmark, standard serving benchmark, cache state, and
generated-artifact checks. The optimized path uses regional `torch.compile`
with explicit packed-weight tensors. Full CPU graph capture is disabled. The
SGLang patch is limited to the RVV compilation policy, model-load setup,
logits dispatch, command-line options, and tests.

The one-batch workload uses batch size 1, an input length of 64 tokens, and an
output length of 8 tokens. The commands spell these out as `--batch-size 1`,
`--input-len 64`, and `--output-len 8`.

The software path is:

```text
SGLang text model
  -> opt-in RVV regional policy
  -> module-owned packed BF16 weights
  -> TorchInductor RVV GEMM/GEMV templates
  -> generated C++ shared objects
```

The user-facing switch is `--enable-cpu-rvv-inductor`. It enables regional
compilation, M=2-15 buckets, and explicit packed BF16 weights as one tested
configuration. On a memory-constrained machine, add
`--cpu-rvv-memory-budget-mib N` to cap packed side-buffer coverage. Without an
explicit cap, SGLang checks currently available host memory and keeps at least
1 GiB or 10% of physical memory, whichever is larger, outside the packing
budget. Lower-level shape-routing and row-major controls remain advanced
ablation options and are not needed for this guide.

## Validated result

The regional implementation was tested from a clean source copy on a Banana Pi
BF3 at implementation revision `0026101a02`. The documentation changes recorded
with these results did not change runtime code.
The policy suite passed 37 tests. Qwen3-0.6B,
DeepSeek-R1-Distill-Qwen-1.5B, and Llama-3.2-1B each completed one cold run that
built the Inductor cache and three new-process warm runs that reused it. All
runs used the one-batch workload defined above.

<details>
<summary>Qwen3 result and generated-artifact checks</summary>

Each model cache contained 10 shared objects and 10 C++ sources. Nine C++
sources used RVV accumulation intrinsics. The supported projection buckets did
not use `extern_kernels.mm`.

Qwen3-0.6B result:

| Cache state | Process wall | Prefill | Decode | Total |
| --- | ---: | ---: | ---: | ---: |
| Artifact-cold | 8:37.72 | 14.10 tok/s | 1.01 tok/s | 6.26 tok/s |
| Artifact-warm, three-process median [range] | 4:01.12 [3:59.37, 4:11.53] | 13.51 [13.15, 13.79] tok/s | 1.02 [0.98, 1.02] tok/s | 6.20 [5.88, 6.24] tok/s |

All runs used the immutable image digest from Step 3. The range is reported so
that a single fast prefill run is not selected as the result. Raw logs also
record process wall time, peak RSS, pack-once time, model revision, and artifact
inventory.

</details>

## 1. Requirements

Use a Banana Pi BF3 or another Linux `riscv64` system with RVV and at least
16 GiB RAM. The commands below require Git and Podman.

```bash
sudo apt-get update
sudo apt-get install -y git podman

uname -m
free -h
podman --version
```

`uname -m` must print `riscv64`.

## 2. Clone the clean branch

```bash
git clone \
  --branch torch_inductor_rvv \
  --single-branch \
  https://github.com/ChenTim1011/sglang.git
cd sglang

git status --short
git rev-parse HEAD
```

The first command should print nothing. Save the SHA from the second command
with the benchmark results.

## 3. Pull the validated runtime image

Use the immutable digest tested with this branch:

```bash
export USER_IMAGE='docker.io/juitingchen/sglang-rvv-torch-native@sha256:d33cb0b13601242efd9449d8bedf6618299cd225f3e38189c568ffb67ebe0289'
podman pull "${USER_IMAGE}"
```

The image contains:

- Python 3.13;
- NumPy 2.2.6 from the SpacemiT riscv64 package index;
- PyTorch `2.14.0a0+git45bd4f6`, built with `USE_NUMPY=ON`;
- torchvision `0.29.0a0`, rebuilt on the Banana Pi against that exact PyTorch
  wheel;
- Transformers 5.12.1, GNU `time`, GCC, and libgomp.

The image labels distinguish compiler roles: the PyTorch/ATen wheel was
cross-built with GCC 14.2; torchvision and TorchInductor generated code use
native GCC 15.2.

## 4. Create host cache directories

```bash
export SGLANG_SOURCE="$(pwd)"
export HF_CACHE="${HOME}/.cache/huggingface"
export INDUCTOR_CACHE="${HOME}/.cache/sglang-rvv/torchinductor"
export RESULT_DIR="${HOME}/sglang-rvv-results"

mkdir -p "${HF_CACHE}" "${INDUCTOR_CACHE}" "${RESULT_DIR}"
```

If the Llama checkpoint is used later, export a Hugging Face token on the host:

```bash
export HF_TOKEN='hf_your_token_here'
```

Keep the token in the shell environment. Leave it out of images, manifests, and
result logs.

## 5. Run the dependency and RVV preflight

```bash
podman run --rm -i \
  --network host \
  --security-opt label=disable \
  --volume "${SGLANG_SOURCE}:/workspace/sglang:ro" \
  --workdir /workspace/sglang \
  --env PYTHONPATH=/workspace/sglang/python:/workspace/sglang \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env ATEN_CPU_CAPABILITY=rvv \
  "${USER_IMAGE}" \
  /opt/.venv/bin/python - <<'PY'
import platform

import torch
import torchvision
from transformers.configuration_utils import PreTrainedConfig

from sglang.srt.server_args import ServerArgs

from torch._inductor import inductor_prims

assert platform.machine() == "riscv64"
assert torch.backends.cpu.get_cpu_capability().upper() == "RVV"
assert torchvision.extension._has_ops()
assert callable(inductor_prims.rvv_pack_bf16_weight)
assert callable(inductor_prims.rvv_packed_bf16_linear)
for name in (
    "enable_cpu_rvv_inductor",
    "cpu_rvv_memory_budget_mib",
):
    assert hasattr(ServerArgs, name), name
print("RVV user-environment preflight: PASS")
PY
```

The `-i` flag passes the heredoc to Python inside the container. Without it,
Python receives an empty program and Podman can exit successfully without
running any assertion. Step 9 checks the generated model kernels.

## 6. Fix the experiment inputs

The default reproducible model is Qwen3-0.6B:

```bash
export MODEL='Qwen/Qwen3-0.6B'
export MODEL_REV='c1899de289a04d12100db370d81485cdf75e47ca'
```

The baseline and candidate use the same inputs:

- real BF16 checkpoint weights;
- TP=1 and the `torch_native` attention backend;
- batch size 1, input length 64 tokens, and output length 8 tokens;
- 8 OpenMP threads;
- no full CPU graph capture;
- the same model revision and Hugging Face cache.

Define the common container arguments:

```bash
rvv_run() {
  podman run --rm \
    --network host \
    --security-opt label=disable \
    --volume "${SGLANG_SOURCE}:/workspace/sglang:ro" \
    --volume "${HF_CACHE}:/cache/huggingface" \
    --volume "${INDUCTOR_CACHE}:/cache/torchinductor" \
    --volume "${RESULT_DIR}:/workspace/results" \
    --workdir /workspace/sglang \
    --env PYTHONPATH=/workspace/sglang/python:/workspace/sglang \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env HF_HOME=/cache/huggingface \
    --env TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    --env TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    --env ATEN_CPU_CAPABILITY=rvv \
    --env OMP_NUM_THREADS=8 \
    "${USER_IMAGE}" "$@"
}
```

If `HF_TOKEN` is needed, add `--env HF_TOKEN` to the `podman run` arguments.

## 7. Run the baseline

The baseline uses the same SGLang model path but leaves the RVV regional policy
and explicit packed state disabled:

```bash
rvv_run /usr/bin/time -v \
  /opt/.venv/bin/python -m sglang.benchmark.one_batch \
  --model-path "${MODEL}" \
  --revision "${MODEL_REV}" \
  --device cpu \
  --dtype bfloat16 \
  --attention-backend torch_native \
  --batch-size 1 \
  --input-len 64 \
  --output-len 8 \
  --mem-fraction-static 0.45 \
  --max-total-tokens 2048 \
  --result-filename /workspace/results/qwen3_baseline.jsonl \
  2>&1 | tee "${RESULT_DIR}/qwen3_baseline.log"
```

## 8. Run the RVV candidate

Start with an empty candidate artifact cache:

```bash
find "${INDUCTOR_CACHE}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
```

Run the candidate once to build the artifacts:

```bash
rvv_run /usr/bin/time -v \
  /opt/.venv/bin/python -m sglang.benchmark.one_batch \
  --model-path "${MODEL}" \
  --revision "${MODEL_REV}" \
  --device cpu \
  --dtype bfloat16 \
  --attention-backend torch_native \
  --enable-cpu-rvv-inductor \
  --batch-size 1 \
  --input-len 64 \
  --output-len 8 \
  --mem-fraction-static 0.45 \
  --max-total-tokens 2048 \
  --result-filename /workspace/results/qwen3_candidate_cold.jsonl \
  2>&1 | tee "${RESULT_DIR}/qwen3_candidate_cold.log"
```

Run the exact command again in a new container with the same cache, changing
only the result filename to `qwen3_candidate_warm.jsonl`. This is the
artifact-warm measurement.

## 9. Verify generated artifacts

```bash
find "${INDUCTOR_CACHE}" -type f -name '*.so' -print
grep -R -l '__riscv_vfmacc' "${INDUCTOR_CACHE}" --include='*.cpp'
```

Expected artifacts include generated C++ with RVV accumulation and matching
shared objects. The supported projection buckets should have no
`extern_kernels.mm` fallback:

```bash
grep -R -n 'extern_kernels.mm' "${INDUCTOR_CACHE}" --include='*.py' || true
```

For reviewer-facing evidence, also disassemble the generated `.so` files with
LLVM objdump and confirm vector-length setup, vector memory operations, and
vector floating-point arithmetic. A single mnemonic match is insufficient.

## 10. Run serving concurrency 1 or 2

Start a server with the same RVV opt-in. It selects M=1 decode, M=2-15
small-batch, and M=16-128 prefill buckets from the actual flattened Linear row
count; users do not need a separate concurrency-specific routing flag.

This step checks the standard SGLang serving path. The exact-token client used
for the final Llama serving table is not public yet.

```bash
export SERVER_NAME='sglang-rvv-server'
export CONCURRENCY=2

podman run --detach \
  --name "${SERVER_NAME}" \
  --network host \
  --security-opt label=disable \
  --volume "${SGLANG_SOURCE}:/workspace/sglang:ro" \
  --volume "${HF_CACHE}:/cache/huggingface" \
  --volume "${INDUCTOR_CACHE}:/cache/torchinductor" \
  --workdir /workspace/sglang \
  --env PYTHONPATH=/workspace/sglang/python:/workspace/sglang \
  --env HF_HOME=/cache/huggingface \
  --env TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  --env TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
  --env ATEN_CPU_CAPABILITY=rvv \
  --env OMP_NUM_THREADS=8 \
  "${USER_IMAGE}" \
  /opt/.venv/bin/sglang serve \
  --model-path "${MODEL}" \
  --revision "${MODEL_REV}" \
  --device cpu \
  --dtype bfloat16 \
  --attention-backend torch_native \
  --enable-cpu-rvv-inductor \
  --mem-fraction-static 0.45 \
  --max-total-tokens 2048 \
  --host 0.0.0.0 \
  --port 30000

podman logs --follow "${SERVER_NAME}"
```

After the server is ready, run the serving benchmark from another shell:

```bash
export TOKENIZER_PATH="/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/${MODEL_REV}"

rvv_run /opt/.venv/bin/python -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --dataset-name random \
  --model "${MODEL}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --num-prompts "${CONCURRENCY}" \
  --warmup-requests "${CONCURRENCY}" \
  --random-input-len 64 \
  --random-output-len 8 \
  --random-range-ratio 1 \
  --request-rate inf \
  --max-concurrency "${CONCURRENCY}" \
  --output-file "/workspace/results/qwen3_serving_c${CONCURRENCY}.jsonl"
```

Stop and remove the server when finished:

```bash
podman stop "${SERVER_NAME}"
podman rm "${SERVER_NAME}"
```

## 11. What to record

Keep the following with each result:

```bash
git rev-parse HEAD
podman image inspect "${USER_IMAGE}" --format '{{.Id}} {{.Architecture}}'
sha256sum "${RESULT_DIR}"/*.jsonl
find "${INDUCTOR_CACHE}" -type f \( -name '*.cpp' -o -name '*.so' \) \
  -print0 | sort -z | xargs -0 -r sha256sum \
  > "${RESULT_DIR}/artifact_inventory.sha256"
```

Report process wall time, peak RSS, prefill/decode/total throughput, model
revision, thread count, cache state, and whether the run is baseline or
candidate. Compare baseline and candidate only when all other inputs match.

## 12. Known boundaries

- The SGLang branch depends on the matching custom PyTorch RVV primitives.
- The pinned image includes NumPy 2.2.6 and a PyTorch wheel built with
  `USE_NUMPY=ON`; this branch does not carry NumPy compatibility workarounds.
- The image contains torchvision built against the installed PyTorch; Step 5
  checks its C++ extension on the target.
- Explicit packed weights increase peak RSS and are opt-in. SGLang logs the
  eligible, packed, and skipped bytes. Use `--cpu-rvv-memory-budget-mib` when a
  deployment needs a stricter cap than the available-memory preflight.
- Packed tensors are derived inference state. They are absent from `state_dict`,
  and `load_state_dict` invalidates them. Code that mutates weights after policy
  installation must call `invalidate_rvv_inductor_packed_weights(model)` before
  inference; arbitrary in-place mutation is not polled on every Linear call.
- The registered adapters cover Qwen3-0.6B, DeepSeek-R1-Distill-Qwen-1.5B, and
  TorchNative Llama. Other model families use the normal SGLang path.
- Serving concurrency 2 improves aggregate capacity but can increase TTFT and
  per-request decode latency.
- Full CPU graph capture is outside this reproduction path.
