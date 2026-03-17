#!/usr/bin/env bash
set -euo pipefail

# Run official SGLang benchmark tools across RVV quant modes:
#   1) Native PyTorch (torch_native)
#   2) BF16
#   3) BF16 + INT8 KV cache
#   4) W8A8
#   5) W8A8 + INT8 KV cache
#
# Tools mapping:
# - bench_serving             : online serving (async, realistic)
# - bench_one_batch_server    : single-batch over HTTP
# - bench_offline_throughput  : in-process engine throughput
# - bench_one_batch           : low-level static batch latency
#
# Usage:
#   bash sgl-kernel/benchmark/rvv/run_sglang_rvv_benchmarks.sh
#   bash sgl-kernel/benchmark/rvv/run_sglang_rvv_benchmarks.sh --tool serving
#   bash sgl-kernel/benchmark/rvv/run_sglang_rvv_benchmarks.sh --tool all --quick
#   bash sgl-kernel/benchmark/rvv/run_sglang_rvv_benchmarks.sh --tool one-batch --batch-sizes "1 2 4 8 16"
#   bash sgl-kernel/benchmark/rvv/run_sglang_rvv_benchmarks.sh --tool all --server-timeout 600
#   bash sgl-kernel/benchmark/rvv/run_sglang_rvv_benchmarks.sh --tool all --native-only
#   bash sgl-kernel/benchmark/rvv/run_sglang_rvv_benchmarks.sh --tool all --skip-native
#
# Optional env:
#   PYTHON_BIN=python
#   BASE_URL=http://127.0.0.1:30000
#   SERVER_READY_TIMEOUT=600
#   SERVER_MEM_FRACTION_STATIC=0.45
#   SERVER_MAX_RUNNING_REQUESTS=1
#   SERVER_MAX_TOTAL_TOKENS=3072
#   SERVING_REQUEST_RATE=0.15

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
TOOL="serving"   # serving | one-batch-server | offline | one-batch | all
QUICK=0
NATIVE_ONLY=0
SKIP_NATIVE=0
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"
SERVER_MEM_FRACTION_STATIC="${SERVER_MEM_FRACTION_STATIC:-0.45}"
SERVER_MAX_RUNNING_REQUESTS="${SERVER_MAX_RUNNING_REQUESTS:-1}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-3072}"
SERVING_REQUEST_RATE="${SERVING_REQUEST_RATE:-0.15}"

# RISC-V low-memory stable defaults.
NUM_PROMPTS=12
MAX_CONCURRENCY=1
BATCH_SIZES="16"
INPUT_LEN=256
OUTPUT_LEN=16
SERVER_LOG_PATH=""
NATIVE_GREEDY_EXTRA_REQUEST_BODY='{"temperature":0.0,"top_k":1,"top_p":1.0,"repetition_penalty":1.0,"ignore_eos":true}'

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool)
      TOOL="$2"
      shift 2
      ;;
    --quick)
      QUICK=1
      shift
      ;;
    --native-only)
      NATIVE_ONLY=1
      shift
      ;;
    --skip-native)
      SKIP_NATIVE=1
      shift
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --batch-sizes)
      shift
      if [[ $# -eq 0 || "$1" == --* ]]; then
        echo "--batch-sizes requires at least one value"
        exit 1
      fi
      local_batch_sizes=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        local_batch_sizes+=("$1")
        shift
      done
      BATCH_SIZES="${local_batch_sizes[*]}"
      ;;
    --server-timeout)
      SERVER_READY_TIMEOUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ "$QUICK" -eq 1 ]]; then
  NUM_PROMPTS=12
  MAX_CONCURRENCY=1
  BATCH_SIZES="16"
  INPUT_LEN=256
  OUTPUT_LEN=16
fi

if [[ ! "$TOOL" =~ ^(serving|one-batch-server|offline|one-batch|all)$ ]]; then
  echo "Invalid --tool: $TOOL"
  exit 1
fi

if [[ "$NATIVE_ONLY" -eq 1 && "$SKIP_NATIVE" -eq 1 ]]; then
  echo "--native-only and --skip-native are mutually exclusive"
  exit 1
fi

if [[ "$BASE_URL" =~ ^https?://[^:]+:([0-9]+)$ ]]; then
  PORT="${BASH_REMATCH[1]}"
else
  echo "BASE_URL must include host and port, e.g. http://127.0.0.1:30000"
  exit 1
fi

MODE_NAMES=(
  "PYTORCH_NATIVE"
  "BF16"
  "BF16+INT8KV"
  "W8A8"
  "W8A8+INT8KV"
)
MODE_MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
)
MODE_QUANTS=(
  ""
  ""
  ""
  "w8a8_int8"
  "w8a8_int8"
)
MODE_KVS=(
  ""
  ""
  "int8"
  ""
  "int8"
)
MODE_ATTN_BACKENDS=(
  "torch_native"
  "rvv"
  "rvv"
  "rvv"
  "rvv"
)
MODE_DISABLE_RVV_KERNELS=(
  "1"
  ""
  ""
  ""
  ""
)

SERVER_PID=""

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

wait_server_ready() {
  local timeout_sec="${SERVER_READY_TIMEOUT}"
  local retries="${timeout_sec}"
  local i
  for ((i=0; i<retries; i++)); do
    if curl -sf "${BASE_URL}/model_info" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Server not ready at ${BASE_URL} after ${retries}s"
  if [[ -n "${SERVER_LOG_PATH}" && -f "${SERVER_LOG_PATH}" ]]; then
    echo "---- server log tail: ${SERVER_LOG_PATH} ----"
    tail -n 120 "${SERVER_LOG_PATH}" || true
    echo "---- end server log tail ----"
  fi
  return 1
}

start_server_for_mode() {
  local model="$1"
  local quant="$2"
  local kv_dtype="$3"
  local attention_backend="$4"
  local disable_rvv_kernels="$5"

  cleanup_server

  local cmd=(
    "${PYTHON_BIN}" -m sglang.launch_server
    --model-path "${model}"
    --device cpu
    --dtype bfloat16
    --attention-backend "${attention_backend}"
    --mem-fraction-static "${SERVER_MEM_FRACTION_STATIC}"
    --max-running-requests "${SERVER_MAX_RUNNING_REQUESTS}"
    --max-total-tokens "${SERVER_MAX_TOTAL_TOKENS}"
    --host 127.0.0.1
    --port "${PORT}"
  )

  if [[ -n "${quant}" ]]; then
    cmd+=(--quantization "${quant}")
  fi
  if [[ -n "${kv_dtype}" ]]; then
    cmd+=(--kv-cache-dtype "${kv_dtype}")
  fi

  SERVER_LOG_PATH="/tmp/rvv_official_server_${PORT}.log"
  echo "[server] ${cmd[*]}"
  echo "[server] wait-timeout=${SERVER_READY_TIMEOUT}s log=${SERVER_LOG_PATH}"
  if [[ -n "${disable_rvv_kernels}" ]]; then
    SGLANG_DISABLE_RVV_KERNELS="${disable_rvv_kernels}" "${cmd[@]}" >"${SERVER_LOG_PATH}" 2>&1 &
  else
    "${cmd[@]}" >"${SERVER_LOG_PATH}" 2>&1 &
  fi
  SERVER_PID=$!
  wait_server_ready
}

run_bench_serving() {
  local model="$1"
  local num_prompts="$2"
  local max_concurrency="$3"
  local input_len="$4"
  local output_len="$5"
  local disable_rvv_kernels="$6"
  local extra_request_body="$7"
  local request_rate="$8"

  local cmd=(
    "${PYTHON_BIN}" -m sglang.bench_serving
    --backend sglang
    --base-url "${BASE_URL}"
    --model "${model}"
    --dataset-name random
    --num-prompts "${num_prompts}"
    --max-concurrency "${max_concurrency}"
    --request-rate "${request_rate}"
    --random-input-len "${input_len}"
    --random-output-len "${output_len}"
  )
  if [[ -n "${extra_request_body}" ]]; then
    cmd+=(--extra-request-body "${extra_request_body}")
  fi

  echo "[bench_serving] mode-model=${model}"
  if [[ -n "${disable_rvv_kernels}" ]]; then
    SGLANG_DISABLE_RVV_KERNELS="${disable_rvv_kernels}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

run_bench_one_batch_server() {
  local batch_sizes="$1"
  local input_len="$2"
  local output_len="$3"
  local disable_rvv_kernels="$4"
  local -a bs_list
  read -r -a bs_list <<< "${batch_sizes}"
  local cmd=(
    "${PYTHON_BIN}" -m sglang.bench_one_batch_server
    --model None
    --base-url "${BASE_URL}"
    --batch-size "${bs_list[@]}"
    --input-len "${input_len}"
    --output-len "${output_len}"
    --temperature 0.0
  )

  echo "[bench_one_batch_server] batch-sizes=${batch_sizes}"
  if [[ -n "${disable_rvv_kernels}" ]]; then
    SGLANG_DISABLE_RVV_KERNELS="${disable_rvv_kernels}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

run_bench_offline_throughput() {
  local model="$1"
  local quant="$2"
  local kv_dtype="$3"
  local attention_backend="$4"
  local num_prompts="$5"
  local input_len="$6"
  local output_len="$7"
  local disable_rvv_kernels="$8"

  local cmd=(
    "${PYTHON_BIN}" -m sglang.bench_offline_throughput
    --model-path "${model}"
    --device cpu
    --dtype bfloat16
    --attention-backend "${attention_backend}"
    --dataset-name random
    --num-prompts "${num_prompts}"
    --random-input-len "${input_len}"
    --random-output-len "${output_len}"
  )
  if [[ -n "${quant}" ]]; then
    cmd+=(--quantization "${quant}")
  fi
  if [[ -n "${kv_dtype}" ]]; then
    cmd+=(--kv-cache-dtype "${kv_dtype}")
  fi

  echo "[bench_offline_throughput] ${cmd[*]}"
  if [[ -n "${disable_rvv_kernels}" ]]; then
    SGLANG_DISABLE_RVV_KERNELS="${disable_rvv_kernels}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

run_bench_one_batch() {
  local model="$1"
  local quant="$2"
  local kv_dtype="$3"
  local attention_backend="$4"
  local batch_sizes="$5"
  local input_len="$6"
  local output_len="$7"
  local disable_rvv_kernels="$8"
  local -a bs_list
  read -r -a bs_list <<< "${batch_sizes}"

  local cmd=(
    "${PYTHON_BIN}" -m sglang.bench_one_batch
    --model-path "${model}"
    --device cpu
    --dtype bfloat16
    --attention-backend "${attention_backend}"
    --batch "${bs_list[@]}"
    --input-len "${input_len}"
    --output-len "${output_len}"
  )
  if [[ -n "${quant}" ]]; then
    cmd+=(--quantization "${quant}")
  fi
  if [[ -n "${kv_dtype}" ]]; then
    cmd+=(--kv-cache-dtype "${kv_dtype}")
  fi

  echo "[bench_one_batch] ${cmd[*]}"
  if [[ -n "${disable_rvv_kernels}" ]]; then
    SGLANG_DISABLE_RVV_KERNELS="${disable_rvv_kernels}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

trap cleanup_server EXIT

for i in "${!MODE_NAMES[@]}"; do
  name="${MODE_NAMES[$i]}"

  if [[ "$NATIVE_ONLY" -eq 1 && "${name}" != "PYTORCH_NATIVE" ]]; then
    continue
  fi
  if [[ "$SKIP_NATIVE" -eq 1 && "${name}" == "PYTORCH_NATIVE" ]]; then
    continue
  fi

  model="${MODE_MODELS[$i]}"
  quant="${MODE_QUANTS[$i]}"
  kv_dtype="${MODE_KVS[$i]}"
  attention_backend="${MODE_ATTN_BACKENDS[$i]}"
  disable_rvv_kernels="${MODE_DISABLE_RVV_KERNELS[$i]}"

  mode_num_prompts="${NUM_PROMPTS}"
  mode_max_concurrency="${MAX_CONCURRENCY}"
  mode_batch_sizes="${BATCH_SIZES}"
  mode_input_len="${INPUT_LEN}"
  mode_output_len="${OUTPUT_LEN}"
  mode_extra_request_body=""
  mode_request_rate="${SERVING_REQUEST_RATE}"

  # Native PyTorch mode is much slower; constrain workload and use explicit greedy sampling.
  if [[ "${name}" == "PYTORCH_NATIVE" ]]; then
    mode_num_prompts=12
    mode_max_concurrency=1
    mode_batch_sizes="16"
    mode_extra_request_body="${NATIVE_GREEDY_EXTRA_REQUEST_BODY}"
    mode_request_rate="0.15"
  fi

  echo
  echo "============================================================"
  echo "Mode: ${name}"
  echo "  model=${model}"
  echo "  quant=${quant:-none}"
  echo "  kv_cache_dtype=${kv_dtype:-bf16}"
  echo "  attention_backend=${attention_backend}"
  echo "  disable_rvv_kernels=${disable_rvv_kernels:-0}"
  echo "  batch_sizes=${mode_batch_sizes}"
  echo "  request_rate=${mode_request_rate}"
  echo "  mem_fraction_static=${SERVER_MEM_FRACTION_STATIC}"
  echo "  max_running_requests=${SERVER_MAX_RUNNING_REQUESTS}"
  echo "  max_total_tokens=${SERVER_MAX_TOTAL_TOKENS}"
  echo "============================================================"

  if [[ "$TOOL" == "serving" || "$TOOL" == "one-batch-server" || "$TOOL" == "all" ]]; then
    start_server_for_mode "${model}" "${quant}" "${kv_dtype}" "${attention_backend}" "${disable_rvv_kernels}"

    if [[ "$TOOL" == "serving" || "$TOOL" == "all" ]]; then
      run_bench_serving "${model}" "${mode_num_prompts}" "${mode_max_concurrency}" "${mode_input_len}" "${mode_output_len}" "${disable_rvv_kernels}" "${mode_extra_request_body}" "${mode_request_rate}"
    fi

    if [[ "$TOOL" == "one-batch-server" || "$TOOL" == "all" ]]; then
      run_bench_one_batch_server "${mode_batch_sizes}" "${mode_input_len}" "${mode_output_len}" "${disable_rvv_kernels}"
    fi

    cleanup_server
  fi

  if [[ "$TOOL" == "offline" || "$TOOL" == "all" ]]; then
    run_bench_offline_throughput "${model}" "${quant}" "${kv_dtype}" "${attention_backend}" "${mode_num_prompts}" "${mode_input_len}" "${mode_output_len}" "${disable_rvv_kernels}"
  fi

  if [[ "$TOOL" == "one-batch" || "$TOOL" == "all" ]]; then
    run_bench_one_batch "${model}" "${quant}" "${kv_dtype}" "${attention_backend}" "${mode_batch_sizes}" "${mode_input_len}" "${mode_output_len}" "${disable_rvv_kernels}"
  fi

done

echo

echo "[DONE] Official RVV benchmark run finished for ${#MODE_NAMES[@]} modes."
