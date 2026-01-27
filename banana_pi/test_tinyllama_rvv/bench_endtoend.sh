#!/bin/bash
set -e

# SGLang benchmark for one_batch, offline_throughput,and serving  for Banana Pi (RVV)


# You can change the model path by setting MODEL_PATH
MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"

# Flags
RUN_ONE_BATCH=false
RUN_OFFLINE=false
RUN_SERVING=false

# Helper function for usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --bench-one-batch    Run 'sglang.bench_one_batch'"
    echo "  --bench-offline      Run 'sglang.bench_offline_throughput'"
    echo "  --bench-serving      Run 'sglang.launch_server' and 'sglang.bench_serving'"
    echo "  --help               Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --bench-serving"
    exit 1
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    usage
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --bench-one-batch)
            RUN_ONE_BATCH=true
            shift
            ;;
        --bench-offline)
            RUN_OFFLINE=true
            shift
            ;;
        --bench-serving)
            RUN_SERVING=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo "Using model: $MODEL_PATH"

# Setup Environment
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${script_dir}/../environment_setting.sh"

CMD_ARGS="--device cpu --mem-fraction-static 0.8 --attention-backend rvv"

if [ "$RUN_ONE_BATCH" = true ]; then
    echo "============================================================"
    echo "1. Benchmark Single Static Batch (No Server)"
    echo "============================================================"
    python3 -m sglang.bench_one_batch --model-path $MODEL_PATH --batch 1 --input-len 128 --output-len 16 --dtype float16 $CMD_ARGS
    echo ""
fi

if [ "$RUN_OFFLINE" = true ]; then
    echo "============================================================"
    echo "2. Benchmark Offline Throughput"
    echo "============================================================"
    python3 -m sglang.bench_offline_throughput --model-path $MODEL_PATH --num-prompts 5 --dtype float16 $CMD_ARGS
    echo ""
fi

if [ "$RUN_SERVING" = true ]; then
    echo "============================================================"
    echo "3. Benchmark Online Serving"
    echo "============================================================"
    echo "Starting Server..."
    python3 -m sglang.launch_server --model-path $MODEL_PATH --port 30000 --host 127.0.0.1 --dtype float16 $CMD_ARGS  &
    SERVER_PID=$!

    echo "Waiting for server ...(timeout: 1800s)"
    for i in {1..360}; do
        if curl -s http://127.0.0.1:30000/health > /dev/null; then
            echo "Server is ready!"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server died!"
            wait $SERVER_PID
            exit 1
        fi
        sleep 5
        echo -n "."
    done

    # Check if loops finished without success (optional check, but curl in loop handles success break)
    if ! curl -s http://127.0.0.1:30000/health > /dev/null; then
         echo "Server failed to start within 1800 seconds."
         kill $SERVER_PID || true
         exit 1
    fi

    echo "Running bench_serving..."
    python3 -m sglang.bench_serving --backend sglang --base-url http://127.0.0.1:30000 --num-prompts 5

    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null || true
fi

echo "Done."
