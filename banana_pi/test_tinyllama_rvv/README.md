# TinyLlama RVV Test Suite

This directory contains scripts for benchmarking and testing the TinyLlama 1.1B model using the high-performance RISC-V Vector (RVV) backend for SGLang.

---

## How to Use (Tutorial)

### Step 1: Initialize the Environment

After connecting to the Banana Pi, you need to source the environment settings script to configure paths and library variables.

```bash
source ../environment_setting.sh
```
*Tip: This sets up `LD_PRELOAD` for OpenMP and activates the Python virtual environment.*

### Step 2: Manage Environment (Stubs & Verification)

We provide a unified tool to install necessary "stub" packages (like `triton` and `vllm` which are simulated on RISC-V) and verify that your system is ready.

```bash
# Verify environment and install missing stubs automatically
python3 manage_rvv_env.py
```

If you see **"All verification steps passed!"**, you are ready to go.

### Step 3: Run Interactive Chat

Test the model in real-time with an interactive chat interface. This script handles the SGLang server lifecycle automatically.

```bash
python3 test_tinyllama_rvv.py
```

### Step 4: Run Performance Benchmarks

Compare the performance of the optimized RVV backend against the standard PyTorch native backend.

```bash
# Run benchmark with default settings
python3 benchmark_rvv_backends.py

# Compare specific backends
python3 benchmark_rvv_backends.py --backend rvv
python3 benchmark_rvv_backends.py --backend torch_native
```

### Step 5: Run Full End-to-End Suite

Run the comprehensive end-to-end benchmark script (moved from `tests_rvv_kernels/`) which supports one-batch, offline, and serving benchmarks.

```bash
# Run serving benchmark
./bench_endtoend.sh --bench-serving
```

---

## File Descriptions

### Core Scripts

*   **`manage_rvv_env.py`**
    **Purpose**: The all-in-one environment manager.
    **Function**: It installs "stub" packages (simulated versions of `triton`, `vllm`, `torchvision` for RISC-V compatibility) and runs a comprehensive health check on your environment (checking imports, libomp, and config generation).
    **Usage**: `python3 manage_rvv_env.py`

*   **`benchmark_rvv_backends.py`**
    **Purpose**: Performance benchmarking tool.
    **Function**: Launches the SGLang server and measures key metrics: Time To First Token (TTFT), Token Generation Throughput, and End-to-End Latency. It can compare RVV vs. Torch Native backends.
    **Usage**: `python3 benchmark_rvv_backends.py [options]`

*   **`test_tinyllama_rvv.py`**
    **Purpose**: Interactive demo.
    **Function**: A chat interface that launches the server and allows you to talk to TinyLlama. It serves as an end-to-end integration test.
    **Usage**: `python3 test_tinyllama_rvv.py`

*   **`launch_server_rvv.py`**
    **Purpose**: RISC-V specific server launcher.
    **Function**: A wrapper used by other scripts to launch the SGLang server. It handles critical setup like monkey-patching `transformers` to avoid circular imports on RISC-V and ensuring the `rvv` backend is correctly registered.

*   **`bench_endtoend.sh`**
    **Purpose**: Comprehensive end-to-end benchmark runner.
    **Function**: Runs single batch, offline throughput, and serving benchmarks. Supports command line flags nicely.
    **Usage**: `./bench_endtoend.sh --bench-serving`

### Helper Files

*   **`../environment_setting.sh`**
    **Purpose**: Shell environment setup.
    **Function**: Sets `LD_PRELOAD`, `LD_LIBRARY_PATH`, and activates the `venv_sglang` virtual environment. Use `source` to run it.

*   **`stubs/`**
    **Purpose**: Compatibility layer.
    **Function**: Contains the source code for the stub packages (`triton_stub.py`, `vllm_stub.py`, etc.) installed by `manage_rvv_env.py`.

### Configuration
