# TinyLlama RVV Test Directory

This directory contains scripts for testing and benchmarking TinyLlama 1.1B model with the RVV backend.

## Files

### `benchmark_rvv_backends.py`
**Purpose**: Compare RVV vs TORCH_NATIVE backend performance for TinyLlama 1.1B

**What it does**:
- Launches SGLang server with specified backend (RVV or TORCH_NATIVE)
- Measures SGLang-style metrics:
  - TTFT (Time To First Token): Prefill latency
  - Token generation throughput: tokens/second
  - Decode latency: Average time per token
  - End-to-end latency: Total generation time
- Compares backends and shows speedup (RVV vs TORCH_NATIVE)

**Usage**:
```bash
# Compare both backends (default)
python benchmark_rvv_backends.py

# Test only RVV or TORCH_NATIVE
python benchmark_rvv_backends.py --backend rvv
python benchmark_rvv_backends.py --backend torch_native

# Custom parameters
python benchmark_rvv_backends.py --warmup 3 --num-runs 10 --max-tokens 100

# Restart server for each backend (clean environment)
python benchmark_rvv_backends.py --restart

# Save results to JSON
python benchmark_rvv_backends.py --output results.json
```

**Output**: Shows comparison table with speedup metrics (e.g., "RVV is 1.25x faster than TORCH_NATIVE")

---

### `test_tinyllama_rvv.py`
**Purpose**: Interactive chat interface with TinyLlama using RVV backend

**What it does**:
- Checks and launches SGLang server with RVV backend
- Provides interactive chat interface
- Manages server lifecycle (startup, shutdown)
- Validates environment (libomp, dependencies)

**Usage**:
```bash
python test_tinyllama_rvv.py
```

**Features**:
- Automatic server management
- Environment validation
- Interactive chat loop
- Graceful shutdown handling

---

### `test_environment.py`
**Purpose**: Validate environment setup for TinyLlama testing

**What it checks**:
- Python dependencies
- SGLang installation
- RVV backend availability
- System configuration

**Usage**:
```bash
python test_environment.py
```

---

### Configuration Files
**Purpose**: SGLang server configuration files (generated dynamically)

**How it works**:
- `benchmark_rvv_backends.py` generates config files dynamically via `create_config_file()`
- Config files are created as `config_benchmark_{backend}.yaml` (e.g., `config_benchmark_rvv.yaml`)
- `test_tinyllama_rvv.py` also uses dynamic config generation

**Configuration** (generated):
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Device: CPU
- Attention backend: RVV or TORCH_NATIVE (depending on test)
- Memory settings optimized for Banana Pi (16GB RAM)
- KV cache: FP16 (auto)

**Note**: No static `config_rvv.yaml` file is needed - configs are generated on-the-fly

---

### `triton_stub.py` & `vllm_stub.py`
**Purpose**: Stub modules for dependencies not available on RISC-V

**What they do**:
- Provide minimal implementations of `triton` and `vllm` modules
- Prevent import errors when these modules are not available
- Required for SGLang to run on RISC-V hardware

**Usage**: Automatically loaded by test scripts

---

### Dependencies
**Purpose**: Python dependencies for TinyLlama testing

**Note**: All dependencies are automatically installed by `setup_banana_pi.sh`. The dependencies include:
- `requests` - HTTP client for API calls
- `psutil` - System and process utilities
- `pyyaml` - YAML parser for configuration files
- And many other SGLang dependencies

**Manual installation** (if needed):
```bash
pip install requests psutil pyyaml
```

---

## Quick Start

**⚠️ Important: Before running any scripts, set up the environment:**

```bash
# Set OpenMP library paths
export LD_PRELOAD=~/.local/lib/libomp.so
export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH

# Activate virtual environment
source ~/.local_riscv_env/workspace/venv_sglang/bin/activate
```

1. **Validate environment**:
   ```bash
   python test_environment.py
   ```

2. **Run backend comparison**:
   ```bash
   python benchmark_rvv_backends.py
   ```

3. **Interactive chat**:
   ```bash
   python test_tinyllama_rvv.py
   ```

---

## Notes

- All scripts require SGLang to be installed and configured
- RVV backend must be compiled and available in SGLang
- Server management scripts handle OpenMP library paths automatically
- Benchmark scripts create temporary config files for each backend
- Results can be saved to JSON for further analysis

---

## Troubleshooting

- **Server fails to start**: Check logs in `benchmark_*.log` files
- **Import errors**: Ensure `triton_stub.py` and `vllm_stub.py` are in the directory
- **OpenMP errors**: Verify `~/.local/lib/libomp.so` exists and is accessible
- **Backend not found**: Ensure RVV backend is registered in SGLang's attention registry
