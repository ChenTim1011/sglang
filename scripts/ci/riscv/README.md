# RISE RISC-V runner smoke

This fork-only workflow checks that the RISE GitHub App can provision a native
`riscv64` runner and that its C++ toolchain is usable. It does not install PyTorch
or SGLang and does not test their dispatch, fallback, or generated kernels.

- Compile a scalar `x*x+1` executable for `rv64gc` with auto-vectorization disabled.
- Execute seven lengths, including empty input and tail-sized inputs, with output
  sentinels and numerical checks.
- Compile a separate RVV 1.0 intrinsic example for `rv64gcv_zvl128b` and inspect
  the function's vector setup, load, multiplication, addition, and store.
- Never execute or load the RVV object. EM-RV1 is not an RVV 1.0 execution target.

The runner label must be exactly `ubuntu-24.04-riscv`; do not add `self-hosted`
or an `rvv` label. The workflow has a 15-minute timeout and read-only repository
permissions. It runs on relevant pushes to this fork's `main` and setup branch,
and supports manual dispatch once the workflow exists on the default branch.

```bash
gh workflow run rise-riscv-smoke.yml --repo ChenTim1011/sglang --ref main
```

The job summary and artifact contain commands, compiler output, CPU information,
source hashes, numerical-test output, and disassembly. A passing run establishes
runner/toolchain readiness, not TorchInductor RVV correctness or performance.

For local cross-compiler validation on an x86 host:

```bash
python3 scripts/ci/riscv/smoke.py --cross-only --output /tmp/riscv-cross-smoke \
  --cxx /path/to/riscv64-linux-gnu-g++ \
  --objdump /path/to/riscv64-linux-gnu-objdump
```

In cross-only mode the scalar executable is compiled but not executed. The
result records this explicitly; this mode is not used by the CI workflow.

The next CI layer needs a baseline-compatible PyTorch installation to test actual
non-RVV fallback. RVV numerical tests need a separate RVV 1.0 machine such as
the Banana Pi. Do not force `ATEN_CPU_CAPABILITY=rvv` on this runner.

References:

- https://ci.riseproject.dev/docs/getting-started/configure.html
- https://ci.riseproject.dev/docs/getting-started/labels
- https://www.scaleway.com/en/docs/elastic-metal/reference-content/elastic-metal-rv1-guidelines/
