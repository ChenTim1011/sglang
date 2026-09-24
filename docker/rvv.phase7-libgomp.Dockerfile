# Reproducible Phase 7 hybrid runtime.
#
# The builder and runtime image arguments must be immutable digests in release
# builds. The PyTorch wheel is a build-context artifact whose SHA256 is required.
ARG BUILDER_IMAGE
ARG RUNTIME_IMAGE

FROM ${BUILDER_IMAGE} AS kernel-builder
ARG PYTORCH_WHEEL
ARG PYTORCH_WHEEL_SHA256
COPY ${PYTORCH_WHEEL} /tmp/torch.whl
COPY python/sglang/kernels/aot /src/aot
RUN echo "${PYTORCH_WHEEL_SHA256}  /tmp/torch.whl" | sha256sum --check --strict - \
    && unset LD_PRELOAD \
    && /opt/.venv/bin/pip install --no-deps --force-reinstall /tmp/torch.whl \
    && cd /src/aot \
    && cp pyproject_riscv64.toml pyproject.toml \
    && CC=gcc-15 CXX=g++-15 CMAKE_BUILD_PARALLEL_LEVEL=8 \
       /opt/.venv/bin/pip wheel . --no-build-isolation --no-deps -w /wheelhouse

FROM ${RUNTIME_IMAGE} AS runtime
ARG PYTORCH_REVISION
ARG SGLANG_REVISION
COPY --from=kernel-builder /wheelhouse/*.whl /opt/rvv-artifacts/
COPY scripts/verify_rvv_openmp_abi.py /usr/local/bin/verify-rvv-openmp-abi

# The inherited image preloads LLVM libomp. Clear it permanently: PyTorch and
# the GCC 15 RVV extension both use GNU libgomp in this image.
ENV LD_PRELOAD="" \
    OMP_NUM_THREADS=8 \
    ATEN_CPU_CAPABILITY=rvv

RUN unset LD_PRELOAD \
    && wheel=$(find /opt/rvv-artifacts -name '*.whl' -print -quit) \
    && sha256sum "${wheel}" > /opt/rvv-artifacts/SHA256SUMS \
    && /opt/.venv/bin/pip install --no-deps --force-reinstall "${wheel}" \
    && env -u LD_PRELOAD /opt/.venv/bin/python /usr/local/bin/verify-rvv-openmp-abi

LABEL org.opencontainers.image.title="SGLang RVV Phase 7 libgomp runtime" \
      org.opencontainers.image.description="riscv64 hybrid runtime with PyTorch and SGLang RVV extension aligned on GNU libgomp" \
      org.opencontainers.image.source="https://github.com/ChenTim1011/sglang" \
      org.opencontainers.image.revision="${SGLANG_REVISION}" \
      io.sglang.rvv.pytorch.revision="${PYTORCH_REVISION}" \
      io.sglang.rvv.sglang.revision="${SGLANG_REVISION}" \
      io.sglang.rvv.openmp.runtime="libgomp" \
      io.sglang.rvv.openmp.preload="none" \
      io.sglang.rvv.kernel.compiler="gcc-15" \
      io.sglang.rvv.abi-gate="verify-rvv-openmp-abi"
