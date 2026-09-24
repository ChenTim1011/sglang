# Fast release-image assembly from an already verified wheel. The source-build
# recipe is rvv.phase7-libgomp.Dockerfile.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG KERNEL_WHEEL
ARG KERNEL_WHEEL_SHA256
ARG PYTORCH_REVISION
ARG SGLANG_REVISION
COPY ${KERNEL_WHEEL} /opt/rvv-artifacts/
COPY verify_rvv_openmp_abi.py /usr/local/bin/verify-rvv-openmp-abi

ENV LD_PRELOAD="" \
    OMP_NUM_THREADS=8 \
    ATEN_CPU_CAPABILITY=rvv

RUN wheel=$(find /opt/rvv-artifacts -name '*.whl' -print -quit) \
    && echo "${KERNEL_WHEEL_SHA256}  ${wheel}" \
       | sha256sum --check --strict - \
    && unset LD_PRELOAD \
    && /opt/.venv/bin/pip install --no-deps --force-reinstall \
       "${wheel}" \
    && env -u LD_PRELOAD /opt/.venv/bin/python \
       /usr/local/bin/verify-rvv-openmp-abi

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
