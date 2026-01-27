FROM python:3.13-slim
SHELL ["/bin/bash", "-c"]

ARG SGLANG_REPO=https://github.com/nthu-pllab/pllab-sglang.git
ARG VER_SGLANG=main

ARG VER_TORCH=2.8.0+spacemit.1
ARG VER_TORCHVISION=0.23.0
ARG VER_TRITON=3.3.0+spacemit.a0
ARG VER_PYARROW=21.0.0
ARG VER_VLLM=0.11.0.post3+spacemit.0.cpu

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    SGLANG_USE_CPU_ENGINE=1

# Because the disk space is limited, we need to use --no-cache and rm -rf /var/lib/apt/lists/* .
# 1. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config gdb lcov \
    ca-certificates curl wget git vim tar gzip unzip \
    libnuma-dev numactl libomp-dev libssl-dev libopenmpi-dev libsleef-dev \
    libsndfile1 \
    clang lld llvm ccache \
    libsqlite3-dev libtbb-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# 3. Setup Workspace & UV VirtualEnv
WORKDIR /sgl-workspace
RUN uv venv /opt/.venv --python 3.13
ENV PATH="/opt/.venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/.venv" \
    UV_HTTP_TIMEOUT=300

# 4. Configure UV (SpacemiT Index)
RUN printf '[[index]]\nname = "spacemit"\nurl = "https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple"\npriority = "default"\n\n[[index]]\nname = "pypi"\nurl = "https://pypi.org/simple"\npriority = "secondary"' > /opt/.venv/uv.toml
ENV UV_CONFIG_FILE=/opt/.venv/uv.toml

# 5. Build Tools & Heavy Dependencies (Pre-install)
RUN uv pip install --no-cache \
    scikit-build-core cmake ninja wheel setuptools \
    "torch==${VER_TORCH}" \
    "torchvision==${VER_TORCHVISION}" \
    "triton==${VER_TRITON}" \
    "pyarrow==${VER_PYARROW}" \
    "vllm==${VER_VLLM}" \
    --index-strategy unsafe-best-match --no-cache

# 6. Install SGLang Source
WORKDIR /sgl-workspace
RUN git clone ${SGLANG_REPO} sglang && \
    cd sglang && \
    git checkout ${VER_SGLANG}

# 7. Compile sgl-kernel (CLANG REQUIRED for RVV)
WORKDIR /sgl-workspace/sglang/sgl-kernel
RUN export CC=clang CXX=clang++ && \
    uv pip install . --no-build-isolation --index-strategy unsafe-best-match --no-cache

# 8. Install SGLang (GCC REQUIRED for XGrammar compatibility)
WORKDIR /sgl-workspace/sglang/python
# Overlay CPU metadata
RUN cp pyproject_cpu.toml pyproject.toml
# Install with GCC and suppressed warnings
RUN unset CC CXX && \
    export CXXFLAGS="-Wno-error" && \
    export RISCV_OMP_LIB_PATH=/usr/lib/riscv64-linux-gnu/libomp.so.5 && \
    uv pip install . --index-strategy unsafe-best-match --no-cache

# 9. Final Configuration
ENV LD_PRELOAD="/usr/lib/riscv64-linux-gnu/libomp.so.5"
RUN echo 'source /opt/.venv/bin/activate' >> /root/.bashrc

WORKDIR /sgl-workspace/sglang
