# Reference oracle for GLM-5.3-Flash: the pinned `transformers` glm5_next code, run on CPU.
#
# Build (network is needed only for the pip step):
#   docker build -t glm53f-oracle:1 --build-arg BASE_IMAGE=<image with Python 3.12 + torch 2.13.0> oracle/
#
# BASE_IMAGE must provide `python3` (3.12) with torch 2.13.0 (x86-64, MKL build), numpy, packaging,
# filelock, fsspec, typing-extensions and jinja2. The published goldens were made from a CUDA 13.2
# base carrying the torch 2.13.0+cu132 wheel; the oracle itself only uses the CPU.
# Weights are never copied into the image: mount them read-only at run time (see README.md).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.13.0

# The base must already carry the right torch; fail the build otherwise.
RUN python3 -c "import sys, torch, numpy, jinja2, packaging, filelock, fsspec, typing_extensions; \
v = torch.__version__.split('+')[0]; \
sys.exit(0 if v == '${TORCH_VERSION}' and sys.version_info[:2] == (3, 12) else \
f'need python 3.12 + torch ${TORCH_VERSION}, found {sys.version.split()[0]} + torch {torch.__version__}')"

# The only step that needs the network: wheels only, no dependency resolution, every file hash-pinned.
COPY requirements.lock /oracle/requirements.lock
RUN python3 -m pip install --no-cache-dir --no-deps --only-binary=:all: --require-hashes \
        -r /oracle/requirements.lock

# The oracle must run the pure-PyTorch reference paths: no hub kernels, no FLA, no causal-conv1d.
RUN python3 -c "import importlib.util as u, sys; \
bad = [m for m in ('kernels', 'fla', 'causal_conv1d', 'flash_attn', 'deep_gemm') if u.find_spec(m)]; \
sys.exit(f'unexpected optional kernels in the base image: {bad}' if bad else 0)"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

COPY golden_layers.py selftest_tiny.py /oracle/
WORKDIR /oracle
ENTRYPOINT ["python3", "/oracle/golden_layers.py"]
