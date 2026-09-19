FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

ENV ARDOR_HOME=/workspace/ArdorRuntime \
    HF_HOME=/workspace/.cache/huggingface \
    UV_CACHE_DIR=/workspace/.cache/uv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/Ardor/scripts \
    ARDOR_CODE_ROOT=/opt/Ardor

WORKDIR /opt/Ardor
COPY . /opt/Ardor
RUN uv sync --frozen

# The repository lock currently resolves torch 2.11 / CUDA 13, while RunPod
# hosts used by the fixed v14a5 diagnostic expose CUDA-12.8-capable drivers.
# Overlay a CUDA 12.8 PyTorch wheel in the image venv. The v14a5 control-plane
# path invokes this venv directly so uv cannot resync it back to the lock.
ARG ARDOR_RUNPOD_TORCH_VERSION=2.7.1
ARG ARDOR_RUNPOD_TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN uv pip install --python .venv/bin/python \
      --index-url "${ARDOR_RUNPOD_TORCH_INDEX}" \
      "torch==${ARDOR_RUNPOD_TORCH_VERSION}" \
    && .venv/bin/python -c "import torch; assert torch.__version__.startswith('2.7.1'); assert torch.version.cuda == '12.8'; print(torch.__version__, torch.version.cuda)"

CMD ["uv", "run", "--frozen", "python", "scripts/runpod_worker.py"]
