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
    ARDOR_CODE_ROOT=/opt/Ardor

WORKDIR /opt/Ardor
COPY . /opt/Ardor
RUN uv sync --frozen

CMD ["uv", "run", "--frozen", "python", "scripts/runpod_worker.py"]
