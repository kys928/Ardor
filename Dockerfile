FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

ENV ARDOR_HOME=/workspace/ArdorRuntime \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app/Ardor
COPY . /app/Ardor

CMD ["bash", "./scripts/start_ardor.sh"]
