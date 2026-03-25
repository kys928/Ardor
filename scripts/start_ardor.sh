#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -d "/workspace" ]]; then
  : "${ARDOR_HOME:=/workspace/ArdorRuntime}"
  : "${HF_HOME:=/workspace/.cache/huggingface}"
else
  FALLBACK_WS="${REPO_ROOT}/.workspace"
  mkdir -p "${FALLBACK_WS}"
  : "${ARDOR_HOME:=${FALLBACK_WS}/ArdorRuntime}"
  : "${HF_HOME:=${FALLBACK_WS}/.cache/huggingface}"
  echo "[start] warning: /workspace not found; using ${FALLBACK_WS}" >&2
fi

export ARDOR_HOME HF_HOME PYTHONUNBUFFERED=1

if [[ ! -f "uv.lock" ]]; then
  echo "[start] ERROR: uv.lock is required for deterministic startup. Run `uv lock` and commit it." >&2
  exit 2
fi

echo "[start] syncing environment with uv.lock (frozen)"
uv sync --frozen

echo "[start] bootstrapping runtime"
uv run python scripts/bootstrap_runtime.py

echo "[start] launching Ardor"
uv run python scripts/launch_ardor.py
