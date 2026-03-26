#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() {
  echo "[start] $*"
}

fail() {
  echo "[start] ERROR: $*" >&2
  exit 1
}

spinner() {
  local pid="$1"
  local label="$2"
  local spin='|/-\'
  local i=0

  while kill -0 "$pid" 2>/dev/null; do
    i=$(((i + 1) % 4))
    printf "\r[start] [%c] %s" "${spin:$i:1}" "$label"
    sleep 0.1
  done
  printf "\r[start] [✓] %s\n" "$label"
}

run_step() {
  local label="$1"
  shift

  local logfile
  logfile="$(mktemp)"

  "$@" >"$logfile" 2>&1 &
  local pid=$!

  spinner "$pid" "$label"

  if ! wait "$pid"; then
    echo
    cat "$logfile" >&2
    rm -f "$logfile"
    fail "$label failed"
  fi

  rm -f "$logfile"
}

if [[ -d "/workspace" ]]; then
  : "${ARDOR_HOME:=/workspace/ArdorRuntime}"
  : "${HF_HOME:=/workspace/.cache/huggingface}"
  : "${UV_CACHE_DIR:=/workspace/.cache/uv}"
else
  FALLBACK_WS="${REPO_ROOT}/.workspace"
  mkdir -p "${FALLBACK_WS}"
  : "${ARDOR_HOME:=${FALLBACK_WS}/ArdorRuntime}"
  : "${HF_HOME:=${FALLBACK_WS}/.cache/huggingface}"
  : "${UV_CACHE_DIR:=${FALLBACK_WS}/.cache/uv}"
  log "warning: /workspace not found; using ${FALLBACK_WS}"
fi

: "${UV_LINK_MODE:=copy}"
: "${ARDOR_BACKEND:=native}"
: "${ARDOR_LAUNCH_TARGET:=cli}"
: "${ARDOR_DEVICE:=auto}"
: "${ARDOR_ENABLE_DMN:=1}"
: "${ARDOR_ENABLE_RETRIEVAL:=1}"

export ARDOR_HOME
export HF_HOME
export UV_CACHE_DIR
export UV_LINK_MODE
export PYTHONUNBUFFERED=1
export ARDOR_BACKEND
export ARDOR_LAUNCH_TARGET
export ARDOR_DEVICE
export ARDOR_ENABLE_DMN
export ARDOR_ENABLE_RETRIEVAL

case "${ARDOR_BACKEND}" in
  native|hf) ;;
  *)
    fail "invalid ARDOR_BACKEND='${ARDOR_BACKEND}'. Expected 'native' or 'hf'."
    ;;
esac

case "${ARDOR_LAUNCH_TARGET}" in
  cli|gui|api) ;;
  *)
    fail "invalid ARDOR_LAUNCH_TARGET='${ARDOR_LAUNCH_TARGET}'. Expected 'cli', 'gui', or 'api'."
    ;;
esac

case "${ARDOR_DEVICE}" in
  auto|cpu|cuda) ;;
  *)
    fail "invalid ARDOR_DEVICE='${ARDOR_DEVICE}'. Expected 'auto', 'cpu', or 'cuda'."
    ;;
esac

if [[ ! -f "${REPO_ROOT}/uv.lock" ]]; then
  fail "uv.lock is required for deterministic startup."
fi

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  fail "pyproject.toml not found at repo root."
fi

mkdir -p "${ARDOR_HOME}" "${HF_HOME}" "${UV_CACHE_DIR}"

UV_LOCAL_BIN="${REPO_ROOT}/.uv"
export PATH="${UV_LOCAL_BIN}:/root/.local/bin:$PATH"

# Critical for imports like `import Aeternum`
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Cerebrum/Cortex:${PYTHONPATH:-}"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  log "uv not found; installing local unmanaged uv"
  export UV_UNMANAGED_INSTALL="${REPO_ROOT}/.uv"

  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m pip install --upgrade uv
  elif command -v python >/dev/null 2>&1; then
    python -m pip install --upgrade uv
  else
    fail "neither curl, wget, python3, nor python is available to install uv"
  fi

  command -v uv >/dev/null 2>&1 || fail "uv installation failed"
}

ensure_uv

SYNC_ARGS=(sync --frozen)
RUN_ARGS=(run --frozen)

if [[ "${ARDOR_BACKEND}" == "hf" ]]; then
  SYNC_ARGS+=(--extra hf)
  RUN_ARGS+=(--extra hf)
fi

if [[ "${ARDOR_LAUNCH_TARGET}" == "gui" ]]; then
  SYNC_ARGS+=(--extra gui)
  RUN_ARGS+=(--extra gui)
fi

printf '\n\033[1;35m==[ Ardor Runtime Boot ]==============================\033[0m\n'
log "repo root: ${REPO_ROOT}"
log "runtime root: ${ARDOR_HOME}"
log "hf cache root: ${HF_HOME}"
log "uv cache root: ${UV_CACHE_DIR}"
log "backend: ${ARDOR_BACKEND}"
log "launch target: ${ARDOR_LAUNCH_TARGET}"
log "device request: ${ARDOR_DEVICE}"
log "dmn enabled: ${ARDOR_ENABLE_DMN}"
log "retrieval enabled: ${ARDOR_ENABLE_RETRIEVAL}"

run_step "Syncing environment" uv "${SYNC_ARGS[@]}"
run_step "Bootstrapping runtime" uv "${RUN_ARGS[@]}" python scripts/bootstrap_runtime.py

log "launching Ardor"
exec uv "${RUN_ARGS[@]}" python scripts/launch_ardor.py