# Ardor

Ardor is a modular, neuro-inspired AI runtime organized as interacting cognitive subsystems rather than a single model checkpoint. The repository combines model backends, orchestration logic, memory/replay layers, and operator interfaces into one runtime that can be bootstrapped and launched deterministically.

## Overview

Ardor treats inference as a system-level process:

- **Cerebrum** provides core cognition and model orchestration.
- **Aeternum** provides memory/state-related components.
- **Praetor** provides operator-facing interfaces (CLI/GUI pathways).
- **Erratum** contains diagnostics, repair, and maintenance tooling.
- **Hephaestus** contains build and infrastructure-oriented support code.

At runtime, the central orchestration layer is the **Prefrontal Cortex** (`ArdorCore` in `Cerebrum/Cortex/prefrontal_cortex.py`), which initializes the selected backend, retrieval path, and generation flow.

## Project structure

```text
Cerebrum/     Core model stack, cortex orchestration, backends, language processing
Aeternum/     Memory and state-oriented systems
Praetor/      GUI and operator-facing runtime interfaces
Erratum/      Diagnostics, repair scripts, migrations, maintenance utilities
Hephaestus/   Build/tooling/infrastructure support
scripts/      Runtime bootstrap and launch entrypoints
tests/        Test coverage for runtime components
```

## Runtime flow

Ardor startup is split into explicit phases:

1. `./scripts/start_ardor.sh` validates environment assumptions, enforces `uv.lock`, and runs `uv sync --frozen`.
2. `scripts/bootstrap_runtime.py` resolves backend/device/runtime paths and writes runtime state to `ARDOR_HOME/runtime/runtime_state.json`.
3. `scripts/launch_ardor.py` reads runtime state and starts the requested target:
   - `cli` (interactive terminal)
   - `gui` (Tkinter-based UI via `Praetor/GUI_Cortex.py`)
   - `api` (reserved path; currently not implemented)

This separation keeps bootstrap concerns (resolution, validation, state capture) distinct from process launch.

## Backends

Ardor supports two backend families through Cortex backend factories:

- **Native backend** (`native_ardor`): loads local checkpoint + tokenizer artifacts.
- **Hugging Face causal LM backend** (`hf_causal_lm`): loads transformer causal LMs through `transformers` and Hugging Face model snapshots.

Backend selection is controlled by environment (`ARDOR_BACKEND`), with runtime validation in bootstrap.

## Configuration

Use environment variables to control runtime behavior. The most important are:

- `ARDOR_BACKEND` = `native` | `hf` (startup script defaults to `native`; bootstrap also supports `auto` resolution)
- `ARDOR_LAUNCH_TARGET` = `cli` | `gui` | `api`
- `ARDOR_DEVICE` = `auto` | `cpu` | `cuda`
- `ARDOR_HOME` = runtime root (default `/workspace/ArdorRuntime` when available)
- `HF_HOME` = Hugging Face cache root (default `/workspace/.cache/huggingface`)
- `ARDOR_MODEL_PATH` / `ARDOR_TOKENIZER_PATH` for native runtime artifacts
- `ARDOR_MODEL_ID` (and optional `ARDOR_MODEL_REVISION`) for HF runtime
- `ARDOR_ENABLE_DMN`, `ARDOR_ENABLE_RETRIEVAL` toggles for runtime cognitive modules

## Quickstart

```bash
# from repo root
./scripts/start_ardor.sh
```

Typical examples:

```bash
# Native backend + CLI
ARDOR_BACKEND=native ARDOR_LAUNCH_TARGET=cli ./scripts/start_ardor.sh

# HF backend + CLI
ARDOR_BACKEND=hf ARDOR_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct ARDOR_LAUNCH_TARGET=cli ./scripts/start_ardor.sh

# GUI launch
ARDOR_LAUNCH_TARGET=gui ./scripts/start_ardor.sh
```

## Direction

Ardor is being developed as an artificial cognitive runtime: a structured system of cooperating modules for reasoning, memory interaction, and operator control, with explicit runtime contracts for reproducible startup and backend portability.
