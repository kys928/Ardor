# Ardor Runtime Artifact Layout

Ardor source code remains in tracked neurobiological modules (`Cerebrum/`, `Aeternum/`, `Praetor/`, `Erratum/`, `Hephaestus/`).

Runtime-generated content is expected outside tracked source trees:

- `data/` — raw/manual corpora and input datasets
- `artifacts/datasets/` — processed tensor/token shard datasets
- `artifacts/models/` — trained checkpoints and exported model weights
- `artifacts/rem/` — REM status, dream/replay outputs, REM checkpoints
- `artifacts/memory/` — runtime dialogue/memory logs
- `logs/` — runtime log files

Tokenizer source remains canonical in:

- `Cerebrum/ProjectTokenizer/ardor_tokenizer/`

## Portable Runtime / RunPod Boot

Ardor now ships with a portable runtime bootstrap contract designed for Pod workflows:

- source code stays in the repo checkout
- persistent model/cache/runtime artifacts live under `/workspace`
- one command handles env sync, bootstrap, and launch

### Runtime roots

- `ARDOR_HOME` defaults to `/workspace/ArdorRuntime`
- `HF_HOME` defaults to `/workspace/.cache/huggingface`

Bootstrap creates:

- `/workspace/ArdorRuntime/models`
- `/workspace/ArdorRuntime/tokenizers`
- `/workspace/ArdorRuntime/logs`
- `/workspace/ArdorRuntime/runtime/runtime_state.json`

### Backend selection

Set `ARDOR_BACKEND` in `.env`/environment:

- `native` uses local `ARDOR_MODEL_PATH` + `ARDOR_TOKENIZER_PATH`
- `hf` uses `ARDOR_MODEL_ID` and reuses/downloads via Hugging Face cache (`HF_HOME`)

### Start command

```bash
./scripts/start_ardor.sh
```

`start_ardor.sh` requires a committed `uv.lock` and will fail fast if it is missing.

`start_ardor.sh` will:

1. run `uv sync --frozen` from `uv.lock`
2. run `scripts/bootstrap_runtime.py`
3. launch Ardor based on `ARDOR_LAUNCH_TARGET` (`cli`, `gui`, `api` placeholder)
