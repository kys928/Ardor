# Ardor Runtime Artifact Layout

Ardor source code remains in the tracked neurobiological modules (`Cerebrum/`, `Aeternum/`, `Praetor/`, `Erratum/`, `Hephaestus/`).

Runtime-generated content is now expected outside tracked source trees:

- `data/` — raw/manual corpora and input datasets
- `artifacts/datasets/` — processed tensor/token shard datasets
- `artifacts/models/` — trained checkpoints and exported model weights
- `artifacts/rem/` — REM status, dream/replay outputs, REM checkpoints
- `artifacts/memory/` — runtime dialogue/memory logs
- `logs/` — runtime log files

Tokenizer source remains canonical in:

- `Cerebrum/ProjectTokenizer/ardor_tokenizer/`
