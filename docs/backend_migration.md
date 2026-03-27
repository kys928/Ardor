# Backend migration note: native + HF decoder loading

Ardor now loads decoder models through a unified backend factory:

```python
from pathlib import Path
from Cerebrum.Cortex.backends.factory import load_backend

backend = load_backend(
    model_path="/path/to/model/or/checkpoint",
    tokenizer_path=None,
    device="cpu",
    repo_root=Path("."),
    backend_type="auto",  # "native", "hf", or "auto"
)

tokenizer = backend.get_tokenizer()
logits = backend.forward_logits(input_ids, attention_mask=attention_mask)
model = backend.unwrap_model()  # when training/probe code needs the raw object
meta = backend.describe()
```

## Native Ardor checkpoint load

```python
backend = load_backend(
    model_path="./Models/native_decoder.pt",
    tokenizer_path="./tokenizer.json",
    device="cuda",
    repo_root=Path("."),
    backend_type="native",
    allow_partial_load=False,
)
```

Strict loading is the default for native checkpoints. If strict loading fails, loading raises with missing/unexpected key details.
Use `allow_partial_load=True` only when you explicitly want a partial load. `backend.describe()` exposes:

- `strict_loaded`
- `partial_loaded`
- `missing_keys`
- `unexpected_keys`
- `checkpoint_path`

## Hugging Face causal LM load

```python
backend = load_backend(
    model_path="./tiny-hf-model-dir",
    tokenizer_path=None,
    device="cpu",
    repo_root=Path("."),
    backend_type="hf",
)
```

HF loading uses official Transformers auto classes:

- `AutoTokenizer.from_pretrained(...)`
- `AutoModelForCausalLM.from_pretrained(...)`

No HF checkpoint keys are remapped into `ArdorDecoder`.
