from __future__ import annotations

from pathlib import Path
from typing import Optional

from backends.base import ModelBackend
from backends.hf_causal_lm import HFCausalLMBackend
from backends.native_ardor import NativeArdorBackend


VALID_BACKENDS = {"native_ardor", "hf_causal_lm"}


def _detect_backend_family(model_path: str) -> str:
    p = Path(model_path)
    if p.is_file() and p.suffix.lower() == ".pt":
        return "native_ardor"
    if p.is_dir() and (p / "config.json").exists():
        return "hf_causal_lm"
    raise ValueError(f"Ambiguous backend for model path: {model_path}")


def load_backend(
    model_path: str,
    tokenizer_path: Optional[str],
    device: str,
    repo_root: Path,
    *,
    backend_family: Optional[str] = None,
    encoder_ckpt: Optional[str] = None,
) -> ModelBackend:
    family = (backend_family or "").strip().lower() or _detect_backend_family(model_path)
    if family not in VALID_BACKENDS:
        raise ValueError(f"Invalid backend family '{backend_family}'. Valid: {sorted(VALID_BACKENDS)}")

    if family == "native_ardor":
        return NativeArdorBackend(model_path=model_path, tokenizer_path=tokenizer_path, device=device, repo_root=repo_root, encoder_ckpt=encoder_ckpt)
    if family == "hf_causal_lm":
        return HFCausalLMBackend(model_path=model_path, device=device)
    raise RuntimeError(f"Unsupported backend family: {family}")
