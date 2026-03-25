from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from backends.base import ModelBackend
from backends.hf_causal_lm import HFCausalLMBackend
from backends.native_ardor import NativeArdorBackend


VALID_BACKENDS = {"native_ardor", "hf_causal_lm"}
_NATIVE_FILE_EXTS = {".pt", ".pth", ".ckpt", ".bin"}


def _looks_like_native_state_dict(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.suffix.lower() in _NATIVE_FILE_EXTS:
        return True
    try:
        raw = torch.load(str(p), map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(str(p), map_location="cpu")
    except Exception:
        return False
    if isinstance(raw, torch.nn.Module):
        return True
    if isinstance(raw, dict):
        for k in ("state_dict", "model_state_dict", "module", "model"):
            v = raw.get(k)
            if isinstance(v, dict) and any(isinstance(t, torch.Tensor) for t in v.values()):
                return True
        if any(isinstance(t, torch.Tensor) for t in raw.values()):
            return True
    return False


def _detect_backend_family(model_path: str) -> str:
    p = Path(model_path)
    if p.is_dir() and (p / "config.json").exists():
        return "hf_causal_lm"
    if _looks_like_native_state_dict(p):
        return "native_ardor"
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
