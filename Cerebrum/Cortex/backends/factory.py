from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from backends.base import ModelBackend
from backends.hf_causal_lm import HFCausalLMBackend
from backends.native_ardor import NativeArdorBackend


VALID_BACKENDS = {"native", "hf", "native_ardor", "hf_causal_lm", "auto"}
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


def _looks_like_hf_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    has_config = (p / "config.json").exists()
    has_tok = (p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists()
    has_weights = any((p / name).exists() for name in ("pytorch_model.bin", "model.safetensors", "tf_model.h5", "flax_model.msgpack"))
    return has_config and (has_tok or has_weights)


def _normalize_backend_type(value: Optional[str]) -> str:
    family = (value or "auto").strip().lower()
    mapping = {
        "native": "native",
        "native_ardor": "native",
        "hf": "hf",
        "hf_causal_lm": "hf",
        "auto": "auto",
        "": "auto",
    }
    if family not in VALID_BACKENDS:
        raise ValueError(f"Invalid backend type '{value}'. Valid: {sorted(VALID_BACKENDS)}")
    return mapping.get(family, family)


def _detect_backend_type(model_path: str) -> str:
    p = Path(model_path)
    hf_like = _looks_like_hf_dir(p)
    native_like = _looks_like_native_state_dict(p)
    if hf_like and not native_like:
        return "hf"
    if native_like and not hf_like:
        return "native"
    if hf_like and native_like:
        raise ValueError(
            f"Ambiguous backend for model path: {model_path}. "
            "Path matches both HF-directory and native-checkpoint heuristics. "
            "Pass backend_type='native' or backend_type='hf' explicitly."
        )
    raise ValueError(
        f"Could not detect backend for model path: {model_path}. "
        "Expected native checkpoint file (.pt/.pth/.ckpt/.bin) or HF model directory with config/tokenizer artifacts."
    )


def load_backend(
    model_path: str,
    tokenizer_path: Optional[str],
    device: str,
    repo_root: Path,
    *,
    backend_type: Optional[str] = None,
    backend_family: Optional[str] = None,
    encoder_ckpt: Optional[str] = None,
    allow_partial_load: bool = False,
) -> ModelBackend:
    requested = backend_type if backend_type is not None else backend_family
    mode = _normalize_backend_type(requested)
    final = _detect_backend_type(model_path) if mode == "auto" else mode

    if final == "native":
        return NativeArdorBackend(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device,
            repo_root=repo_root,
            encoder_ckpt=encoder_ckpt,
            allow_partial_load=allow_partial_load,
        )
    if final == "hf":
        return HFCausalLMBackend(model_path=model_path, tokenizer_path=tokenizer_path, device=device)
    raise RuntimeError(f"Unsupported backend type: {final}")
