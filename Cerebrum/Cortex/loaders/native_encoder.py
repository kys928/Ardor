from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from ardor_config import ArdorConfig
from posterior_parietal_cortex import ArdorEncoder
from loaders.native_checkpoint import best_heads, read_checkpoint_meta, remap_to_model_schema, resolve_state_dict_from_raw


def infer_encoder_config_from_state(sd: Dict[str, torch.Tensor], fallback_cfg: Optional[ArdorConfig] = None) -> ArdorConfig:
    if "token_embed.weight" in sd:
        vocab = int(sd["token_embed.weight"].shape[0])
        hidden = int(sd["token_embed.weight"].shape[1])
    else:
        vocab = int(getattr(fallback_cfg, "vocab_size", 32000))
        hidden = int(getattr(fallback_cfg, "hidden_size", 384))
    try:
        layers = max(int(k.split('.')[1]) for k in sd.keys() if k.startswith('layers.') and '.attn.' in k) + 1
    except Exception:
        layers = int(getattr(fallback_cfg, "n_layers", 8))
    try:
        max_len = int(sd.get("position_embed.weight").shape[0])
    except Exception:
        max_len = int(getattr(fallback_cfg, "max_len", 1024))
    heads = int(getattr(fallback_cfg, "n_heads", best_heads(hidden)))
    if hidden % max(1, heads) != 0:
        heads = best_heads(hidden)
    return ArdorConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        n_layers=layers,
        n_heads=heads,
        max_len=max_len,
        dropout=float(getattr(fallback_cfg, "dropout", 0.1)),
        attn_dropout=float(getattr(fallback_cfg, "attn_dropout", 0.1)),
        resid_dropout=float(getattr(fallback_cfg, "resid_dropout", 0.1)),
        ff_mult=int(getattr(fallback_cfg, "ff_mult", 4)),
        layernorm_eps=float(getattr(fallback_cfg, "layernorm_eps", 1e-5)),
        use_rope=bool(getattr(fallback_cfg, "use_rope", False)),
        rope_theta=float(getattr(fallback_cfg, "rope_theta", 10000.0)),
    )


def encoder_forward_pooled(encoder: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    out = encoder(ids, return_pooled=True, pool="mean")
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        return out[1]
    if isinstance(out, dict) and "pooled" in out:
        return out["pooled"]
    if isinstance(out, torch.Tensor):
        return out.mean(dim=1)
    raise TypeError(f"Unsupported encoder output type: {type(out)}")


def load_encoder_cached(encoder_ckpt: Optional[str], device: str, fallback_decoder_cfg: Optional[ArdorConfig] = None):
    if not encoder_ckpt or not os.path.isfile(encoder_ckpt):
        return None
    raw = torch.load(encoder_ckpt, map_location=device)
    meta = read_checkpoint_meta(raw)

    if isinstance(raw, torch.nn.Module):
        return raw.to(device).eval()

    sd = resolve_state_dict_from_raw(raw)
    try:
        cfg = ArdorConfig.from_dict(meta)
    except Exception:
        cfg = infer_encoder_config_from_state(sd, fallback_decoder_cfg)

    # Required path: config-based API + use_cls_token=False for retrieval mode.
    enc = ArdorEncoder(cfg, use_cls_token=False)
    remapped = remap_to_model_schema(sd, set(enc.state_dict().keys()))
    enc.load_state_dict(remapped, strict=False)
    return enc.to(device).eval()
