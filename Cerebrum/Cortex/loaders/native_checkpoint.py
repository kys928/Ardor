from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ardor_config import ArdorConfig
from broca_decoder import ArdorDecoder

MODEL_REGISTRY = {
    "ArdorDecoder": ArdorDecoder,
}


def unwrap_state_dict(raw) -> Union[dict, torch.nn.Module]:
    if isinstance(raw, torch.nn.Module):
        return raw
    if isinstance(raw, dict):
        for k in ("state_dict", "model_state_dict", "module", "model"):
            v = raw.get(k)
            if isinstance(v, dict) and any(isinstance(t, torch.Tensor) for t in v.values()):
                return v
        if any(isinstance(t, torch.Tensor) for t in raw.values()):
            return raw
    raise ValueError("Unsupported checkpoint format: cannot find a flat state_dict.")


def remap_to_model_schema(sd: dict, model_state_keys: set[str]) -> dict:
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {(k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

    new = dict(sd)

    def _rename_prefix(d, old, newp):
        if any(k.startswith(old) for k in d):
            out = {}
            for k, v in d.items():
                out[newp + k[len(old):] if k.startswith(old) else k] = v
            return out
        return d

    for alias in ("token.", "tok_embeddings.", "embed_tokens.", "embedding."):
        new = _rename_prefix(new, alias, "token_embed.")

    expects_blocks = any(k.startswith("blocks.") for k in model_state_keys)
    expects_layers = any(k.startswith("layers.") for k in model_state_keys)
    has_blocks = any(k.startswith("blocks.") for k in new)
    has_layers = any(k.startswith("layers.") for k in new)
    if expects_blocks and expects_layers:
        if has_blocks and not has_layers:
            extra = {f"layers.{k[len('blocks.'):]}": v for k, v in new.items() if k.startswith("blocks.")}
            new = {**new, **extra}
        elif has_layers and not has_blocks:
            extra = {f"blocks.{k[len('layers.'):]}": v for k, v in new.items() if k.startswith("layers.")}
            new = {**new, **extra}
    elif expects_blocks and has_layers:
        new = _rename_prefix(new, "layers.", "blocks.")
    elif expects_layers and has_blocks:
        new = _rename_prefix(new, "blocks.", "layers.")

    expects_attn = any(".attn." in k for k in model_state_keys)
    expects_attention = any(".attention." in k for k in model_state_keys)
    has_attn = any(".attn." in k for k in new)
    has_attention = any(".attention." in k for k in new)
    if expects_attn and has_attention:
        new = {k.replace(".attention.", ".attn."): v for k, v in new.items()}
    elif expects_attention and has_attn:
        new = {k.replace(".attn.", ".attention."): v for k, v in new.items()}

    tmp = {}
    for k, v in new.items():
        k2 = (
            k.replace(".q_proj.", ".q.")
            .replace(".k_proj.", ".k.")
            .replace(".v_proj.", ".v.")
            .replace(".o_proj.", ".out.")
            .replace(".out_proj.", ".out.")
        )
        tmp[k2] = v
    new = tmp

    expects_lm = any(k.startswith("lm_head.") for k in model_state_keys)
    expects_vocab = any(k.startswith("to_vocab.") for k in model_state_keys)
    has_lm = any(k.startswith("lm_head.") for k in new)
    has_vocab = any(k.startswith("to_vocab.") for k in new)
    for alias in ("to_logits.", "output.", "generator."):
        if any(k.startswith(alias) for k in new) and not has_lm and expects_lm:
            new = _rename_prefix(new, alias, "lm_head.")
            has_lm = True
    if expects_lm and has_vocab:
        new = _rename_prefix(new, "to_vocab.", "lm_head.")
    elif expects_vocab and has_lm:
        new = _rename_prefix(new, "lm_head.", "to_vocab.")

    tmp = {}
    for k, v in new.items():
        tmp[k.replace(".mlp.fc1.", ".ff.0.").replace(".mlp.fc2.", ".ff.2.")] = v
    new = tmp

    expects_posparam = "pos" in model_state_keys
    expects_posembed = "position_embed.weight" in model_state_keys
    has_posparam = "pos" in new
    has_pos_embed = "position_embed.weight" in new

    if expects_posparam and has_pos_embed and "pos" not in new:
        w = new["position_embed.weight"]
        try:
            if getattr(w, "ndim", None) == 2:
                new["pos"] = w.unsqueeze(0)
        except Exception:
            pass
        new.pop("position_embed.weight", None)

    if expects_posembed and has_posparam and "position_embed.weight" not in new:
        w = new["pos"]
        try:
            if getattr(w, "ndim", None) == 3 and w.shape[0] == 1:
                new["position_embed.weight"] = w.squeeze(0)
        except Exception:
            pass
        new.pop("pos", None)

    return new


def infer_dims_from_state(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int]:
    for k in ("token_embed.weight", "lm_head.weight", "to_vocab.weight"):
        t = sd.get(k)
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            vocab = int(t.shape[0]); hidden = int(t.shape[1]); break
    else:
        twoD = [(k, t) for k, t in sd.items() if isinstance(t, torch.Tensor) and t.ndim == 2]
        nonsq = [(k, t) for k, t in twoD if int(t.shape[0]) != int(t.shape[1])]
        pool = nonsq or twoD
        if not pool:
            raise KeyError("No 2D tensors found in checkpoint; cannot infer vocab/hidden.")
        _, t_big = max(pool, key=lambda kv: int(kv[1].shape[0]))
        vocab = int(t_big.shape[0]); hidden = int(t_big.shape[1])

    idxs: List[int] = []
    for k in sd.keys():
        m = re.search(r"(?:layers|blocks|h|layer)\.(\d+)\.", k)
        if m:
            idxs.append(int(m.group(1)))
    layers = (max(idxs) + 1) if idxs else 8

    max_len = None
    for k, t in sd.items():
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            r, c = int(t.shape[0]), int(t.shape[1])
            if c == hidden and re.search(r"(pos|position).*emb.*weight", k, re.I):
                max_len = r; break
        if isinstance(t, torch.Tensor) and t.ndim == 3 and t.shape[0] in (1,):
            r, c = int(t.shape[1]), int(t.shape[2])
            if c == hidden:
                max_len = r; break
    if max_len is None:
        max_len = 2048

    return vocab, hidden, layers, int(max_len)


def best_heads(hidden: int, prefer: int = 6) -> int:
    if prefer > 1 and hidden % prefer == 0:
        return prefer
    divisors = [d for d in range(2, 65) if hidden % d == 0]
    return max(divisors) if divisors else 1


def introspect_model(model: torch.nn.Module, sd: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {"layers": None, "heads": None, "hidden": None, "max_len": None, "vocab": None}
    for a in ("layers", "num_layers", "n_layers"):
        info["layers"] = info["layers"] or getattr(model, a, None)
    for a in ("heads", "num_heads", "n_heads"):
        info["heads"] = info["heads"] or getattr(model, a, None)
    for a in ("hidden", "hidden_dim", "embed_dim", "d_model"):
        info["hidden"] = info["hidden"] or getattr(model, a, None)
    for a in ("max_len", "max_seq_len", "context_len", "ctx_len"):
        info["max_len"] = info["max_len"] or getattr(model, a, None)

    try:
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            info["vocab"] = int(model.lm_head.weight.shape[0])
        elif hasattr(model, "token_embed") and hasattr(model.token_embed, "weight"):
            info["vocab"] = int(model.token_embed.weight.shape[0])
    except Exception:
        pass

    if sd is None:
        try:
            sd = model.state_dict()
        except Exception:
            sd = None
    if sd is not None:
        try:
            v, h, L, T = infer_dims_from_state(sd)
            info["vocab"] = info["vocab"] or v
            info["hidden"] = info["hidden"] or h
            info["layers"] = info["layers"] or L
            info["max_len"] = info["max_len"] or T
            if not info.get("heads") and info.get("hidden"):
                info["heads"] = best_heads(int(info["hidden"]))
        except Exception:
            pass
    return info


def read_checkpoint_meta(raw) -> dict:
    meta: Dict[str, Any] = {}
    if isinstance(raw, torch.nn.Module):
        if hasattr(raw, "model_config") and callable(raw.model_config):
            try:
                meta.update(raw.model_config() or {})
            except Exception:
                pass
        meta.setdefault("arch", type(raw).__name__)
        return dict(meta)

    if not isinstance(raw, dict):
        return meta

    top_keys = [
        "arch", "vocab_size", "hidden_size", "hidden", "n_layers", "layers", "n_heads", "heads",
        "ff_mult", "max_len", "dropout", "attn_dropout", "resid_dropout", "layernorm_eps",
        "use_rope", "rope_theta", "tokenizer_path", "tokenizer_vocab_size", "tokenizer_hash",
        "tokenizer_sha256", "positional_encoding",
    ]
    for src in (raw.get("meta") if isinstance(raw.get("meta"), dict) else {},
                raw.get("model_config") if isinstance(raw.get("model_config"), dict) else {},
                raw.get("config") if isinstance(raw.get("config"), dict) else {}, raw):
        if not isinstance(src, dict):
            continue
        for k in top_keys:
            if k in src and k not in meta:
                meta[k] = src[k]

    cfg = raw.get("config")
    if isinstance(cfg, dict):
        meta["config"] = dict(cfg)

    if "hidden_size" not in meta and "hidden" in meta:
        meta["hidden_size"] = meta["hidden"]
    if "n_layers" not in meta and "layers" in meta:
        meta["n_layers"] = meta["layers"]
    if "n_heads" not in meta and "heads" in meta:
        meta["n_heads"] = meta["heads"]

    if isinstance(meta.get("config"), dict):
        c = meta["config"]
        for k in ("vocab_size", "hidden_size", "hidden", "n_layers", "layers", "n_heads", "heads",
                  "ff_mult", "max_len", "dropout", "attn_dropout", "resid_dropout", "layernorm_eps",
                  "use_rope", "rope_theta"):
            if k not in meta and k in c:
                meta[k] = c[k]
    return dict(meta)


def resolve_state_dict_from_raw(raw) -> dict:
    unwrapped = unwrap_state_dict(raw)
    if isinstance(unwrapped, dict):
        return unwrapped
    raise ValueError("Expected dict state_dict, got module.")


def describe_model(
    model: torch.nn.Module,
    mismatch: Optional[dict] = None,
    *,
    strict_loaded: bool = True,
    partial_loaded: bool = False,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    desc = introspect_model(model)
    if hasattr(model, "model_config") and callable(model.model_config):
        try:
            meta = model.model_config() or {}
            if isinstance(meta, dict):
                desc.update({
                    "vocab": meta.get("vocab_size", desc.get("vocab")),
                    "hidden": meta.get("hidden_size", meta.get("hidden", desc.get("hidden"))),
                    "layers": meta.get("n_layers", meta.get("layers", desc.get("layers"))),
                    "heads": meta.get("n_heads", meta.get("heads", desc.get("heads"))),
                    "max_len": meta.get("max_len", desc.get("max_len")),
                })
        except Exception:
            pass
    mm = mismatch or {"missing": [], "unexpected": []}
    desc["mismatch"] = mm
    desc["strict_loaded"] = bool(strict_loaded)
    desc["partial_loaded"] = bool(partial_loaded)
    desc["missing_keys"] = list(mm.get("missing") or [])
    desc["unexpected_keys"] = list(mm.get("unexpected") or [])
    if checkpoint_path is not None:
        desc["checkpoint_path"] = checkpoint_path
    return desc


def load_native_decoder(model_path: str, device: str, *, allow_partial_load: bool = False):
    try:
        raw = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        raw = torch.load(model_path, map_location=device)
    checkpoint_meta = read_checkpoint_meta(raw)

    if isinstance(raw, torch.nn.Module):
        model = raw.to(device).eval()
        try:
            want_vocab = int(model.lm_head.weight.shape[0])
        except Exception:
            want_vocab = int(model.token_embed.weight.shape[0])
        return (
            model,
            describe_model(model, strict_loaded=True, partial_loaded=False, checkpoint_path=model_path),
            want_vocab,
            checkpoint_meta,
        )

    sd = resolve_state_dict_from_raw(raw)
    model = None
    try:
        arch = checkpoint_meta.get("arch") or checkpoint_meta.get("model")
        if arch in MODEL_REGISTRY:
            cfg = ArdorConfig.from_dict(checkpoint_meta)
            model = MODEL_REGISTRY[arch](cfg)
    except Exception:
        model = None

    if model is None:
        vocab, hidden, layers, max_len = infer_dims_from_state(sd)
        use_rope = "position_embed.weight" not in sd
        heads = best_heads(hidden)
        cfg = ArdorConfig(
            vocab_size=vocab, hidden_size=hidden, n_layers=layers, n_heads=heads, max_len=max_len,
            dropout=float(checkpoint_meta.get("dropout", 0.15) or 0.15),
            attn_dropout=float(checkpoint_meta.get("attn_dropout", checkpoint_meta.get("dropout", 0.15)) or 0.15),
            resid_dropout=float(checkpoint_meta.get("resid_dropout", checkpoint_meta.get("dropout", 0.15)) or 0.15),
            ff_mult=int(checkpoint_meta.get("ff_mult", 4) or 4),
            layernorm_eps=float(checkpoint_meta.get("layernorm_eps", 1e-5) or 1e-5),
            use_rope=bool(checkpoint_meta.get("use_rope", use_rope)),
            rope_theta=float(checkpoint_meta.get("rope_theta", 10000.0) or 10000.0),
        )
        model = ArdorDecoder(cfg)

    remapped = remap_to_model_schema(sd, set(model.state_dict().keys()))
    strict_loaded = False
    partial_loaded = False
    mismatch = {"missing": [], "unexpected": []}
    try:
        model.load_state_dict(remapped, strict=True)
        strict_loaded = True
    except Exception as strict_err:
        if not allow_partial_load:
            try:
                missing, unexpected = model.load_state_dict(remapped, strict=False)
                mismatch = {
                    "missing": list(missing) if isinstance(missing, list) else list(missing or []),
                    "unexpected": list(unexpected) if isinstance(unexpected, list) else list(unexpected or []),
                }
            except Exception:
                mismatch = {"missing": [], "unexpected": []}
            raise RuntimeError(
                "Strict native checkpoint load failed. "
                f"Set allow_partial_load=True to opt into partial loading. "
                f"missing_keys={mismatch['missing']} unexpected_keys={mismatch['unexpected']}"
            ) from strict_err
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        mismatch = {
            "missing": list(missing) if isinstance(missing, list) else list(missing or []),
            "unexpected": list(unexpected) if isinstance(unexpected, list) else list(unexpected or []),
        }
        partial_loaded = True

    model = model.to(device).eval()
    want_vocab = int(getattr(model, "vocab_size", 0) or infer_dims_from_state(sd)[0])
    return (
        model,
        describe_model(
            model,
            mismatch,
            strict_loaded=strict_loaded,
            partial_loaded=partial_loaded,
            checkpoint_path=model_path,
        ),
        want_vocab,
        checkpoint_meta,
    )
