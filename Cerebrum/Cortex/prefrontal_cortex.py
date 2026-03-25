from __future__ import annotations

import os, sys, time, json, random, subprocess, re, glob
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import deque

# ─────────────────────────────────────────────────────────────────────
# ✅ LOGGING (ADDED ONLY — no behavior changes)
# ─────────────────────────────────────────────────────────────────────
import logging
from datetime import datetime

_ARDOR_LOG_LEVEL = os.environ.get("ARDOR_LOG_LEVEL", "INFO").strip().upper()
_ARDOR_LOG_JSON = os.environ.get("ARDOR_LOG_JSON", "0").strip() in ("1", "true", "TRUE", "yes", "YES")
_ARDOR_LOG_FILE = os.environ.get("ARDOR_LOG_FILE", "").strip()

def _setup_ardor_logger() -> logging.Logger:
    lg = logging.getLogger("ARDOR_PFC")
    if getattr(lg, "_ardor_configured", False):
        return lg

    level = getattr(logging, _ARDOR_LOG_LEVEL, logging.INFO)
    lg.setLevel(level)
    lg.propagate = False

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    lg.addHandler(sh)

    if _ARDOR_LOG_FILE:
        try:
            fh = logging.FileHandler(_ARDOR_LOG_FILE, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            lg.addHandler(fh)
        except Exception as e:
            # print only; do not alter execution
            print(f"[LOG] failed to create file logger at {_ARDOR_LOG_FILE}: {e}")

    lg._ardor_configured = True  # type: ignore
    lg.info(f"[boot] logger configured level={_ARDOR_LOG_LEVEL} json={_ARDOR_LOG_JSON} file={'ON' if _ARDOR_LOG_FILE else 'OFF'}")
    return lg

_LOG = _setup_ardor_logger()

def _jlog(level: int, event: str, **kv: Any) -> None:
    """
    Structured logging helper. Never raises. Never changes control-flow.
    """
    try:
        if _ARDOR_LOG_JSON:
            payload = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event, **kv}
            _LOG.log(level, json.dumps(payload, ensure_ascii=False))
        else:
            # lightweight key=value tail
            tail = ""
            if kv:
                tail = " | " + " ".join([f"{k}={repr(v)[:220]}" for k, v in kv.items()])
            _LOG.log(level, f"{event}{tail}")
    except Exception:
        # fail-silent
        pass

def _trace_enabled() -> bool:
    return _LOG.isEnabledFor(logging.DEBUG) and os.environ.get("ARDOR_TRACE", "0").strip() in ("1", "true", "TRUE", "yes", "YES")

# ── Optional safety/env toggles ──────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RUST_BACKTRACE", "1")

import torch
import torch.nn.functional as F
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from ardor_config import ArdorConfig
import inspect

# Encoder (Parietal)
from Cerebrum.Cortex.posterior_parietal_cortex import ArdorEncoder

# # Surface polisher (Anterior Cingulate)
# sys.path.append("../Cerebrum/LanguageProcessing")
# from Anterior_Cingulate import polish  # noqa: E402
from Cerebrum.LanguageProcessing.Anterior_Cingulate import polish
from Cerebrum.Cortex.broca_decoder import ArdorDecoder  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# PFC GLOBAL SINGLETON (one brain per process)
# ─────────────────────────────────────────────────────────────────────

_PFC_SINGLETON: Optional["ArdorCore"] = None
_PFC_SIGNATURE: Optional[tuple] = None

def get_global_core(
    *,
    model_path: str,
    tokenizer_path: Optional[str],
    device: str = "cpu",
    enable_retrieval: bool = True,
    encoder_ckpt: Optional[str] = None,
    max_len: int = 300,
    enable_dmn: bool = False,
    force_reload: bool = False,
) -> "ArdorCore":
    """
    Boots (or returns) the one true ArdorCore for this process.
    GUI must call ONLY this function. Never instantiate ArdorCore directly from GUI.
    """
    global _PFC_SINGLETON, _PFC_SIGNATURE

    _jlog(logging.INFO, "[singleton] get_global_core called",
          model_path=model_path, tokenizer_path=tokenizer_path, device=device,
          enable_retrieval=enable_retrieval, encoder_ckpt=encoder_ckpt, max_len=max_len,
          enable_dmn=enable_dmn, force_reload=force_reload)

    sig = (
        os.path.abspath(model_path),
        os.path.abspath(tokenizer_path) if tokenizer_path else None,
        device,
        bool(enable_retrieval),
        os.path.abspath(encoder_ckpt) if encoder_ckpt else None,
        int(max_len),
        bool(enable_dmn),
    )

    _jlog(logging.DEBUG, "[singleton] computed signature", sig=sig, prev=_PFC_SIGNATURE)

    if _PFC_SINGLETON is None or force_reload or (_PFC_SIGNATURE != sig):
        _jlog(logging.INFO, "[singleton] creating/reloading ArdorCore",
              reason=("none" if _PFC_SINGLETON is None else ("force_reload" if force_reload else "sig_changed")))
        _PFC_SIGNATURE = sig
        _PFC_SINGLETON = ArdorCore(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device,
            max_len=max_len,
            enable_retrieval=enable_retrieval,
            encoder_ckpt=encoder_ckpt,
            enable_dmn=enable_dmn,
            aeternum=None,  # IMPORTANT: ArdorCore will init via bridge
        )
    else:
        _jlog(logging.INFO, "[singleton] returning existing ArdorCore", sig=_PFC_SIGNATURE)

    return _PFC_SINGLETON


def get_core_singleton() -> Optional["ArdorCore"]:
    _jlog(logging.DEBUG, "[singleton] get_core_singleton", exists=_PFC_SINGLETON is not None)
    return _PFC_SINGLETON


# Decoder model (Broca)


# ── stopwords (fail-soft if nltk unavailable) ────────────────────────
try:
    import nltk  # type: ignore
    from nltk.corpus import stopwords as nltk_sw  # type: ignore

    _jlog(logging.INFO, "[stopwords] nltk available; downloading stopwords")
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(nltk_sw.words("english"))
    _jlog(logging.INFO, "[stopwords] loaded nltk stopwords", count=len(STOPWORDS))
except Exception as e:
    _jlog(logging.WARNING, "[stopwords] nltk unavailable or failed; using fallback stopwords", err=str(e))
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "if", "then", "so", "because", "as",
        "of", "in", "on", "for", "to", "from", "by", "with", "about", "into", "over",
        "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    }


# ── EOS helpers ──────────────────────────────────────────────────────
EOS_CANDIDATES = ("<eos>", "</s>", "<|endoftext|>", "<|eot|>")

def _find_eos_id(tokenizer: Tokenizer) -> Optional[int]:
    _jlog(logging.DEBUG, "[eos] searching eos id", candidates=EOS_CANDIDATES)
    for tok in EOS_CANDIDATES:
        tid = tokenizer.token_to_id(tok)
        _jlog(logging.DEBUG, "[eos] candidate", tok=tok, tid=tid)
        if tid is not None:
            _jlog(logging.INFO, "[eos] found eos token id", tok=tok, tid=tid)
            return tid
    _jlog(logging.WARNING, "[eos] no eos token id found")
    return None


# ── conversation log (Hippocampus source) ────────────────────────────
LOG_FILE = Path("../Dataset/Conversations/ardor_dialogues.jsonl")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
_jlog(logging.INFO, "[hippocampus] LOG_FILE ready", path=str(LOG_FILE), exists=LOG_FILE.exists())

def slow_type(txt: str, delay: float = 0.005):
    _jlog(logging.DEBUG, "[cli] slow_type", n_chars=len(txt), delay=delay)
    for ch in txt:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                     Encoder checkpoint resolver                    ║
# ╚═══════════════════════════════════════════════════════════════════╝
def _resolve_encoder_ckpt(user_path: Optional[str]) -> str:
    """
    Resolve encoder checkpoint path.
    Priority:
      1) explicit user_path (if exists)
      2) env ARDOR_ENCODER_CKPT (if exists)
      3) auto-search common locations for newest "*encoder*.pt" or "*parietal*.pt"

    Raises FileNotFoundError if nothing found.
    """
    _jlog(logging.INFO, "[encoder_ckpt] resolve start", user_path=user_path, env=os.environ.get("ARDOR_ENCODER_CKPT", "").strip())

    if user_path:
        p = Path(user_path).expanduser()
        _jlog(logging.DEBUG, "[encoder_ckpt] user_path expanded", path=str(p), is_file=p.is_file(), is_dir=p.is_dir())
        if p.is_file():
            _jlog(logging.INFO, "[encoder_ckpt] using explicit file", path=str(p))
            return str(p)
        if p.is_dir():
            hits = []
            for patt in ("*encoder*.pt", "*Encoder*.pt", "*parietal*.pt", "*Parietal*.pt"):
                found = list(p.glob(patt))
                _jlog(logging.DEBUG, "[encoder_ckpt] scanning dir", dir=str(p), pattern=patt, found=len(found))
                hits += found
            if hits:
                hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                _jlog(logging.INFO, "[encoder_ckpt] using newest in directory", picked=str(hits[0]), n=len(hits))
                return str(hits[0])
        _jlog(logging.ERROR, "[encoder_ckpt] explicit path invalid", user_path=user_path)
        raise FileNotFoundError(f"encoder_ckpt was provided but not found: {user_path}")

    envp = os.environ.get("ARDOR_ENCODER_CKPT", "").strip()
    if envp:
        p = Path(envp).expanduser()
        _jlog(logging.DEBUG, "[encoder_ckpt] env expanded", envp=envp, path=str(p), is_file=p.is_file())
        if p.is_file():
            _jlog(logging.INFO, "[encoder_ckpt] using env file", path=str(p))
            return str(p)
        _jlog(logging.ERROR, "[encoder_ckpt] env set but missing", envp=envp)
        raise FileNotFoundError(f"ARDOR_ENCODER_CKPT is set but file not found: {envp}")

    roots = [
        Path("../Cerebrum/Models/Encoders"),
        Path("../Models/Encoders"),
        Path("../Cerebrum/Models"),
        Path("../Models"),
        Path("../runs"),
        Path("../Cerebrum/runs"),
        Path("./runs"),
        Path("."),
    ]
    patterns = ("*encoder*.pt", "*Encoder*.pt", "*parietal*.pt", "*Parietal*.pt")

    candidates: List[Path] = []
    for r in roots:
        try:
            _jlog(logging.DEBUG, "[encoder_ckpt] root scan", root=str(r), exists=r.exists(), is_dir=r.is_dir())
            if not r.exists() or not r.is_dir():
                continue
            for patt in patterns:
                got = list(r.glob(patt))
                if got:
                    _jlog(logging.DEBUG, "[encoder_ckpt] hits in root", root=str(r), patt=patt, n=len(got))
                candidates += got
        except Exception as e:
            _jlog(logging.WARNING, "[encoder_ckpt] root scan error", root=str(r), err=str(e))

    # bounded depth (one level)
    for r in roots:
        try:
            if not r.exists() or not r.is_dir():
                continue
            for sub in r.iterdir():
                if sub.is_dir():
                    for patt in patterns:
                        got = list(sub.glob(patt))
                        if got:
                            _jlog(logging.DEBUG, "[encoder_ckpt] hits in subdir", sub=str(sub), patt=patt, n=len(got))
                        candidates += got
        except Exception as e:
            _jlog(logging.WARNING, "[encoder_ckpt] subdir scan error", root=str(r), err=str(e))

    candidates = [c for c in candidates if c.is_file()]
    _jlog(logging.INFO, "[encoder_ckpt] candidates collected", n=len(candidates))

    if candidates:
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        _jlog(logging.INFO, "[encoder_ckpt] picked newest", path=str(candidates[0]))
        return str(candidates[0])

    _jlog(logging.ERROR, "[encoder_ckpt] no checkpoint found", roots=[str(r) for r in roots], patterns=patterns)
    raise FileNotFoundError(
        "No encoder checkpoint found. Provide encoder_ckpt=... or set ARDOR_ENCODER_CKPT to a valid .pt file. "
        "Searched in: " + ", ".join(str(r) for r in roots)
    )


# ╔═══════════════════════════════════════════════════════════════════╗
# ║             Checkpoint compatibility / key remapping              ║
# ╚═══════════════════════════════════════════════════════════════════╝
def _unwrap_state_dict(raw) -> Union[dict, torch.nn.Module]:
    _jlog(logging.DEBUG, "[ckpt] unwrap_state_dict start", raw_type=str(type(raw)))
    if isinstance(raw, torch.nn.Module):
        _jlog(logging.INFO, "[ckpt] checkpoint is module instance")
        return raw
    if isinstance(raw, dict):
        for k in ("state_dict", "model_state_dict", "module", "model"):
            v = raw.get(k)
            if isinstance(v, dict) and any(isinstance(t, torch.Tensor) for t in v.values()):
                _jlog(logging.INFO, "[ckpt] found nested state_dict", key=k, n=len(v))
                return v
        if any(isinstance(t, torch.Tensor) for t in raw.values()):
            _jlog(logging.INFO, "[ckpt] checkpoint looks like flat state_dict", n=len(raw))
            return raw
    _jlog(logging.ERROR, "[ckpt] unsupported format", keys=(list(raw.keys())[:20] if isinstance(raw, dict) else None))
    raise ValueError("Unsupported checkpoint format: cannot find a flat state_dict.")


def _infer_layers_from_keys(sd: Dict[str, torch.Tensor]) -> int:
    idxs: List[int] = []
    patts = [r"(?:layers|blocks|h|layer)\.(\d+)\."]
    for k in sd.keys():
        for p in patts:
            m = re.search(p, k)
            if m:
                idxs.append(int(m.group(1)))
                break
    L = (max(idxs) + 1) if idxs else 4
    _jlog(logging.DEBUG, "[ckpt] infer_layers_from_keys", inferred=L, n_matches=len(idxs))
    return L


def _infer_maxlen_from_pos(sd: Dict[str, torch.Tensor], *, hidden: int, default: int = 2048) -> int:
    for k, t in sd.items():
        if not isinstance(t, torch.Tensor):
            continue
        if t.ndim == 2:
            r, c = int(t.shape[0]), int(t.shape[1])
            if c == hidden and re.search(r"(pos|position).*emb.*weight", k, re.I):
                _jlog(logging.DEBUG, "[ckpt] infer_maxlen_from_pos hit", key=k, rows=r, cols=c)
                return r
        if t.ndim == 3 and int(t.shape[0]) == 1:
            r, c = int(t.shape[1]), int(t.shape[2])
            if c == hidden:
                _jlog(logging.DEBUG, "[ckpt] infer_maxlen_from_pos hit(3d)", key=k, rows=r, cols=c)
                return r
    _jlog(logging.DEBUG, "[ckpt] infer_maxlen_from_pos default", default=int(default))
    return int(default)


def _best_heads(hidden: int, prefer: int = 6) -> int:
    if prefer > 1 and hidden % prefer == 0:
        _jlog(logging.DEBUG, "[arch] best_heads using prefer", hidden=hidden, heads=prefer)
        return prefer
    divisors = [d for d in range(2, 65) if hidden % d == 0]
    h = max(divisors) if divisors else 1
    _jlog(logging.DEBUG, "[arch] best_heads computed", hidden=hidden, heads=h, n_div=len(divisors))
    return h


def _remap_to_model_schema(sd: dict, model_state_keys: set[str]) -> dict:
    _jlog(logging.INFO, "[ckpt] remap_to_model_schema start", n_keys=len(sd), model_keys=len(model_state_keys))

    if any(k.startswith("_orig_mod.") for k in sd):
        _jlog(logging.DEBUG, "[ckpt] removing _orig_mod prefix")
        sd = {(k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

    new = dict(sd)

    def _rename_prefix(d, old, newp):
        if any(k.startswith(old) for k in d):
            out = {}
            for k, v in d.items():
                out[newp + k[len(old):] if k.startswith(old) else k] = v
            _jlog(logging.DEBUG, "[ckpt] rename_prefix applied", old=old, newp=newp)
            return out
        return d

    for alias in ("token.", "tok_embeddings.", "embed_tokens.", "embedding."):
        new = _rename_prefix(new, alias, "token_embed.")

    expects_blocks = any(k.startswith("blocks.") for k in model_state_keys)
    expects_layers = any(k.startswith("layers.") for k in model_state_keys)
    has_blocks = any(k.startswith("blocks.") for k in new)
    has_layers = any(k.startswith("layers.") for k in new)
    _jlog(logging.DEBUG, "[ckpt] block/layer schema", expects_blocks=expects_blocks, expects_layers=expects_layers, has_blocks=has_blocks, has_layers=has_layers)
    if expects_blocks and has_layers:
        new = _rename_prefix(new, "layers.", "blocks.")
    elif expects_layers and has_blocks:
        new = _rename_prefix(new, "blocks.", "layers.")

    expects_attn = any(".attn." in k for k in model_state_keys)
    expects_attention = any(".attention." in k for k in model_state_keys)
    has_attn = any(".attn." in k for k in new)
    has_attention = any(".attention." in k for k in new)
    _jlog(logging.DEBUG, "[ckpt] attn schema", expects_attn=expects_attn, expects_attention=expects_attention, has_attn=has_attn, has_attention=has_attention)
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
    _jlog(logging.DEBUG, "[ckpt] head schema", expects_lm=expects_lm, expects_vocab=expects_vocab, has_lm=has_lm, has_vocab=has_vocab)

    for alias in ("to_logits.", "output.", "generator."):
        if any(k.startswith(alias) for k in new) and (not has_lm) and expects_lm:
            new = _rename_prefix(new, alias, "lm_head.")
            has_lm = True
    if expects_lm and has_vocab:
        new = _rename_prefix(new, "to_vocab.", "lm_head.")
    elif expects_vocab and has_lm:
        new = _rename_prefix(new, "lm_head.", "to_vocab.")

    tmp = {}
    for k, v in new.items():
        k2 = k.replace(".mlp.fc1.", ".ff.0.").replace(".mlp.fc2.", ".ff.2.")
        tmp[k2] = v
    new = tmp

    expects_posembed = "position_embed.weight" in model_state_keys
    has_posparam = "pos" in new
    has_posembed = "position_embed.weight" in new
    _jlog(logging.DEBUG, "[ckpt] pos embed schema", expects_posembed=expects_posembed, has_posparam=has_posparam, has_posembed=has_posembed)
    if expects_posembed and has_posparam and not has_posembed:
        w = new["pos"]
        try:
            if getattr(w, "ndim", None) == 3 and w.shape[0] == 1:
                new["position_embed.weight"] = w.squeeze(0)
                _jlog(logging.INFO, "[ckpt] pos -> position_embed.weight remapped", shape=tuple(new["position_embed.weight"].shape))
        except Exception as e:
            _jlog(logging.WARNING, "[ckpt] pos remap failed", err=str(e))
        new.pop("pos", None)

    _jlog(logging.INFO, "[ckpt] remap_to_model_schema done", n_keys=len(new))
    return new


def _infer_dims_from_state(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int]:
    twoD = [(k, t) for k, t in sd.items() if isinstance(t, torch.Tensor) and t.ndim == 2]
    nonsq = [(k, t) for k, t in twoD if int(t.shape[0]) != int(t.shape[1])]
    pool = nonsq or twoD
    if not pool:
        _jlog(logging.ERROR, "[ckpt] infer_dims_from_state failed: no 2D tensors")
        raise KeyError("No 2D tensors found in checkpoint; cannot infer vocab/hidden.")
    _, t_big = max(pool, key=lambda kv: int(kv[1].shape[0]))
    vocab = int(t_big.shape[0])
    hidden = int(t_big.shape[1])

    idxs: List[int] = []
    for k in sd.keys():
        m = re.search(r"(?:layers|blocks|h|layer)\.(\d+)\.", k)
        if m:
            idxs.append(int(m.group(1)))
    layers = (max(idxs) + 1) if idxs else 8

    max_len = _infer_maxlen_from_pos(sd, hidden=hidden, default=2048)
    _jlog(logging.INFO, "[ckpt] inferred dims", vocab=vocab, hidden=hidden, layers=layers, max_len=max_len)
    return vocab, hidden, layers, int(max_len)


def _introspect_model(model: torch.nn.Module, sd: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {"layers": None, "heads": None, "hidden": None, "max_len": None, "vocab": None}
    _jlog(logging.DEBUG, "[arch] introspect_model start", model_type=str(type(model)))

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
    except Exception as e:
        _jlog(logging.WARNING, "[arch] vocab introspection failed", err=str(e))

    if sd is None:
        try:
            sd = model.state_dict()
        except Exception as e:
            _jlog(logging.WARNING, "[arch] model.state_dict failed", err=str(e))
            sd = None
    if sd is not None:
        try:
            v, h, L, T = _infer_dims_from_state(sd)
            info["vocab"] = info["vocab"] or v
            info["hidden"] = info["hidden"] or h
            info["layers"] = info["layers"] or L
            info["max_len"] = info["max_len"] or T
            if not info.get("heads") and info.get("hidden"):
                info["heads"] = _best_heads(int(info["hidden"]))
        except Exception as e:
            _jlog(logging.WARNING, "[arch] infer dims from sd failed", err=str(e))

    _jlog(logging.INFO, "[arch] introspect_model done", **info)
    return info


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                        Global brain caches                         ║
# ╚═══════════════════════════════════════════════════════════════════╝
_MODEL_CACHE: Dict[Tuple[str, str], Tuple[nn.Module, Dict[str, Any], Optional[Dict[str, torch.Tensor]]]] = {}
_TOKENIZER_CACHE: Dict[Tuple[str, int], Tokenizer] = {}
_ENCODER_CACHE: Dict[Tuple[str, str, int, int, int, int, int], nn.Module] = {}  # (ckpt, device, vocab, hidden, heads, layers, max_len)

_GLOBAL_CORE: Optional["ArdorCore"] = None


def _abs(p: str) -> str:
    ap = os.path.abspath(os.path.expanduser(p))
    _jlog(logging.DEBUG, "[path] abs", inp=p, out=ap)
    return ap


def _load_broca_cached(model_path: str, device: str) -> Tuple[nn.Module, Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
    key = (_abs(model_path), str(device))
    _jlog(logging.INFO, "[broca] load requested", key=key)

    if key in _MODEL_CACHE:
        _jlog(logging.INFO, "[broca] cache hit", key=key)
        return _MODEL_CACHE[key]

    _jlog(logging.INFO, "[broca] cache miss; torch.load starting", path=model_path, device=device)
    t0 = time.time()
    raw = torch.load(model_path, map_location=device)
    _jlog(logging.INFO, "[broca] torch.load done", seconds=round(time.time() - t0, 4), raw_type=str(type(raw)))

    sd_or_mod = _unwrap_state_dict(raw)

    missing_list: Optional[List[str]] = None
    unexpected_list: Optional[List[str]] = None
    model_sd: Optional[Dict[str, torch.Tensor]] = None

    if isinstance(sd_or_mod, torch.nn.Module):
        _jlog(logging.INFO, "[broca] checkpoint provided a Module", module_type=str(type(sd_or_mod)))
        model = sd_or_mod.to(device).eval()
        try:
            want_vocab = int(model.lm_head.weight.shape[0])
        except Exception:
            want_vocab = int(model.token_embed.weight.shape[0])
        _jlog(logging.INFO, "[broca] module ready", want_vocab=want_vocab)
    else:
        sd = sd_or_mod
        _jlog(logging.INFO, "[broca] checkpoint is state_dict", n_keys=len(sd))
        vocab, hidden, layers, maxlen = _infer_dims_from_state(sd)
        heads = _best_heads(hidden, prefer=6)

        _jlog(logging.INFO, "[broca] constructing ArdorDecoder", vocab=vocab, hidden=hidden, layers=layers, heads=heads, max_len=maxlen)

        cfg_sig = inspect.signature(ArdorConfig)

        cfg_kwargs = {}
        for name in cfg_sig.parameters:
            if name == "vocab_size":
                cfg_kwargs[name] = vocab
            elif name == "hidden_size":
                cfg_kwargs[name] = hidden
            elif name == "n_layers":
                cfg_kwargs[name] = layers
            elif name == "n_heads":
                cfg_kwargs[name] = heads
            elif name == "max_len":
                cfg_kwargs[name] = maxlen
            elif name == "ff_mult":
                cfg_kwargs[name] = 4
            elif name == "dropout":
                cfg_kwargs[name] = 0.12
            elif name == "attn_dropout":
                cfg_kwargs[name] = 0.0
            elif name == "resid_dropout":
                cfg_kwargs[name] = 0.0
            elif name == "layernorm_eps":
                cfg_kwargs[name] = 1e-5
            elif name == "use_rope":
                cfg_kwargs[name] = True
            elif name == "rope_theta":
                cfg_kwargs[name] = 10000.0

        cfg = ArdorConfig(**cfg_kwargs)
        model = ArdorDecoder(cfg)
        remapped = _remap_to_model_schema(sd, set(model.state_dict().keys()))
        _jlog(logging.INFO, "[broca] remap complete; loading state_dict", remapped_keys=len(remapped))

        try:
            model.load_state_dict(remapped, strict=True)
            _jlog(logging.INFO, "[broca] strict load ok")
        except Exception as e:
            missing, unexpected = model.load_state_dict(remapped, strict=False)
            missing_list = list(missing) if isinstance(missing, (list, tuple)) else list(missing or [])
            unexpected_list = list(unexpected) if isinstance(unexpected, (list, tuple)) else list(unexpected or [])
            print(f"[load] non-strict: missing={len(missing_list or [])} unexpected={len(unexpected_list or [])}")
            _jlog(logging.WARNING, "[broca] strict load failed; non-strict used",
                  err=str(e), missing=len(missing_list or []), unexpected=len(unexpected_list or []))

        model = model.to(device).eval()
        model_sd = remapped
        want_vocab = int(vocab)
        _jlog(logging.INFO, "[broca] decoder ready", want_vocab=want_vocab)

    # tie head ⇄ embedding if shapes agree
    try:
        if hasattr(model, "lm_head") and hasattr(model, "token_embed"):
            if model.lm_head.weight.shape == model.token_embed.weight.shape:
                model.lm_head.weight = model.token_embed.weight
                _jlog(logging.INFO, "[broca] weight tying applied", shape=tuple(model.lm_head.weight.shape))
    except Exception as e:
        _jlog(logging.WARNING, "[broca] weight tying failed", err=str(e))

    schema = _introspect_model(model, sd=model_sd)
    schema["mismatch"] = {"missing": missing_list, "unexpected": unexpected_list}
    schema["want_vocab"] = want_vocab

    _MODEL_CACHE[key] = (model, schema, model_sd)
    _jlog(logging.INFO, "[broca] cached", key=key, schema=schema)
    return model, schema, model_sd


def _load_tokenizer_matching_vocab(tokenizer_path: Optional[str], want_vocab: int) -> Tuple[Tokenizer, str]:
    requested_tok = tokenizer_path if (tokenizer_path and os.path.isfile(tokenizer_path)) else None
    _jlog(logging.INFO, "[tok] load tokenizer requested", tokenizer_path=tokenizer_path, requested_tok=requested_tok, want_vocab=want_vocab)

    roots: List[str] = []
    if requested_tok:
        roots.append(os.path.dirname(requested_tok))
    roots += [
        "../Cerebrum/ProjectTokenizer/ardor_tokenizer",
        "../ProjectTokenizer/ardor_tokenizer",
        "./Cerebrum/ProjectTokenizer/ardor_tokenizer",
        "./ProjectTokenizer/ardor_tokenizer",
    ]
    _jlog(logging.DEBUG, "[tok] roots", roots=roots)

    # Prefer v9 first, but allow any tokenizer_v*.json that matches vocab
    cand_files: List[str] = []
    for r in roots:
        rr = os.path.abspath(r)
        if not os.path.isdir(rr):
            _jlog(logging.DEBUG, "[tok] root not dir", root=rr)
            continue
        found = glob.glob(os.path.join(rr, "tokenizer_v9.json"))
        _jlog(logging.DEBUG, "[tok] glob v9", root=rr, found=len(found))
        cand_files += found

    # If user provided a tokenizer, try it first
    if requested_tok:
        try:
            _jlog(logging.INFO, "[tok] trying override tokenizer", path=requested_tok)
            t0 = Tokenizer.from_file(requested_tok)
            v0 = t0.get_vocab_size()
            _jlog(logging.INFO, "[tok] override vocab", vocab=v0, want=want_vocab)
            if v0 == want_vocab:
                tok = t0
                # add ByteLevel decoder for BPE if needed
                try:
                    name = type(tok.model).__name__.lower()
                    if name == "bpe" and getattr(tok, "decoder", None) is None:
                        tok.decoder = ByteLevel()
                        _jlog(logging.INFO, "[tok] added ByteLevel decoder to override tokenizer")
                except Exception as e:
                    _jlog(logging.WARNING, "[tok] decoder attach failed", err=str(e))
                _jlog(logging.INFO, "[tok] using override tokenizer", path=requested_tok)
                return tok, requested_tok
        except Exception as _e:
            print(f"[tok] failed to open override {requested_tok}: {_e}. Searching for a match…")
            _jlog(logging.WARNING, "[tok] override open failed", path=requested_tok, err=str(_e))

    # Search
    _jlog(logging.INFO, "[tok] searching candidates", n=len(cand_files))
    for p in cand_files:
        k = (_abs(p), int(want_vocab))
        if k in _TOKENIZER_CACHE:
            _jlog(logging.INFO, "[tok] cache hit", key=k, path=p)
            return _TOKENIZER_CACHE[k], p
        try:
            t = Tokenizer.from_file(p)
            vv = t.get_vocab_size()
            _jlog(logging.DEBUG, "[tok] candidate loaded", path=p, vocab=vv)
            if vv == want_vocab:
                try:
                    name = type(t.model).__name__.lower()
                    if name == "bpe" and getattr(t, "decoder", None) is None:
                        t.decoder = ByteLevel()
                        _jlog(logging.INFO, "[tok] added ByteLevel decoder to matched tokenizer", path=p)
                except Exception as e:
                    _jlog(logging.WARNING, "[tok] decoder attach failed", path=p, err=str(e))
                _TOKENIZER_CACHE[k] = t
                _jlog(logging.INFO, "[tok] matched tokenizer selected", path=p, vocab=vv)
                return t, p
        except Exception as e:
            _jlog(logging.WARNING, "[tok] candidate open failed", path=p, err=str(e))

    _jlog(logging.ERROR, "[tok] no matching tokenizer found", want_vocab=want_vocab, roots=roots)
    raise FileNotFoundError(
        f"No tokenizer with vocab size {want_vocab} found. Put tokenizer_v*.json under ProjectTokenizer/ardor_tokenizer."
    )


def _encoder_forward_pooled(encoder: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """
    Return pooled [B,H] for ArdorEncoder, robust to minor signature differences.
    """
    _jlog(logging.DEBUG, "[encoder] forward pooled start", ids_shape=tuple(ids.shape), encoder_type=str(type(encoder)))
    with torch.no_grad():
        try:
            out = encoder(ids, return_pooled=True, pool="mean")
            _jlog(logging.DEBUG, "[encoder] forward pooled primary ok", out_type=str(type(out)))
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                pooled = out[1]
                _jlog(logging.DEBUG, "[encoder] pooled from tuple", pooled_shape=tuple(pooled.shape))
                return pooled
            if isinstance(out, dict) and "pooled" in out:
                pooled = out["pooled"]
                _jlog(logging.DEBUG, "[encoder] pooled from dict", pooled_shape=tuple(pooled.shape))
                return pooled
        except Exception as e:
            _jlog(logging.WARNING, "[encoder] forward pooled primary failed", err=str(e))

        # fallback: assume encoder(ids) -> hidden [B,T,H]
        out = encoder(ids)
        _jlog(logging.DEBUG, "[encoder] forward fallback ok", out_type=str(type(out)))
        if isinstance(out, (list, tuple)) and len(out) > 0:
            out = out[0]
        if isinstance(out, dict) and "last_hidden_state" in out:
            out = out["last_hidden_state"]
        # mean pool tokens
        if isinstance(out, torch.Tensor) and out.ndim == 3:
            pooled = out.mean(dim=1)
            _jlog(logging.DEBUG, "[encoder] pooled fallback mean", pooled_shape=tuple(pooled.shape))
            return pooled
        _jlog(logging.ERROR, "[encoder] forward pooled unusable output", out_type=str(type(out)))
        raise RuntimeError("Encoder forward did not produce a usable pooled embedding.")


def _load_encoder_cached(
    encoder_ckpt: str,
    device: str,
    *,
    vocab_size: int,
    hidden: int,
    heads: int,
) -> nn.Module:
    """
    Loads encoder ONCE per process for a given (ckpt, device, vocab/hidden/heads/...).
    """
    ckpt_path = _abs(encoder_ckpt)
    _jlog(logging.INFO, "[encoder] load requested", ckpt=ckpt_path, device=device, vocab=vocab_size, hidden=hidden, heads=heads)

    quick_key = (ckpt_path, str(device), int(vocab_size), int(hidden), int(heads))
    for k, enc in _ENCODER_CACHE.items():
        if k[:5] == quick_key:
            return enc

    t0 = time.time()
    enc_raw = torch.load(ckpt_path, map_location=device)
    _jlog(logging.INFO, "[encoder] torch.load done", seconds=round(time.time() - t0, 4), raw_type=str(type(enc_raw)))

    enc_sd_or_mod = _unwrap_state_dict(enc_raw)

    if isinstance(enc_sd_or_mod, torch.nn.Module):
        _jlog(logging.INFO, "[encoder] checkpoint provided module; returning eval()")
        return enc_sd_or_mod.to(device).eval()

    enc_sd: Dict[str, torch.Tensor] = enc_sd_or_mod  # type: ignore
    num_layers = _infer_layers_from_keys(enc_sd)
    max_len = _infer_maxlen_from_pos(enc_sd, hidden=hidden, default=2048)

    key = (ckpt_path, str(device), int(vocab_size), int(hidden), int(heads), int(num_layers), int(max_len))
    if key in _ENCODER_CACHE:
        _jlog(logging.INFO, "[encoder] cache hit", key=key)
        return _ENCODER_CACHE[key]

    _jlog(logging.INFO, "[encoder] cache miss; constructing ArdorEncoder", layers=num_layers, max_len=max_len)
    encoder = ArdorEncoder(
        vocab_size=int(vocab_size),
        hidden_dim=int(hidden),
        num_layers=int(num_layers),
        heads=int(heads),
        max_len=int(max_len),
        dropout=0.10,
        use_cls_token=False,
    ).to(device).eval()

    # forgiving load
    msd = encoder.state_dict()
    loadable: Dict[str, torch.Tensor] = {}
    pe_keys = [k for k in enc_sd.keys() if k.endswith("position_embed.weight") and k in msd]
    _jlog(logging.DEBUG, "[encoder] pe_keys", n=len(pe_keys))

    for k, v in enc_sd.items():
        if k not in msd:
            continue
        mv = msd[k]
        if hasattr(v, "shape") and hasattr(mv, "shape") and tuple(v.shape) == tuple(mv.shape):
            loadable[k] = v

    _jlog(logging.INFO, "[encoder] loadable params", n=len(loadable), total=len(enc_sd))
    encoder.load_state_dict(loadable, strict=False)

    # position embedding overlap copy
    for pe_key in pe_keys:
        src = enc_sd[pe_key]
        dst = msd[pe_key]
        if src.ndim == 2 and dst.ndim == 2:
            n = min(src.shape[0], dst.shape[0])
            d = min(src.shape[1], dst.shape[1])
            with torch.no_grad():
                dst[:n, :d].copy_(src[:n, :d])
            encoder.load_state_dict({pe_key: dst}, strict=False)
            _jlog(logging.INFO, "[encoder] position embedding overlap copied", key=pe_key, rows=int(n), cols=int(d))

    _ENCODER_CACHE[key] = encoder
    print(f"[encoder] loaded (cached): {ckpt_path}")
    _jlog(logging.INFO, "[encoder] cached", key=key)
    return encoder


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                     Hippocampus / ParietalMemory                  ║
# ╚═══════════════════════════════════════════════════════════════════╝
class ParietalMemory:
    """
    Hippocampus index:
      - Stores episodes from JSONL (prompt/response)
      - Encodes USER PROMPTS into vectors
      - Cosine similarity = dot product on normalized vectors
    Key rule: encoder called ONCE per turn for the user query. (No per-token calls.)
    """

    def __init__(
        self,
        tok: Tokenizer,
        device: str,
        *,
        broca_model: nn.Module,
        encoder: Optional[nn.Module],
        memory_jsonl: Optional[str],
        max_items: int = 2000,
        max_len: int = 192,
    ):
        self.tok = tok
        self.device = device
        self.broca = broca_model
        self.encoder = encoder
        self.memory_jsonl = Path(memory_jsonl) if memory_jsonl else None
        self.max_items = int(max_items)
        self.max_len = int(max_len)

        self.episodes: List[Dict[str, Any]] = []  # each: {"snippet": str, "ts": float}
        self.index_emb: Optional[torch.Tensor] = None  # (N,H) normalized

        _jlog(logging.INFO, "[memory] ParietalMemory init",
              device=device, has_encoder=encoder is not None, memory_jsonl=str(self.memory_jsonl) if self.memory_jsonl else None,
              max_items=self.max_items, max_len=self.max_len)

        if self.memory_jsonl and self.memory_jsonl.exists():
            _jlog(logging.INFO, "[memory] rebuilding from jsonl", path=str(self.memory_jsonl))
            self.rebuild_from_jsonl(self.memory_jsonl, max_items=self.max_items)
        else:
            _jlog(logging.INFO, "[memory] no jsonl to rebuild", path=str(self.memory_jsonl) if self.memory_jsonl else None)

    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        _jlog(logging.DEBUG, "[memory] tokenize_batch start", n=len(texts))
        encs = [self.tok.encode(t) for t in texts]
        ids_list = [e.ids[: self.max_len] for e in encs]

        B = len(ids_list)
        L = max(1, max(len(x) for x in ids_list))
        pad_id = 0

        ids = torch.full((B, L), pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((B, L), dtype=torch.float32, device=self.device)

        for i, seq in enumerate(ids_list):
            n = len(seq)
            if n == 0:
                continue
            ids[i, :n] = torch.tensor(seq, dtype=torch.long, device=self.device)
            mask[i, :n] = 1.0

        _jlog(logging.DEBUG, "[memory] tokenize_batch done", ids_shape=tuple(ids.shape), mask_shape=tuple(mask.shape))
        return ids, mask

    @staticmethod
    def _embed_text_semantic(
            *,
            text: str,
            tok: Tokenizer,
            encoder: nn.Module,
            device: str,
            max_len: int = 192,
    ) -> torch.Tensor:
        """
        Returns normalized embedding [1,H] for a single text using your tokenizer + your encoder pooling.
        Uses _encoder_forward_pooled(encoder, ids) from your file.
        """
        text = (text or "").strip()
        if not text:
            # return a safe zero vector (will yield cosine=0 if paired with non-zero)
            # We must know H, so infer from encoder output by doing a tiny forward if needed.
            # But easiest: do a dummy encode of "." (still cheap).
            text = "."

        enc = tok.encode(text)
        ids_list = enc.ids[:max_len]

        # pad to at least 1 token
        if not ids_list:
            ids_list = tok.encode(".").ids[:max_len] or [0]

        ids = torch.tensor([ids_list], dtype=torch.long, device=device)  # [1,T]

        pooled = _encoder_forward_pooled(encoder, ids)  # [1,H]
        pooled = pooled.to(device)

        # normalize
        vec = F.normalize(pooled, dim=1)
        return vec  # [1,H]
    @staticmethod
    def semantic_relevance_cosine(
            *,
            prompt: str,
            response: str,
            tok: Tokenizer,
            encoder: Optional[nn.Module],
            broca_model: Optional[nn.Module],
            device: str,
            max_len: int = 192,
    ) -> float:
        """
        Cosine similarity between prompt and response embeddings.

        Uses encoder if available; otherwise falls back to Broca token_embed mean-pool.
        Returns cosine in [-1,1].
        """
        p = (prompt or "").strip()
        r = (response or "").strip()
        if not p or not r:
            return 0.0

        if encoder is not None:
            v_p = ParietalMemory._embed_text_semantic(text=p, tok=tok, encoder=encoder, device=device, max_len=max_len)  # [1,H]
            v_r = ParietalMemory._embed_text_semantic(text=r, tok=tok, encoder=encoder, device=device, max_len=max_len)  # [1,H]
            sim = float(torch.matmul(v_p, v_r.transpose(0, 1)).item())  # cosine since normalized
            return sim

        # ---- fallback: Broca token embedding mean pool ----
        if broca_model is None:
            return 0.0

        def _broca_embed(text: str) -> torch.Tensor:
            enc = tok.encode(text)
            ids_list = enc.ids[:max_len] or [0]
            ids = torch.tensor([ids_list], dtype=torch.long, device=device)  # [1,T]
            with torch.no_grad():
                tok_emb = broca_model.token_embed(ids)  # [1,T,H]
                pooled = tok_emb.mean(dim=1)  # [1,H]
            return F.normalize(pooled, dim=1)

        v_p = _broca_embed(p)
        v_r = _broca_embed(r)
        sim = float(torch.matmul(v_p, v_r.transpose(0, 1)).item())
        return sim


    def _embed_prompts_batch(self, prompts: List[str]) -> torch.Tensor:
        _jlog(logging.DEBUG, "[memory] embed_prompts_batch start", n=len(prompts), has_encoder=self.encoder is not None)
        ids, mask = self._tokenize_batch(prompts)

        with torch.no_grad():
            if self.encoder is not None:
                pooled = _encoder_forward_pooled(self.encoder, ids)  # [B,H]
                vec = F.normalize(pooled, dim=1)
                _jlog(logging.DEBUG, "[memory] embedded via encoder", vec_shape=tuple(vec.shape))
                return vec

            # fallback: mean-pool Broca token embeddings
            tok_emb = self.broca.token_embed(ids)  # [B,L,H]
            denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            pooled = (tok_emb * mask.unsqueeze(-1)).sum(dim=1) / denom
            vec = F.normalize(pooled, dim=1)
            _jlog(logging.DEBUG, "[memory] embedded via broca token_embed fallback", vec_shape=tuple(vec.shape))
            return vec

    def encode(self, text: str) -> torch.Tensor:
        """
        Turn-level encoder call (THIS is your e_query): returns [1,H], normalized.
        """
        _jlog(logging.INFO, "[memory] encode turn", n_chars=len(text))
        vec = self._embed_prompts_batch([text])
        _jlog(logging.DEBUG, "[memory] encode done", vec_shape=tuple(vec.shape))
        return vec  # [1,H]

    def rebuild_from_jsonl(self, path: Path, *, max_items: int = 2000):
        _jlog(logging.INFO, "[memory] rebuild_from_jsonl start", path=str(path), max_items=max_items)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            _jlog(logging.INFO, "[memory] jsonl read", n_lines=len(lines))
        except Exception as e:
            _jlog(logging.WARNING, "[memory] jsonl read failed", err=str(e))
            return

        lines = lines[-max_items:]
        prompts: List[str] = []
        episodes: List[Dict[str, Any]] = []

        bad_json = 0
        kept = 0
        for ln in lines:
            try:
                obj = json.loads(ln)
            except Exception:
                bad_json += 1
                continue
            p = str(obj.get("prompt", "")).strip()
            r = str(obj.get("response", "")).strip()
            ts = float(obj.get("ts", 0.0) or 0.0)

            if not p:
                continue

            snippet = f"User: {p}"
            if r:
                snippet += f"\nArdor: {r}"
            episodes.append({"snippet": snippet, "ts": ts})
            prompts.append(p)
            kept += 1

        _jlog(logging.INFO, "[memory] jsonl parsed", kept=kept, bad_json=bad_json)

        if not prompts:
            self.episodes = []
            self.index_emb = None
            _jlog(logging.WARNING, "[memory] no prompts after parse; memory empty")
            return

        # batch embed prompts
        all_vecs: List[torch.Tensor] = []
        bs = 32
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i + bs]
            _jlog(logging.DEBUG, "[memory] embedding chunk", i=i, chunk_size=len(chunk))
            all_vecs.append(self._embed_prompts_batch(chunk))
        emb = torch.cat(all_vecs, dim=0)  # (N,H)

        self.episodes = episodes
        self.index_emb = emb
        _jlog(logging.INFO, "[memory] rebuild done", episodes=len(self.episodes), emb_shape=tuple(self.index_emb.shape))

    def ingest_episode(self, prompt: str, response: str, *, prompt_vec: Optional[torch.Tensor] = None, ts: Optional[float] = None):
        """
        Add a new episode WITHOUT calling encoder again if prompt_vec is provided.
        prompt_vec should be [1,H] normalized (i.e., your e_query).
        """
        p = (prompt or "").strip()
        if not p:
            _jlog(logging.DEBUG, "[memory] ingest_episode skipped: empty prompt")
            return
        r = (response or "").strip()
        snippet = f"User: {p}"
        if r:
            snippet += f"\nArdor: {r}"

        if ts is None:
            ts = time.time()

        self.episodes.append({"snippet": snippet, "ts": float(ts)})
        _jlog(logging.DEBUG, "[memory] ingested episode", episodes=len(self.episodes), has_prompt_vec=prompt_vec is not None)

        if prompt_vec is None:
            # fallback: embed prompt (this does call encoder) — avoid if you pass e_query
            _jlog(logging.WARNING, "[memory] ingest_episode embedding prompt because prompt_vec missing")
            prompt_vec = self.encode(p)

        if self.index_emb is None:
            self.index_emb = prompt_vec.detach().to(self.device)
        else:
            self.index_emb = torch.cat([self.index_emb, prompt_vec.detach().to(self.device)], dim=0)

        # bound
        MAX = 6000
        if len(self.episodes) > MAX:
            extra = len(self.episodes) - MAX
            self.episodes = self.episodes[extra:]
            if self.index_emb is not None:
                self.index_emb = self.index_emb[extra:, :]
            _jlog(logging.INFO, "[memory] bounded memory applied", MAX=MAX, dropped=extra, now=len(self.episodes))

    def topk_from_vec(self, q_vec: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """
        q_vec: [1,H] normalized
        cosine sim = dot(index_emb, q_vec)
        """
        if self.index_emb is None or not self.episodes:
            _jlog(logging.DEBUG, "[memory] topk_from_vec empty", has_index=self.index_emb is not None, episodes=len(self.episodes))
            return []
        q = q_vec[0]  # [H]
        sims = torch.matmul(self.index_emb, q)  # [N]
        kk = min(int(k), int(sims.numel()))
        vals, idx = torch.topk(sims, k=kk)
        out: List[Tuple[str, float]] = []
        for score, j in zip(vals.tolist(), idx.tolist()):
            out.append((self.episodes[j]["snippet"], float(score)))
        _jlog(logging.INFO, "[memory] topk_from_vec", k=k, returned=len(out),
              best=(out[0][1] if out else None), worst=(out[-1][1] if out else None))
        return out


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                           Decoding helpers                        ║
# ╚═══════════════════════════════════════════════════════════════════╝
ROLE_PRIMER = (
    "Hi, You are Ardor. "
    "Answer my questions cleanly. Respond to me in friendly manner. "
    "Prefer 3-6 sentences at most, however you can extend it if you deem necessary. "
    "Always start the conversation."
)

_SPEAKER_RE_LEAD = re.compile(r"^[\s\W]*(?:User|Assistant|System)\s*:\s*", re.I)
_SPEAKER_RE_NEXT = re.compile(r"(?mi)\n(?:User|Assistant|System)\s*:\s*")

BAD_PHRASES = [
    "can you provide",
    "could you provide",
    "please provide more information",
    "please provide more details",
    "if you're looking",
    "if you are looking",
    "can you please",
    "let me know if you have any other questions",
    "feel free to ask",
    "i'm not sure if",
    "here are some examples",
    "this story should be told",
    "the story should",
    "in this story",
    "chapter ",
    "once upon a time",
    "the first step is to",
    "as the story progresses",
]

def _keywords(text: str, stopwords: set[str]) -> set[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    out = {t for t in toks if t not in stopwords}
    _jlog(logging.DEBUG, "[kw] keywords", in_len=len(text), toks=len(toks), out=len(out))
    return out

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    v = len(a & b) / max(len(a | b), 1)
    return v

def _token_ids_for_terms(tokenizer: Tokenizer, terms: set[str]) -> set[int]:
    ids: set[int] = set()
    for t in terms:
        for cand in (t, "Ġ" + t):
            tid = tokenizer.token_to_id(cand)
            if tid is not None:
                ids.add(tid)
        enc = tokenizer.encode(t).ids
        if enc:
            ids.add(enc[0])
    _jlog(logging.DEBUG, "[tok] token_ids_for_terms", terms=len(terms), ids=len(ids))
    return ids

def _token_ids_for_chars(tokenizer: Tokenizer, chars: set[str]) -> set[int]:
    ids: set[int] = set()
    for ch in chars:
        try:
            enc = tokenizer.encode(ch).ids
            if enc:
                ids.add(enc[0])
        except Exception:
            pass
    _jlog(logging.DEBUG, "[tok] token_ids_for_chars", chars="".join(sorted(chars))[:64], n_ids=len(ids))
    return ids

def _strip_speaker_tags(text: str) -> str:
    s = text.strip()
    s0 = s
    s = _SPEAKER_RE_LEAD.sub("", s)
    m = _SPEAKER_RE_NEXT.search(s)
    if m:
        s = s[: m.start()].rstrip()
    _jlog(logging.DEBUG, "[text] strip_speaker_tags", before=s0[:120], after=s[:120])
    return s

def _count_questions(text: str) -> int:
    # counts explicit question marks + common question words at line starts
    qm = text.count("?")
    qw = len(re.findall(r"(?mi)^\s*(what|why|how|when|where|who|which|can|could|should|do|does|is|are)\b", text))
    return max(qm, qw)

def _has_code(text: str) -> bool:
    t = text.lower()
    return ("```" in t) or any(x in t for x in ["def ", "class ", "#include", "import ", "using ", "{", "};"])

def _estimate_desired_tokens(prompt: str, profile: str, prompt_tok_len: int) -> int:
    """
    Predict 'appropriate' completion length (in tokens), independent of context constraints.
    Tunable via env vars.
    """
    # base per profile (these are sane defaults for your “3–6 sentences” persona)
    base_map = {
        "code":        int(os.environ.get("ARDOR_BUDGET_CODE_BASE", "220")),
        "math":        int(os.environ.get("ARDOR_BUDGET_MATH_BASE", "180")),
        "instruction": int(os.environ.get("ARDOR_BUDGET_INST_BASE", "220")),
        "qa":          int(os.environ.get("ARDOR_BUDGET_QA_BASE", "150")),
        "creative":    int(os.environ.get("ARDOR_BUDGET_CREAT_BASE", "260")),
        "general":     int(os.environ.get("ARDOR_BUDGET_GEN_BASE", "160")),
    }
    base = base_map.get(profile, base_map["general"])

    # complexity features
    qn = _count_questions(prompt)
    long_prompt_bonus = int(10 * (prompt_tok_len ** 0.5))  # grows slowly: sqrt tokens
    multi_question_bonus = min(160, 40 * max(0, qn - 1))

    code_bonus = 120 if _has_code(prompt) else 0
    list_bonus = 80 if re.search(r"(?mi)^\s*[-*]\s+", prompt) else 0

    # "short answer" signals: if user asks "shorten", "brief", etc, cut hard
    short_signal = bool(re.search(r"\b(short|brief|tl;dr|one sentence|few sentences)\b", prompt.lower()))
    if short_signal:
        base = min(base, 120)

    desired = base + long_prompt_bonus + multi_question_bonus + code_bonus + list_bonus

    # final clamp
    desired_min = int(os.environ.get("ARDOR_BUDGET_DESIRED_MIN", "64"))
    desired_max = int(os.environ.get("ARDOR_BUDGET_DESIRED_MAX", "420"))
    return max(desired_min, min(desired, desired_max))



class NgramBlocker:
    """Efficient map-based n-gram blocker (O(#blocked))."""

    def __init__(self, n: int = 4):
        self.n = max(1, int(n))
        self.window = deque(maxlen=max(0, self.n - 1))
        self.map: Dict[Tuple[int, ...], set[int]] = {}
        _jlog(logging.DEBUG, "[ngram] init", n=self.n)

    def reset(self):
        self.window.clear()
        self.map.clear()
        _jlog(logging.DEBUG, "[ngram] reset")

    def update(self, tok_id: int):
        if self.n <= 1:
            return
        if len(self.window) == self.n - 1:
            prefix = tuple(self.window)
            s = self.map.get(prefix)
            if s is None:
                self.map[prefix] = {tok_id}
            else:
                s.add(tok_id)
        self.window.append(tok_id)

    def apply(self, logits_1d: torch.Tensor):
        if self.n <= 1 or len(self.window) < self.n - 1:
            return
        prefix = tuple(self.window)
        blocked = self.map.get(prefix)
        if not blocked:
            return
        idx = torch.tensor(list(blocked), device=logits_1d.device, dtype=torch.long)
        logits_1d.index_fill_(0, idx, -float("inf"))
        if _trace_enabled():
            _jlog(logging.DEBUG, "[ngram] applied", prefix=prefix, blocked_n=len(blocked))


class PhraseBias:
    def __init__(self, tokenizer: Tokenizer, phrases: List[str], bias: float = -2.5, max_len: int = 8):
        self.tk = tokenizer
        self.bias = float(bias)
        self.max_len = int(max_len)
        self.phr_ids = [tuple(self.tk.encode(p, add_special_tokens=False).ids) for p in phrases if p.strip()]
        self.phr_ids = [p for p in self.phr_ids if p]
        _jlog(logging.INFO, "[phrase] bias init", n_phrases=len(self.phr_ids), bias=self.bias, max_len=self.max_len)

    def apply(self, logits_1d: torch.Tensor, generated_ids: List[int]):
        if not generated_ids:
            return
        tail = generated_ids[-self.max_len:]
        for ids in self.phr_ids:
            k = min(len(ids), len(tail))
            for m in range(k, 0, -1):
                if tuple(tail[-m:]) == tuple(ids[:m]):
                    if m < len(ids):
                        nxt = ids[m]
                        if 0 <= nxt < logits_1d.shape[-1]:
                            logits_1d[nxt] += self.bias
                            if _trace_enabled():
                                _jlog(logging.DEBUG, "[phrase] applied", match_len=m, next=nxt, bias=self.bias)
                    break


class EarlyQuestionTamer:
    def __init__(self, tokenizer: Tokenizer, until_tokens: int = 60, penalty: float = 0.8):
        self.tk = tokenizer
        self.until = int(until_tokens)
        self.penalty = float(penalty)
        qids = self.tk.encode("?", add_special_tokens=False).ids
        self.q_id = qids[0] if qids else None
        _jlog(logging.INFO, "[tamer] init", until=self.until, penalty=self.penalty, q_id=self.q_id)

    def apply(self, logits_1d: torch.Tensor, step: int):
        if self.q_id is None:
            return
        if step < self.until and 0 <= self.q_id < logits_1d.shape[-1]:
            logits_1d[self.q_id] -= self.penalty
            if _trace_enabled():
                _jlog(logging.DEBUG, "[tamer] applied", step=step, q_id=self.q_id, penalty=self.penalty)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                              ArdorCore                            ║
# ╚═══════════════════════════════════════════════════════════════════╝
class ArdorCore:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str],
        device: str = "cpu",
        max_len: int = 300,
        *,
        enable_retrieval: bool = True,
        encoder_ckpt: Optional[str] = None,
        enable_dmn: bool = False,
        aeternum=None,
    ):
        _jlog(logging.INFO, "[PFC] ArdorCore.__init__ start",
              model_path=model_path, tokenizer_path=tokenizer_path, device=device,
              max_len=max_len, enable_retrieval=enable_retrieval, encoder_ckpt=encoder_ckpt,
              enable_dmn=enable_dmn, aeternum_is_none=(aeternum is None))

        self.device = device
        self.gen_max_tokens = int(max_len)
        self.model_path = model_path

        # --- Load decoder (cached) ---
        self.model, schema, _model_sd = _load_broca_cached(model_path, device)
        want_vocab = int(schema.get("want_vocab") or schema.get("vocab") or 0)

        # --- Load tokenizer (cached) ---
        self.tokenizer, chosen_tok_path = _load_tokenizer_matching_vocab(tokenizer_path, want_vocab)
        self.tokenizer_path = chosen_tok_path

        # schema fields
        self.layers = schema.get("layers")
        self.heads = schema.get("heads")
        self.hidden = schema.get("hidden")
        self.model_ctx_len = schema.get("max_len")
        self.vocab_size = schema.get("vocab") or want_vocab
        self.schema = {
            "layers": self.layers,
            "heads": self.heads,
            "hidden": self.hidden,
            "max_len": self.model_ctx_len,
            "vocab": self.vocab_size,
            "mismatch": schema.get("mismatch"),
        }

        miss = self.schema["mismatch"]["missing"] if self.schema.get("mismatch") else None
        unex = self.schema["mismatch"]["unexpected"] if self.schema.get("mismatch") else None
        miss_ct = None if miss is None else len(miss)
        unex_ct = None if unex is None else len(unex)

        print(
            f"🧠 Model schema: layers={self.layers} heads={self.heads} hidden={self.hidden} "
            f"max_len={self.model_ctx_len} mismatch: missing={miss_ct} unexpected={unex_ct}"
        )
        _jlog(logging.INFO, "[PFC] schema", layers=self.layers, heads=self.heads, hidden=self.hidden, max_len=self.model_ctx_len, missing=miss_ct, unexpected=unex_ct)

        try:
            emb_rows = getattr(getattr(self.model, "token_embed", None), "weight", None)
            emb_rows = int(emb_rows.shape[0]) if emb_rows is not None else int(self.vocab_size)
        except Exception as e:
            _jlog(logging.WARNING, "[PFC] token_embed rows inspect failed", err=str(e))
            emb_rows = int(self.vocab_size)
        print(f"🧩 Tokenizer: {self.tokenizer_path}  | vocab={self.vocab_size}  embed={emb_rows}")
        _jlog(logging.INFO, "[PFC] tokenizer", path=self.tokenizer_path, vocab=self.vocab_size, embed_rows=emb_rows)

        # --- Retrieval enablement ---
        resolved_encoder_ckpt: Optional[str] = None
        if encoder_ckpt is not None:
            _jlog(logging.INFO, "[PFC] encoder_ckpt provided explicitly", encoder_ckpt=encoder_ckpt)
            resolved_encoder_ckpt = _resolve_encoder_ckpt(encoder_ckpt)
            self.enable_retrieval = True
        else:
            if enable_retrieval:
                _jlog(logging.INFO, "[PFC] retrieval enabled; auto-resolving encoder ckpt")
                resolved_encoder_ckpt = _resolve_encoder_ckpt(None)  # may raise
                self.enable_retrieval = True
            else:
                _jlog(logging.INFO, "[PFC] retrieval disabled by flag")
                self.enable_retrieval = False

        # --- Load encoder (cached) ---
        self.encoder: Optional[nn.Module] = None
        if self.enable_retrieval and resolved_encoder_ckpt:
            try:
                _jlog(logging.INFO, "[PFC] loading encoder", ckpt=resolved_encoder_ckpt)
                self.encoder = _load_encoder_cached(
                    resolved_encoder_ckpt,
                    device=self.device,
                    vocab_size=int(self.vocab_size),
                    hidden=int(self.hidden or 384),
                    heads=int(self.heads or 6),
                )
                _jlog(logging.INFO, "[PFC] encoder ready", encoder_type=str(type(self.encoder)))
            except Exception as e:
                print(f"[encoder] failed to load, retrieval OFF: {e}")
                _jlog(logging.ERROR, "[PFC] encoder load failed; retrieval OFF", err=str(e))
                self.enable_retrieval = False
                self.encoder = None

        # --- Aeternum (lazy import to avoid circular double-brain construction) ---
        if aeternum is not None:
            _jlog(logging.INFO, "[PFC] aeternum injected externally", aeternum_type=str(type(aeternum)))
            self.aet = aeternum
        else:
            # IMPORTANT: import here, not at module import time.
            try:
                _jlog(logging.INFO, "[PFC] importing/init Aeternum bridge")
                from Aeternum.aeternum_bridge import init_aeternum  # type: ignore
                self.aet = init_aeternum(
                    device=self.device,
                    vad_csv_path=os.environ.get("ARDOR_VAD_CSV", None),
                    state_path=os.environ.get("ARDOR_AET_STATE", None),
                )
                _jlog(logging.INFO, "[PFC] Aeternum init OK", aet_type=str(type(self.aet)))
            except Exception as e:
                print(f"[PFC] Aeternum init failed, continuing without emotion core: {e}")
                _jlog(logging.ERROR, "[PFC] Aeternum init failed", err=str(e))
                self.aet = None

        # --- Hippocampus / ParietalMemory ---
        self.parietal: Optional[ParietalMemory] = None
        if self.enable_retrieval:
            mem_jsonl = os.environ.get("ARDOR_MEMORY_JSONL", str(LOG_FILE))
            _jlog(logging.INFO, "[PFC] init ParietalMemory", mem_jsonl=mem_jsonl)
            self.parietal = ParietalMemory(
                self.tokenizer,
                self.device,
                broca_model=self.model,
                encoder=self.encoder,
                memory_jsonl=mem_jsonl,
                max_items=int(os.environ.get("ARDOR_MEMORY_MAX_ITEMS", "2000")),
                max_len=int(os.environ.get("ARDOR_ENCODER_MAXLEN", "192")),
            )
        else:
            _jlog(logging.INFO, "[PFC] ParietalMemory skipped (retrieval disabled)")

        # rolling convo memory
        self.recent_texts = deque(maxlen=128)
        self.chat_turns: List[Tuple[str, str]] = []  # list of ("user"/"assistant", text)
        self.max_chat_turns = int(os.environ.get("ARDOR_CHAT_TURNS", "12"))  # last N user+assistant turns
        self.stopwords = STOPWORDS
        self._digit_ids = _token_ids_for_chars(self.tokenizer, set("0123456789"))
        self._eos_id = _find_eos_id(self.tokenizer)
        self._eot_id = self.tokenizer.token_to_id("<|eot|>")

        self.enable_dmn = bool(enable_dmn)
        self.dmn = None
        self._dmn_blocker = None
        self._last_dmn_takeaway = None
        self._last_dmn_state = None
        if self.enable_dmn:
            try:
                from Cerebrum.DMN import InsideASyntheticThought
                self.dmn = InsideASyntheticThought()
                self._last_dmn_state = self.dmn.get_state()
                _jlog(logging.INFO, "[PFC] DMN init OK")
            except Exception as e:
                self.dmn = None
                self._dmn_blocker = str(e)
                print(f"[PFC] DMN init failed, continuing without DMN: {e}")
                _jlog(logging.ERROR, "[PFC] DMN init failed", err=str(e))
        else:
            _jlog(logging.INFO, "[PFC] DMN disabled")

        _jlog(logging.INFO, "[PFC] init done",
              gen_max_tokens=self.gen_max_tokens, max_chat_turns=self.max_chat_turns,
              eos_id=self._eos_id, eot_id=self._eot_id, digit_ids=len(self._digit_ids),
              dmn_enabled=self.enable_dmn, dmn_ready=self.dmn is not None)



    # ───────────────────── prompt classification (heuristics) ─────────────────────
    @staticmethod
    def classify_prompt(prompt: str) -> str:
        p = prompt.strip().lower()
        if ("```" in prompt) or any(t in p for t in ["def ", "class ", "import ", ";", "{", "};", "#include", ">>>"]):
            return "code"
        if re.search(r"\b(sum|integral|limit|theorem|lemma|proof|equation|matrix|vector)\b", p):
            return "math"
        if re.search(r"\b(write|story|poem|metaphor|imagine|creative)\b", p):
            return "creative"
        if re.match(r"^(what|who|when|where|why|how)\b", p):
            return "qa"
        if re.search(r"\b(step|steps|guide|instructions|checklist|bullet)\b", p):
            return "instruction"
        return "general"



    # ───────────────────── metrics/scoring ─────────────────────
    def _text_metrics(self, text: str, prompt: str) -> Dict[str, float]:
        toks = re.findall(r"\w+|[^\w\s]", text)
        n = len(toks)
        if n == 0:
            return {"d1": 0.0, "d2": 0.0, "rep3": 1.0, "closure": 0.0, "imbalance": 1.0, "rel": 0.0}
        d1 = len(set(toks)) / max(1, n)
        bigrams = [tuple(toks[i: i + 2]) for i in range(max(0, n - 1))]
        d2 = (len(set(bigrams)) / max(1, len(bigrams))) if bigrams else 0.0
        trigrams = [tuple(toks[i: i + 3]) for i in range(max(0, n - 2))]
        rep3 = 0.0
        if trigrams:
            counts: Dict[Tuple[str, str, str], int] = {}
            for t in trigrams:
                counts[t] = counts.get(t, 0) + 1
            repeats = sum(c - 1 for c in counts.values() if c > 1)
            rep3 = repeats / max(1, len(trigrams))
        closure = 1.0 if re.search(r"[.!?]\"?$|[.!?]$|\n$", text.strip()) and n > 6 else 0.0

        def _imb(a, b):
            return abs(text.count(a) - text.count(b))

        imb = (_imb("(", ")") + _imb("[", "]") + _imb("{", "}") + (_imb('"', '"') % 2))
        imbalance = min(1.0, imb / 3.0)
        jac = _jaccard(_keywords(prompt, self.stopwords), _keywords(text, self.stopwords))

        # semantic cosine in [-1,1]
        cos = ParietalMemory.semantic_relevance_cosine(
            prompt=prompt,
            response=text,
            tok=self.tokenizer,
            encoder=self.encoder,  # uses your encoder if present
            broca_model=self.model,  # fallback
            device=self.device,
            max_len=int(os.environ.get("ARDOR_ENCODER_MAXLEN", "192")),
        )

        # map cosine -> [0,1] so it blends nicely with jaccard
        cos01 = (cos + 1.0) * 0.5

        # blend weight (env-tunable)
        w = float(os.environ.get("ARDOR_REL_SEM_W", "0.70"))  # 0.7 means mostly semantic
        rel = w * cos01 + (1.0 - w) * jac

        m = {"d1": d1, "d2": d2, "rep3": rep3, "closure": closure, "imbalance": imbalance, "rel": rel}
        _jlog(logging.DEBUG, "[metrics] computed", **m)
        return m

    @staticmethod
    def _apply_top_k_(logits: torch.Tensor, k: int):
        if k and k > 0 and k < logits.size(-1):
            th = torch.topk(logits, k).values[..., -1, None]
            logits[logits < th] = -float("inf")

    @staticmethod
    def _nucleus_pick(probs: torch.Tensor, top_p: float) -> int:
        sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
        keep = sorted_p.cumsum(-1) <= top_p
        keep[..., 0] = True
        sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        denom = sorted_p.sum()
        sorted_p = sorted_p / denom if denom.item() != 0 else torch.softmax(sorted_p, dim=-1)
        pick = torch.multinomial(sorted_p.squeeze(0), 1).item()
        return sorted_idx.squeeze(0)[pick].item()

    def log(self, prompt: str, resp: str, *, prompt_vec: Optional[torch.Tensor] = None):
        _jlog(logging.INFO, "[log] writing episode", prompt_len=len(prompt), resp_len=len(resp), has_prompt_vec=prompt_vec is not None)
        with LOG_FILE.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"prompt": prompt, "response": resp, "ts": time.time()}, ensure_ascii=False) + "\n")
            self.recent_texts.append(prompt[:600])
            self.recent_texts.append(resp[:600])

        # live hippocampus update WITHOUT calling encoder again if prompt_vec provided
        if self.enable_retrieval and self.parietal is not None:
            try:
                self.parietal.ingest_episode(prompt, resp, prompt_vec=prompt_vec)
                _jlog(logging.DEBUG, "[log] memory ingest ok")
            except Exception as e:
                _jlog(logging.WARNING, "[log] memory ingest failed", err=str(e))

    def pick_decoding_config(self, prompt: str, *, profile: Optional[str] = None, probe_len: int = 56) -> Dict[str, Any]:
        profile = profile or self.classify_prompt(prompt)
        _jlog(logging.INFO, "[decode] pick_decoding_config start", profile=profile, probe_len=probe_len)
        base = [
            {"name": "precise", "temperature": 0.55, "top_p": 0.80, "top_k": 30, "rep_penalty": 1.25},
            {"name": "balanced", "temperature": 0.70, "top_p": 0.90, "top_k": 40, "rep_penalty": 1.15},
            {"name": "creative", "temperature": 0.95, "top_p": 0.95, "top_k": 60, "rep_penalty": 1.10},
            {"name": "focused", "temperature": 0.60, "top_p": 0.85, "top_k": 20, "rep_penalty": 1.25},
        ]
        if profile in ("code", "math"):
            cand = [base[0], base[3], {**base[0], "top_p": 0.75, "top_k": 25}]
        elif profile == "creative":
            cand = [base[2], base[1], {**base[2], "temperature": 1.05, "top_p": 0.97}]
        elif profile in ("qa", "instruction"):
            cand = [base[1], base[0], base[3]]
        else:
            cand = [base[1], base[0], base[3], base[2]]

        results = []
        for cfg in cand:
            _jlog(logging.INFO, "[decode] probe cfg", **cfg)
            txt = self.generate_text(
                prompt,
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                top_k=cfg["top_k"],
                rep_penalty=cfg["rep_penalty"],
                min_new_tokens=8,
                max_new_tokens=probe_len,
                suppress_vague=True,
                typical_p=0.30,
                auto_pick=False,
                profile=profile,
                log_response=False,
                polish_output=False,
            )
            m = self._text_metrics(txt, prompt)
            score = (
                0.40 * m["rel"]
                + 0.20 * m["d1"]
                + 0.15 * m["d2"]
                + 0.15 * m["closure"]
                - 0.15 * m["rep3"]
                - 0.10 * m["imbalance"]
            )
            results.append({**cfg, "score": score, "metrics": m, "sample": txt})
            _jlog(logging.INFO, "[decode] probe result", name=cfg["name"], score=score, rel=m["rel"], rep3=m["rep3"], closure=m["closure"])
        results.sort(key=lambda r: (-r["score"], r["temperature"] if profile in ("code", "math") else 0.0))
        _jlog(logging.INFO, "[decode] best cfg selected", best=results[0]["name"], score=results[0]["score"])
        return results[0]

    def conversation_memory_last_N_chunks(self, N: int = 32) -> List[str]:
        seq = list(self.recent_texts)
        if N <= 0:
            return []
        out = seq[-2 * N:]
        _jlog(logging.DEBUG, "[memory] last N chunks", N=N, returned=len(out))
        return out

    def _strip_boilerplate(self, text: str) -> str:
        """
        Removes recurring instruction boilerplate the model may emit.
        Also prevents storing it back into history (contamination loop).
        """
        t = (text or "").strip()

        # drop leading instruction boilerplate (common in some fine-tunes)
        bad_leads = [
            r"^Answer directly and concretely\.\s*No disclaimers\.\s*",
            r"^Answer directly and concretely\.\s*",
            r"^No disclaimers\.\s*",
        ]
        for pat in bad_leads:
            t2 = re.sub(pat, "", t, flags=re.IGNORECASE)
            if t2 != t:
                _jlog(logging.INFO, "[text] boilerplate stripped", pattern=pat)
            t = t2

        return t.strip()

    def _should_store_in_history(self, prompt: str, resp: str, relevance_floor: float, rep3_max: float = 0.18) -> bool:
        m = self._text_metrics(resp, prompt)
        # slightly softer for short prompts (relevance is harder to hit)
        short = len(prompt.strip()) < 60
        rel_floor = relevance_floor * (0.60 if short else 1.0)

        ok = (m["rel"] >= rel_floor) and (m["rep3"] <= rep3_max) and (m["imbalance"] <= 0.6)
        _jlog(logging.INFO, "[history] store gate", ok=ok, rel=m["rel"], rel_floor=rel_floor, rep3=m["rep3"],
              imbalance=m["imbalance"])
        return bool(ok)

    def _clean_for_history(self, text: str) -> str:
        """
        What we store into chat history must be 'safe' (no meta headers, no special tokens).
        """
        t = (text or "").strip()

        # remove accidental special tokens if emitted
        for tok in ("<|system|>", "<|user|>", "<|assistant|>", "<|eot|>"):
            if tok in t:
                _jlog(logging.DEBUG, "[text] removing special token", tok=tok)
            t = t.replace(tok, "")

        # remove boilerplate if it slipped in
        t = self._strip_boilerplate(t)

        return t.strip()



    def _build_chat_prompt(self, persona_primer: str, user_text: str) -> str:
        """
        Builds a stable dialogue prompt using role tags when available.
        Falls back to 'User:' / 'Ardor:' tags if special tokens don't exist.
        """
        sys_block = (persona_primer or "").strip()
        if not sys_block:
            sys_block = "You are Ardor. Stay in-context. Be helpful. Speak naturally."

        # detect whether tokenizer supports special chat tokens
        has_user = self.tokenizer.token_to_id("<|user|>") is not None
        has_asst = self.tokenizer.token_to_id("<|assistant|>") is not None
        has_eot = self.tokenizer.token_to_id("<|eot|>") is not None

        turns = self.chat_turns[-(self.max_chat_turns * 2):]
        _jlog(logging.DEBUG, "[prompt] build_chat_prompt", has_user=has_user, has_asst=has_asst, has_eot=has_eot, turns=len(turns))

        if has_user and has_asst and has_eot:
            parts = [f"<|system|>\n{sys_block}\n<|eot|>\n"]
            for role, msg in turns:
                msg = self._clean_for_history(msg)
                if not msg:
                    continue
                if role == "user":
                    parts.append(f"<|user|>\n{msg}\n<|eot|>\n")
                else:
                    parts.append(f"<|assistant|>\n{msg}\n<|eot|>\n")
            user_text = self._clean_for_history(user_text)
            parts.append(f"<|user|>\n{user_text}\n<|eot|>\n<|assistant|>\n")
            composed = "".join(parts)
            _jlog(logging.DEBUG, "[prompt] composed with chat tokens", chars=len(composed))
            return composed

        # fallback (no special tokens)
        parts = [f"System: {sys_block}\n\n"]
        for role, msg in turns:
            msg = self._clean_for_history(msg)
            if not msg:
                continue
            if role == "user":
                parts.append(f"User: {msg}\n")
            else:
                parts.append(f"Ardor: {msg}\n")
        user_text = self._clean_for_history(user_text)
        parts.append(f"User: {user_text}\nArdor: ")
        composed = "".join(parts)
        _jlog(logging.DEBUG, "[prompt] composed fallback", chars=len(composed))
        return composed

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.65,
        top_p: float = 0.90,
        rep_penalty: float = 1.2,
        ngram_block: int = 0,
        persona_primer: str = ROLE_PRIMER,
        relevance_floor: float = 0.25,
        retry_tighter: Tuple[float, float] = (0.55, 0.70),
        suppress_vague: bool = True,
        min_new_tokens: int = 16,
        max_new_tokens: int = 300,
        top_k: int = 40,
        typical_p: float = 0.95,
        min_temp: float = 0.35,
        *,
        auto_pick: bool = False,
        stop_on_eos: bool = True,
        profile: Optional[str] = None,
        log_response: bool = True,
        polish_output: bool = True,
        enable_retrieval: Optional[bool] = None,
    ) -> str:

        t_start = time.time()
        orig_user_prompt = prompt.strip()
        use_retrieval = self.enable_retrieval if enable_retrieval is None else bool(enable_retrieval)
        hard_fallback = False
        dmn_retrieved_traces: List[str] = []
        aet_decision = None

        _jlog(logging.INFO, "[PFC] generate_text start",
              prompt_len=len(orig_user_prompt), temperature=temperature, top_p=top_p, rep_penalty=rep_penalty,
              ngram_block=ngram_block, relevance_floor=relevance_floor, retry_tighter=retry_tighter,
              suppress_vague=suppress_vague, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens,
              top_k=top_k, typical_p=typical_p, min_temp=min_temp, auto_pick=auto_pick, stop_on_eos=stop_on_eos,
              profile=profile, log_response=log_response, polish_output=polish_output, use_retrieval=use_retrieval)

        # after: generated = ids[0].tolist(); prompt_len = len(generated)

        # AUTO-BUDGET: if max_new_tokens is None or <=0, compute from context
        if max_new_tokens is None or int(max_new_tokens) <= 0:
            # keep a little safety margin so we don't hit the hard context edge
            safety = int(os.environ.get("ARDOR_CTX_SAFETY", "64"))
            ctx = int(self.model_ctx_len or 2048)
            budget = max(16, ctx - len(orig_user_prompt) - safety)
            max_new_tokens = int(os.environ.get("ARDOR_MAX_NEW_TOKENS", str(budget)))
        else:
            max_new_tokens = int(max_new_tokens)

        # ALSO: gen_max_tokens should not be the bottleneck
        # cap the step-loop to the same effective budget (or keep it larger)
        self.gen_max_tokens = max(self.gen_max_tokens, max_new_tokens)

        # ─────────────────────────────────────────────────────────────
        # ✅ TURN-LEVEL encoder call ONCE:
        # e_query = parietal.encode(user_text) -> [1,384] normalized
        # ─────────────────────────────────────────────────────────────
        e_query: Optional[torch.Tensor] = None
        if self.parietal is not None:
            try:
                _jlog(logging.INFO, "[PFC] parietal.encode begin")
                e_query = self.parietal.encode(orig_user_prompt).to(self.device)  # [1,H]
                _jlog(logging.INFO, "[PFC] parietal.encode ok", e_shape=tuple(e_query.shape))
            except Exception as e:
                print(f"[PFC] parietal.encode failed: {e}")
                _jlog(logging.ERROR, "[PFC] parietal.encode failed", err=str(e))
                e_query = None
        else:
            _jlog(logging.INFO, "[PFC] parietal is None (no retrieval memory object)")

        # ─────────────────────────────────────────────────────────────
        # ✅ Hippocampus cosine top-k using the SAME e_query
        # Feed snippets back as context
        # ─────────────────────────────────────────────────────────────
        if use_retrieval and (self.parietal is not None) and (e_query is not None):
            _jlog(logging.INFO, "[PFC] memory topk begin", k=8)
            hits = self.parietal.topk_from_vec(e_query, k=8)
            # ── STRONG RETRIEVAL GATE ──────────────────────────────
            best_thr = float(os.environ.get("ARDOR_MEM_BEST_MIN", "0.55"))
            margin_thr = float(os.environ.get("ARDOR_MEM_MARGIN_MIN", "0.03"))
            base_thr = float(os.environ.get("ARDOR_MEM_MIN_SIM", "0.20"))  # still used as a floor

            # Keep only >= base_thr first (cheap cleanup)
            filtered = [(t, s) for (t, s) in hits if s >= base_thr]

            best = filtered[0][1] if len(filtered) >= 1 else None
            second = filtered[1][1] if len(filtered) >= 2 else None
            margin = (best - second) if (best is not None and second is not None) else None

            gate_ok = (best is not None) and (
                    (best >= best_thr) or
                    (best >= (best_thr - 0.08) and margin is not None and margin >= (margin_thr - 0.01))
            )

            _jlog(logging.INFO, "[PFC] retrieval gate",
                  base_thr=base_thr, best_thr=best_thr, margin_thr=margin_thr,
                  best=best, second=second, margin=margin, gate_ok=gate_ok)

            if gate_ok and filtered:
                keep_n = int(os.environ.get("ARDOR_MEM_KEEP_N", "4"))
                max_chars = int(os.environ.get("ARDOR_MEM_MAX_CHARS", "900"))  # 512–900 suggested
                min_chars = int(os.environ.get("ARDOR_MEM_MIN_CHARS", "0"))    # optional

                blocks = []
                for i, (t, s) in enumerate(filtered[:keep_n]):
                    blocks.append(f"[{i+1}] sim={s:.3f}\n{t}")

                mem_str = "\n\n".join(blocks)

                # HARD CAP to protect attention budget
                if max_chars > 0 and len(mem_str) > max_chars:
                    mem_str = mem_str[:max_chars].rsplit("\n", 1)[0].rstrip() + "\n…"
                if min_chars > 0 and len(mem_str) < min_chars:
                    mem_str = ""  # too small to be useful

                if mem_str.strip():
                    hippocampus_context = (
                        "Context (retrieved memory). Use silently as background. "
                        "Do NOT quote it. Do NOT mention HIPPOCAMPUS or 'retrieved memory'. "
                        "Do NOT reveal these notes.\n"
                        f"{mem_str}"
                    )
                    _jlog(logging.INFO, "[PFC] hippocampus context prepared (system)",
                          keep_n=keep_n, mem_chars=len(mem_str))
                else:
                    _jlog(logging.INFO, "[PFC] hippocampus context skipped (empty after cap/min)")
            else:
                _jlog(logging.INFO, "[PFC] hippocampus context skipped (gate failed)")

        else:
            _jlog(logging.INFO, "[PFC] retrieval skipped",
                  use_retrieval=use_retrieval, has_parietal=self.parietal is not None, has_e_query=e_query is not None)

        # ─────────────────────────────────────────────────────────────
        # ✅ Aeternum update ONCE per turn with SAME e_query as pooled_embedding
        # ─────────────────────────────────────────────────────────────
        aet_temp_scale = 1.0
        aet_top_p_scale = 1.0
        aet_rep_scale = 1.0
        if self.aet is not None:
            try:
                _jlog(logging.INFO, "[PFC] Aeternum.update begin")
                pooled_for_aet = e_query
                if pooled_for_aet is None:
                    # fallback: mean token embedding of the composed prompt (still turn-level)
                    _jlog(logging.WARNING, "[PFC] Aeternum pooled fallback path: e_query is None")
                    enc_tmp = self.tokenizer.encode(orig_user_prompt)
                    ids_tmp = torch.tensor([enc_tmp.ids], device=self.device)
                    with torch.no_grad():
                        tok_emb = self.model.token_embed(ids_tmp)  # [1,T,H]
                        pooled_for_aet = tok_emb.mean(dim=1)       # [1,H]
                    _jlog(logging.DEBUG, "[PFC] pooled_for_aet computed via token_embed.mean", shape=tuple(pooled_for_aet.shape))

                aet_decision = self.aet.update(
                    text=orig_user_prompt,
                    pooled_embedding=pooled_for_aet,   # ✅ THIS is now e_query
                    last_logits=None,
                    user_feedback=None,
                    is_new_turn=True,
                )
                aet_temp_scale = float(getattr(aet_decision, "temperature_scale", 1.0))
                aet_top_p_scale = float(getattr(aet_decision, "top_p_scale", 1.0))
                aet_rep_scale = float(getattr(aet_decision, "rep_penalty_scale", 1.0))
                _jlog(logging.INFO, "[PFC] Aeternum.update ok", temp_scale=aet_temp_scale, top_p_scale=aet_top_p_scale, rep_scale=aet_rep_scale)
            except Exception as e:
                print(f"[PFC] Aeternum turn-update failed: {e}")
                _jlog(logging.ERROR, "[PFC] Aeternum.update failed", err=str(e))
        else:
            _jlog(logging.INFO, "[PFC] Aeternum is None (emotion core disabled)")

        # ─────────────────────────────────────────────────────────────
        # Persona / scaffolding
        # ─────────────────────────────────────────────────────────────
        spec = {
            "user": self.tokenizer.token_to_id("<|user|>"),
            "asst": self.tokenizer.token_to_id("<|assistant|>"),
            "eot": self.tokenizer.token_to_id("<|eot|>"),
        }
        _jlog(logging.DEBUG, "[prompt] token ids", user=spec["user"], asst=spec["asst"], eot=spec["eot"])

        ctx = int(self.model_ctx_len or 2048)
        safety = int(os.environ.get("ARDOR_CTX_SAFETY", "96"))

        sys_primer = (persona_primer or "")
        if 'hippocampus_context' in locals() and hippocampus_context:
            sys_primer = sys_primer.rstrip() + "\n\n[CONTEXT]\n" + hippocampus_context + "\n[/CONTEXT]\n"

        composed = self._build_chat_prompt(sys_primer, orig_user_prompt)

        _jlog(logging.INFO, "[prompt] composed ready", chars=len(composed), head=composed[:160])

        enc = self.tokenizer.encode(composed)
        ids = torch.tensor([enc.ids], device=self.device)
        _jlog(logging.INFO, "[tok] encoded composed", n_ids=len(enc.ids), ids_shape=tuple(ids.shape))

        prompt_tok_len = len(enc.ids)
        ctx_cap = max(1, ctx - prompt_tok_len - safety)

        # If caller didn't specify max_new_tokens, compute an appropriate amount
        if max_new_tokens is None or int(max_new_tokens) <= 0:
            prof = profile or self.classify_prompt(orig_user_prompt)
            desired = _estimate_desired_tokens(orig_user_prompt, prof, prompt_tok_len)
            hard_max = int(os.environ.get("ARDOR_MAX_NEW_TOKENS_HARD", "520"))
            max_new_tokens = min(desired, ctx_cap, hard_max)
        else:
            max_new_tokens = min(int(max_new_tokens), ctx_cap)

        # enforce minimum new tokens if possible
        min_new_tokens = int(min_new_tokens)
        max_new_tokens = max(min_new_tokens, int(max_new_tokens))

        # gen loop cap should follow budget, not exceed it wildly
        self.gen_max_tokens = int(max_new_tokens)

        _jlog(logging.INFO, "[budget] computed",
              ctx=ctx, safety=safety, prompt_tok_len=prompt_tok_len, ctx_cap=ctx_cap,
              max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, profile=(profile or None))

        want = _keywords(orig_user_prompt, self.stopwords)
        key_ids = _token_ids_for_terms(self.tokenizer, want)
        BAD_START = _token_ids_for_chars(self.tokenizer, {'"', '“', '”', '—', '–', '…', "''", "'", '’', '•'})

        nblock = NgramBlocker(n=max(1, int(ngram_block)))
        phrase_bias = PhraseBias(self.tokenizer, BAD_PHRASES, bias=-2.5, max_len=8)
        early_q = EarlyQuestionTamer(self.tokenizer, until_tokens=60, penalty=0.8)
        nblock.reset()

        eos_id = _find_eos_id(self.tokenizer)
        eot_id = self.tokenizer.token_to_id("<|eot|>")
        eos_like = [t for t in (eos_id, eot_id) if t is not None]
        _jlog(logging.INFO, "[decode] eos_like", eos_id=eos_id, eot_id=eot_id, eos_like=eos_like)

        generated = ids[0].tolist()
        prompt_len = len(generated)
        first_token = True
        eos_bias_steps = 32

        role_user_tok = spec["user"]
        role_asst_tok = spec["asst"]

        def _soft_bias_logits(logits: torch.Tensor, ids_set: set[int], delta: float) -> torch.Tensor:
            if ids_set:
                logits = logits.clone()
                logits[0, list(ids_set)] += delta
            return logits

        def _model_forward(input_ids: torch.Tensor) -> torch.Tensor:
            out = self.model(input_ids)
            if isinstance(out, (list, tuple)) and len(out) > 0:
                out = out[0]
            if isinstance(out, dict) and "logits" in out:
                out = out["logits"]
            return out

        def decode_once(temp: float, p: float) -> str:
            nonlocal ids, generated, first_token

            # ✅ FIX: use the correct Aeternum scales
            temp_local = float(temp) * float(aet_temp_scale)
            p_local = float(p) * float(aet_top_p_scale)
            rp_penalty = float(rep_penalty) * float(aet_rep_scale)

            # safety clamps
            temp_local = max(0.05, min(2.0, temp_local))
            p_local = max(0.05, min(0.999, p_local))
            rp_penalty = max(1.0, min(2.0, rp_penalty))

            _jlog(logging.INFO, "[decode] decode_once begin",
                  temp=temp, top_p=p, temp_local=temp_local, p_local=p_local, rp_penalty=rp_penalty,
                  gen_max_tokens=self.gen_max_tokens)

            for step in range(self.gen_max_tokens):
                with torch.no_grad():
                    logits = _model_forward(ids)[:, -1, :]  # (1, V)

                    if _trace_enabled() and step < 4:
                        _jlog(logging.DEBUG, "[decode] logits snapshot", step=step, logits_shape=tuple(logits.shape))

                    # ✅ Aeternum bias is allowed per-token, but we do NOT call encoder here.
                    if self.aet is not None:
                        try:
                            logits_1d = logits[0]
                            logits_1d = self.aet.apply_bias(self.tokenizer, logits_1d)
                            logits = logits_1d.unsqueeze(0)
                            if _trace_enabled() and step < 6:
                                _jlog(logging.DEBUG, "[decode] Aeternum bias applied", step=step)
                        except Exception as e:
                            print(f"[PFC] Aeternum bias failed: {e}")
                            _jlog(logging.WARNING, "[decode] Aeternum bias failed", err=str(e))

                    # repetition penalty over last 256 tokens
                    win = generated[-256:]
                    if win and rp_penalty and rp_penalty != 1.0:
                        rp = max(1.0, float(rp_penalty))
                        idx = torch.tensor(list(set(win)), device=logits.device, dtype=torch.long)
                        vals = logits[0, idx]
                        logits[0, idx] = torch.where(vals > 0, vals / rp, vals * rp)
                        if _trace_enabled() and step < 6:
                            _jlog(logging.DEBUG, "[decode] repetition penalty applied", step=step, unique=len(set(win)), rp=rp)

                    # discourage immediate token repeat
                    if len(generated) > 0:
                        last_tok = generated[-1]
                        logits[0, last_tok] -= 1.25

                    # no-repeat-ngram
                    if ngram_block and ngram_block > 1:
                        nblock.apply(logits[0])

                    # topical bias + digit suppression + role suppression
                    logits = _soft_bias_logits(logits, key_ids, +0.10)
                    logits = _soft_bias_logits(logits, self._digit_ids, -0.50)
                    if role_user_tok is not None:
                        logits[0, role_user_tok] -= 5.0
                    if role_asst_tok is not None:
                        logits[0, role_asst_tok] -= 5.0

                    # avoid opening with stray quotes/dashes
                    if first_token and BAD_START:
                        logits = _soft_bias_logits(logits, BAD_START, -1.5)
                    if first_token:
                        alpha_ids = _token_ids_for_chars(
                            self.tokenizer,
                            set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
                        )
                        logits = _soft_bias_logits(logits, alpha_ids, +0.15)

                    # style suppressors
                    if suppress_vague:
                        phrase_bias.apply(logits[0], generated)
                        gen_steps = len(generated) - prompt_len
                        early_q.apply(logits[0], step=gen_steps)

                    # early EOS/EOT control (EOS is NEVER banned permanently)
                    if eos_like:
                        logits = logits.clone()
                        gen_steps = len(generated) - prompt_len
                        for eos_tok in eos_like:
                            if gen_steps < min_new_tokens:
                                logits[0, eos_tok] = -float("inf")
                            elif gen_steps < eos_bias_steps:
                                logits[0, eos_tok] -= 3.0

                    # top-k + nucleus + annealed temperature
                    ArdorCore._apply_top_k_(logits, top_k)
                    frac = step / max(1, self.gen_max_tokens - 1)
                    dyn_temp = max(min_temp, temp_local * (0.7 + 0.3 * (1 - frac)))
                    probs = torch.softmax(logits / max(dyn_temp, 1e-5), dim=-1)

                    # typical sampling (optional)
                    if typical_p and 0.0 < typical_p < 1.0:
                        eps = 1e-8
                        logp = (probs + eps).log()
                        H = -(probs * logp).sum(dim=-1, keepdim=True)
                        typicality = (-(logp) - H).abs().squeeze(0)
                        order = typicality.argsort()
                        csum = probs[0, order].cumsum(dim=0)
                        keep = order[csum <= float(typical_p)]
                        if keep.numel() > 0:
                            mask = torch.zeros_like(probs)
                            mask[0, keep] = probs[0, keep]
                            denom = mask.sum()
                            probs = mask / (denom + eps)

                    next_tok = ArdorCore._nucleus_pick(probs, p_local)

                generated.append(next_tok)
                first_token = False
                if ngram_block and ngram_block > 1:
                    nblock.update(next_tok)

                new_len = len(generated) - prompt_len
                if _trace_enabled() and step < 10:
                    _jlog(logging.DEBUG, "[decode] step", step=step, next_tok=next_tok, new_len=new_len)

                if new_len >= max_new_tokens:
                    _jlog(logging.INFO, "[decode] stopping: max_new_tokens reached", new_len=new_len, max_new_tokens=max_new_tokens)
                    break

                if eos_like and (next_tok in eos_like) and new_len >= min_new_tokens:
                    _jlog(logging.INFO, "[decode] stopping: eos_like token generated", tok=next_tok, new_len=new_len)
                    if stop_on_eos or True:
                        break

                ids = torch.tensor([generated], device=self.device)

            out = self.tokenizer.decode(generated[prompt_len:], skip_special_tokens=True)
            _jlog(logging.INFO, "[decode] decode_once done", out_chars=len(out))
            return out.strip()

        # first pass
        _jlog(logging.INFO, "[decode] first pass begin")
        out1 = decode_once(temperature, top_p)
        _jlog(logging.INFO, "[decode] first pass done", out_chars=len(out1))

        out1 = _strip_speaker_tags(out1).strip()
        if re.match(r"(?i)^(this|the)\s+story\s+(should|must)\b", out1) or out1.lower().startswith("chapter "):
            _jlog(logging.INFO, "[decode] story/chapter guard triggered")
            out1 = re.sub(r"^.*?(?=\b[a-z])", "", out1, flags=re.I).strip()
        out1 = polish(out1) if polish_output else out1
        _jlog(logging.INFO, "[post] polish applied" if polish_output else "[post] polish skipped", out_chars=len(out1))

        # retry if low topicality
        got1 = _keywords(out1, self.stopwords)
        rel1 = _jaccard(want, got1)
        _jlog(logging.INFO, "[score] relevance after first pass", rel1=rel1, floor=relevance_floor)

        if rel1 < relevance_floor and not auto_pick:
            _jlog(logging.INFO, "[decode] retry path entered", retry_tighter=retry_tighter)
            enc2 = self.tokenizer.encode(composed)
            ids = torch.tensor([enc2.ids], device=self.device)
            generated = ids[0].tolist()
            first_token = True
            nblock.reset()

            out2 = decode_once(retry_tighter[0], retry_tighter[1])

            rel2 = _jaccard(want, _keywords(out2, self.stopwords))
            _jlog(logging.INFO, "[decode] retry done", rel2=rel2, rel1=rel1, out2_chars=len(out2))

            short = len(orig_user_prompt) < 60
            floor = relevance_floor * (0.60 if short else 1.0)

            if (rel2 >= floor) or (rel2 >= rel1 + 0.06):
                out1 = out2
                _jlog(logging.INFO, "[decode] retry accepted")
            else:
                _jlog(logging.INFO, "[decode] retry rejected")

        final_out = polish(_strip_speaker_tags(out1).strip()) if polish_output else _strip_speaker_tags(out1).strip()
        final_out = self._strip_boilerplate(final_out)
        _jlog(logging.INFO, "[final] output ready", chars=len(final_out), head=final_out[:140])

        # ── MEMORY WRITE GATE ──────────────────────────────────────
        metrics_final = self._text_metrics(final_out, orig_user_prompt)
        rel_final = float(metrics_final.get("rel", 0.0))
        rep3_final = float(metrics_final.get("rep3", 1.0))

        rep3_max = float(os.environ.get("ARDOR_MEM_REP3_MAX", "0.18"))
        min_chars_log = int(os.environ.get("ARDOR_MEM_MIN_LOG_CHARS", "20"))

        allow_log = (
                log_response
                and (not hard_fallback)
                and (len(final_out.strip()) >= min_chars_log)
                and (rel_final >= float(relevance_floor))
                and (rep3_final <= rep3_max)
        )

        _jlog(logging.INFO, "[memory] write gate",
              allow_log=allow_log, hard_fallback=hard_fallback,
              rel_final=rel_final, rep3_final=rep3_final,
              floor=relevance_floor, rep3_max=rep3_max, min_chars_log=min_chars_log)

        if allow_log:
            self.log(orig_user_prompt, final_out, prompt_vec=e_query)
        else:
            _jlog(logging.WARNING, "[memory] skipped logging to hippocampus", reason="quality_gate")



        self._maybe_summarize_with_dmn(
            prompt=orig_user_prompt,
            response=final_out,
            recent_turns=list(self.chat_turns),
            recent_texts=list(self.recent_texts),
            retrieved_memory_summary=dmn_retrieved_traces,
            aet_state=self._safe_aeternum_state(aet_decision),
        )

        # keep bounded
        if len(self.chat_turns) > self.max_chat_turns * 2:
            self.chat_turns = self.chat_turns[-(self.max_chat_turns * 2):]
            _jlog(logging.INFO, "[history] chat_turns bounded", max_turns=self.max_chat_turns, now=len(self.chat_turns))

        _jlog(logging.INFO, "[PFC] generate_text end", seconds=round(time.time() - t_start, 4))

        # Store the turn so the next generation stays in-context (clean history!)
        self.chat_turns.append(("user", self._clean_for_history(orig_user_prompt)))

        # Only store assistant if it passed quality
        clean_final = self._clean_for_history(final_out)
        if self._should_store_in_history(orig_user_prompt, clean_final, relevance_floor=relevance_floor):
            self.chat_turns.append(("assistant", clean_final))
        else:
            # critical: do NOT store junk assistant replies
            _jlog(logging.WARNING, "[history] assistant reply NOT stored (failed gate)")

        return final_out


    def _safe_aeternum_state(self, aet_state: Any = None) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        src = aet_state if aet_state is not None else getattr(self, 'aet', None)
        if src is None:
            return state
        if isinstance(src, dict):
            for key in ('valence', 'arousal', 'dominance', 'salience', 'activation', 'temperature_scale', 'top_p_scale', 'rep_penalty_scale'):
                if key in src:
                    state[key] = src.get(key)
            return state
        for key in ('valence', 'arousal', 'dominance', 'salience', 'activation', 'temperature_scale', 'top_p_scale', 'rep_penalty_scale'):
            try:
                val = getattr(src, key, None)
                if val is not None:
                    state[key] = val
            except Exception:
                continue
        return state

    def get_dmn_state(self) -> Dict[str, Any]:
        if self.dmn is None:
            return {
                'mode': 'DISABLED',
                'last_seed': '',
                'last_retrieved_traces': [],
                'current_narrative_takeaway': None,
                'self_model_state': {},
                'salience_info': {},
                'last_error': self._dmn_blocker,
            }
        try:
            state = self.dmn.get_state()
            self._last_dmn_state = state
            return state.to_dict() if hasattr(state, 'to_dict') else dict(state)
        except Exception as e:
            return {
                'mode': 'ERROR',
                'last_seed': '',
                'last_retrieved_traces': [],
                'current_narrative_takeaway': None,
                'self_model_state': {},
                'salience_info': {},
                'last_error': str(e),
            }

    def _maybe_summarize_with_dmn(
        self,
        *,
        prompt: str,
        response: str,
        recent_turns: Optional[List[Tuple[str, str]]] = None,
        recent_texts: Optional[List[str]] = None,
        retrieved_memory_summary: Optional[List[str]] = None,
        aet_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.dmn is None:
            return
        try:
            takeaway = self.dmn.summarize_active_context(
                prompt=prompt,
                response=response,
                recent_turns=recent_turns or list(self.chat_turns),
                recent_texts=recent_texts or list(self.recent_texts),
                retrieved_memory_summary=retrieved_memory_summary or [],
                aet_state=aet_state or self._safe_aeternum_state(),
                parietal=self.parietal,
                retrieval_enabled=bool(self.enable_retrieval and self.parietal is not None),
            )
            self._last_dmn_takeaway = takeaway.to_dict() if hasattr(takeaway, 'to_dict') else takeaway
            self._last_dmn_state = self.dmn.get_state()
            _jlog(logging.INFO, '[PFC] DMN active summary ok', theme=getattr(takeaway, 'theme', None), confidence=getattr(takeaway, 'confidence', None))
        except Exception as e:
            _jlog(logging.WARNING, '[PFC] DMN active summary failed', err=str(e))

    def run_idle_dmn_cycle(self, prompt: str = '') -> Dict[str, Any]:
        if self.dmn is None:
            return self.get_dmn_state()
        try:
            self.dmn.step_idle_cycle(
                prompt=prompt,
                recent_turns=list(self.chat_turns),
                recent_texts=list(self.recent_texts),
                parietal=self.parietal,
                retrieval_enabled=bool(self.enable_retrieval and self.parietal is not None),
                aet_state=self._safe_aeternum_state(),
            )
            self._last_dmn_state = self.dmn.get_state()
        except Exception as e:
            _jlog(logging.WARNING, '[PFC] DMN idle cycle failed', err=str(e))
        return self.get_dmn_state()

    def tick_idle_cognition(self) -> Dict[str, Any]:
        return self.run_idle_dmn_cycle()

        def model_schema(self) -> Dict[str, Any]:
            _jlog(logging.DEBUG, "[PFC] model_schema called")
            return dict(self.schema)




# ╔═══════════════════════════════════════════════════════════════════╗
# ║                               CLI                                 ║
# ╚═══════════════════════════════════════════════════════════════════╝
try:
    import typer  # type: ignore
    app = typer.Typer()
    _jlog(logging.INFO, "[cli] typer available; CLI enabled")
except Exception as e:
    app = None
    _jlog(logging.WARNING, "[cli] typer unavailable; CLI disabled", err=str(e))

if app:

    @app.command()
    def cli():
        _jlog(logging.INFO, "[cli] starting CLI")
        slow_type("\n🧠 Welcome to Ardor CLI — Synthesizer of Minds\n")
        roots = ["../Models", "../Cerebrum/Models", "../Cerebrum/Models/Ardor"]
        models = []
        for r in roots:
            if os.path.isdir(r):
                found = [os.path.join(r, f) for f in os.listdir(r) if f.endswith(".pt")]
                _jlog(logging.INFO, "[cli] scanning models", root=r, found=len(found))
                models += found
        if not models:
            print("No .pt models found.")
            _jlog(logging.ERROR, "[cli] no models found in roots", roots=roots)
            return
        for i, path in enumerate(models, 1):
            print(f"  {i}. {os.path.basename(path)}")

        import typer as _ty
        idx = int(_ty.prompt(f"\n🔎 Choose a model [1–{len(models)}]")) - 1
        model_path = models[idx]
        _jlog(logging.INFO, "[cli] model selected", idx=idx, model_path=model_path)

        tok_candidates = [
            "../ProjectTokenizer/ardor_tokenizer/tokenizer_v9.json",
            "../ProjectTokenizer/ardor_tokenizer/tokenizer_v8.json",
            "../ProjectTokenizer/ardor_tokenizer/tokenizer_v7.json",
            "../Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v9.json",
            "../Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v8.json",
            "../Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v7.json",
        ]
        tokenizer_path = next((p for p in tok_candidates if os.path.isfile(p)), None)
        _jlog(logging.INFO, "[cli] tokenizer candidate resolved", tokenizer_path=tokenizer_path)
        if tokenizer_path is None:
            print("Tokenizer not found. Please place tokenizer_v*.json in ProjectTokenizer/ardor_tokenizer.")
            _jlog(logging.ERROR, "[cli] tokenizer not found", candidates=tok_candidates)
            return

        encoder_ckpt = os.environ.get("ARDOR_ENCODER_CKPT", None) or r"..\Cerebrum\Models\Encoders\ArdorEncoder.pt"
        _jlog(logging.INFO, "[cli] encoder_ckpt", encoder_ckpt=encoder_ckpt)

        core = get_global_core(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device="cpu",
            enable_retrieval=True,
            encoder_ckpt=encoder_ckpt,
            force_reload=True,
        )

        print("\n💡 Type 'train', 'rem', 'exit', or just chat.\n")
        while True:
            prompt = input("🗨️  > ").strip().rstrip()
            if not prompt:
                continue
            cmd = prompt.lower()
            _jlog(logging.INFO, "[cli] input", cmd=cmd, prompt_len=len(prompt))
            if cmd in ("exit", "quit"):
                _jlog(logging.INFO, "[cli] exit")
                break
            if cmd == "train":
                _jlog(logging.INFO, "[cli] launching training subprocess")
                subprocess.call(["python", "../Cerebrum/Cortex/neural_plasticity_training.py"])
                continue
            if cmd in ("rem", "sleep"):
                _jlog(logging.INFO, "[cli] launching REM subprocess")
                subprocess.call(["python", "../Cerebrum/CorticalIntegration/REM.py"])
                continue

            slow_type("\n🧠 Ardor:", 0.03)
            ans = core.generate_text(prompt, persona_primer="")
            slow_type(ans, 0.01)
            print("\n" + "-" * 60 + "\n")


def _selftest():
    _jlog(logging.INFO, "[selftest] basic run start")
    print("[selftest] basic run")
    logits = torch.zeros(1, 10)
    logits[0, [3, 7]] += 1.0
    nb = NgramBlocker(3)
    nb.reset()
    nb.update(1)
    nb.update(2)
    l1 = torch.zeros(10)
    nb.apply(l1)
    print("[selftest] OK")
    _jlog(logging.INFO, "[selftest] done")


if __name__ == "__main__":
    _jlog(logging.INFO, "[main] entry", argv=sys.argv)
    if "--selftest" in sys.argv:
        _selftest()
    elif app:
        app()
    else:
        print("Typer not installed; CLI disabled.")
        _jlog(logging.WARNING, "[main] typer missing; nothing to run")
