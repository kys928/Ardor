#!/usr/bin/env python3
# prefrontal_cortex.py — Ardor inference core (PFC control + biasing rails)
# PATCHSET (2026-03-02):
# - ✅ Working Memory: rolling short-term chat history injected into _build_chat_prompt()
# - ✅ Split memory systems:
#     A) Semantic facts KV-store (deterministic query + injection)
#     B) Episodic vector store stores ONLY user prompts + safe summaries (no raw assistant output)
# - ✅ High-integrity memory acceptance:
#     - hard gating + task-aware correctness hooks (extensible)
# - ✅ Backward-compatible: can read old JSONL entries (prompt/response) but will embed prompt, inject prompt-only summaries
from __future__ import annotations

import os, sys, time, json, random, subprocess, re, glob
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import deque

# ─────────────────────────────────────────────────────────────────────
# ✅ LOGGING (behavior-neutral)
# ─────────────────────────────────────────────────────────────────────
import logging
from datetime import datetime

_ARDOR_LOG_LEVEL = os.environ.get("ARDOR_LOG_LEVEL", "DEBUG").strip().upper()
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
            print(f"[LOG] failed to create file logger at {_ARDOR_LOG_FILE}: {e}")

    lg._ardor_configured = True  # type: ignore
    lg.info(f"[boot] logger configured level={_ARDOR_LOG_LEVEL} json={_ARDOR_LOG_JSON} file={'ON' if _ARDOR_LOG_FILE else 'OFF'}")
    return lg


_LOG = _setup_ardor_logger()


def _jlog(level: int, event: str, **kv: Any) -> None:
    """Structured logging helper. Never raises. Never changes control-flow."""
    try:
        if _ARDOR_LOG_JSON:
            payload = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event, **kv}
            _LOG.log(level, json.dumps(payload, ensure_ascii=False))
        else:
            tail = ""
            if kv:
                tail = " | " + " ".join([f"{k}={repr(v)[:220]}" for k, v in kv.items()])
            _LOG.log(level, f"{event}{tail}")
    except Exception:
        pass


def _trace_enabled() -> bool:
    return _LOG.isEnabledFor(logging.DEBUG) and os.environ.get("ARDOR_TRACE", "1").strip() in ("1", "true", "TRUE", "yes", "YES")


def _dump_text_file(path: Union[str, Path], text: str) -> None:
    """
    Debug helper: writes text to disk. Never raises. Behavior-neutral.
    Enabled only when ARDOR_DUMP_TEXT=1.
    """
    try:
        if os.environ.get("ARDOR_DUMP_TEXT", "0").strip() not in ("1", "true", "TRUE", "yes", "YES"):
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    except Exception:
        pass


# ── Optional safety/env toggles ──────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RUST_BACKTRACE", "1")

import torch
import torch.nn.functional as F
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

# Encoder (Parietal)
from Cerebrum.Cortex.posterior_parietal_cortex import ArdorEncoder

# Surface polisher (Anterior Cingulate)
from Cerebrum.LanguageProcessing.Anterior_Cingulate import polish
from Cerebrum.Cortex.broca_decoder import ArdorDecoder  # noqa: E402

from Cerebrum.Cortex.ardor_config import ArdorConfig
from Cerebrum.Cortex.inferior_parietal_analog import ParietalMemory


# ─────────────────────────────────────────────────────────────────────
# Project root + robust path resolution (fixes relative-path logging bugs)
# ─────────────────────────────────────────────────────────────────────
def _find_project_root(start: Path) -> Path:
    env = os.environ.get("ARDOR_HOME", "").strip()
    if env:
        try:
            p = Path(env).expanduser().resolve()
            if p.exists():
                return p
        except Exception:
            pass

    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "Dataset").exists() and (p / "Cerebrum").exists():
            return p
    return start.parents[1] if len(start.parents) > 1 else start.parent


_PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
_DATASET_CONV_DIR = (_PROJECT_ROOT / "Dataset" / "Conversations")


def _resolve_path_maybe_relative(p: Union[str, Path], base: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base / pp).resolve()


# ─────────────────────────────────────────────────────────────────────
# PFC GLOBAL SINGLETON (one brain per process)
# ─────────────────────────────────────────────────────────────────────
_PFC_SINGLETON: Optional["ArdorCore"] = None
_PFC_SIGNATURE: Optional[tuple] = None


def _norm_tok_for_sig(tokenizer_path: Optional[str]) -> Optional[str]:
    if tokenizer_path:
        try:
            return os.path.abspath(tokenizer_path)
        except Exception:
            return tokenizer_path
    if _PFC_SINGLETON is not None and getattr(_PFC_SINGLETON, "tokenizer_path", None):
        try:
            return os.path.abspath(str(_PFC_SINGLETON.tokenizer_path))
        except Exception:
            return str(_PFC_SINGLETON.tokenizer_path)
    return None


def get_global_core(
    *,
    model_path: str,
    tokenizer_path: Optional[str],
    device: str = "cpu",
    enable_retrieval: bool = True,
    encoder_ckpt: Optional[str] = None,
    max_len: int = 300,
    force_reload: bool = False,
) -> "ArdorCore":
    global _PFC_SINGLETON, _PFC_SIGNATURE

    _jlog(logging.INFO, "[singleton] get_global_core called",
          model_path=model_path, tokenizer_path=tokenizer_path, device=device,
          enable_retrieval=enable_retrieval, encoder_ckpt=encoder_ckpt, max_len=max_len, force_reload=force_reload)

    sig = (
        os.path.abspath(model_path),
        _norm_tok_for_sig(tokenizer_path),
        device,
        bool(enable_retrieval),
        os.path.abspath(encoder_ckpt) if encoder_ckpt else None,
        int(max_len),
    )

    _jlog(logging.DEBUG, "[singleton] computed signature", sig=sig, prev=_PFC_SIGNATURE)

    if _PFC_SINGLETON is None or force_reload or (_PFC_SIGNATURE != sig):
        _jlog(logging.INFO, "[singleton] creating/reloading ArdorCore",
              reason=("none" if _PFC_SINGLETON is None else ("force_reload" if force_reload else "sig_changed")))
        _PFC_SINGLETON = ArdorCore(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device,
            max_len=max_len,
            enable_retrieval=enable_retrieval,
            encoder_ckpt=encoder_ckpt,
            aeternum=None,
        )
        _PFC_SIGNATURE = (
            os.path.abspath(model_path),
            os.path.abspath(str(_PFC_SINGLETON.tokenizer_path)) if getattr(_PFC_SINGLETON, "tokenizer_path", None) else None,
            device,
            bool(enable_retrieval),
            os.path.abspath(encoder_ckpt) if encoder_ckpt else None,
            int(max_len),
        )
        _jlog(logging.INFO, "[singleton] signature normalized", sig=_PFC_SIGNATURE)
    else:
        _jlog(logging.INFO, "[singleton] returning existing ArdorCore", sig=_PFC_SIGNATURE)

    return _PFC_SINGLETON


def get_core_singleton() -> Optional["ArdorCore"]:
    _jlog(logging.DEBUG, "[singleton] get_core_singleton", exists=_PFC_SINGLETON is not None)
    return _PFC_SINGLETON


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


def _get_stop_ids(tokenizer: Tokenizer) -> List[int]:
    V = int(tokenizer.get_vocab_size())

    def _env_int(name: str) -> Optional[int]:
        v = os.environ.get(name, "").strip()
        if not v:
            return None
        try:
            return int(v)
        except Exception:
            return None

    eos_env = _env_int("ARDOR_EOS_ID")
    eot_env = _env_int("ARDOR_EOT_ID")

    out: List[int] = []
    for x in (eos_env, eot_env):
        if x is not None and 0 <= x < V:
            out.append(x)

    for x in (3, 7):
        if 0 <= x < V:
            out.append(x)

    eos_str = _find_eos_id(tokenizer)
    if eos_str is not None:
        out.append(eos_str)
    eot_str = tokenizer.token_to_id("<|eot|>")
    if eot_str is not None:
        out.append(eot_str)

    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)

    _jlog(logging.INFO, "[eos] stop ids resolved", stop_ids=dedup, vocab_size=V)
    return dedup


# ─────────────────────────────────────────────────────────────────────
# Conversation logs (Hippocampus sources) — robust paths
# ─────────────────────────────────────────────────────────────────────
BAD_LOG_FILE = _resolve_path_maybe_relative(
    os.environ.get("ARDOR_BAD_LOG_JSONL", str(_DATASET_CONV_DIR / "ardor_dialogues.jsonl")),
    _PROJECT_ROOT
)
BAD_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

GOOD_LOG_FILE = _resolve_path_maybe_relative(
    os.environ.get("ARDOR_GOOD_MEMORY_JSONL", str(_DATASET_CONV_DIR / "ardor_dpo_300.jsonl")),
    _PROJECT_ROOT
)
GOOD_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_jlog(logging.INFO, "[hippocampus] files ready",
      bad_path=str(BAD_LOG_FILE), bad_exists=BAD_LOG_FILE.exists(),
      good_path=str(GOOD_LOG_FILE), good_exists=GOOD_LOG_FILE.exists(),
      project_root=str(_PROJECT_ROOT))


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
    _jlog(logging.INFO, "[encoder_ckpt] resolve start", user_path=user_path,
          env=os.environ.get("ARDOR_ENCODER_CKPT", "").strip())

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
        _PROJECT_ROOT / "Cerebrum" / "Models" / "Encoders",
        _PROJECT_ROOT / "Models" / "Encoders",
        _PROJECT_ROOT / "Cerebrum" / "Models",
        _PROJECT_ROOT / "Models",
        _PROJECT_ROOT / "runs",
        _PROJECT_ROOT / "Cerebrum" / "runs",
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
    _jlog(logging.DEBUG, "[ckpt] block/layer schema", expects_blocks=expects_blocks, expects_layers=expects_layers,
          has_blocks=has_blocks, has_layers=has_layers)
    if expects_blocks and has_layers:
        new = _rename_prefix(new, "layers.", "blocks.")
    elif expects_layers and has_blocks:
        new = _rename_prefix(new, "blocks.", "layers.")

    expects_attn = any(".attn." in k for k in model_state_keys)
    expects_attention = any(".attention." in k for k in model_state_keys)
    has_attn = any(".attn." in k for k in new)
    has_attention = any(".attention." in k for k in new)
    _jlog(logging.DEBUG, "[ckpt] attn schema", expects_attn=expects_attn, expects_attention=expects_attention,
          has_attn=has_attn, has_attention=has_attention)
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
_ENCODER_CACHE: Dict[Tuple[str, str, int, int, int, int, int], nn.Module] = {}
_GLOBAL_CORE: Optional["ArdorCore"] = None


def _abs(p: str) -> str:
    ap = os.path.abspath(os.path.expanduser(p))
    _jlog(logging.DEBUG, "[path] abs", inp=p, out=ap)
    return ap


# ─────────────────────────────────────────────────────────────────────
# model_meta.json (required for 1B+ schema)
# ─────────────────────────────────────────────────────────────────────
def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _find_model_meta(model_path: str) -> Path:
    p = Path(model_path)
    if p.is_dir():
        cand = p / "model_meta.json"
        return cand
    return p.parent / "model_meta.json"

def _load_required_meta(model_path: str, tokenizer_path_hint: str | None) -> dict:
    meta_path = _find_model_meta(model_path)
    if not meta_path.exists() and not _env_flag("ARDOR_ALLOW_LEGACY_NO_META", "0"):
        raise FileNotFoundError(f"model_meta.json missing next to checkpoint: {meta_path}")
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # minimal required keys
    req = ["vocab_size","hidden_size","n_layers","n_heads","ff_mult","max_len","use_rope","rope_theta","tokenizer_path","tokenizer_sha256"]
    missing = [k for k in req if k not in meta]
    if missing and not _env_flag("ARDOR_ALLOW_LEGACY_NO_META", "0"):
        raise ValueError(f"model_meta.json missing keys: {missing}")
    # verify tokenizer sha if possible
    tok_path = Path(str(meta.get("tokenizer_path") or tokenizer_path_hint or "")).expanduser()
    if tok_path.exists() and meta.get("tokenizer_sha256"):
        sha = _sha256_file(tok_path)
        if sha != meta["tokenizer_sha256"] and not _env_flag("ARDOR_ALLOW_TOKENIZER_SHA_MISMATCH","0"):
            raise ValueError("Tokenizer SHA256 mismatch vs model_meta.json (set ARDOR_ALLOW_TOKENIZER_SHA_MISMATCH=1 to override).")
    return meta

def _config_from_meta(meta: dict, *, default_vocab: int) -> ArdorConfig:
    if not meta:
        # legacy fallback: infer small config
        return ArdorConfig(
            vocab_size=default_vocab, hidden_size=384, n_layers=8, n_heads=6, ff_mult=4,
            max_len=2048, dropout=0.12, attn_dropout=0.12, resid_dropout=0.12,
            layernorm_eps=1e-5, use_rope=False, rope_theta=10000.0,
        )
    cfg = ArdorConfig(
        vocab_size=int(meta["vocab_size"]),
        hidden_size=int(meta["hidden_size"]),
        n_layers=int(meta["n_layers"]),
        n_heads=int(meta["n_heads"]),
        ff_mult=int(meta["ff_mult"]),
        max_len=int(meta["max_len"]),
        dropout=float(meta.get("dropout", 0.1)),
        attn_dropout=float(meta.get("attn_dropout", meta.get("dropout",0.1))),
        resid_dropout=float(meta.get("resid_dropout", meta.get("dropout",0.1))),
        layernorm_eps=float(meta.get("layernorm_eps", 1e-5)),
        use_rope=bool(meta.get("use_rope", True)),
        rope_theta=float(meta.get("rope_theta", 10000.0)),
    )
    cfg.validate()
    return cfg

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
        
        # model_meta.json is the authoritative config for 1B+ runs
        meta = _load_required_meta(model_path, tokenizer_path_hint=None)
        # if meta missing (legacy override), infer dims as before
        if not meta:
            vocab, hidden, layers, maxlen = _infer_dims_from_state(sd)
            cfg = _config_from_meta({}, default_vocab=int(vocab))
        else:
            cfg = _config_from_meta(meta, default_vocab=int(meta.get("vocab_size", 0) or 0))
            vocab = cfg.vocab_size
            hidden = cfg.hidden_size
            layers = cfg.n_layers
            maxlen = cfg.max_len

        _jlog(logging.INFO, "[broca] constructing ArdorDecoder", vocab=int(vocab), hidden=int(hidden), layers=int(layers),
              heads=int(cfg.n_heads), max_len=int(maxlen), use_rope=bool(cfg.use_rope))
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
        str(_PROJECT_ROOT / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer"),
        str(_PROJECT_ROOT / "ProjectTokenizer" / "ardor_tokenizer"),
        "../Cerebrum/ProjectTokenizer/ardor_tokenizer",
        "../ProjectTokenizer/ardor_tokenizer",
        "./Cerebrum/ProjectTokenizer/ardor_tokenizer",
        "./ProjectTokenizer/ardor_tokenizer",
    ]
    _jlog(logging.DEBUG, "[tok] roots", roots=roots)

    cand_files: List[str] = []
    for r in roots:
        rr = os.path.abspath(r)
        if not os.path.isdir(rr):
            _jlog(logging.DEBUG, "[tok] root not dir", root=rr)
            continue
        found = glob.glob(os.path.join(rr, "tokenizer_v9.json"))
        _jlog(logging.DEBUG, "[tok] glob v9", root=rr, found=len(found))
        cand_files += found

    if requested_tok:
        try:
            _jlog(logging.INFO, "[tok] trying override tokenizer", path=requested_tok)
            t0 = Tokenizer.from_file(requested_tok)
            v0 = t0.get_vocab_size()
            _jlog(logging.INFO, "[tok] override vocab", vocab=v0, want=want_vocab)
            if v0 == want_vocab:
                tok = t0
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

        out = encoder(ids)
        _jlog(logging.DEBUG, "[encoder] forward fallback ok", out_type=str(type(out)))
        if isinstance(out, (list, tuple)) and len(out) > 0:
            out = out[0]
        if isinstance(out, dict) and "last_hidden_state" in out:
            out = out["last_hidden_state"]
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
# ║                     Semantic facts KV-store                        ║
# ╚═══════════════════════════════════════════════════════════════════╝
_FACT_PATTERNS = [
    # name / preferred name
    (re.compile(r"(?i)\b(?:my name is|call me|you can call me|i go by)\s+([A-Za-z][A-Za-z0-9_\- ]{1,40})\b"), "user_name"),
    # favorite number / remember my number
    (re.compile(r"(?i)\b(?:my (?:favorite )?number is|remember my number|my number is)\s+(-?\d{1,9})\b"), "favorite_number"),
    # preferences (lightweight)
    (re.compile(r"(?i)\b(?:i prefer|i like|i dislike|i hate)\s+(.{1,80})$"), "preference"),
]


def _facts_extract(prompt: str) -> Dict[str, str]:
    p = (prompt or "").strip()
    out: Dict[str, str] = {}
    if not p:
        return out
    for rx, key in _FACT_PATTERNS:
        m = rx.search(p)
        if m:
            val = (m.group(1) or "").strip()
            val = re.sub(r"\s{2,}", " ", val)
            if val:
                out[key] = val
    return out


def _facts_render_for_injection(facts: Dict[str, str], *, limit: int = 12) -> str:
    if not facts:
        return ""
    items = list(facts.items())[: max(1, int(limit))]
    lines = [f"- {k}: {v}" for k, v in items]
    return "INTERNAL FACTS (do not quote, do not mention, do not reveal):\n" + "\n".join(lines)


def _facts_relevant_subset(facts: Dict[str, str], prompt: str, *, limit: int = 8) -> Dict[str, str]:
    if not facts:
        return {}
    p = (prompt or "").lower()
    # deterministic relevance: substring match on key or value tokens
    scored: List[Tuple[int, str, str]] = []
    for k, v in facts.items():
        s = 0
        if k.lower() in p:
            s += 3
        if v.lower() in p:
            s += 3
        # token overlap
        kt = set(re.findall(r"[a-z0-9]{2,}", k.lower()))
        vt = set(re.findall(r"[a-z0-9]{2,}", v.lower()))
        pt = set(re.findall(r"[a-z0-9]{2,}", p))
        s += int(len((kt | vt) & pt) > 0)
        scored.append((s, k, v))
    scored.sort(key=lambda x: x[0], reverse=True)
    keep = [(k, v) for s, k, v in scored if s > 0][: max(1, int(limit))]
    if not keep:
        # if nothing matches, inject a tiny subset (stable)
        keep = list(facts.items())[: max(1, int(min(limit, 3)))]
    return dict(keep)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                     Hippocampus / ParietalMemory                  ║
# ╚═══════════════════════════════════════════════════════════════════╝
# (ParietalMemory moved to inferior_parietal_analog.py)

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
        aeternum=None,
    ):
        _jlog(logging.INFO, "[PFC] ArdorCore.__init__ start",
              model_path=model_path, tokenizer_path=tokenizer_path, device=device,
              max_len=max_len, enable_retrieval=enable_retrieval, encoder_ckpt=encoder_ckpt,
              aeternum_is_none=(aeternum is None))

        self.device = device
        self.gen_max_tokens = int(max_len)
        self.model_path = model_path

        self.model, schema, _model_sd = _load_broca_cached(model_path, device)
        want_vocab = int(schema.get("want_vocab") or schema.get("vocab") or 0)

        self.tokenizer, chosen_tok_path = _load_tokenizer_matching_vocab(tokenizer_path, want_vocab)
        self.tokenizer_path = chosen_tok_path

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
        _jlog(logging.INFO, "[PFC] schema", layers=self.layers, heads=self.heads, hidden=self.hidden,
              max_len=self.model_ctx_len, missing=miss_ct, unexpected=unex_ct)

        try:
            emb_rows = getattr(getattr(self.model, "token_embed", None), "weight", None)
            emb_rows = int(emb_rows.shape[0]) if emb_rows is not None else int(self.vocab_size)
        except Exception as e:
            _jlog(logging.WARNING, "[PFC] token_embed rows inspect failed", err=str(e))
            emb_rows = int(self.vocab_size)

        print(f"🧩 Tokenizer: {self.tokenizer_path}  | vocab={self.vocab_size}  embed={emb_rows}")
        _jlog(logging.INFO, "[PFC] tokenizer", path=self.tokenizer_path, vocab=self.vocab_size, embed_rows=emb_rows)

        resolved_encoder_ckpt: Optional[str] = None
        if encoder_ckpt is not None:
            _jlog(logging.INFO, "[PFC] encoder_ckpt provided explicitly", encoder_ckpt=encoder_ckpt)
            resolved_encoder_ckpt = _resolve_encoder_ckpt(encoder_ckpt)
            self.enable_retrieval = True
        else:
            if enable_retrieval:
                _jlog(logging.INFO, "[PFC] retrieval enabled; auto-resolving encoder ckpt")
                resolved_encoder_ckpt = _resolve_encoder_ckpt(None)
                self.enable_retrieval = True
            else:
                _jlog(logging.INFO, "[PFC] retrieval disabled by flag")
                self.enable_retrieval = False

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

        if aeternum is not None:
            _jlog(logging.INFO, "[PFC] aeternum injected externally", aeternum_type=str(type(aeternum)))
            self.aet = aeternum
        else:
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

        # ✅ Episodic memory (CLEAN) — embed prompt or summary, inject snippet only
        self.parietal: Optional[ParietalMemory] = None
        if self.enable_retrieval:
            mem_jsonl = os.environ.get("ARDOR_MEMORY_JSONL", str(GOOD_LOG_FILE))
            embed_field = os.environ.get("ARDOR_MEM_EMBED_FIELD", "prompt").strip().lower()
            _jlog(logging.INFO, "[PFC] init ParietalMemory (CLEAN)", mem_jsonl=mem_jsonl, embed_field=embed_field)
            self.parietal = ParietalMemory(
                self.tokenizer,
                self.device,
                broca_model=self.model,
                encoder=self.encoder,
                memory_jsonl=mem_jsonl,
                max_items=int(os.environ.get("ARDOR_MEMORY_MAX_ITEMS", "2000")),
                max_len=int(os.environ.get("ARDOR_ENCODER_MAXLEN", "128")),
                embed_field=("summary" if embed_field == "summary" else "prompt"),
            )
        else:
            _jlog(logging.INFO, "[PFC] ParietalMemory skipped (retrieval disabled)")

        self.recent_texts = deque(maxlen=128)
        self.stopwords = STOPWORDS

        self._digit_ids = _token_ids_for_chars(self.tokenizer, set("0123456789"))
        self._stop_ids = _get_stop_ids(self.tokenizer)
        self._eos_id = self._stop_ids[0] if self._stop_ids else None
        self._eot_id = (self._stop_ids[1] if len(self._stop_ids) > 1 else None)

        _jlog(logging.INFO, "[PFC] init done",
              gen_max_tokens=self.gen_max_tokens, eos_id=self._eos_id, eot_id=self._eot_id, stop_ids=getattr(self, "_stop_ids", None),
              digit_ids=len(self._digit_ids))

        # ✅ Memory write policy
        self.write_memory = _env_flag("ARDOR_WRITE_MEMORY", default="1")
        _jlog(logging.INFO, "[PFC] memory write policy", write_memory=self.write_memory, good_log=str(GOOD_LOG_FILE), bad_log=str(BAD_LOG_FILE))

        # ✅ Working memory (rolling chat context)
        # Stored inside PFC so GUI does not have to do it (but GUI can mirror it).
        self.chat_turn_pairs = int(os.environ.get("ARDOR_CHAT_TURNS", "8"))  # number of (user,assistant) pairs to keep
        self.chat_turns: deque[Dict[str, str]] = deque(maxlen=max(2, self.chat_turn_pairs * 2))
        _jlog(logging.INFO, "[PFC] working memory initialized", pairs=self.chat_turn_pairs, max_items=self.chat_turns.maxlen)

        # ✅ Semantic facts store (deterministic)
        self.facts: Dict[str, str] = {}
        _jlog(logging.INFO, "[PFC] facts store initialized")

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
            return {"d1": 0.0, "d2": 0.0, "rep2": 1.0, "rep3": 1.0, "closure": 1.0, "imbalance": 0.6, "rel": 0.0, "and_dup": 1.0}

        d1 = len(set(toks)) / max(1, n)

        bigrams = [tuple(toks[i: i + 2]) for i in range(max(0, n - 1))]
        d2 = (len(set(bigrams)) / max(1, len(bigrams))) if bigrams else 0.0

        rep2 = 0.0
        if bigrams:
            bcounts: Dict[Tuple[str, str], int] = {}
            for b in bigrams:
                bcounts[b] = bcounts.get(b, 0) + 1
            brepeats = sum(c - 1 for c in bcounts.values() if c > 1)
            rep2 = brepeats / max(1, len(bigrams))

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

        # semantic cosine relevance (prompt vs response) using encoder or broca embedding
        cos = 0.0
        try:
            # quick, robust: embed prompt and response via ParietalMemory static pattern if encoder present
            if self.encoder is not None:
                v_p = self.parietal.encode(prompt) if self.parietal is not None else None
                if v_p is None:
                    v_p = None
                # fallback compute in-place if needed
                if v_p is None:
                    cos = 0.0
                else:
                    # embed response via same embedder (prompt embedder)
                    v_r = self.parietal.encode(text) if self.parietal is not None else None
                    if v_r is not None:
                        cos = float(torch.matmul(v_p, v_r.transpose(0, 1)).item())
            else:
                cos = 0.0
        except Exception:
            cos = 0.0

        cos01 = (cos + 1.0) * 0.5
        w = float(os.environ.get("ARDOR_REL_SEM_W", "0.70"))
        rel = w * cos01 + (1.0 - w) * jac

        and_dups = re.findall(r"\b([A-Za-z]{3,})\s+and\s+\1\b", text, flags=re.IGNORECASE)
        and_dup_score = min(1.0, len(and_dups) / 2.0)

        m = {"d1": d1, "d2": d2, "rep2": rep2, "rep3": rep3, "closure": closure, "imbalance": imbalance, "rel": rel, "and_dup": and_dup_score}
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

    # ─────────────────────────────────────────────────────────────
    # ✅ Working memory + facts updates
    # ─────────────────────────────────────────────────────────────
    def _wm_add_user(self, user_text: str) -> None:
        user_text = self._clean_for_history(user_text)
        if not user_text:
            return
        self.chat_turns.append({"role": "user", "content": user_text})
        _jlog(logging.DEBUG, "[wm] add user", n=len(self.chat_turns), tail=user_text[:80])

        # Extract semantic facts deterministically
        newly = _facts_extract(user_text)
        if newly:
            self.facts.update(newly)
            _jlog(logging.INFO, "[facts] updated", new=newly, total=len(self.facts))

    def _wm_add_assistant(self, assistant_text: str, *, allow_store: bool) -> None:
        # Poison control: only store assistant turn if allow_store (i.e., passed integrity checks)
        if not allow_store:
            _jlog(logging.INFO, "[wm] assistant turn NOT stored (low-integrity)")
            return
        a = self._clean_for_history(assistant_text)
        if not a:
            return
        self.chat_turns.append({"role": "assistant", "content": a})
        _jlog(logging.DEBUG, "[wm] add assistant", n=len(self.chat_turns), tail=a[:80])

    def _wm_recent_turns(self) -> List[Dict[str, str]]:
        # Return in-order list
        return list(self.chat_turns)

    # ─────────────────────────────────────────────────────────────
    # ✅ Task-aware correctness checks (objective, conservative)
    # These are deliberately narrow + strict. Expand over time.
    # ─────────────────────────────────────────────────────────────
    def _correctness_checks(self, prompt: str, answer: str) -> Tuple[bool, Dict[str, Any]]:
        p = (prompt or "").strip()
        a = (answer or "").strip()
        meta: Dict[str, Any] = {"checks": {}, "ok": True}

        def fail(name: str, reason: str):
            meta["checks"][name] = {"ok": False, "reason": reason}
            meta["ok"] = False

        def ok(name: str):
            meta["checks"][name] = {"ok": True}

        # 1) refuse “email/corporate template” drift
        low = a.lower()
        if any(x in low for x in ["sincerely", "best regards", "dear ", "to whom it may concern", "[your name]"]):
            fail("no_email_style", "email-like phrasing detected")
        else:
            ok("no_email_style")

        # 2) if user asks for stored number and we have it, answer should include it
        if re.search(r"(?i)\b(my number|favorite number|what number)\b", p) and ("favorite_number" in self.facts):
            want = str(self.facts["favorite_number"])
            if want and (want not in a):
                fail("facts_consistency_number", f"expected favorite_number={want} to appear")
            else:
                ok("facts_consistency_number")
        else:
            ok("facts_consistency_number")

        # 3) specific numeric-pay sanity check (your known failure mode scenario)
        # Pattern: "worked 31 hours and 28 minutes ... paid 14.78€ per hour ... how much"
        m = re.search(r"(?i)\bworked\s+(\d+)\s+hours?\s+and\s+(\d+)\s+minutes?\b", p)
        r = re.search(r"(?i)\b(?:paid|get paid)\s+(\d+(?:\.\d+)?)\s*€?\s*(?:per|\/)\s*hour\b", p)
        if m and r and re.search(r"(?i)\bhow much\b", p):
            hrs = int(m.group(1))
            mins = int(m.group(2))
            rate = float(r.group(1))
            total = (hrs + mins / 60.0) * rate
            # if answer contains an amount close to expected, accept
            nums = [float(x.replace(",", ".")) for x in re.findall(r"(\d+(?:[.,]\d+)?)", a)]
            close = any(abs(x - total) <= max(0.05, 0.01 * total) for x in nums)
            if not close:
                fail("pay_calc_sanity", f"expected approx {total:.2f}, not found")
            else:
                ok("pay_calc_sanity")
        else:
            ok("pay_calc_sanity")

        return bool(meta["ok"]), meta

    # ─────────────────────────────────────────────────────────────
    # ✅ Logging: good memory stores only prompt+summary+facts
    # ─────────────────────────────────────────────────────────────
    def log(self, prompt: str, resp: str, *, prompt_vec: Optional[torch.Tensor] = None, good: bool, facts: Optional[Dict[str, str]] = None, quality: Optional[Dict[str, Any]] = None):
        if not getattr(self, "write_memory", True):
            _jlog(logging.INFO, "[log] skipped: ARDOR_WRITE_MEMORY=0", good=good)
            return

        prompt = _strip_special_text_tokens((prompt or "").strip())
        resp = _strip_special_text_tokens((resp or "").strip())
        if not prompt:
            _jlog(logging.WARNING, "[log] skipped (empty prompt)", prompt_len=len(prompt))
            return

        # BAD log: keep raw prompt/response for debugging (not used in retrieval)
        if not good:
            path = BAD_LOG_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps({"prompt": prompt, "response": resp, "ts": time.time()}, ensure_ascii=False) + "\n")
            except Exception as e:
                _jlog(logging.ERROR, "[log] bad file write failed", path=str(path), err=str(e))
            return

        # GOOD log: store only high-integrity, safe fields
        # Summary MUST NOT contain raw assistant output (poison vector); keep it prompt-focused.
        summary_max = int(os.environ.get("ARDOR_MEM_SUMMARY_MAX_CHARS", "240"))
        safe_summary = f"User asked: {prompt}"
        safe_summary = safe_summary[:summary_max].strip()

        entry = {
            "prompt": prompt,
            "summary": safe_summary,
            "facts": facts or {},
            "quality": quality or {},
            "ts": time.time(),
        }

        path = GOOD_LOG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            _jlog(logging.ERROR, "[log] good file write failed", path=str(path), err=str(e))

        # Ingest into ParietalMemory (clean index)
        if self.enable_retrieval and self.parietal is not None:
            try:
                # Embed prompt by default (embed_field controls it)
                embed_text = prompt if getattr(self.parietal, "embed_field", "prompt") == "prompt" else safe_summary
                self.parietal.ingest_episode(prompt=prompt, summary=safe_summary, embed_text=embed_text, prompt_vec=prompt_vec)
                _jlog(logging.DEBUG, "[log] memory ingest ok (good)")
            except Exception as e:
                _jlog(logging.WARNING, "[log] memory ingest failed", err=str(e))

    # ─────────────────────────────────────────────────────────────
    # text cleaning helpers
    # ─────────────────────────────────────────────────────────────
    def _strip_boilerplate(self, text: str) -> str:
        t = _strip_special_text_tokens((text or "").strip())
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

    def _trim_to_sentence_boundary(self, text: str, max_chars: int) -> str:
        t = (text or "").strip()
        if max_chars <= 0 or len(t) <= max_chars:
            return t

        cut = t[:max_chars].rstrip()
        matches = list(re.finditer(r"[.!?](?=\s|$)", cut))
        if matches:
            cut = cut[:matches[-1].end()].rstrip()
        else:
            cut = re.sub(r"\s+\S*$", "", cut).rstrip()
            if not cut:
                cut = t[:max_chars].rstrip()

        return cut

    def _clean_for_history(self, text: str) -> str:
        t = _strip_special_text_tokens((text or "").strip())
        t = self._strip_boilerplate(t)
        t = re.sub(r"[ \t]+\n", "\n", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    # ─────────────────────────────────────────────────────────────
    # ✅ Prompt builder now injects rolling dialogue + facts + retrieval
    # ─────────────────────────────────────────────────────────────
    def _build_chat_prompt(self, persona_primer: str, user_text: str, *, history: List[Dict[str, str]], facts_block: str = "", retrieval_block: str = "") -> str:
        sys_block = (persona_primer or "").strip() or ROLE_PRIMER
        user_text = self._clean_for_history(user_text)

        has_user = self.tokenizer.token_to_id("<|user|>") is not None
        has_asst = self.tokenizer.token_to_id("<|assistant|>") is not None
        has_eot = self.tokenizer.token_to_id("<|eot|>") is not None

        _jlog(logging.DEBUG, "[prompt] build_chat_prompt MULTI_TURN",
              has_user=has_user, has_asst=has_asst, has_eot=has_eot, history_items=len(history),
              has_facts=bool(facts_block), has_retrieval=bool(retrieval_block))

        # Merge blocks into system message (NOT as separate roles; prevents quote magnets)
        sys_aug = sys_block
        if facts_block:
            sys_aug = sys_aug.rstrip() + "\n\n" + facts_block.strip() + "\n"
        if retrieval_block:
            sys_aug = sys_aug.rstrip() + "\n\n" + retrieval_block.strip() + "\n"

        if has_user and has_asst and has_eot:
            parts: List[str] = []
            parts.append(f"<|system|>\n{sys_aug}\n<|eot|>\n")

            # Rolling history (bounded by deque maxlen already)
            for turn in history:
                role = turn.get("role", "").strip().lower()
                content = self._clean_for_history(turn.get("content", ""))
                if not content:
                    continue
                if role == "user":
                    parts.append(f"<|user|>\n{content}\n<|eot|>\n")
                elif role == "assistant":
                    parts.append(f"<|assistant|>\n{content}\n<|eot|>\n")

            # Current user message last
            parts.append(f"<|user|>\n{user_text}\n<|eot|>\n")
            parts.append(f"<|assistant|>\n")
            return "".join(parts)

        # Fallback non-special chat template
        hist_txt = ""
        for turn in history[-12:]:
            role = turn.get("role", "")
            content = self._clean_for_history(turn.get("content", ""))
            if not content:
                continue
            if role == "user":
                hist_txt += f"User: {content}\n"
            elif role == "assistant":
                hist_txt += f"Ardor: {content}\n"

        sys_lines = sys_aug.strip()
        return f"System: {sys_lines}\n\n{hist_txt}\nUser: {user_text}\nArdor: "

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.65,
        top_p: float = 0.90,
        rep_penalty: float = 1.2,
        ngram_block: int = 0,
        persona_primer: str = ROLE_PRIMER,
        relevance_floor: float = 0.0100,
        retry_tighter: Tuple[float, float] = (0.55, 0.70),
        suppress_vague: bool = True,
        min_new_tokens: int = 8,
        max_new_tokens: Optional[int] = None,
        top_k: int = 40,
        typical_p: float = 0.95,
        min_temp: float = 0.35,
        *,
        auto_pick: bool = False,
        stop_on_eos: bool = True,
        profile: Optional[str] = None,
        log_response: bool = True,
        polish_output: bool = False,
        enable_retrieval: Optional[bool] = None,
    ) -> str:

        t_start = time.time()
        orig_user_prompt = _strip_special_text_tokens(prompt.strip())

        # ✅ Working memory: record user turn immediately (so facts extracted even if generation fails)
        self._wm_add_user(orig_user_prompt)

        # ✅ Facts injection: deterministic subset relevant to current prompt
        facts_subset = _facts_relevant_subset(self.facts, orig_user_prompt, limit=int(os.environ.get("ARDOR_FACTS_INJECT_LIMIT", "8")))
        facts_block = _facts_render_for_injection(facts_subset, limit=int(os.environ.get("ARDOR_FACTS_INJECT_LIMIT", "8")))

        use_retrieval = self.enable_retrieval if enable_retrieval is None else bool(enable_retrieval)
        hard_fallback = False
        hippocampus_context = ""
        retrieved_snippets: List[Dict[str, Any]] = []

        len_req = _parse_length_request(orig_user_prompt)
        target_sentences = len_req.get("sentences")
        target_words = len_req.get("words")
        target_tokens = len_req.get("tokens")
        _jlog(logging.INFO, "[PFC] length request parsed", sentences=target_sentences, words=target_words, tokens=target_tokens)

        _jlog(logging.INFO, "[PFC] retrieval status",
              self_enable_retrieval=self.enable_retrieval,
              use_retrieval=use_retrieval,
              has_encoder=self.encoder is not None,
              has_parietal=self.parietal is not None)

        max_new_tokens = None if (max_new_tokens is None or int(max_new_tokens) <= 0) else int(max_new_tokens)
        hippocampus_context = ""

        # ✅ TURN-LEVEL encoder call ONCE
        e_query: Optional[torch.Tensor] = None
        if self.parietal is not None:
            try:
                _jlog(logging.INFO, "[PFC] parietal.encode begin")
                e_query = self.parietal.encode(orig_user_prompt).to(self.device)
                _jlog(logging.INFO, "[PFC] parietal.encode ok", e_shape=tuple(e_query.shape))
            except Exception as e:
                print(f"[PFC] parietal.encode failed: {e}")
                _jlog(logging.ERROR, "[PFC] parietal.encode failed", err=str(e))
                e_query = None
        else:
            _jlog(logging.INFO, "[PFC] parietal is None (no retrieval memory object)")

        # ✅ Hippocampus cosine top-k (snippets are SAFE summaries only)
        if use_retrieval and (self.parietal is not None) and (e_query is not None):
            _jlog(logging.INFO, "[PFC] memory topk begin", k=8)
            hits = self.parietal.topk_from_vec(e_query, k=8)

            retrieved_snippets = [
                {"rank": i + 1, "sim": float(s), "text": _strip_special_text_tokens((t or "").strip())}
                for i, (t, s) in enumerate(hits)
            ]
            _jlog(logging.INFO, "[RETRIEVAL] raw hits", n=len(retrieved_snippets),
                  best=(retrieved_snippets[0]["sim"] if retrieved_snippets else None),
                  head=(retrieved_snippets[0]["text"][:180] if retrieved_snippets else ""))

            best_thr = float(os.environ.get("ARDOR_MEM_BEST_MIN", "0.55"))
            margin_thr = float(os.environ.get("ARDOR_MEM_MARGIN_MIN", "0.07"))
            base_thr = float(os.environ.get("ARDOR_MEM_MIN_SIM", "0.42"))

            filtered = [(t, s) for (t, s) in hits if float(s) >= base_thr]

            used_snips = [
                {"rank": i + 1, "sim": float(s), "text": _strip_special_text_tokens((t or "").strip())}
                for i, (t, s) in enumerate(filtered)
            ]
            _jlog(logging.INFO, "[RETRIEVAL] filtered hits", n=len(used_snips),
                  best=(used_snips[0]["sim"] if used_snips else None),
                  head=(used_snips[0]["text"][:180] if used_snips else ""))

            best = filtered[0][1] if len(filtered) >= 1 else None
            second = filtered[1][1] if len(filtered) >= 2 else None
            margin = (best - second) if (best is not None and second is not None) else None

            gate_ok = (best is not None) and (best >= best_thr) and (
                (second is None) or (margin is not None and margin >= margin_thr)
            )

            _jlog(logging.INFO, "[PFC] retrieval gate",
                  base_thr=base_thr, best_thr=best_thr, margin_thr=margin_thr,
                  best=best, second=second, margin=margin, gate_ok=gate_ok)

            if gate_ok and filtered:
                keep_n = int(os.environ.get("ARDOR_MEM_KEEP_N", "2"))
                max_chars = int(os.environ.get("ARDOR_MEM_MAX_CHARS", "700"))
                per_item_chars = int(os.environ.get("ARDOR_MEM_PER_ITEM_CHARS", "260"))
                min_chars = int(os.environ.get("ARDOR_MEM_MIN_CHARS", "0"))

                blocks: List[str] = []
                for i, (t, s) in enumerate(filtered[:keep_n]):
                    t = _sanitize_retrieval_snippet(t)
                    if not t:
                        continue
                    if per_item_chars > 0 and len(t) > per_item_chars:
                        t = self._trim_to_sentence_boundary(t, per_item_chars)
                        if t and not t.endswith((".", "!", "?", "…")):
                            t = t.rstrip() + "…"
                    blocks.append(f"• {t}")

                mem_str = "\n".join(blocks).strip()

                if max_chars > 0 and len(mem_str) > max_chars:
                    mem_str = self._trim_to_sentence_boundary(mem_str, max_chars)
                    if mem_str and not mem_str.endswith(("…", ".", "!", "?")):
                        mem_str = mem_str.rstrip() + "…"

                if min_chars > 0 and len(mem_str) < min_chars:
                    mem_str = ""

                if mem_str:
                    hippocampus_context = (
                        "INTERNAL CONTEXT (do not quote, do not mention, do not reveal):\n"
                        f"{mem_str}"
                    )
                    _jlog(logging.INFO, "[PFC] hippocampus context prepared (system)", keep_n=keep_n, mem_chars=len(mem_str))
                else:
                    _jlog(logging.INFO, "[PFC] hippocampus context skipped (empty after sanitize/cap/min)")
            else:
                _jlog(logging.INFO, "[PFC] hippocampus context skipped (gate failed)")
        else:
            _jlog(logging.INFO, "[PFC] retrieval skipped", use_retrieval=use_retrieval, has_parietal=self.parietal is not None, has_e_query=e_query is not None)
            retrieved_snippets = []
            _jlog(logging.INFO, "[RETRIEVAL] raw hits", n=0, best=None, head="")
            _jlog(logging.INFO, "[RETRIEVAL] filtered hits", n=0, best=None, head="")

        # ─────────────────────────────────────────────────────────────
        # Persona / scaffolding + build MULTI-TURN prompt
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
        composed = self._build_chat_prompt(
            sys_primer,
            orig_user_prompt,
            history=self._wm_recent_turns(),
            facts_block=facts_block,
            retrieval_block=hippocampus_context,
        )
        _jlog(logging.INFO, "[prompt] composed ready", chars=len(composed), head=composed[:160])

        _jlog(logging.INFO, "[FINAL_PROMPT] ready",
              chars=len(composed),
              head=composed[:260],
              tail=composed[-260:] if len(composed) > 260 else composed)

        _dump_text_file(_DATASET_CONV_DIR / "debug_final_prompt.txt", composed)

        _jlog(logging.INFO, "[FINAL_PROMPT] retrieved_snippets",
              n=len(retrieved_snippets),
              sims=[round(float(x.get("sim", 0.0)), 4) for x in retrieved_snippets[:6]])

        enc = self.tokenizer.encode(composed)
        ids = torch.tensor([enc.ids], device=self.device, dtype=torch.long)
        _jlog(logging.INFO, "[tok] encoded composed", n_ids=len(enc.ids), ids_shape=tuple(ids.shape))

        prompt_tok_len = len(enc.ids)
        user_tok_len = len(self.tokenizer.encode(orig_user_prompt).ids)
        ctx_cap = max(1, ctx - prompt_tok_len - safety)

        hard_max = int(os.environ.get("ARDOR_HARD_MAX_NEW", str(self.gen_max_tokens)))

        requested_cap: Optional[int] = None
        if isinstance(target_tokens, int) and target_tokens > 0:
            requested_cap = int(target_tokens)
        elif isinstance(target_words, int) and target_words > 0:
            requested_cap = int(max(24, min(420, int(target_words * 1.3))))
        elif isinstance(target_sentences, int) and target_sentences > 0:
            requested_cap = int(max(40, min(360, target_sentences * 30 + 20)))

        if max_new_tokens is None:
            prof = profile or "general"
            desired = int(_estimate_desired_tokens(orig_user_prompt, prof, user_tok_len))
            max_new_tokens = min(desired, ctx_cap, hard_max)
        else:
            max_new_tokens = min(int(max_new_tokens), ctx_cap, hard_max)

        if requested_cap is not None:
            max_new_tokens = min(int(max_new_tokens), int(requested_cap))

        min_new_tokens = int(min_new_tokens)
        if max_new_tokens < min_new_tokens:
            min_new_tokens = max(1, max_new_tokens)
        max_new_tokens = max(min_new_tokens, int(max_new_tokens))

        self.gen_max_tokens = int(max_new_tokens)

        _jlog(logging.INFO, "[budget] computed",
              ctx=ctx, safety=safety, prompt_tok_len=prompt_tok_len, ctx_cap=ctx_cap,
              max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, profile=(profile or None),
              requested_cap=requested_cap)

        want = _keywords(orig_user_prompt, self.stopwords)
        key_ids = _token_ids_for_terms(self.tokenizer, want)
        BAD_START = _token_ids_for_chars(self.tokenizer, {'"', '“', '”', '—', '–', '…', "''", "'", '’', '•'})

        nblock = NgramBlocker(n=max(1, int(ngram_block)))
        phrase_bias = PhraseBias(self.tokenizer, BAD_PHRASES, bias=-2.5, max_len=16)
        early_q = EarlyQuestionTamer(self.tokenizer, until_tokens=60, penalty=0.8)
        nblock.reset()

        eos_like = list(getattr(self, "_stop_ids", None) or _get_stop_ids(self.tokenizer))
        _jlog(logging.INFO, "[decode] eos_like", eos_like=eos_like)

        generated = ids[0].tolist()
        prompt_len = len(generated)
        first_token = True

        role_user_tok = spec["user"]
        role_asst_tok = spec["asst"]

        eos_bias_after = int(os.environ.get("ARDOR_EOS_BIAS_AFTER", "20"))
        eos_bonus_max = float(os.environ.get("ARDOR_EOS_BONUS_MAX", "2.5"))
        eos_punct_bonus = float(os.environ.get("ARDOR_EOS_PUNCT_BONUS", "0.7"))

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

            temp_local = float(temp)
            p_local = float(p)
            rp_penalty = float(rep_penalty)

            temp_local = max(0.05, min(2.0, temp_local))
            p_local = max(0.05, min(0.999, p_local))
            rp_penalty = max(1.0, min(2.0, rp_penalty))

            _jlog(logging.INFO, "[decode] decode_once begin",
                  temp=temp, top_p=p, temp_local=temp_local, p_local=p_local, rp_penalty=rp_penalty,
                  gen_max_tokens=self.gen_max_tokens)

            out_parts: List[str] = []
            sentences_seen = 0
            words_seen = 0
            stop_delim = os.environ.get("ARDOR_STOP_DELIM", "<|eot|>").strip()

            def _update_counts():
                nonlocal sentences_seen, words_seen
                txt = "".join(out_parts)
                sentences_seen = len(re.findall(r"[.!?](?=\s|$)", txt))
                words_seen = len(re.findall(r"\b\w+\b", txt))

            for step in range(self.gen_max_tokens):
                with torch.no_grad():
                    logits = _model_forward(ids)[:, -1, :]

                    win = generated[-256:]
                    if win and rp_penalty and rp_penalty != 1.0:
                        rp = max(1.0, float(rp_penalty))
                        idx = torch.tensor(list(set(win)), device=logits.device, dtype=torch.long)
                        vals = logits[0, idx]
                        logits[0, idx] = torch.where(vals > 0, vals / rp, vals * rp)

                    if len(generated) > 0:
                        last_tok = generated[-1]
                        logits[0, last_tok] -= 1.25

                    if ngram_block and ngram_block > 1:
                        nblock.apply(logits[0], never_block=set(eos_like))

                    logits = _soft_bias_logits(logits, key_ids, +0.10)

                    want_digits = bool(re.search(r"\d", orig_user_prompt)) or bool(re.search(
                        r"\b(number|digit|math|calculate|hours|minutes|percent|%|€|\$)\b",
                        orig_user_prompt.lower()
                    ))
                    if not want_digits:
                        logits = _soft_bias_logits(logits, self._digit_ids, -0.50)

                    if role_user_tok is not None:
                        logits[0, role_user_tok] -= 5.0
                    if role_asst_tok is not None:
                        logits[0, role_asst_tok] -= 5.0

                    if first_token and BAD_START:
                        logits = _soft_bias_logits(logits, BAD_START, -1.5)
                    if first_token:
                        alpha_ids = _token_ids_for_chars(self.tokenizer, set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))
                        logits = _soft_bias_logits(logits, alpha_ids, +0.15)

                    if suppress_vague:
                        phrase_bias.apply(logits[0], generated)
                        gen_steps = len(generated) - prompt_len
                        early_q.apply(logits[0], step=gen_steps)

                    gen_steps = len(generated) - prompt_len
                    if eos_like:
                        logits = logits.clone()
                        for eos_tok in eos_like:
                            if gen_steps < min_new_tokens:
                                logits[0, eos_tok] = -float("inf")
                            else:
                                if gen_steps >= eos_bias_after and max_new_tokens > eos_bias_after:
                                    frac = (gen_steps - eos_bias_after) / max(1, (max_new_tokens - eos_bias_after))
                                    frac = max(0.0, min(1.0, frac))
                                    logits[0, eos_tok] += (eos_bonus_max * frac)

                                if out_parts:
                                    tail_txt = "".join(out_parts).rstrip()
                                    if tail_txt.endswith((".", "!", "?")):
                                        logits[0, eos_tok] += eos_punct_bonus

                    ArdorCore._apply_top_k_(logits, top_k)
                    frac = step / max(1, self.gen_max_tokens - 1)
                    dyn_temp = max(min_temp, temp_local * (0.7 + 0.3 * (1 - frac)))
                    probs = torch.softmax(logits / max(dyn_temp, 1e-5), dim=-1)

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

                piece = self.tokenizer.decode([next_tok], skip_special_tokens=True)
                if piece:
                    out_parts.append(piece)
                    _update_counts()
                    if stop_delim:
                        joined = "".join(out_parts)
                        pos = joined.find(stop_delim)
                        if pos != -1 and (len(generated) - prompt_len) >= min_new_tokens:
                            out_parts = [joined[:pos]]
                            _jlog(logging.INFO, "[decode] stopping: stop_delim matched", stop_delim=stop_delim, at=pos)
                            break

                new_len = len(generated) - prompt_len

                if isinstance(target_sentences, int) and target_sentences > 0:
                    if sentences_seen >= target_sentences and new_len >= min_new_tokens:
                        tail_txt = "".join(out_parts).rstrip()
                        if tail_txt.endswith((".", "!", "?")):
                            _jlog(logging.INFO, "[decode] stopping: sentence target reached", target=target_sentences, sentences=sentences_seen, new_len=new_len)
                            break

                if isinstance(target_words, int) and target_words > 0:
                    if words_seen >= target_words and new_len >= min_new_tokens:
                        tail_txt = "".join(out_parts).rstrip()
                        if tail_txt.endswith((".", "!", "?")) or words_seen >= (target_words + 8):
                            _jlog(logging.INFO, "[decode] stopping: word target reached", target=target_words, words=words_seen, new_len=new_len)
                            break

                if new_len >= max_new_tokens:
                    _jlog(logging.INFO, "[decode] stopping: max_new_tokens reached", new_len=new_len, max_new_tokens=max_new_tokens)
                    break

                if eos_like and (next_tok in eos_like) and new_len >= min_new_tokens:
                    _jlog(logging.INFO, "[decode] stopping: eos_like token generated", tok=next_tok, new_len=new_len)
                    if stop_on_eos:
                        break

                ids = torch.tensor([generated], device=self.device, dtype=torch.long)

            out = "".join(out_parts)
            if not out:
                out = self.tokenizer.decode(generated[prompt_len:], skip_special_tokens=True)
            _jlog(logging.INFO, "[decode] decode_once done", out_chars=len(out))
            return out.strip()

        _jlog(logging.INFO, "[decode] first pass begin")
        out1 = decode_once(temperature, top_p)
        _jlog(logging.INFO, "[decode] first pass done", out_chars=len(out1))

        out1 = _strip_speaker_tags(out1).strip()
        out1 = polish(out1) if polish_output else out1

        final_out = _strip_speaker_tags(out1).strip()
        final_out = self._strip_boilerplate(final_out)

        if isinstance(target_sentences, int) and target_sentences > 0:
            s = final_out.strip()
            ends = list(re.finditer(r"[.!?](?=\s|$)", s))
            if len(ends) > target_sentences:
                cut_i = ends[target_sentences - 1].end()
                final_out = s[:cut_i].strip()

        out_cap = int(os.environ.get("ARDOR_OUT_MAX_CHARS", "1600"))
        if out_cap > 0 and len(final_out) > out_cap:
            final_out = self._trim_to_sentence_boundary(final_out, out_cap)

        final_out = _strip_special_text_tokens(final_out)
        _jlog(logging.INFO, "[final] output ready", chars=len(final_out), head=final_out[:140])

        # ─────────────────────────────────────────────────────────────
        # ✅ Memory integrity gating
        # - heuristics (fluency/repetition/closure/rel)
        # - PLUS task-aware correctness checks (objective hooks)
        # ─────────────────────────────────────────────────────────────
        metrics_final = self._text_metrics(final_out, orig_user_prompt)
        rel_final = float(metrics_final.get("rel", 0.0))
        rep2_final = float(metrics_final.get("rep2", 1.0))
        rep3_final = float(metrics_final.get("rep3", 1.0))
        and_dup_final = float(metrics_final.get("and_dup", 1.0))

        rep3_max = float(os.environ.get("ARDOR_MEM_REP3_MAX", "0.12"))
        rep2_max = float(os.environ.get("ARDOR_MEM_REP2_MAX", "0.10"))
        min_chars_log = int(os.environ.get("ARDOR_MEM_MIN_LOG_CHARS", "40"))

        good_rel_min = float(os.environ.get("ARDOR_GOOD_REL_MIN", "0.25"))
        if len(orig_user_prompt) < 60:
            good_rel_min *= 0.85

        violates_len = False
        if isinstance(target_sentences, int) and target_sentences > 0:
            scount = len(re.findall(r"[.!?](?=\s|$)", final_out.strip()))
            if scount > target_sentences:
                violates_len = True

        max_good_chars = int(os.environ.get("ARDOR_MEM_MAX_GOOD_CHARS", "300"))
        closure_final = float(metrics_final.get("closure", 0.0))

        # ✅ Task-aware correctness hooks
        ok_correct, correctness = self._correctness_checks(orig_user_prompt, final_out)

        is_good = (
            (not hard_fallback)
            and (len(final_out.strip()) >= min_chars_log)
            and (len(final_out) <= max_good_chars)
            and (closure_final >= 1.0)
            and (rel_final >= max(float(relevance_floor), good_rel_min))
            and (rep3_final <= rep3_max)
            and (rep2_final <= rep2_max)
            and (and_dup_final <= 0.0)
            and (not violates_len)
            and ok_correct
        )

        _jlog(logging.INFO, "[memory] classify",
              is_good=is_good, hard_fallback=hard_fallback,
              rel_final=rel_final, good_rel_min=good_rel_min, floor=relevance_floor,
              rep2_final=rep2_final, rep2_max=rep2_max,
              rep3_final=rep3_final, rep3_max=rep3_max,
              and_dup=and_dup_final, violates_len=violates_len,
              min_chars_log=min_chars_log,
              correctness=correctness)

        # ✅ Working memory store: only store assistant if high-integrity
        self._wm_add_assistant(final_out, allow_store=is_good)

        # ✅ Write logs: GOOD is safe prompt+summary+facts only
        if log_response and getattr(self, "write_memory", True):
            self.log(
                orig_user_prompt,
                final_out,
                prompt_vec=e_query,
                good=is_good,
                facts=facts_subset,
                quality={"metrics": metrics_final, "correctness": correctness, "good": is_good},
            )
        else:
            _jlog(logging.INFO, "[memory] logging suppressed",
                  log_response=log_response,
                  write_memory=getattr(self, "write_memory", True))

        _jlog(logging.INFO, "[PFC] generate_text end", seconds=round(time.time() - t_start, 4))
        return final_out

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
        roots = [str(_PROJECT_ROOT / "Models"), str(_PROJECT_ROOT / "Cerebrum" / "Models"), str(_PROJECT_ROOT / "Cerebrum" / "Models" / "Ardor")]
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
            str(_PROJECT_ROOT / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v9.json"),
            str(_PROJECT_ROOT / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v8.json"),
            str(_PROJECT_ROOT / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v7.json"),
            str(_PROJECT_ROOT / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v9.json"),
            str(_PROJECT_ROOT / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v8.json"),
            str(_PROJECT_ROOT / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v7.json"),
        ]
        tokenizer_path = next((p for p in tok_candidates if os.path.isfile(p)), None)
        _jlog(logging.INFO, "[cli] tokenizer candidate resolved", tokenizer_path=tokenizer_path)
        if tokenizer_path is None:
            print("Tokenizer not found. Please place tokenizer_v*.json in ProjectTokenizer/ardor_tokenizer.")
            _jlog(logging.ERROR, "[cli] tokenizer not found", candidates=tok_candidates)
            return

        encoder_ckpt = os.environ.get("ARDOR_ENCODER_CKPT", None) or str(_PROJECT_ROOT / "Cerebrum" / "Models" / "Encoders" / "ArdorEncoder.pt")
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
                subprocess.call(["python", str(_PROJECT_ROOT / "Cerebrum" / "Cortex" / "neural_plasticity_training.py")])
                continue
            if cmd in ("rem", "sleep"):
                _jlog(logging.INFO, "[cli] launching REM subprocess")
                subprocess.call(["python", str(_PROJECT_ROOT / "Cerebrum" / "CorticalIntegration" / "REM.py")])
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