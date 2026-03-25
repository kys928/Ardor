from __future__ import annotations

import os
import sys
import time
import json
import subprocess
import re
import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

try:
    from .ardor_config import ArdorConfig
except Exception:
    from ardor_config import ArdorConfig

try:
    from .backends.factory import load_backend
    from .backends.retrieval import load_retrieval_backend
    from .loaders.native_checkpoint import load_native_decoder as _loader_load_native_decoder
    from .loaders.native_encoder import load_encoder_cached as _loader_load_encoder_cached
    from .loaders.native_tokenizer import load_tokenizer_matching_vocab as _loader_match_tokenizer
except Exception:
    from backends.factory import load_backend
    from backends.retrieval import load_retrieval_backend
    from loaders.native_checkpoint import load_native_decoder as _loader_load_native_decoder
    from loaders.native_encoder import load_encoder_cached as _loader_load_encoder_cached
    from loaders.native_tokenizer import load_tokenizer_matching_vocab as _loader_match_tokenizer

try:
    from Cerebrum.LanguageProcessing.Anterior_Cingulate import polish
except Exception:
    try:
        from .LanguageProcessing.Anterior_Cingulate import polish  # type: ignore
    except Exception:
        def polish(text: str) -> str:
            return text

REPO_ROOT = Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────────────────────────────
# ✅ LOGGING
# ─────────────────────────────────────────────────────────────────────
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
            print(f"[LOG] failed to create file logger at {_ARDOR_LOG_FILE}: {e}")

    lg._ardor_configured = True  # type: ignore[attr-defined]
    lg.info(
        f"[boot] logger configured level={_ARDOR_LOG_LEVEL} json={_ARDOR_LOG_JSON} "
        f"file={'ON' if _ARDOR_LOG_FILE else 'OFF'}"
    )
    return lg


_LOG = _setup_ardor_logger()


def _jlog(level: int, event: str, **kv: Any) -> None:
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
    return _LOG.isEnabledFor(logging.DEBUG) and os.environ.get("ARDOR_TRACE", "0").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    )


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    supported = {}
    accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_varkw:
        supported = dict(kwargs)
    else:
        for k, v in kwargs.items():
            if k in sig.parameters:
                supported[k] = v
    return fn(**supported)


# ── Optional safety/env toggles ──────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RUST_BACKTRACE", "1")

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
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "because",
        "as",
        "of",
        "in",
        "on",
        "for",
        "to",
        "from",
        "by",
        "with",
        "about",
        "into",
        "over",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "this",
        "that",
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


# ─────────────────────────────────────────────────────────────────────
# Compatibility wrappers for moved loader helpers
# ─────────────────────────────────────────────────────────────────────
def _read_checkpoint_meta(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        if isinstance(raw.get("config"), dict):
            return dict(raw["config"])
        if isinstance(raw.get("meta"), dict):
            return dict(raw["meta"])
        keys = ("vocab_size", "hidden_size", "n_layers", "n_heads", "max_len")
        if any(k in raw for k in keys):
            return {k: raw.get(k) for k in keys if k in raw}
    return {}


def _config_from_meta(meta: Dict[str, Any]) -> ArdorConfig:
    sig = inspect.signature(ArdorConfig)
    cfg_kwargs: Dict[str, Any] = {}
    defaults = {
        "ff_mult": 4,
        "dropout": 0.12,
        "attn_dropout": 0.0,
        "resid_dropout": 0.0,
        "layernorm_eps": 1e-5,
        "use_rope": True,
        "rope_theta": 10000.0,
    }
    for name in sig.parameters:
        if name in meta:
            cfg_kwargs[name] = meta[name]
        elif name in defaults:
            cfg_kwargs[name] = defaults[name]
    return ArdorConfig(**cfg_kwargs)


def _infer_dims_from_state(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int]:
    preferred_vocab_keys = (
        "token_embed.weight",
        "tok_emb.weight",
        "embedding.weight",
        "wte.weight",
        "embeddings.weight",
        "lm_head.weight",
        "to_vocab.weight",
    )

    vocab = hidden = None
    for key in preferred_vocab_keys:
        t = sd.get(key)
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            vocab = int(t.shape[0])
            hidden = int(t.shape[1])
            break

    if vocab is None or hidden is None:
        two_d = [(k, t) for k, t in sd.items() if isinstance(t, torch.Tensor) and t.ndim == 2]
        nonsq = [(k, t) for k, t in two_d if int(t.shape[0]) != int(t.shape[1])]
        pool = nonsq or two_d
        if not pool:
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

    max_len = 2048
    for k, t in sd.items():
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            r, c = int(t.shape[0]), int(t.shape[1])
            if c == hidden and re.search(r"(pos|position).*emb.*weight", k, re.I):
                max_len = r
                break

    _jlog(logging.INFO, "[ckpt] inferred dims", vocab=vocab, hidden=hidden, layers=layers, max_len=max_len)
    return vocab, hidden, layers, int(max_len)


def _load_broca_cached(model_path: str, device: str, tokenizer_path_hint: Optional[str] = None):
    return _loader_load_native_decoder(model_path, device)


def _load_tokenizer_matching_vocab(
    tokenizer_path: Optional[str], want_vocab: int, checkpoint_meta: Optional[dict] = None
) -> Tuple[Tokenizer, str]:
    return _loader_match_tokenizer(REPO_ROOT, tokenizer_path, want_vocab, checkpoint_meta)


def _load_encoder_cached(encoder_ckpt: Optional[str], device: str, fallback_decoder_cfg: Optional[ArdorConfig] = None):
    return _loader_load_encoder_cached(encoder_ckpt, device, fallback_decoder_cfg)


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


def _resolve_encoder_ckpt(user_path: Optional[str]) -> str:
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
                hits += found
            if hits:
                hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                _jlog(logging.INFO, "[encoder_ckpt] using newest in directory", picked=str(hits[0]), n=len(hits))
                return str(hits[0])
        raise FileNotFoundError(f"encoder_ckpt was provided but not found: {user_path}")

    envp = os.environ.get("ARDOR_ENCODER_CKPT", "").strip()
    if envp:
        p = Path(envp).expanduser()
        if p.is_file():
            _jlog(logging.INFO, "[encoder_ckpt] using env file", path=str(p))
            return str(p)
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
            if not r.exists() or not r.is_dir():
                continue
            for patt in patterns:
                candidates += list(r.glob(patt))
        except Exception:
            pass

    for r in roots:
        try:
            if not r.exists() or not r.is_dir():
                continue
            for sub in r.iterdir():
                if sub.is_dir():
                    for patt in patterns:
                        candidates += list(sub.glob(patt))
        except Exception:
            pass

    candidates = [c for c in candidates if c.is_file()]
    _jlog(logging.INFO, "[encoder_ckpt] candidates collected", n=len(candidates))
    if candidates:
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        _jlog(logging.INFO, "[encoder_ckpt] picked newest", path=str(candidates[0]))
        return str(candidates[0])

    raise FileNotFoundError(
        "No encoder checkpoint found. Provide encoder_ckpt=... or set ARDOR_ENCODER_CKPT to a valid .pt file."
    )


def _keywords(text: str, stopwords: set[str]) -> set[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    out = {t for t in toks if t not in stopwords}
    _jlog(logging.DEBUG, "[kw] keywords", in_len=len(text), toks=len(toks), out=len(out))
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


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
    qm = text.count("?")
    qw = len(re.findall(r"(?mi)^\s*(what|why|how|when|where|who|which|can|could|should|do|does|is|are)\b", text))
    return max(qm, qw)


def _has_code(text: str) -> bool:
    t = text.lower()
    return ("```" in t) or any(x in t for x in ["def ", "class ", "#include", "import ", "using ", "{", "};"])


def _estimate_desired_tokens(prompt: str, profile: str, prompt_tok_len: int) -> int:
    base_map = {
        "code": int(os.environ.get("ARDOR_BUDGET_CODE_BASE", "220")),
        "math": int(os.environ.get("ARDOR_BUDGET_MATH_BASE", "180")),
        "instruction": int(os.environ.get("ARDOR_BUDGET_INST_BASE", "220")),
        "qa": int(os.environ.get("ARDOR_BUDGET_QA_BASE", "150")),
        "creative": int(os.environ.get("ARDOR_BUDGET_CREAT_BASE", "260")),
        "general": int(os.environ.get("ARDOR_BUDGET_GEN_BASE", "160")),
    }
    base = base_map.get(profile, base_map["general"])
    qn = _count_questions(prompt)
    long_prompt_bonus = int(10 * (prompt_tok_len ** 0.5))
    multi_question_bonus = min(160, 40 * max(0, qn - 1))
    code_bonus = 120 if _has_code(prompt) else 0
    list_bonus = 80 if re.search(r"(?mi)^\s*[-*]\s+", prompt) else 0
    short_signal = bool(re.search(r"\b(short|brief|tl;dr|one sentence|few sentences)\b", prompt.lower()))
    if short_signal:
        base = min(base, 120)
    desired = base + long_prompt_bonus + multi_question_bonus + code_bonus + list_bonus
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
    backend_family: Optional[str] = None,
) -> "ArdorCore":
    global _PFC_SINGLETON, _PFC_SIGNATURE

    _jlog(
        logging.INFO,
        "[singleton] get_global_core called",
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        enable_retrieval=enable_retrieval,
        encoder_ckpt=encoder_ckpt,
        max_len=max_len,
        enable_dmn=enable_dmn,
        force_reload=force_reload,
        backend_family=backend_family,
    )

    sig = (
        os.path.abspath(model_path),
        os.path.abspath(tokenizer_path) if tokenizer_path else None,
        device,
        bool(enable_retrieval),
        os.path.abspath(encoder_ckpt) if encoder_ckpt else None,
        int(max_len),
        bool(enable_dmn),
        backend_family,
    )

    if _PFC_SINGLETON is None or force_reload or (_PFC_SIGNATURE != sig):
        _jlog(logging.INFO, "[singleton] creating/reloading ArdorCore")
        _PFC_SIGNATURE = sig
        _PFC_SINGLETON = ArdorCore(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device,
            max_len=max_len,
            enable_retrieval=enable_retrieval,
            encoder_ckpt=encoder_ckpt,
            enable_dmn=enable_dmn,
            aeternum=None,
            backend_family=backend_family,
        )
    else:
        _jlog(logging.INFO, "[singleton] returning existing ArdorCore", sig=_PFC_SIGNATURE)

    return _PFC_SINGLETON


def get_core_singleton() -> Optional["ArdorCore"]:
    _jlog(logging.DEBUG, "[singleton] get_core_singleton", exists=_PFC_SINGLETON is not None)
    return _PFC_SINGLETON


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
        backend_family: Optional[str] = None,
    ):
        _jlog(
            logging.INFO,
            "[PFC] ArdorCore.__init__ start",
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device,
            max_len=max_len,
            enable_retrieval=enable_retrieval,
            encoder_ckpt=encoder_ckpt,
            enable_dmn=enable_dmn,
            aeternum_is_none=(aeternum is None),
            backend_family=backend_family,
        )

        self.device = device
        self.gen_max_tokens = int(max_len)
        self.enable_retrieval = bool(enable_retrieval)
        self.model_path = model_path

        self.backend = load_backend(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device,
            repo_root=REPO_ROOT,
            backend_family=backend_family,
            encoder_ckpt=encoder_ckpt,
        )
        self.model = getattr(self.backend, "model", None)
        self.tokenizer = getattr(self.backend, "tokenizer", None)
        self.tokenizer_path = self.backend.tokenizer_path()

        self.schema = self.backend.schema()
        self.layers = int(self.schema.get("layers")) if self.schema.get("layers") is not None else None
        self.heads = int(self.schema.get("heads")) if self.schema.get("heads") is not None else None
        self.hidden = int(self.schema.get("hidden")) if self.schema.get("hidden") is not None else None
        self.model_ctx_len = int(self.schema.get("max_len")) if self.schema.get("max_len") is not None else None
        self.vocab_size = int(self.schema.get("vocab") or self.backend.vocab_size())

        mismatch = self.schema.get("mismatch") or {"missing": [], "unexpected": []}
        miss_ct = len(mismatch.get("missing") or [])
        unex_ct = len(mismatch.get("unexpected") or [])

        print(
            f"🧠 Model schema: layers={self.layers} heads={self.heads} hidden={self.hidden} "
            f"max_len={self.model_ctx_len} mismatch: missing={miss_ct} unexpected={unex_ct}"
        )
        print(f"🧩 Tokenizer: {self.tokenizer_path}  | vocab={self.vocab_size}")
        _jlog(
            logging.INFO,
            "[PFC] schema",
            layers=self.layers,
            heads=self.heads,
            hidden=self.hidden,
            max_len=self.model_ctx_len,
            missing=miss_ct,
            unexpected=unex_ct,
        )

        self.retrieval_backend = load_retrieval_backend(self.backend, device=self.device, enabled=self.enable_retrieval)
        if self.retrieval_backend is not None:
            print("🧠 Retrieval: ENABLED")
        else:
            _jlog(logging.INFO, "[PFC] retrieval backend unavailable or disabled")

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

        self.recent_texts = deque(maxlen=128)
        self.chat_turns: List[Tuple[str, str]] = []
        self.max_chat_turns = int(os.environ.get("ARDOR_CHAT_TURNS", "12"))

        self.stopwords = STOPWORDS
        self._digit_ids = (
            _token_ids_for_chars(self.tokenizer, set("0123456789")) if isinstance(self.tokenizer, Tokenizer) else set()
        )
        self._eos_id = _find_eos_id(self.tokenizer) if isinstance(self.tokenizer, Tokenizer) else None
        self._eot_id = self.tokenizer.token_to_id("<|eot|>") if isinstance(self.tokenizer, Tokenizer) else None

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

        _jlog(
            logging.INFO,
            "[PFC] init done",
            gen_max_tokens=self.gen_max_tokens,
            max_chat_turns=self.max_chat_turns,
            eos_id=self._eos_id,
            eot_id=self._eot_id,
            digit_ids=len(self._digit_ids),
            dmn_enabled=self.enable_dmn,
            dmn_ready=self.dmn is not None,
        )

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

    def _backend_encode_text(self, text: str) -> Optional[torch.Tensor]:
        try:
            if hasattr(self.backend, "encode_text"):
                vec = self.backend.encode_text(text)
                if vec is None:
                    return None
                if isinstance(vec, torch.Tensor):
                    if vec.ndim == 1:
                        vec = vec.unsqueeze(0)
                    vec = vec.to(self.device)
                    return F.normalize(vec, dim=1)
        except Exception as e:
            _jlog(logging.WARNING, "[backend] encode_text failed", err=str(e))
        return None

    # ───────────────────── metrics/scoring ─────────────────────
    def _text_metrics(self, text: str, prompt: str) -> Dict[str, float]:
        toks = re.findall(r"\w+|[^\w\s]", text)
        n = len(toks)
        if n == 0:
            return {"d1": 0.0, "d2": 0.0, "rep3": 1.0, "closure": 0.0, "imbalance": 1.0, "rel": 0.0}

        d1 = len(set(toks)) / max(1, n)
        bigrams = [tuple(toks[i : i + 2]) for i in range(max(0, n - 1))]
        d2 = (len(set(bigrams)) / max(1, len(bigrams))) if bigrams else 0.0
        trigrams = [tuple(toks[i : i + 3]) for i in range(max(0, n - 2))]
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

        rel = jac
        try:
            v_p = self._backend_encode_text(prompt)
            v_r = self._backend_encode_text(text)
            if v_p is not None and v_r is not None:
                cos = float(torch.matmul(v_p, v_r.transpose(0, 1)).item())
                cos01 = (cos + 1.0) * 0.5
                w = float(os.environ.get("ARDOR_REL_SEM_W", "0.70"))
                rel = w * cos01 + (1.0 - w) * jac
        except Exception as e:
            _jlog(logging.WARNING, "[metrics] semantic relevance fallback to jaccard", err=str(e))

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

    def log(self, prompt: str, resp: str):
        _jlog(logging.INFO, "[log] writing episode", prompt_len=len(prompt), resp_len=len(resp))
        with LOG_FILE.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"prompt": prompt, "response": resp, "ts": time.time()}, ensure_ascii=False) + "\n")
        self.recent_texts.append(prompt[:600])
        self.recent_texts.append(resp[:600])

    def pick_decoding_config(self, prompt: str, *, profile: Optional[str] = None, probe_len: int = 56) -> Dict[str, Any]:
        profile = profile or self.classify_prompt(prompt)
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
            txt = self.generate_text(
                prompt,
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
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
        results.sort(key=lambda r: (-r["score"], r["temperature"] if profile in ("code", "math") else 0.0))
        return results[0]

    def conversation_memory_last_N_chunks(self, N: int = 32) -> List[str]:
        seq = list(self.recent_texts)
        if N <= 0:
            return []
        out = seq[-2 * N :]
        _jlog(logging.DEBUG, "[memory] last N chunks", N=N, returned=len(out))
        return out

    def _strip_boilerplate(self, text: str) -> str:
        t = (text or "").strip()
        bad_leads = [
            r"^Answer directly and concretely\.\s*No disclaimers\.\s*",
            r"^Answer directly and concretely\.\s*",
            r"^No disclaimers\.\s*",
        ]
        for pat in bad_leads:
            t = re.sub(pat, "", t, flags=re.IGNORECASE)
        return t.strip()

    def _should_store_in_history(self, prompt: str, resp: str, relevance_floor: float, rep3_max: float = 0.18) -> bool:
        m = self._text_metrics(resp, prompt)
        short = len(prompt.strip()) < 60
        rel_floor = relevance_floor * (0.60 if short else 1.0)
        ok = (m["rel"] >= rel_floor) and (m["rep3"] <= rep3_max) and (m["imbalance"] <= 0.6)
        _jlog(logging.INFO, "[history] store gate", ok=ok, rel=m["rel"], rel_floor=rel_floor, rep3=m["rep3"], imbalance=m["imbalance"])
        return bool(ok)

    def _clean_for_history(self, text: str) -> str:
        t = (text or "").strip()
        for tok in ("<|system|>", "<|user|>", "<|assistant|>", "<|eot|>"):
            t = t.replace(tok, "")
        t = self._strip_boilerplate(t)
        return t.strip()

    def _build_chat_prompt(self, persona_primer: str, user_text: str) -> str:
        sys_block = (persona_primer or "").strip()
        if not sys_block:
            sys_block = "You are Ardor. Stay in-context. Be helpful. Speak naturally."

        has_user = self.tokenizer.token_to_id("<|user|>") is not None
        has_asst = self.tokenizer.token_to_id("<|assistant|>") is not None
        has_eot = self.tokenizer.token_to_id("<|eot|>") is not None

        turns = self.chat_turns[-(self.max_chat_turns * 2) :]
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
            return "".join(parts)

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
        return "".join(parts)

    def _dmn_on_prompt(self, prompt: str) -> str:
        if self.dmn is None:
            return prompt
        for name in ("on_prompt", "observe_prompt", "dmn_on_prompt"):
            fn = getattr(self.dmn, name, None)
            if callable(fn):
                try:
                    out = _call_with_supported_kwargs(
                        fn,
                        prompt=prompt,
                        recent_turns=list(self.chat_turns),
                        recent_texts=list(self.recent_texts),
                        retrieval_enabled=bool(self.retrieval_backend is not None),
                    )
                    if isinstance(out, str) and out.strip():
                        return out
                except Exception as e:
                    _jlog(logging.WARNING, "[PFC] DMN prompt hook failed", hook=name, err=str(e))
        return prompt

    def _safe_aeternum_state(self, aet_state: Any = None) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        src = aet_state if aet_state is not None else getattr(self, "aet", None)
        if src is None:
            return state
        if isinstance(src, dict):
            for key in (
                "valence",
                "arousal",
                "dominance",
                "salience",
                "activation",
                "temperature_scale",
                "top_p_scale",
                "rep_penalty_scale",
            ):
                if key in src:
                    state[key] = src.get(key)
            return state
        for key in (
            "valence",
            "arousal",
            "dominance",
            "salience",
            "activation",
            "temperature_scale",
            "top_p_scale",
            "rep_penalty_scale",
        ):
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
                "mode": "DISABLED",
                "last_seed": "",
                "last_retrieved_traces": [],
                "current_narrative_takeaway": None,
                "self_model_state": {},
                "salience_info": {},
                "last_error": self._dmn_blocker,
            }
        try:
            state = self.dmn.get_state()
            self._last_dmn_state = state
            return state.to_dict() if hasattr(state, "to_dict") else dict(state)
        except Exception as e:
            return {
                "mode": "ERROR",
                "last_seed": "",
                "last_retrieved_traces": [],
                "current_narrative_takeaway": None,
                "self_model_state": {},
                "salience_info": {},
                "last_error": str(e),
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
            takeaway = _call_with_supported_kwargs(
                self.dmn.summarize_active_context,
                prompt=prompt,
                response=response,
                recent_turns=recent_turns or list(self.chat_turns),
                recent_texts=recent_texts or list(self.recent_texts),
                retrieved_memory_summary=retrieved_memory_summary or [],
                aet_state=aet_state or self._safe_aeternum_state(),
                retrieval_enabled=bool(self.retrieval_backend is not None),
            )
            self._last_dmn_takeaway = takeaway.to_dict() if hasattr(takeaway, "to_dict") else takeaway
            self._last_dmn_state = self.dmn.get_state()
            _jlog(
                logging.INFO,
                "[PFC] DMN active summary ok",
                theme=getattr(takeaway, "theme", None),
                confidence=getattr(takeaway, "confidence", None),
            )
        except Exception as e:
            _jlog(logging.WARNING, "[PFC] DMN active summary failed", err=str(e))

    def run_idle_dmn_cycle(self, prompt: str = "") -> Dict[str, Any]:
        if self.dmn is None:
            return self.get_dmn_state()
        try:
            _call_with_supported_kwargs(
                self.dmn.step_idle_cycle,
                prompt=prompt,
                recent_turns=list(self.chat_turns),
                recent_texts=list(self.recent_texts),
                retrieval_enabled=bool(self.retrieval_backend is not None),
                aet_state=self._safe_aeternum_state(),
            )
            self._last_dmn_state = self.dmn.get_state()
        except Exception as e:
            _jlog(logging.WARNING, "[PFC] DMN idle cycle failed", err=str(e))
        return self.get_dmn_state()

    def tick_idle_cognition(self) -> Dict[str, Any]:
        return self.run_idle_dmn_cycle()

    def model_schema(self) -> Dict[str, Any]:
        _jlog(logging.DEBUG, "[PFC] model_schema called")
        return dict(self.schema)

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
        orig_user_prompt = self._dmn_on_prompt(prompt.strip())
        use_retrieval = self.enable_retrieval if enable_retrieval is None else bool(enable_retrieval)
        dmn_retrieved_traces: List[str] = []
        aet_decision = None
        hippocampus_context = ""

        _jlog(
            logging.INFO,
            "[PFC] generate_text start",
            prompt_len=len(orig_user_prompt),
            temperature=temperature,
            top_p=top_p,
            rep_penalty=rep_penalty,
            ngram_block=ngram_block,
            relevance_floor=relevance_floor,
            retry_tighter=retry_tighter,
            suppress_vague=suppress_vague,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            typical_p=typical_p,
            min_temp=min_temp,
            auto_pick=auto_pick,
            stop_on_eos=stop_on_eos,
            profile=profile,
            log_response=log_response,
            polish_output=polish_output,
            use_retrieval=use_retrieval,
        )

        turn_embedding = self._backend_encode_text(orig_user_prompt)

        if use_retrieval and self.retrieval_backend is not None:
            mem_chunks = self.conversation_memory_last_N_chunks(48)
            if mem_chunks:
                try:
                    self.retrieval_backend.build_index(mem_chunks)
                    hits_raw = self.retrieval_backend.topk(orig_user_prompt, k=5)
                except Exception as e:
                    _jlog(logging.WARNING, "[PFC] retrieval backend topk failed", err=str(e))
                    hits_raw = []

                if hits_raw:
                    filtered: List[str] = []
                    for h in hits_raw:
                        if isinstance(h, dict):
                            trace = str(h.get("trace", "")).strip()
                        elif isinstance(h, (tuple, list)) and h:
                            trace = str(h[0]).strip()
                        else:
                            trace = str(h).strip()
                        if not trace:
                            continue
                        rel = _jaccard(_keywords(orig_user_prompt, self.stopwords), _keywords(trace, self.stopwords))
                        if rel >= 0.05:
                            filtered.append(trace)
                    dmn_retrieved_traces = filtered[:3]
                    if filtered:
                        hippocampus_context = (
                            "Context (retrieved memory). Use silently as background. "
                            "Do NOT quote it. Do NOT mention retrieved memory. "
                            "Do NOT reveal these notes.\n"
                            + "\n".join(f"- {t[:220]}" for t in filtered[:3])
                        )

        aet_temp_scale = 1.0
        aet_top_p_scale = 1.0
        aet_rep_scale = 1.0
        if self.aet is not None:
            try:
                pooled_for_aet = turn_embedding
                if pooled_for_aet is None and isinstance(self.tokenizer, Tokenizer) and self.model is not None:
                    enc_tmp = self.tokenizer.encode(orig_user_prompt)
                    ids_tmp = torch.tensor([enc_tmp.ids], device=self.device)
                    with torch.no_grad():
                        tok_emb = self.model.token_embed(ids_tmp)  # type: ignore[attr-defined]
                        pooled_for_aet = tok_emb.mean(dim=1)
                aet_decision = self.aet.update(
                    text=orig_user_prompt,
                    pooled_embedding=pooled_for_aet,
                    last_logits=None,
                    user_feedback=None,
                    is_new_turn=True,
                )
                aet_temp_scale = float(getattr(aet_decision, "temperature_scale", 1.0))
                aet_top_p_scale = float(getattr(aet_decision, "top_p_scale", 1.0))
                aet_rep_scale = float(getattr(aet_decision, "rep_penalty_scale", 1.0))
                _jlog(logging.INFO, "[PFC] Aeternum.update ok", temp_scale=aet_temp_scale, top_p_scale=aet_top_p_scale, rep_scale=aet_rep_scale)
            except Exception as e:
                _jlog(logging.ERROR, "[PFC] Aeternum.update failed", err=str(e))

        # Opaque backend path
        if not isinstance(self.tokenizer, Tokenizer):
            backend_prompt = orig_user_prompt
            if persona_primer:
                backend_prompt = f"{persona_primer.strip()}\n\nUser: {orig_user_prompt}"
            if hippocampus_context:
                backend_prompt = f"{persona_primer.strip()}\n\n[CONTEXT]\n{hippocampus_context}\n[/CONTEXT]\n\nUser: {orig_user_prompt}"

            out = self.backend.generate(
                backend_prompt,
                temperature=max(0.05, min(2.0, float(temperature) * float(aet_temp_scale))),
                top_p=max(0.05, min(0.999, float(top_p) * float(aet_top_p_scale))),
                max_new_tokens=max_new_tokens,
            )
            final_out = polish(out) if polish_output else out
            final_out = self._strip_boilerplate(final_out)

            if log_response:
                self.log(orig_user_prompt, final_out)

            self._maybe_summarize_with_dmn(
                prompt=orig_user_prompt,
                response=final_out,
                recent_turns=list(self.chat_turns),
                recent_texts=list(self.recent_texts),
                retrieved_memory_summary=dmn_retrieved_traces,
                aet_state=self._safe_aeternum_state(aet_decision),
            )

            self.chat_turns.append(("user", self._clean_for_history(orig_user_prompt)))
            clean_final = self._clean_for_history(final_out)
            if self._should_store_in_history(orig_user_prompt, clean_final, relevance_floor=relevance_floor):
                self.chat_turns.append(("assistant", clean_final))
            if len(self.chat_turns) > self.max_chat_turns * 2:
                self.chat_turns = self.chat_turns[-(self.max_chat_turns * 2) :]

            return final_out

        spec = {
            "user": self.tokenizer.token_to_id("<|user|>"),
            "asst": self.tokenizer.token_to_id("<|assistant|>"),
            "eot": self.tokenizer.token_to_id("<|eot|>"),
        }

        ctx = int(self.model_ctx_len or 2048)
        safety = int(os.environ.get("ARDOR_CTX_SAFETY", "96"))

        sys_primer = (persona_primer or "")
        if hippocampus_context:
            sys_primer = sys_primer.rstrip() + "\n\n[CONTEXT]\n" + hippocampus_context + "\n[/CONTEXT]\n"

        composed = self._build_chat_prompt(sys_primer, orig_user_prompt)
        enc = self.tokenizer.encode(composed)
        ids = torch.tensor([enc.ids], device=self.device)

        prompt_tok_len = len(enc.ids)
        ctx_cap = max(1, ctx - prompt_tok_len - safety)

        if max_new_tokens is None or int(max_new_tokens) <= 0:
            prof = profile or self.classify_prompt(orig_user_prompt)
            desired = _estimate_desired_tokens(orig_user_prompt, prof, prompt_tok_len)
            hard_max = int(os.environ.get("ARDOR_MAX_NEW_TOKENS_HARD", "520"))
            max_new_tokens = min(desired, ctx_cap, hard_max)
        else:
            max_new_tokens = min(int(max_new_tokens), ctx_cap)

        min_new_tokens = int(min_new_tokens)
        max_new_tokens = max(min_new_tokens, int(max_new_tokens))
        self.gen_max_tokens = int(max_new_tokens)

        want = _keywords(orig_user_prompt, self.stopwords)
        key_ids = _token_ids_for_terms(self.tokenizer, want)
        bad_start = _token_ids_for_chars(self.tokenizer, {'"', "“", "”", "—", "–", "…", "''", "'", "’", "•"})

        nblock = NgramBlocker(n=max(1, int(ngram_block)))
        phrase_bias = PhraseBias(self.tokenizer, BAD_PHRASES, bias=-2.5, max_len=8)
        early_q = EarlyQuestionTamer(self.tokenizer, until_tokens=60, penalty=0.8)
        nblock.reset()

        eos_id = _find_eos_id(self.tokenizer)
        eot_id = self.tokenizer.token_to_id("<|eot|>")
        eos_like = [t for t in (eos_id, eot_id) if t is not None]

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
            if hasattr(self.backend, "forward_logits"):
                return self.backend.forward_logits(input_ids)
            out = self.model(input_ids)  # type: ignore[operator]
            if isinstance(out, (list, tuple)) and len(out) > 0:
                out = out[0]
            if isinstance(out, dict) and "logits" in out:
                out = out["logits"]
            return out

        def decode_once(temp: float, p: float) -> str:
            nonlocal ids, generated, first_token

            temp_local = max(0.05, min(2.0, float(temp) * float(aet_temp_scale)))
            p_local = max(0.05, min(0.999, float(p) * float(aet_top_p_scale)))
            rp_pen = max(1.0, min(2.0, float(rep_penalty) * float(aet_rep_scale)))

            for step in range(self.gen_max_tokens):
                with torch.no_grad():
                    logits = _model_forward(ids)[:, -1, :]

                    if self.aet is not None:
                        try:
                            logits_1d = logits[0]
                            logits_1d = self.aet.apply_bias(self.tokenizer, logits_1d)
                            logits = logits_1d.unsqueeze(0)
                        except Exception as e:
                            _jlog(logging.WARNING, "[decode] Aeternum bias failed", err=str(e))

                    win = generated[-256:]
                    if win and rp_pen != 1.0:
                        idx = torch.tensor(list(set(win)), device=logits.device, dtype=torch.long)
                        vals = logits[0, idx]
                        logits[0, idx] = torch.where(vals > 0, vals / rp_pen, vals * rp_pen)

                    if len(generated) > 0:
                        last_tok = generated[-1]
                        logits[0, last_tok] -= 1.25

                    if ngram_block and ngram_block > 1:
                        nblock.apply(logits[0])

                    logits = _soft_bias_logits(logits, key_ids, +0.10)
                    logits = _soft_bias_logits(logits, self._digit_ids, -0.50)

                    if role_user_tok is not None:
                        logits[0, role_user_tok] -= 5.0
                    if role_asst_tok is not None:
                        logits[0, role_asst_tok] -= 5.0

                    if first_token and bad_start:
                        logits = _soft_bias_logits(logits, bad_start, -1.5)
                    if first_token:
                        alpha_ids = _token_ids_for_chars(
                            self.tokenizer,
                            set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
                        )
                        logits = _soft_bias_logits(logits, alpha_ids, +0.15)

                    if suppress_vague:
                        phrase_bias.apply(logits[0], generated)
                        gen_steps = len(generated) - prompt_len
                        early_q.apply(logits[0], step=gen_steps)

                    if eos_like:
                        logits = logits.clone()
                        gen_steps = len(generated) - prompt_len
                        for eos_tok in eos_like:
                            if gen_steps < min_new_tokens:
                                logits[0, eos_tok] = -float("inf")
                            elif gen_steps < eos_bias_steps:
                                logits[0, eos_tok] -= 3.0

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

                new_len = len(generated) - prompt_len
                if new_len >= max_new_tokens:
                    break

                if eos_like and (next_tok in eos_like) and new_len >= min_new_tokens:
                    if stop_on_eos or True:
                        break

                ids = torch.tensor([generated], device=self.device)

            out = self.tokenizer.decode(generated[prompt_len:], skip_special_tokens=True)
            return out.strip()

        try:
            out1 = self.backend.generate(
                orig_user_prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                decode_fn=decode_once,
            )
        except TypeError:
            out1 = decode_once(temperature, top_p)

        out1 = _strip_speaker_tags(out1).strip()
        if re.match(r"(?i)^(this|the)\s+story\s+(should|must)\b", out1) or out1.lower().startswith("chapter "):
            out1 = re.sub(r"^.*?(?=\b[a-z])", "", out1, flags=re.I).strip()
        out1 = polish(out1) if polish_output else out1

        got1 = _keywords(out1, self.stopwords)
        rel1 = _jaccard(want, got1)

        if rel1 < relevance_floor and not auto_pick:
            enc2 = self.tokenizer.encode(composed)
            ids = torch.tensor([enc2.ids], device=self.device)
            generated = ids[0].tolist()
            first_token = True
            nblock.reset()

            try:
                out2 = self.backend.generate(
                    orig_user_prompt,
                    temperature=retry_tighter[0],
                    top_p=retry_tighter[1],
                    max_new_tokens=max_new_tokens,
                    decode_fn=decode_once,
                )
            except TypeError:
                out2 = decode_once(retry_tighter[0], retry_tighter[1])

            rel2 = _jaccard(want, _keywords(out2, self.stopwords))
            short = len(orig_user_prompt) < 60
            floor = relevance_floor * (0.60 if short else 1.0)
            if (rel2 >= floor) or (rel2 >= rel1 + 0.06):
                out1 = out2

        final_out = polish(_strip_speaker_tags(out1).strip()) if polish_output else _strip_speaker_tags(out1).strip()
        final_out = self._strip_boilerplate(final_out)

        metrics_final = self._text_metrics(final_out, orig_user_prompt)
        rel_final = float(metrics_final.get("rel", 0.0))
        rep3_final = float(metrics_final.get("rep3", 1.0))
        rep3_max = float(os.environ.get("ARDOR_MEM_REP3_MAX", "0.18"))
        min_chars_log = int(os.environ.get("ARDOR_MEM_MIN_LOG_CHARS", "20"))

        allow_log = (
            log_response
            and (len(final_out.strip()) >= min_chars_log)
            and (rel_final >= float(relevance_floor))
            and (rep3_final <= rep3_max)
        )

        if allow_log:
            self.log(orig_user_prompt, final_out)
        else:
            _jlog(logging.WARNING, "[memory] skipped logging to conversation history", reason="quality_gate")

        self._maybe_summarize_with_dmn(
            prompt=orig_user_prompt,
            response=final_out,
            recent_turns=list(self.chat_turns),
            recent_texts=list(self.recent_texts),
            retrieved_memory_summary=dmn_retrieved_traces,
            aet_state=self._safe_aeternum_state(aet_decision),
        )

        self.chat_turns.append(("user", self._clean_for_history(orig_user_prompt)))
        clean_final = self._clean_for_history(final_out)
        if self._should_store_in_history(orig_user_prompt, clean_final, relevance_floor=relevance_floor):
            self.chat_turns.append(("assistant", clean_final))
        else:
            _jlog(logging.WARNING, "[history] assistant reply NOT stored (failed gate)")

        if len(self.chat_turns) > self.max_chat_turns * 2:
            self.chat_turns = self.chat_turns[-(self.max_chat_turns * 2) :]

        _jlog(logging.INFO, "[PFC] generate_text end", seconds=round(time.time() - t_start, 4))
        return final_out


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