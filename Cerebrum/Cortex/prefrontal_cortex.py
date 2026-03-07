#!/usr/bin/env python3
"""
prefrontal_cortex.py — Ardor inference core (PFC control + biasing rails)
Compatible with LoRA-merged checkpoints produced by selfsupervised_multitask_lora.py

PATCH NOTES (stability-first):
- Fix EOS retry: retry path never bans EOS.
- Retrieval fully disabled behind a flag (default OFF); skips index + hits entirely.
- N-gram blocker replaced with O(#blocked) map-based version (no O(V) scans).
- Removed the drift detector / anchor-based tightening (was destabilizing).

PY38-COMPAT: uses from __future__ annotations and avoids PEP 695 features.
"""

from __future__ import annotations

import os, sys, time, json, random, subprocess, re, glob
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import deque

# ── Optional safety/env toggles for tokenizers (helps avoid thread warnings and gives better traces) ──
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RUST_BACKTRACE", "1")

# NOTE: Encoder import kept, but retrieval is OFF by default and the encoder is not instantiated unless enabled.
from posterior_parietal_cortex import ArdorEncoder

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

# Surface polisher (Anterior Cingulate)
sys.path.append("../Cerebrum/LanguageProcessing")
from Anterior_Cingulate import polish  # noqa: E402

# Decoder model (Broca)
from broca_decoder import ArdorDecoder  # noqa: E402


# ── stopwords (fail-soft if nltk unavailable) ────────────────────────
try:
    import nltk  # type: ignore
    from nltk.corpus import stopwords as nltk_sw  # type: ignore
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(nltk_sw.words("english"))
except Exception:
    STOPWORDS = {
        "the","a","an","and","or","but","if","then","so","because","as",
        "of","in","on","for","to","from","by","with","about","into","over",
        "is","are","was","were","be","been","being","it","this","that"
    }


# ── EOS helpers ──────────────────────────────────────────────────────
EOS_CANDIDATES = ("<eos>", "</s>", "<|endoftext|>", "<|eot|>")

def _find_eos_id(tokenizer: Tokenizer) -> Optional[int]:
    for tok in EOS_CANDIDATES:
        tid = tokenizer.token_to_id(tok)
        if tid is not None:
            return tid
    return None


# ── conversation log ─────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_MODELS_DIR = ARTIFACTS_DIR / "models"
ARTIFACTS_MEMORY_DIR = ARTIFACTS_DIR / "memory"

LOG_FILE = ARTIFACTS_MEMORY_DIR / "ardor_dialogues.jsonl"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def slow_type(txt: str, delay: float = 0.005):
    for ch in txt:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                     Lightweight retrieval memory                  ║
# ╚═══════════════════════════════════════════════════════════════════╝
class ParietalMemory:
    """Text embedding retrieval (DISABLED by default).

    IMPORTANT:
      - Do NOT enable until you have trained/loaded encoder weights.
      - Random/untrained encoder embeddings = random memory injection.
    """

    def __init__(self, tokenizer, device, vocab_size, hidden=384, layers=8, heads=6, max_len=1024):
        self.device = device
        self.tok = tokenizer
        self.encoder = ArdorEncoder(
            vocab_size,
            hidden_dim=hidden,
            num_layers=layers,
            heads=heads,
            max_len=max_len,
            dropout=0.1,
        ).to(device).eval()
        self.index_texts: List[str] = []
        self.index_emb: Optional[torch.Tensor] = None

    def _encode(self, text: str) -> torch.Tensor:
        ids = self.tok.encode(text).ids[:1024] or [self.tok.token_to_id("<pad>") or 0]
        x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1,T]
        with torch.no_grad():
            _, pooled = self.encoder(x, return_pooled=True, pool="mean")
            pooled = F.normalize(pooled, dim=-1)
        return pooled[0]

    def build_index(self, texts: List[str]):
        self.index_texts = list(texts)
        embs = [self._encode(t).unsqueeze(0) for t in texts]
        self.index_emb = torch.cat(embs, dim=0) if embs else None

    def topk(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.index_emb is None or len(self.index_texts) == 0:
            return []
        q = self._encode(query).unsqueeze(0)
        sims = (q @ self.index_emb.T).squeeze(0)
        vals, idx = torch.topk(sims, k=min(k, sims.numel()))
        return [(self.index_texts[i], float(vals[j])) for j, i in enumerate(idx.tolist())]


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
    "can you provide","could you provide","please provide more information",
    "please provide more details","if you're looking","if you are looking",
    "can you please","let me know if you have any other questions",
    "feel free to ask","i'm not sure if","here are some examples",
    # story template suppressors
    "this story should be told","the story should","in this story","chapter ",
    "once upon a time","the first step is to","as the story progresses"
]


def _keywords(text: str, stopwords: set[str]) -> set[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    return {t for t in toks if t not in stopwords}


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
    return ids


def _strip_speaker_tags(text: str) -> str:
    s = text.strip()
    s = _SPEAKER_RE_LEAD.sub("", s)
    m = _SPEAKER_RE_NEXT.search(s)
    if m:
        s = s[: m.start()].rstrip()
    return s


class NgramBlocker:
    """Efficient map-based n-gram blocker (O(#blocked)).

    Stores: prefix (n-1 tokens) -> set(next_tokens) that would complete a seen n-gram.
    """

    def __init__(self, n: int = 4):
        self.n = max(1, int(n))
        self.window = deque(maxlen=max(0, self.n - 1))
        self.map: Dict[Tuple[int, ...], set[int]] = {}

    def reset(self):
        self.window.clear()
        self.map.clear()

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


class PhraseBias:
    def __init__(self, tokenizer: Tokenizer, phrases: List[str], bias: float = -2.5, max_len: int = 8):
        self.tk = tokenizer
        self.bias = float(bias)
        self.max_len = int(max_len)
        self.phr_ids = [tuple(self.tk.encode(p, add_special_tokens=False).ids) for p in phrases if p.strip()]
        self.phr_ids = [p for p in self.phr_ids if p]

    def apply(self, logits_1d: torch.Tensor, generated_ids: List[int]):
        if not generated_ids:
            return
        tail = generated_ids[-self.max_len :]
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

    def apply(self, logits_1d: torch.Tensor, step: int):
        if self.q_id is None:
            return
        if step < self.until and 0 <= self.q_id < logits_1d.shape[-1]:
            logits_1d[self.q_id] -= self.penalty


# ╔═══════════════════════════════════════════════════════════════════╗
# ║             Checkpoint compatibility / key remapping              ║
# ╚═══════════════════════════════════════════════════════════════════╝

def _unwrap_state_dict(raw) -> Union[dict, torch.nn.Module]:
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


def _remap_to_model_schema(sd: dict, model_state_keys: set[str]) -> dict:
    """Map common naming variants to this model's expected schema using `model_state_keys`."""

    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {
            (k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()
        }

    new = dict(sd)

    def _rename_prefix(d, old, newp):
        if any(k.startswith(old) for k in d):
            out = {}
            for k, v in d.items():
                out[newp + k[len(old):] if k.startswith(old) else k] = v
            return out
        return d

    # token embeddings → token_embed.
    for alias in ("token.", "tok_embeddings.", "embed_tokens.", "embedding."):
        new = _rename_prefix(new, alias, "token_embed.")

    # blocks/layers
    expects_blocks = any(k.startswith("blocks.") for k in model_state_keys)
    expects_layers = any(k.startswith("layers.") for k in model_state_keys)
    has_blocks = any(k.startswith("blocks.") for k in new)
    has_layers = any(k.startswith("layers.") for k in new)
    if expects_blocks and has_layers:
        new = _rename_prefix(new, "layers.", "blocks.")
    elif expects_layers and has_blocks:
        new = _rename_prefix(new, "blocks.", "layers.")

    # attention/attn
    expects_attn = any(".attn." in k for k in model_state_keys)
    expects_attention = any(".attention." in k for k in model_state_keys)
    has_attn = any(".attn." in k for k in new)
    has_attention = any(".attention." in k for k in new)
    if expects_attn and has_attention:
        new = {k.replace(".attention.", ".attn."): v for k, v in new.items()}
    elif expects_attention and has_attn:
        new = {k.replace(".attn.", ".attention."): v for k, v in new.items()}

    # q_proj/k_proj/v_proj/out_proj → q/k/v/out
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

    # heads/output naming
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

    # MLP aliases → ff.0 / ff.2
    tmp = {}
    for k, v in new.items():
        k2 = k.replace(".mlp.fc1.", ".ff.0.").replace(".mlp.fc2.", ".ff.2.")
        tmp[k2] = v
    new = tmp

    # position embedding param/weight
    expects_posparam = "pos" in model_state_keys
    expects_posembed = "position_embed.weight" in model_state_keys
    has_posparam = "pos" in new
    has_pos_embed = "position_embed.weight" in new

    if expects_posparam and has_pos_embed and "pos" not in new:
        w = new["position_embed.weight"]
        try:
            if getattr(w, "ndim", None) == 2:
                new["pos"] = w.unsqueeze(0)  # [T,H] → [1,T,H]
        except Exception:
            pass
        new.pop("position_embed.weight", None)

    if expects_posembed and has_posparam and "position_embed.weight" not in new:
        w = new["pos"]
        try:
            if getattr(w, "ndim", None) == 3 and w.shape[0] == 1:
                new["position_embed.weight"] = w.squeeze(0)  # [1,T,H] → [T,H]
        except Exception:
            pass
        new.pop("pos", None)

    return new


def _infer_dims_from_state(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int]:
    """Return vocab, hidden, layers, max_len using only tensor shapes/names."""

    # (1) pick the largest-row 2D non-square matrix as [vocab, hidden]
    twoD = [(k, t) for k, t in sd.items() if isinstance(t, torch.Tensor) and t.ndim == 2]
    nonsq = [(k, t) for k, t in twoD if int(t.shape[0]) != int(t.shape[1])]
    pool = nonsq or twoD
    if not pool:
        raise KeyError("No 2D tensors found in checkpoint; cannot infer vocab/hidden.")
    _, t_big = max(pool, key=lambda kv: int(kv[1].shape[0]))
    vocab = int(t_big.shape[0])
    hidden = int(t_big.shape[1])

    # (2) infer layers from attention/MLP weights with block indices
    patts = [r"(?:layers|blocks|h|layer)\.(\d+)\."]
    idxs: List[int] = []
    for k in sd.keys():
        for p in patts:
            m = re.search(p, k)
            if m:
                idxs.append(int(m.group(1)))
                break
    layers = (max(idxs) + 1) if idxs else 8

    # (3) infer max_len from positional embedding shapes
    max_len = None
    for k, t in sd.items():
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            r, c = int(t.shape[0]), int(t.shape[1])
            if c == hidden and re.search(r"(pos|position).*emb.*weight", k, re.I):
                max_len = r
                break
        if isinstance(t, torch.Tensor) and t.ndim == 3 and t.shape[0] in (1,):
            r, c = int(t.shape[1]), int(t.shape[2])
            if c == hidden:
                max_len = r
                break
    if max_len is None:
        max_len = 2048

    return vocab, hidden, layers, int(max_len)


def _best_heads(hidden: int, prefer: int = 6) -> int:
    if prefer > 1 and hidden % prefer == 0:
        return prefer
    divisors = [d for d in range(2, 65) if hidden % d == 0]
    return max(divisors) if divisors else 1


def _introspect_model(model: torch.nn.Module, sd: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    """Extract layers/heads/hidden/max_len robustly from module or fallback to state_dict."""

    info: Dict[str, Any] = {"layers": None, "heads": None, "hidden": None, "max_len": None, "vocab": None}

    # direct attributes
    for a in ("layers", "num_layers", "n_layers"):
        info["layers"] = info["layers"] or getattr(model, a, None)
    for a in ("heads", "num_heads", "n_heads"):
        info["heads"] = info["heads"] or getattr(model, a, None)
    for a in ("hidden", "hidden_dim", "embed_dim", "d_model"):
        info["hidden"] = info["hidden"] or getattr(model, a, None)
    for a in ("max_len", "max_seq_len", "context_len", "ctx_len"):
        info["max_len"] = info["max_len"] or getattr(model, a, None)

    # vocab via embedding/LM head
    try:
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            info["vocab"] = int(model.lm_head.weight.shape[0])
        elif hasattr(model, "token_embed") and hasattr(model.token_embed, "weight"):
            info["vocab"] = int(model.token_embed.weight.shape[0])
    except Exception:
        pass

    # fallback via state_dict
    if sd is None:
        try:
            sd = model.state_dict()
        except Exception:
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
        except Exception:
            pass

    return info


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
        enable_retrieval: bool = False,
    ):
        # max_len here is *generation* budget; model context is reported separately.
        self.device = device
        self.gen_max_tokens = int(max_len)
        self.enable_retrieval = bool(enable_retrieval)

        raw = torch.load(model_path, map_location=device)
        sd_or_mod = _unwrap_state_dict(raw)

        missing_list: Optional[List[str]] = None
        unexpected_list: Optional[List[str]] = None

        if isinstance(sd_or_mod, torch.nn.Module):
            self.model = sd_or_mod.to(device).eval()
            try:
                want_vocab = int(self.model.lm_head.weight.shape[0])
            except Exception:
                want_vocab = int(self.model.token_embed.weight.shape[0])
            model_sd = None
        else:
            sd = sd_or_mod  # {name: tensor}
            vocab, hidden, layers, maxlen = _infer_dims_from_state(sd)
            heads = _best_heads(hidden, prefer=6)

            model = ArdorDecoder(
                vocab_size=vocab,
                hidden=hidden,
                layers=layers,
                heads=heads,
                max_len=maxlen,
                dropout=0.12,
            )

            remapped = _remap_to_model_schema(sd, set(model.state_dict().keys()))
            try:
                model.load_state_dict(remapped, strict=True)
            except Exception:
                missing, unexpected = model.load_state_dict(remapped, strict=False)
                missing_list = list(missing) if isinstance(missing, list) else list(missing or [])
                unexpected_list = list(unexpected) if isinstance(unexpected, list) else list(unexpected or [])
                print(f"[load] non-strict: missing={len(missing_list or [])} unexpected={len(unexpected_list or [])}")

            self.model = model.to(device).eval()
            want_vocab = int(vocab)
            model_sd = remapped

        # Tie head ⇄ embedding if shapes agree (saves params / consistency)
        try:
            if hasattr(self.model, "lm_head") and hasattr(self.model, "token_embed"):
                if self.model.lm_head.weight.shape == self.model.token_embed.weight.shape:
                    self.model.lm_head.weight = self.model.token_embed.weight
        except Exception:
            pass

        # ---------- tokenizer that matches vocab ----------
        requested_tok = tokenizer_path if (tokenizer_path and os.path.isfile(tokenizer_path)) else None

        def _find_tok(start_dir: Optional[str], want: int) -> Optional[str]:
            roots: List[str] = []
            if start_dir and os.path.isdir(start_dir):
                roots.append(start_dir)
            roots += [
                str(REPO_ROOT / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer"),
                "../Cerebrum/ProjectTokenizer/ardor_tokenizer",
                "../ProjectTokenizer/ardor_tokenizer",
                "./Cerebrum/ProjectTokenizer/ardor_tokenizer",
                "./ProjectTokenizer/ardor_tokenizer",
            ]
            seen, cands = set(), []
            for r in roots:
                rr = os.path.abspath(r)
                if not os.path.isdir(rr):
                    continue
                for p in glob.glob(os.path.join(rr, "tokenizer_v*.json")):
                    if p in seen:
                        continue
                    seen.add(p)
                    try:
                        t = Tokenizer.from_file(p)
                        if t.get_vocab_size() == want:
                            cands.append(p)
                    except Exception:
                        pass

            # Prefer explicit path first if it matches; else best candidate
            if requested_tok:
                try:
                    t0 = Tokenizer.from_file(requested_tok)
                    if t0.get_vocab_size() == want:
                        return requested_tok
                    else:
                        print(
                            f"[tok] override provided but vocab mismatch: {requested_tok} "
                            f"(vocab={t0.get_vocab_size()}) != model({want}). Searching for a match…"
                        )
                except Exception as _e:
                    print(f"[tok] failed to open override {requested_tok}: {_e}. Searching for a match…")

            return cands[0] if cands else None

        tok_path = _find_tok(os.path.dirname(requested_tok) if requested_tok else None, want_vocab)
        if not tok_path:
            raise FileNotFoundError(
                f"No tokenizer with vocab size {want_vocab} found. "
                "Place tokenizer_v*.json under ProjectTokenizer/ardor_tokenizer."
            )

        self.tokenizer_requested_path = requested_tok
        self.tokenizer = Tokenizer.from_file(tok_path)
        try:
            name = type(self.tokenizer.model).__name__.lower()
            # Attach a ByteLevel decoder only for actual BPE models and if no decoder is already set.
            if name == "bpe" and getattr(self.tokenizer, "decoder", None) is None:
                self.tokenizer.decoder = ByteLevel()
        except Exception:
            pass
        self.tokenizer_path = tok_path

        if requested_tok and os.path.abspath(requested_tok) != os.path.abspath(tok_path):
            try:
                rv = Tokenizer.from_file(requested_tok).get_vocab_size()
            except Exception:
                rv = "?"
            print(
                f"[tok] requested={requested_tok} (vocab={rv}) → chosen={tok_path} "
                f"(vocab={self.tokenizer.get_vocab_size()})"
            )

        # ---------- model schema export (fills the 'None' you saw) ----------
        schema = _introspect_model(self.model, sd=model_sd)
        self.layers = int(schema.get("layers")) if schema.get("layers") is not None else None
        self.heads = int(schema.get("heads")) if schema.get("heads") is not None else None
        self.hidden = int(schema.get("hidden")) if schema.get("hidden") is not None else None
        self.model_ctx_len = int(schema.get("max_len")) if schema.get("max_len") is not None else None
        self.vocab_size = int(schema.get("vocab")) if schema.get("vocab") is not None else want_vocab
        self.schema = {
            "layers": self.layers,
            "heads": self.heads,
            "hidden": self.hidden,
            "max_len": self.model_ctx_len,
            "vocab": self.vocab_size,
            "mismatch": {
                "missing": missing_list,
                "unexpected": unexpected_list,
            },
        }
        miss_ct = None if missing_list is None else len(missing_list)
        unex_ct = None if unexpected_list is None else len(unexpected_list)
        print(
            f"🧠 Model schema: layers={self.layers} heads={self.heads} hidden={self.hidden} "
            f"max_len={self.model_ctx_len} mismatch: missing={miss_ct} unexpected={unex_ct}"
        )
        try:
            emb_rows = getattr(getattr(self.model, "token_embed", None), "weight", None)
            emb_rows = int(emb_rows.shape[0]) if emb_rows is not None else self.vocab_size
        except Exception:
            emb_rows = self.vocab_size
        print(f"🧩 Tokenizer: {self.tokenizer_path}  | vocab={self.vocab_size}  embed={emb_rows}")

        # ---------- Parietal memory (RETRIEVAL OFF by default) ----------
        self.parietal: Optional[ParietalMemory] = None
        if self.enable_retrieval:
            try:
                vocab_size_tok = self.tokenizer.get_vocab_size()
            except Exception:
                vocab_size_tok = len(self.tokenizer.get_vocab())
            self.parietal = ParietalMemory(self.tokenizer, self.device, vocab_size_tok)
            enc = getattr(self.parietal, "encoder", None)
            if hasattr(enc, "tie_from_broca"):
                try:
                    enc.tie_from_broca(self.model)
                except Exception:
                    pass
            print("🧠 Retrieval: ENABLED (make sure encoder weights are trained/loaded!)")
        else:
            print("🧠 Retrieval: DISABLED (safe default)")

        # Rolling conversation memory + cached bias ids
        self.recent_texts = deque(maxlen=64)
        self.stopwords = STOPWORDS
        self._digit_ids = _token_ids_for_chars(self.tokenizer, set("0123456789"))
        self._eos_id = _find_eos_id(self.tokenizer)
        self._eot_id = self.tokenizer.token_to_id("<|eot|>")

    # ───────────── prompt classification (heuristics) ────────────────
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

    # ─────────────────────── probe metrics/scoring ────────────────────
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
        rel = _jaccard(_keywords(prompt, self.stopwords), _keywords(text, self.stopwords))
        return {"d1": d1, "d2": d2, "rep3": rep3, "closure": closure, "imbalance": imbalance, "rel": rel}

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
            score = 0.40 * m["rel"] + 0.20 * m["d1"] + 0.15 * m["d2"] + 0.15 * m["closure"] - 0.15 * m["rep3"] - 0.10 * m["imbalance"]
            results.append({**cfg, "score": score, "metrics": m, "sample": txt})
        results.sort(key=lambda r: (-r["score"], r["temperature"] if profile in ("code", "math") else 0.0))
        return results[0]

    def conversation_memory_last_N_chunks(self, N: int = 32) -> List[str]:
        seq = list(self.recent_texts)[-N:]
        return seq[::2]

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
        max_new_tokens: int = 120,
        top_k: int = 40,
        typical_p: float = 0.95,
        min_temp: float = 0.35,
        *,
        auto_pick: bool = False,
        stop_on_eos: bool = False,
        profile: Optional[str] = None,
        log_response: bool = True,
        polish_output: bool = True,
        enable_retrieval: Optional[bool] = None,
    ) -> str:

        # === Retrieval (DISABLED by default) ===
        # IMPORTANT: skip building hits_raw entirely when retrieval is off.
        orig_user_prompt = prompt
        use_retrieval = self.enable_retrieval if enable_retrieval is None else bool(enable_retrieval)

        # (Memory reinjection stays DISABLED even if retrieval is enabled — reintroduce only after stability.)
        # if use_retrieval and self.parietal is not None:
        #     self.parietal.build_index(self.conversation_memory_last_N_chunks(48))
        #     hits_raw = self.parietal.topk(orig_user_prompt, k=5)
        #
        #     # SAFE REINJECTION (future): inject in a separate channel with hard delimiters, only if relevant.
        #     # filtered = ... (jaccard / relevance checks)
        #     # if filtered:
        #     #     prompt = (
        #     #         orig_user_prompt
        #     #         + "\n\n[MEMORY_CONTEXT_BEGIN]\n"
        #     #         + "\n".join(f"- {t}" for t in filtered[:3])
        #     #         + "\n[MEMORY_CONTEXT_END]\n"
        #     #     )

        # === Persona / conversation scaffolding ===
        spec = {
            "user": self.tokenizer.token_to_id("<|user|>"),
            "asst": self.tokenizer.token_to_id("<|assistant|>"),
            "eot": self.tokenizer.token_to_id("<|eot|>"),
        }
        if spec["user"] is not None and spec["asst"] is not None:
            composed = f"{persona_primer}<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            composed = f"{prompt}\n"

        enc = self.tokenizer.encode(composed)
        ids = torch.tensor([enc.ids], device=self.device)

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

        generated = ids[0].tolist()
        prompt_len = len(generated)
        first_token = True
        eos_bias_steps = 32

        # Special-role token suppression during generation (prevents role-echo)
        role_user_tok = spec["user"]
        role_asst_tok = spec["asst"]

        def _soft_bias_logits(logits: torch.Tensor, ids_set: set[int], delta: float) -> torch.Tensor:
            if ids_set:
                logits = logits.clone()
                logits[0, list(ids_set)] += delta
            return logits

        def _model_forward(input_ids: torch.Tensor) -> torch.Tensor:
            """Be tolerant to different model forward signatures; return logits [B,T,V]."""
            out = self.model(input_ids)
            if isinstance(out, (list, tuple)) and len(out) > 0:
                out = out[0]
            if isinstance(out, dict) and "logits" in out:
                out = out["logits"]
            return out

        def decode_once(temp: float, p: float) -> str:
            nonlocal ids, generated, first_token

            for step in range(self.gen_max_tokens):
                with torch.no_grad():
                    logits = _model_forward(ids)[:, -1, :]  # (1, V)

                    # a) repetition penalty (GPT-NeoX style, sign-aware) over last 256 tokens
                    win = generated[-256:]
                    if win and rep_penalty and rep_penalty != 1.0:
                        rp = max(1.0, float(rep_penalty))
                        idx = torch.tensor(list(set(win)), device=logits.device, dtype=torch.long)
                        vals = logits[0, idx]
                        logits[0, idx] = torch.where(vals > 0, vals / rp, vals * rp)

                    # a2) discourage immediate token repeat a bit (not a hard ban)
                    if len(generated) > 0:
                        last_tok = generated[-1]
                        logits[0, last_tok] -= 1.25

                    # b) no-repeat-ngram (efficient)
                    if ngram_block and ngram_block > 1:
                        nblock.apply(logits[0])

                    # c) gentle topical bias (kept minimal) + digit suppression + role suppression
                    # NOTE: drift detector removed; no tightening loops.
                    logits = _soft_bias_logits(logits, key_ids, +0.10)
                    logits = _soft_bias_logits(logits, self._digit_ids, -0.50)
                    if role_user_tok is not None:
                        logits[0, role_user_tok] -= 5.0
                    if role_asst_tok is not None:
                        logits[0, role_asst_tok] -= 5.0

                    # d) avoid opening with stray quotes/dashes
                    if first_token and BAD_START:
                        logits = _soft_bias_logits(logits, BAD_START, -1.5)
                    if first_token:
                        alpha_ids = _token_ids_for_chars(self.tokenizer, set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))
                        logits = _soft_bias_logits(logits, alpha_ids, +0.15)

                    # e) style suppressors
                    if suppress_vague:
                        phrase_bias.apply(logits[0], generated)
                        gen_steps = len(generated) - prompt_len
                        early_q.apply(logits[0], step=gen_steps)

                    # f) early EOS/EOT control (EOS is NEVER banned permanently)
                    if eos_like:
                        logits = logits.clone()
                        gen_steps = len(generated) - prompt_len
                        for eos_tok in eos_like:
                            if gen_steps < min_new_tokens:
                                logits[0, eos_tok] = -float("inf")
                            elif gen_steps < eos_bias_steps:
                                logits[0, eos_tok] -= 3.0

                    # g) top-k + nucleus with annealed temperature
                    ArdorCore._apply_top_k_(logits, top_k)
                    frac = step / max(1, self.gen_max_tokens - 1)
                    dyn_temp = max(min_temp, temp * (0.7 + 0.3 * (1 - frac)))
                    probs = torch.softmax(logits / max(dyn_temp, 1e-5), dim=-1)

                    if typical_p and 0.0 < typical_p < 1.0:
                        eps = 1e-8
                        logp = (probs + eps).log()  # [1,V]
                        H = -(probs * logp).sum(dim=-1, keepdim=True)  # entropy per step
                        typicality = (-(logp) - H).abs().squeeze(0)  # [V]
                        order = typicality.argsort()  # low→high typicality
                        csum = probs[0, order].cumsum(dim=0)
                        keep = order[csum <= float(typical_p)]
                        if keep.numel() > 0:
                            mask = torch.zeros_like(probs)
                            mask[0, keep] = probs[0, keep]
                            denom = mask.sum()
                            probs = mask / (denom + eps)

                    next_tok = ArdorCore._nucleus_pick(probs, p)

                generated.append(next_tok)
                first_token = False
                if ngram_block and ngram_block > 1:
                    nblock.update(next_tok)

                new_len = len(generated) - prompt_len
                if new_len >= max_new_tokens:
                    break

                # Stop on EOS/EOT if requested OR if stop_on_eos is False but EOS was sampled after min tokens.
                if eos_like and (next_tok in eos_like) and new_len >= min_new_tokens:
                    if stop_on_eos or True:
                        break

                ids = torch.tensor([generated], device=self.device)

            out = self.tokenizer.decode(generated[prompt_len:], skip_special_tokens=True)
            return out.strip()

        # === First pass generation ===
        out1 = decode_once(temperature, top_p)

        # One-time boilerplate guard
        out1 = _strip_speaker_tags(out1).strip()
        if re.match(r"(?i)^(this|the)\s+story\s+(should|must)\b", out1) or out1.lower().startswith("chapter "):
            out1 = re.sub(r"^.*?(?=\b[a-z])", "", out1, flags=re.I).strip()
        out1 = polish(out1) if polish_output else out1

        # === Re-try with tighter params if low topicality ===
        got1 = _keywords(out1, self.stopwords)
        rel1 = _jaccard(want, got1)
        if rel1 < relevance_floor and not auto_pick:
            enc2 = self.tokenizer.encode(composed)
            ids = torch.tensor([enc2.ids], device=self.device)
            generated = ids[0].tolist()
            first_token = True
            nblock.reset()

            # ✅ EOS retry fix: retry NEVER bans EOS (decode_once has no ban_eos_entirely path).
            out2 = decode_once(retry_tighter[0], retry_tighter[1])

            if _jaccard(want, _keywords(out2, self.stopwords)) >= rel1:
                out1 = out2

        final_out = polish(_strip_speaker_tags(out1).strip()) if polish_output else _strip_speaker_tags(out1).strip()
        if log_response:
            self.log(orig_user_prompt, final_out)
        return final_out

    # Convenience for GUI
    def model_schema(self) -> Dict[str, Any]:
        return dict(self.schema)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                               CLI                                 ║
# ╚═══════════════════════════════════════════════════════════════════╝
try:
    import typer  # type: ignore

    app = typer.Typer()
except Exception:
    app = None

if app:

    @app.command()
    def cli():
        slow_type("\n🧠 Welcome to Ardor CLI — Synthesizer of Minds\n")
        roots = [
            str(ARTIFACTS_MODELS_DIR),
            "../Models",
            "../Cerebrum/Models",
            "../Cerebrum/Models/Ardor",
        ]
        models = []
        for r in roots:
            if os.path.isdir(r):
                models += [os.path.join(r, f) for f in os.listdir(r) if f.endswith(".pt")]
        if not models:
            print("No .pt models found.")
            return
        for i, path in enumerate(models, 1):
            print(f"  {i}. {os.path.basename(path)}")

        import typer as _ty

        idx = int(_ty.prompt(f"\n🔎 Choose a model [1–{len(models)}]")) - 1
        model_path = models[idx]

        # Look nearby for a matching tokenizer; CLI keeps it simple
        tok_candidates = [
            str(REPO_ROOT / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v9.json"),
            str(REPO_ROOT / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v8.json"),
            str(REPO_ROOT / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v7.json"),
            "../ProjectTokenizer/ardor_tokenizer/tokenizer_v9.json",
            "../ProjectTokenizer/ardor_tokenizer/tokenizer_v8.json",
            "../ProjectTokenizer/ardor_tokenizer/tokenizer_v7.json",
            "../Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v9.json",
            "../Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v8.json",
            "../Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v7.json",
        ]
        tokenizer_path = next((p for p in tok_candidates if os.path.isfile(p)), None)
        if tokenizer_path is None:
            print("Tokenizer not found. Please place tokenizer_v*.json in ProjectTokenizer/ardor_tokenizer.")
            return

        # Retrieval is OFF by default.
        core = ArdorCore(model_path=model_path, tokenizer_path=tokenizer_path, device="cpu", enable_retrieval=False)

        print("\n💡 Type 'train', 'rem', 'exit', or just chat. (/set is handled in GUI)\n")
        while True:
            prompt = input("🗨️  > ").strip().rstrip()
            if not prompt:
                continue
            cmd = prompt.lower()
            if cmd in ("exit", "quit"):
                break
            if cmd == "train":
                subprocess.call(["python", "../Cerebrum/Cortex/neural_plasticity_training.py"])
                continue
            if cmd in ("rem", "sleep"):
                subprocess.call(["python", str((REPO_ROOT / "Cerebrum" / "CorticalIntegration" / "REM.py").resolve())])
                continue

            slow_type("\n🧠 Ardor:", 0.03)
            ans = core.generate_text(prompt, persona_primer="")

            slow_type(ans, 0.01)
            print("\n" + "-" * 60 + "\n")


def _selftest():
    print("[selftest] basic suppression & biasing")
    a = _keywords("Set temperature to 0.8 and repetition penalty", STOPWORDS)
    assert "temperature" in a and "repetition" in a
    b = _keywords("Please raise temp to 0.8", STOPWORDS)
    j = _jaccard(a, b)
    assert 0.0 <= j <= 1.0
    logits = torch.zeros(1, 10)
    ids = {3, 7}
    logits[0, list(ids)] += 1.0

    nb = NgramBlocker(3)
    nb.reset()
    nb.update(1)
    nb.update(2)
    l1 = torch.zeros(10)
    nb.apply(l1)
    print("[selftest] OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif app:
        app()
    else:
        print("Typer not installed; CLI disabled.")
