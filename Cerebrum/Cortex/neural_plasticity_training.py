#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Card
Goal: Train Ardor toward coherent, bounded, non-looping prompt-conditioned generation.
Command:
  python /workspace/Ardor/Cerebrum/Cortex/ardor_promptgen_trainer.py --stage lm_base
  python /workspace/Ardor/Cerebrum/Cortex/ardor_promptgen_trainer.py --stage stabilize --resume /workspace/Ardor/runs/ardor_promptgen/checkpoints/ckpt_full_latest.pt
  python /workspace/Ardor/Cerebrum/Cortex/ardor_promptgen_trainer.py --stage sft --sft_jsonl /workspace/Ardor/data/ardor_sft.jsonl --resume /workspace/Ardor/runs/ardor_promptgen/checkpoints/ckpt_full_latest.pt
Inputs:
  - tokens.bin/meta.json for base LM + stabilization
  - optional held-out val tokens.bin/meta.json
  - optional SFT jsonl with {prompt,response[,system]}
Artifacts:
  - run_dir/checkpoints/*.pt
  - run_dir/metrics.jsonl
  - run_dir/run_state.json
Notes:
  - Stage lm_base: standard CE + label smoothing on stream data
  - Stage stabilize: CE + repeated n-gram unlikelihood on stream data
  - Stage sft: masked assistant-only supervised finetuning on prompt/response pairs
  - # TODO(OBSERVE): compare rep3/rep4, distinct-n, eos-rate, and raw generations at each stage
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# -------------------------
# Environment
# -------------------------
os.environ.setdefault("PYTHONPATH", "/workspace/Ardor")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.9")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/workspace/.inductor_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/.triton_cache")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")

# -------------------------
# Paths / defaults
# -------------------------
ARDOR_ROOT = Path("/workspace/Ardor")
DEFAULT_TRAIN_BIN = Path("/workspace/Ardor/bin_dataset_20B/tokens.bin")
DEFAULT_TRAIN_META = Path("/workspace/Ardor/bin_dataset_20B/meta.json")
DEFAULT_VAL_BIN = Path("/workspace/Ardor/bin_dataset_heldout_25M/tokens.bin")
DEFAULT_VAL_META = Path("/workspace/Ardor/bin_dataset_heldout_25M/meta.json")
DEFAULT_TOKENIZER = Path("/workspace/Ardor/tokenizer_v9.json")
DEFAULT_RUN_DIR = Path("/workspace/Ardor/runs/ardor_promptgen")

# -------------------------
# Utils
# -------------------------
def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_repo_on_path() -> None:
    if str(ARDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(ARDOR_ROOT))


def import_project_config() -> Any:
    ensure_repo_on_path()
    from Cerebrum.Cortex.ardor_config import ArdorConfig
    return ArdorConfig


def import_project_decoder() -> Any:
    ensure_repo_on_path()
    candidates = [
        ("Cerebrum.Cortex.broca_decoder", "ArdorDecoder"),
        ("Cerebrum.Cortex.broca_decoder", "BrocaDecoder"),
        ("broca_decoder", "ArdorDecoder"),
    ]
    for mod_name, sym in candidates:
        try:
            mod = __import__(mod_name, fromlist=[sym])
            return getattr(mod, sym)
        except Exception:
            continue
    raise SystemExit("Could not import ArdorDecoder from project broca_decoder.py")


def build_model(dec_cls: Any, proj_cfg: Any) -> nn.Module:
    sig = inspect.signature(dec_cls.__init__)
    params = set(sig.parameters.keys())
    if "cfg" in params:
        return dec_cls(cfg=proj_cfg)
    if "config" in params:
        return dec_cls(config=proj_cfg)
    try:
        return dec_cls(proj_cfg)
    except TypeError:
        pass
    kwargs = {}
    for k in (
        "vocab_size", "hidden_size", "n_layers", "n_heads", "ff_mult", "max_len",
        "dropout", "attn_dropout", "resid_dropout", "use_rope", "rope_theta",
        "hidden", "layers", "heads",
    ):
        if k in params:
            if hasattr(proj_cfg, k):
                kwargs[k] = getattr(proj_cfg, k)
            elif k == "hidden" and hasattr(proj_cfg, "hidden_size"):
                kwargs[k] = proj_cfg.hidden_size
            elif k == "layers" and hasattr(proj_cfg, "n_layers"):
                kwargs[k] = proj_cfg.n_layers
            elif k == "heads" and hasattr(proj_cfg, "n_heads"):
                kwargs[k] = proj_cfg.n_heads
    if kwargs:
        return dec_cls(**kwargs)
    raise SystemExit(f"Unsupported decoder constructor signature: {sig}")

# -------------------------
# Tokenizer helpers
# -------------------------
def tokenizer_vocab_size(tokenizer_json: Path) -> int:
    tok = load_json(tokenizer_json)
    return int(len(tok["model"]["vocab"]))


def tokenizer_special_ids(tokenizer_json: Path) -> Dict[str, Optional[int]]:
    tok = load_json(tokenizer_json)
    vocab = tok["model"]["vocab"]
    return {
        "pad_id": vocab.get("<pad>"),
        "unk_id": vocab.get("<unk>"),
        "bos_id": vocab.get("<bos>"),
        "eos_id": vocab.get("<eos>"),
        "user_id": vocab.get("<|user|>"),
        "assistant_id": vocab.get("<|assistant|>"),
        "system_id": vocab.get("<|system|>"),
        "eot_id": vocab.get("<|eot|>"),
    }


def build_simple_tokenizer(tokenizer_json: Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(tokenizer_json))

# -------------------------
# Data
# -------------------------
class TokenStream:
    def __init__(self, tokens_bin: Path):
        self.tokens_mm = np.memmap(tokens_bin, dtype=np.uint16, mode="r")
        self.N = int(self.tokens_mm.shape[0])

    def slice(self, start: int, length: int) -> np.ndarray:
        start = int(start) % self.N
        end = start + int(length)
        if end <= self.N:
            return np.asarray(self.tokens_mm[start:end], dtype=np.uint16)
        n1 = self.N - start
        part1 = np.asarray(self.tokens_mm[start:self.N], dtype=np.uint16)
        part2 = np.asarray(self.tokens_mm[0:end - self.N], dtype=np.uint16)
        return np.concatenate([part1, part2], axis=0)


class StreamBatcher:
    def __init__(self, stream: TokenStream, batch_size: int, seq_len: int, seed: int, random_starts: bool = True):
        self.stream = stream
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.random_starts = bool(random_starts)
        self.rng = np.random.default_rng(seed)
        self.cursor = int(self.rng.integers(0, max(1, stream.N - 1)))
        self.chunk = self.batch_size * (self.seq_len + 1)

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.random_starts:
            start = int(self.rng.integers(0, max(1, self.stream.N - self.chunk - 1)))
        else:
            start = self.cursor
            self.cursor = (self.cursor + self.chunk) % self.stream.N
        flat = self.stream.slice(start, self.chunk).astype(np.int64, copy=False)
        arr = flat.reshape(self.batch_size, self.seq_len + 1)
        x = torch.from_numpy(arr[:, :-1].copy())
        y = torch.from_numpy(arr[:, 1:].copy())
        return x, y, start


class SFTExampleDataset(Dataset):
    def __init__(self, path: Path, tokenizer_json: Path, seq_len: int):
        self.path = path
        self.seq_len = int(seq_len)
        self.tok = build_simple_tokenizer(tokenizer_json)
        self.sp = tokenizer_special_ids(tokenizer_json)
        self.examples = self._load_examples()

    def _enc_plain(self, text: str) -> List[int]:
        return list(self.tok.encode(text).ids)

    def _enc_no_special(self, text: str) -> List[int]:
        ids = self._enc_plain(text)
        bos = self.sp["bos_id"]
        eos = self.sp["eos_id"]
        if bos is not None and ids and ids[0] == bos:
            ids = ids[1:]
        if eos is not None and ids and ids[-1] == eos:
            ids = ids[:-1]
        return ids

    def _pack_one(self, obj: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        user_id = self.sp["user_id"]
        assistant_id = self.sp["assistant_id"]
        system_id = self.sp["system_id"]
        eot_id = self.sp["eot_id"]
        bos_id = self.sp["bos_id"]
        eos_id = self.sp["eos_id"]
        if user_id is None or assistant_id is None:
            raise ValueError("Tokenizer missing <|user|> or <|assistant|> special ids needed for SFT.")

        prompt = str(obj.get("prompt", ""))
        response = str(obj.get("response", ""))
        system = str(obj.get("system", "")).strip()

        ids: List[int] = []
        labels: List[int] = []
        if bos_id is not None:
            ids.append(bos_id)
            labels.append(-100)
        if system:
            ids.append(system_id if system_id is not None else user_id)
            labels.append(-100)
            sys_ids = self._enc_no_special(system)
            ids.extend(sys_ids)
            labels.extend([-100] * len(sys_ids))
            if eot_id is not None:
                ids.append(eot_id)
                labels.append(-100)
        ids.append(user_id)
        labels.append(-100)
        p_ids = self._enc_no_special(prompt)
        ids.extend(p_ids)
        labels.extend([-100] * len(p_ids))
        if eot_id is not None:
            ids.append(eot_id)
            labels.append(-100)
        ids.append(assistant_id)
        labels.append(-100)
        r_ids = self._enc_no_special(response)
        ids.extend(r_ids)
        labels.extend(r_ids)
        if eot_id is not None:
            ids.append(eot_id)
            labels.append(eot_id)
        elif eos_id is not None:
            ids.append(eos_id)
            labels.append(eos_id)

        ids = ids[:self.seq_len + 1]
        labels = labels[:self.seq_len + 1]
        if len(ids) < 2:
            raise ValueError("Degenerate SFT example after packing.")
        return ids, labels

    def _load_examples(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids, labels = self._pack_one(obj)
                x = np.full((self.seq_len,), fill_value=self.sp["pad_id"] or 0, dtype=np.int64)
                y = np.full((self.seq_len,), fill_value=-100, dtype=np.int64)
                xin = ids[:-1]
                yin = labels[1:]
                L = min(self.seq_len, len(xin))
                x[:L] = np.asarray(xin[:L], dtype=np.int64)
                y[:L] = np.asarray(yin[:L], dtype=np.int64)
                out.append((x, y))
        if not out:
            raise SystemExit(f"No usable SFT examples found in {self.path}")
        return out

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.examples[idx]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

# -------------------------
# Losses / eval
# -------------------------
def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


def ce_loss(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        label_smoothing=label_smoothing,
    )


def repeated_ngram_unlikelihood_loss(logits: torch.Tensor, targets: torch.Tensor, ngram: int = 4, tail: int = 256) -> torch.Tensor:
    """
    Penalize assigning high probability to target tokens that would complete a repeated n-gram.
    This is a lightweight training-time anti-loop term, not a full sequence-level objective.
    """
    B, T, V = logits.shape
    if T < ngram or ngram < 2:
        return logits.new_zeros(())
    start_t = max(ngram - 1, T - tail)
    log_probs = F.log_softmax(logits, dim=-1)
    losses: List[torch.Tensor] = []
    for b in range(B):
        y = targets[b]
        for t in range(start_t, T):
            prefix = y[max(0, t - ngram + 1):t]
            if prefix.numel() != ngram - 1:
                continue
            cand = y[t]
            if cand.item() < 0:
                continue
            repeated = False
            for j in range(0, t - ngram + 1):
                prev = y[j:j + ngram - 1]
                if prev.numel() != ngram - 1:
                    continue
                if torch.equal(prev, prefix) and y[j + ngram - 1].item() == cand.item():
                    repeated = True
                    break
            if repeated:
                p = log_probs[b, t, cand].exp().clamp(max=1 - 1e-6)
                losses.append(-torch.log1p(-p))
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


@torch.no_grad()
def sample_generate(model: nn.Module, prompt_ids: Sequence[int], max_new_tokens: int, eos_id: Optional[int], top_p: float = 0.9, temperature: float = 0.9) -> List[int]:
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(ids[:, -2048:])[:, -1, :]
        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            mask = cdf > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_local = torch.multinomial(sorted_probs, num_samples=1)
            next_id = sorted_idx.gather(-1, next_local)
        ids = torch.cat([ids, next_id], dim=1)
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break
    return ids[0].tolist()


def repetition_metrics(ids: Sequence[int]) -> Dict[str, float]:
    arr = list(ids)
    out: Dict[str, float] = {}
    for n in (2, 3, 4):
        if len(arr) < n:
            out[f"rep{n}"] = 0.0
            out[f"distinct{n}"] = 1.0
            continue
        ngrams = [tuple(arr[i:i+n]) for i in range(len(arr) - n + 1)]
        uniq = len(set(ngrams))
        total = len(ngrams)
        out[f"distinct{n}"] = float(uniq / max(1, total))
        out[f"rep{n}"] = float(1.0 - (uniq / max(1, total)))
    return out


@torch.no_grad()
def eval_stream_loss(model: nn.Module, batcher: StreamBatcher, steps: int, device: torch.device, label_smoothing: float) -> float:
    model.eval()
    losses: List[float] = []
    for _ in range(steps):
        x, y, _ = batcher.next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = ce_loss(logits, y, label_smoothing=label_smoothing)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_sft_loss(model: nn.Module, loader: DataLoader, device: torch.device, label_smoothing: float, max_batches: int = 32) -> float:
    model.eval()
    losses: List[float] = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = masked_ce_loss(logits, y, label_smoothing=label_smoothing)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")

# -------------------------
# Config
# -------------------------
@dataclass
class StageConfig:
    name: str
    max_train_tokens: int
    lr: float
    weight_decay: float
    dropout: float
    label_smoothing: float
    ul_weight: float
    ul_ngram: int
    ul_tail: int
    batch_size: int
    target_tokens_per_opt_step: int
    max_grad_accum: int
    grad_clip: float
    warmup_fraction: float
    log_every: int
    eval_every: int
    gen_eval_every: int
    save_every: int
    random_starts: bool


STAGE_PRESETS: Dict[str, StageConfig] = {
    "lm_base": StageConfig(
        name="lm_base",
        max_train_tokens=3_000_000_000,
        lr=3.0e-5,
        weight_decay=0.03,
        dropout=0.10,
        label_smoothing=0.02,
        ul_weight=0.00,
        ul_ngram=4,
        ul_tail=256,
        batch_size=8,
        target_tokens_per_opt_step=65_536,
        max_grad_accum=8,
        grad_clip=0.5,
        warmup_fraction=0.03,
        log_every=50,
        eval_every=250,
        gen_eval_every=500,
        save_every=250,
        random_starts=True,
    ),
    "stabilize": StageConfig(
        name="stabilize",
        max_train_tokens=3_600_000_000,
        lr=2.0e-5,
        weight_decay=0.02,
        dropout=0.10,
        label_smoothing=0.03,
        ul_weight=0.15,
        ul_ngram=4,
        ul_tail=256,
        batch_size=8,
        target_tokens_per_opt_step=65_536,
        max_grad_accum=8,
        grad_clip=0.5,
        warmup_fraction=0.02,
        log_every=50,
        eval_every=250,
        gen_eval_every=250,
        save_every=250,
        random_starts=True,
    ),
    "sft": StageConfig(
        name="sft",
        max_train_tokens=250_000_000,
        lr=1.0e-5,
        weight_decay=0.01,
        dropout=0.10,
        label_smoothing=0.02,
        ul_weight=0.05,
        ul_ngram=4,
        ul_tail=128,
        batch_size=8,
        target_tokens_per_opt_step=32_768,
        max_grad_accum=8,
        grad_clip=0.5,
        warmup_fraction=0.05,
        log_every=20,
        eval_every=100,
        gen_eval_every=100,
        save_every=100,
        random_starts=False,
    ),
}


class WarmupCosine:
    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
        self.base_lr = float(base_lr)
        self.warmup_steps = int(max(1, warmup_steps))
        self.total_steps = int(max(self.warmup_steps + 1, total_steps))

    def lr_at(self, step: int) -> float:
        s = int(step)
        if s < self.warmup_steps:
            return self.base_lr * (s + 1) / self.warmup_steps
        t = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        t = min(max(t, 0.0), 1.0)
        return 0.5 * self.base_lr * (1.0 + math.cos(math.pi * t))

# -------------------------
# Args
# -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["lm_base", "stabilize", "sft"], required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--train_tokens", type=Path, default=DEFAULT_TRAIN_BIN)
    ap.add_argument("--train_meta", type=Path, default=DEFAULT_TRAIN_META)
    ap.add_argument("--val_tokens", type=Path, default=DEFAULT_VAL_BIN)
    ap.add_argument("--val_meta", type=Path, default=DEFAULT_VAL_META)
    ap.add_argument("--sft_jsonl", type=Path, default=None)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--hidden_size", type=int, default=1536)
    ap.add_argument("--n_layers", type=int, default=33)
    ap.add_argument("--n_heads", type=int, default=24)
    ap.add_argument("--ff_mult", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--use_compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--gen_probe_prompts", type=Path, default=None)
    return ap.parse_args()


def maybe_strip_compile_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            out[k[len("_orig_mod."):]] = v
        else:
            out[k] = v
    return out

def resolve_model_config(
    ArdorConfig: Any,
    *,
    args: argparse.Namespace,
    cfg_stage: StageConfig,
    vocab_size: int,
    special_ids: Dict[str, Optional[int]],
    resume_meta: Dict[str, Any],
) -> Any:
    declared = resume_meta.get("model_config")
    if declared is None:
        cfg = ArdorConfig(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            ff_mult=args.ff_mult,
            max_len=args.ctx,
            dropout=cfg_stage.dropout,
            attn_dropout=cfg_stage.dropout,
            resid_dropout=cfg_stage.dropout,
            use_rope=True,
            rope_theta=10000.0,
        )
        cfg.validate()
        return cfg

    if not isinstance(declared, dict):
        raise SystemExit("Resume checkpoint meta.model_config must be an object.")
    cfg = ArdorConfig.from_dict(declared)

    cli_contract = {
        "hidden_size": int(args.hidden_size),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "ff_mult": int(args.ff_mult),
        "max_len": int(args.ctx),
    }
    checkpoint_contract = {
        "hidden_size": int(cfg.hidden_size),
        "n_layers": int(cfg.n_layers),
        "n_heads": int(cfg.n_heads),
        "ff_mult": int(cfg.ff_mult),
        "max_len": int(cfg.max_len),
    }
    conflicts = {
        name: {"cli": cli_contract[name], "checkpoint": checkpoint_contract[name]}
        for name in cli_contract
        if cli_contract[name] != checkpoint_contract[name]
    }
    if conflicts:
        raise SystemExit(
            "Resume checkpoint model_config conflicts with canonical architecture CLI: "
            + json.dumps(conflicts, sort_keys=True)
        )
    if int(cfg.vocab_size) != int(vocab_size):
        raise SystemExit(
            f"Resume checkpoint vocab_size={cfg.vocab_size} does not match tokenizer vocab_size={vocab_size}."
        )

    checkpoint_special_ids = resume_meta.get("special_ids")
    if checkpoint_special_ids is not None:
        if not isinstance(checkpoint_special_ids, dict):
            raise SystemExit("Resume checkpoint meta.special_ids must be an object when present.")
        normalized = {key: checkpoint_special_ids.get(key) for key in special_ids}
        if normalized != special_ids:
            raise SystemExit(
                "Resume checkpoint special-token ids do not match the selected tokenizer: "
                + json.dumps({"checkpoint": normalized, "tokenizer": special_ids}, sort_keys=True)
            )
    return cfg


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda")

    set_all_seeds(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_dir / "metrics.jsonl"
    run_state_path = args.run_dir / "run_state.json"
    ckpt_last = ckpt_dir / "ckpt_last.pt"
    ckpt_latest = ckpt_dir / "ckpt_full_latest.pt"
    ckpt_final = ckpt_dir / "ckpt_full_final.pt"

    sp = tokenizer_special_ids(args.tokenizer)
    vocab_size = tokenizer_vocab_size(args.tokenizer)
    cfg_stage = STAGE_PRESETS[args.stage]

    resume_ckpt: Optional[Dict[str, Any]] = None
    resume_meta: Dict[str, Any] = {}
    if args.resume is not None:
        if not args.resume.exists():
            raise SystemExit(f"Missing resume checkpoint: {args.resume}")
        loaded = torch.load(args.resume, map_location="cpu")
        if not isinstance(loaded, dict):
            raise SystemExit("Resume checkpoint must be a dictionary.")
        resume_ckpt = loaded
        raw_meta = loaded.get("meta", {})
        if raw_meta is None:
            raw_meta = {}
        if not isinstance(raw_meta, dict):
            raise SystemExit("Resume checkpoint meta must be an object.")
        resume_meta = raw_meta

    ArdorConfig = import_project_config()
    model_cfg = resolve_model_config(
        ArdorConfig,
        args=args,
        cfg_stage=cfg_stage,
        vocab_size=vocab_size,
        special_ids=sp,
        resume_meta=resume_meta,
    )
    if resume_ckpt is not None and "model_config" not in resume_meta:
        print(f"[{now_str()}] [resume] checkpoint has no meta.model_config; using canonical CLI/stage model config")


    dec_cls = import_project_decoder()
    model = build_model(dec_cls, model_cfg).to(device)
    if args.use_compile:
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
            print(f"[{now_str()}] [compile] enabled")
        except Exception as e:
            print(f"[{now_str()}] [compile] failed: {e}")

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg_stage.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=cfg_stage.weight_decay,
            fused=True,
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg_stage.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=cfg_stage.weight_decay,
        )

    global_step = 0
    tokens_seen = 0
    best_val = float("inf")
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"], strict=True)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        global_step = int(resume_meta.get("step", 0))
        tokens_seen = int(resume_meta.get("tokens_seen", 0))
        best_val = float(resume_meta.get("best_val", best_val))
        print(f"[{now_str()}] [resume] ckpt={args.resume} step={global_step} tokens_seen={tokens_seen:,}")


    train_batcher = None
    val_batcher = None
    sft_loader = None
    sft_val_loader = None

    if args.stage in ("lm_base", "stabilize"):
        if not args.train_tokens.exists():
            raise SystemExit(f"Missing train tokens: {args.train_tokens}")
        train_stream = TokenStream(args.train_tokens)
        train_batcher = StreamBatcher(train_stream, cfg_stage.batch_size, args.ctx, seed=args.seed, random_starts=cfg_stage.random_starts)
        if args.val_tokens.exists():
            val_stream = TokenStream(args.val_tokens)
            val_batcher = StreamBatcher(val_stream, cfg_stage.batch_size, args.ctx, seed=args.seed + 1, random_starts=True)
    else:
        if args.sft_jsonl is None or not args.sft_jsonl.exists():
            raise SystemExit("Stage sft requires --sft_jsonl with prompt/response JSONL.")
        ds = SFTExampleDataset(args.sft_jsonl, args.tokenizer, args.ctx)
        n_val = max(64, int(0.02 * len(ds))) if len(ds) >= 512 else max(8, len(ds) // 10)
        n_train = len(ds) - n_val
        if n_train <= 0:
            raise SystemExit("SFT dataset too small after split.")
        train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
        sft_loader = DataLoader(train_ds, batch_size=cfg_stage.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        sft_val_loader = DataLoader(val_ds, batch_size=cfg_stage.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    micro_tokens = cfg_stage.batch_size * args.ctx
    grad_accum = int(math.ceil(cfg_stage.target_tokens_per_opt_step / max(1, micro_tokens)))
    grad_accum = max(1, min(cfg_stage.max_grad_accum, grad_accum))
    effective_tokens = micro_tokens * grad_accum

    final_tokens_target = cfg_stage.max_train_tokens
    remaining_tokens = max(0, final_tokens_target - tokens_seen)
    stop_step = global_step + int(math.ceil(remaining_tokens / max(1, effective_tokens)))
    sched = WarmupCosine(cfg_stage.lr, max(1, int(cfg_stage.warmup_fraction * max(1, stop_step))), max(1, stop_step))

    append_jsonl(metrics_path, {
        "ts": now_str(),
        "event": "boot",
        "stage": cfg_stage.name,
        "cfg_stage": asdict(cfg_stage),
        "model_cfg": asdict(model_cfg),
        "special_ids": sp,
        "tokens_seen_start": tokens_seen,
        "stop_step": stop_step,
        "effective_tokens_per_opt_step": effective_tokens,
    })

    prompt_tok = build_simple_tokenizer(args.tokenizer)
    probe_prompts = [
        "The nature of consciousness is",
        "What is the purpose of memory?",
        "Explain why repetition in language models happens.",
    ]
    if args.gen_probe_prompts and args.gen_probe_prompts.exists():
        probe_prompts = [ln.strip() for ln in args.gen_probe_prompts.read_text(encoding="utf-8").splitlines() if ln.strip()]

    model.train()
    loss_sum = 0.0
    ce_sum = 0.0
    ul_sum = 0.0
    time_sum = 0.0
    batch_iter: Optional[Iterable] = iter(sft_loader) if sft_loader is not None else None
    t0 = time.time()

    while global_step < stop_step and tokens_seen < final_tokens_target:
        lr = sched.lr_at(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        remaining_tokens = max(0, final_tokens_target - tokens_seen)
        current_grad_accum = int(math.ceil(remaining_tokens / max(1, micro_tokens)))
        current_grad_accum = max(1, min(grad_accum, current_grad_accum))
        current_effective_tokens = current_grad_accum * micro_tokens

        step_ce = 0.0
        step_ul = 0.0

        for _ in range(current_grad_accum):
            if args.stage in ("lm_base", "stabilize"):
                assert train_batcher is not None
                x, y, _ = train_batcher.next_batch()
            else:
                assert batch_iter is not None
                try:
                    x, y = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(sft_loader)
                    x, y = next(batch_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                if args.stage == "sft":
                    ce = masked_ce_loss(logits, y, label_smoothing=cfg_stage.label_smoothing)
                    ul = repeated_ngram_unlikelihood_loss(logits, torch.where(y >= 0, y, torch.zeros_like(y)), ngram=cfg_stage.ul_ngram, tail=cfg_stage.ul_tail)
                else:
                    ce = ce_loss(logits, y, label_smoothing=cfg_stage.label_smoothing)
                    ul = repeated_ngram_unlikelihood_loss(logits, y, ngram=cfg_stage.ul_ngram, tail=cfg_stage.ul_tail)
                loss = (ce + cfg_stage.ul_weight * ul) / current_grad_accum
            loss.backward()
            step_ce += float(ce.item())
            step_ul += float(ul.item())

        if cfg_stage.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_stage.grad_clip)
        optimizer.step()

        global_step += 1
        tokens_seen += current_effective_tokens
        dt = time.time() - t0
        t0 = time.time()
        time_sum += dt
        loss_sum += step_ce + cfg_stage.ul_weight * step_ul
        ce_sum += step_ce
        ul_sum += step_ul

        if global_step % cfg_stage.log_every == 0:
            avg_loss = loss_sum / cfg_stage.log_every
            avg_ce = ce_sum / cfg_stage.log_every
            avg_ul = ul_sum / cfg_stage.log_every
            tok_s = (cfg_stage.log_every * effective_tokens) / max(1e-9, time_sum)
            print(f"[{now_str()}] stage={cfg_stage.name} step={global_step}/{stop_step} tok/s={tok_s:,.0f} loss={avg_loss:.4f} ce={avg_ce:.4f} ul={avg_ul:.4f}")
            append_jsonl(metrics_path, {
                "ts": now_str(),
                "event": "train",
                "stage": cfg_stage.name,
                "step": global_step,
                "tokens_seen": tokens_seen,
                "loss": avg_loss,
                "ce": avg_ce,
                "ul": avg_ul,
                "lr": lr,
                "tok_s": tok_s,
            })
            loss_sum = ce_sum = ul_sum = 0.0
            time_sum = 0.0

        if global_step % cfg_stage.eval_every == 0:
            if args.stage in ("lm_base", "stabilize") and val_batcher is not None:
                val_loss = eval_stream_loss(model, val_batcher, steps=24, device=device, label_smoothing=cfg_stage.label_smoothing)
            elif args.stage == "sft" and sft_val_loader is not None:
                val_loss = eval_sft_loss(model, sft_val_loader, device=device, label_smoothing=cfg_stage.label_smoothing)
            else:
                val_loss = float("nan")
            best_val = min(best_val, val_loss) if not math.isnan(val_loss) else best_val
            append_jsonl(metrics_path, {
                "ts": now_str(),
                "event": "eval",
                "stage": cfg_stage.name,
                "step": global_step,
                "tokens_seen": tokens_seen,
                "val_loss": val_loss,
                "best_val": best_val,
            })
            print(f"[{now_str()}] [eval] stage={cfg_stage.name} step={global_step} val_loss={val_loss:.4f} best={best_val:.4f}")

        if global_step % cfg_stage.gen_eval_every == 0:
            model.eval()
            gen_rows = []
            for prompt in probe_prompts[:3]:
                enc = prompt_tok.encode(prompt)
                ids = list(enc.ids)
                out = sample_generate(model, ids, max_new_tokens=96, eos_id=sp["eos_id"], top_p=0.9, temperature=0.8)
                gen_metrics = repetition_metrics(out)
                gen_rows.append({"prompt": prompt, "metrics": gen_metrics, "ids_tail": out[-32:]})
            append_jsonl(metrics_path, {
                "ts": now_str(),
                "event": "gen_eval",
                "stage": cfg_stage.name,
                "step": global_step,
                "tokens_seen": tokens_seen,
                "samples": gen_rows,
            })
            model.train()

        if global_step % cfg_stage.save_every == 0:
            state = {
                "model": maybe_strip_compile_prefix(model.state_dict()),
                "meta": {
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                    "stage": cfg_stage.name,
                    "best_val": best_val,
                    "special_ids": sp,
                    "model_config": asdict(model_cfg),
                },
            }
            safe_torch_save(state, ckpt_last)
            full = {
                "model": maybe_strip_compile_prefix(model.state_dict()),
                "optimizer": optimizer.state_dict(),
                "meta": {
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                    "stage": cfg_stage.name,
                    "best_val": best_val,
                    "special_ids": sp,
                    "model_config": asdict(model_cfg),
                },
            }
            safe_torch_save(full, ckpt_latest)
            latest_path = str(ckpt_latest)
            atomic_write_json(run_state_path, {
                "global_step": global_step,
                "tokens_seen": tokens_seen,
                "stage": cfg_stage.name,
                "lr": lr,
                "best_val": best_val,
                "ckpt_last": latest_path,
            })

    final = {
        "model": maybe_strip_compile_prefix(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "meta": {
            "step": global_step,
            "tokens_seen": tokens_seen,
            "stage": cfg_stage.name,
            "best_val": best_val,
            "special_ids": sp,
            "model_config": asdict(model_cfg),
        },
    }
    safe_torch_save(final, ckpt_final)
    atomic_write_json(run_state_path, {
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "stage": cfg_stage.name,
        "best_val": best_val,
        "ckpt_last": str(ckpt_final),
    })
    append_jsonl(metrics_path, {"ts": now_str(), "event": "done", "stage": cfg_stage.name, "step": global_step, "tokens_seen": tokens_seen})
    print(f"[{now_str()}] [done] stage={cfg_stage.name} step={global_step} tokens_seen={tokens_seen:,}")


if __name__ == "__main__":
    main()
