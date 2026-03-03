#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neural_plasticity_training.py — B200 Throughput-First Trainer (single GPU, T=2048 fixed)

Run Card
Goal:
  Train Ardor (≈1B params) as fast as possible on a single B200 using a contiguous token stream.
Command:
  python /workspace/Ardor/neural_plasticity_training.py
  python /workspace/Ardor/neural_plasticity_training.py --resume
Inputs:
  - /workspace/Ardor/bin_dataset_20B/tokens.bin  (uint16 stream)
Artifacts:
  - /workspace/Ardor/runs/b200_stream_2048/run_state.json
  - /workspace/Ardor/runs/b200_stream_2048/metrics.jsonl
  - /workspace/Ardor/runs/b200_stream_2048/checkpoints/
Notes:
  - Hot loop is pure GPU: no tokenization, no JSON parsing, no torch.load, no per-step file I/O.
  - Data pipeline: memmap → pinned ring → async H2D (dedicated CUDA stream) → small GPU queue.
  - Resume is O(1): uses token_cursor into tokens.bin.
  - EOS=3, EOT=7 are measured once at startup but ignored for packing (fastest policy).
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import hashlib
import random
import threading
import queue
import subprocess
import inspect
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Hard invariants (edit the file to change; only --resume is supported)
# ==============================================================================

# Dataset (bin)
BIN_DIR = Path("/workspace/Ardor/bin_dataset_20B")
TOKENS_BIN = BIN_DIR / "tokens.bin"
META_JSON = BIN_DIR / "meta.json"

# Run output
RUN_DIR = Path(os.environ.get("ARDOR_RUN_DIR", "/workspace/Ardor/runs/b200_stream_2048"))
CKPT_DIR = RUN_DIR / "checkpoints"
RUN_STATE_PATH = RUN_DIR / "run_state.json"
METRICS_PATH = RUN_DIR / "metrics.jsonl"

# Model config (fixed by user)
HIDDEN_SIZE = 1536
N_LAYERS = 33
N_HEADS = 24
FF_MULT = 4
SEQ_LEN = 2048  # fixed and persistent

EOS_ID = 3
EOT_ID = 7

# Training defaults (override via env, not CLI)
DROPOUT = float(os.environ.get("ARDOR_DROPOUT", "0.0"))
LABEL_SMOOTHING = float(os.environ.get("ARDOR_LABEL_SMOOTHING", "0.0"))
GRAD_CLIP_NORM = float(os.environ.get("ARDOR_GRAD_CLIP", "1.0"))

BASE_LR = float(os.environ.get("ARDOR_LR", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("ARDOR_WD", "0.1"))
BETA1 = float(os.environ.get("ARDOR_BETA1", "0.9"))
BETA2 = float(os.environ.get("ARDOR_BETA2", "0.95"))
ADAM_EPS = float(os.environ.get("ARDOR_ADAM_EPS", "1e-8"))

WARMUP_FRACTION = float(os.environ.get("ARDOR_WARMUP_FRACTION", "0.015"))

PINNED_RING = int(os.environ.get("ARDOR_PINNED_RING", "16"))
GPU_QUEUE = int(os.environ.get("ARDOR_GPU_QUEUE", "3"))

AUTOTUNE = os.environ.get("ARDOR_AUTOTUNE", "1").strip().lower() in ("1", "true", "yes", "y", "on")
AUTOTUNE_ITERS = int(os.environ.get("ARDOR_AUTOTUNE_ITERS", "4"))
AUTOTUNE_WARMUP = int(os.environ.get("ARDOR_AUTOTUNE_WARMUP", "2"))

TARGET_TOKENS_PER_OPT_STEP = int(os.environ.get("ARDOR_TARGET_TOKENS_PER_OPT_STEP", "2000000"))
MAX_GRAD_ACCUM = int(os.environ.get("ARDOR_MAX_GRAD_ACCUM", "16"))

# FASTEST defaults (less chatty, less ckpt overhead)
LOG_EVERY_STEPS = int(os.environ.get("ARDOR_LOG_EVERY", "200"))
SYNC_LOG_EVERY = int(os.environ.get("ARDOR_SYNC_LOG_EVERY", "0"))  # 0 disables explicit sync
RUN_STATE_EVERY_STEPS = int(os.environ.get("ARDOR_RUN_STATE_EVERY", "200"))
WEIGHTS_CKPT_EVERY_STEPS = int(os.environ.get("ARDOR_WEIGHTS_CKPT_EVERY", "20000"))
FULL_CKPT_EVERY_STEPS = int(os.environ.get("ARDOR_FULL_CKPT_EVERY", "100000"))

USE_COMPILE = os.environ.get("ARDOR_COMPILE", "1").strip().lower() in ("1", "true", "yes", "y", "on")
COMPILE_MODE = os.environ.get("ARDOR_COMPILE_MODE", "max-autotune")

BASE_SEED = int(os.environ.get("ARDOR_SEED", "1337"))

# Optional perf toggle: keep pinned/gpu batches as uint16 and cast to long on GPU (often faster overall).
# Options: "uint16" (default, smaller transfers), "int64" (old behavior).
TOKEN_BATCH_DTYPE = os.environ.get("ARDOR_TOKEN_BATCH_DTYPE", "uint16").strip().lower()


# ==============================================================================
# Utilities
# ==============================================================================

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state_full() -> Dict[str, Any]:
    # Stored only inside full checkpoints, not in run_state.json
    st: Dict[str, Any] = {
        "torch_cpu_state": torch.get_rng_state().cpu().numpy().tolist(),
        "python_state": repr(random.getstate()),
        "numpy_state": repr(np.random.get_state()),
    }
    st["torch_cuda_state"] = [t.cpu().numpy().tolist() for t in torch.cuda.get_rng_state_all()]
    return st


def safe_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ==============================================================================
# Project model import (canonical)
# ==============================================================================

def _ensure_repo_on_path() -> None:
    root = Path(__file__).resolve().parents[0]
    if (root / "Cerebrum").exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def import_project_config() -> Any:
    _ensure_repo_on_path()
    try:
        from Cerebrum.Cortex.ardor_config import ArdorConfig  # canonical
        return ArdorConfig
    except Exception as e:
        raise SystemExit(
            "Could not import Cerebrum.Cortex.ardor_config.ArdorConfig.\n"
            "Make sure New Cortex exists at /workspace/Ardor/Cerebrum/Cortex/ardor_config.py\n"
            f"Import error: {e}"
        )


def import_project_decoder() -> Any:
    _ensure_repo_on_path()
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
    raise SystemExit(
        "Could not import ArdorDecoder from the project.\n"
        "Expected Cerebrum/Cortex/broca_decoder.py with ArdorDecoder (or BrocaDecoder)."
    )


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
    ):
        if k in params and hasattr(proj_cfg, k):
            kwargs[k] = getattr(proj_cfg, k)
    if kwargs:
        return dec_cls(**kwargs)

    raise SystemExit(
        f"Decoder constructor signature not supported: {sig}\n"
        "Expected it to accept (cfg=...), (config=...), or (cfg) positional."
    )


# ==============================================================================
# Dataset stream (memmap)
# ==============================================================================

class TokenStream:
    def __init__(self, tokens_bin: Path):
        self.tokens_mm = np.memmap(tokens_bin, dtype=np.uint16, mode="r")
        self.N = int(self.tokens_mm.shape[0])

    def copy_flat_into(self, dst_flat: np.ndarray, start: int) -> None:
        n = dst_flat.size
        start = int(start) % self.N
        end = start + n
        if end <= self.N:
            np.copyto(dst_flat, self.tokens_mm[start:end], casting="unsafe")
            return
        n1 = self.N - start
        n2 = n - n1
        np.copyto(dst_flat[:n1], self.tokens_mm[start:self.N], casting="unsafe")
        np.copyto(dst_flat[n1:], self.tokens_mm[0:n2], casting="unsafe")


# ==============================================================================
# Prefetch pipeline (memmap → pinned → async H2D → GPU queue)
# ==============================================================================

class Prefetcher:
    def __init__(
        self,
        stream: TokenStream,
        B: int,
        T: int,
        ring: int,
        gpu_queue: int,
        device: torch.device,
        start_cursor: int,
        token_batch_dtype: str = "uint16",
    ):
        self.stream = stream
        self.B = int(B)
        self.T = int(T)
        self.device = device
        self.chunk = self.B * (self.T + 1)
        self._stop = threading.Event()

        if token_batch_dtype not in ("uint16", "int64"):
            raise ValueError("ARDOR_TOKEN_BATCH_DTYPE must be 'uint16' or 'int64'")
        self.token_batch_dtype = token_batch_dtype

        cpu_dtype = torch.uint16 if token_batch_dtype == "uint16" else torch.int64
        gpu_dtype = torch.uint16 if token_batch_dtype == "uint16" else torch.int64

        # CPU pinned ring (uint16 default = smaller H2D + less CPU work)
        self.cpu_bufs: List[torch.Tensor] = [
            torch.empty((self.B, self.T + 1), dtype=cpu_dtype, pin_memory=True)
            for _ in range(ring)
        ]
        # numpy view for flat fill
        self.cpu_np_flat: List[np.ndarray] = [buf.view(-1).numpy() for buf in self.cpu_bufs]

        # GPU buffers
        self.gpu_bufs: List[torch.Tensor] = [
            torch.empty((self.B, self.T + 1), dtype=gpu_dtype, device=self.device)
            for _ in range(gpu_queue)
        ]
        self.free_slots: "queue.Queue[int]" = queue.Queue()
        for i in range(gpu_queue):
            self.free_slots.put(i)

        self.ready: "queue.Queue[Tuple[int, torch.cuda.Event, int]]" = queue.Queue(maxsize=gpu_queue)

        self.copy_stream = torch.cuda.Stream()
        self.cursor = int(start_cursor) % self.stream.N
        self.consumed_cursor = int(start_cursor) % self.stream.N
        self.cursor_lock = threading.Lock()

        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.thread.join(timeout=2.0)
        except Exception:
            pass

    def _advance_cursor(self, delta: int) -> int:
        with self.cursor_lock:
            start = int(self.cursor)
            self.cursor = (self.cursor + int(delta)) % self.stream.N
            return start

    def mark_consumed(self, end_cursor: int) -> None:
        with self.cursor_lock:
            self.consumed_cursor = int(end_cursor) % self.stream.N

    def release(self, slot: int) -> None:
        self.free_slots.put(slot)

    def _worker(self) -> None:
        cpu_i = 0
        while not self._stop.is_set():
            try:
                slot = self.free_slots.get(timeout=0.1)
            except queue.Empty:
                continue

            start = self._advance_cursor(self.chunk)

            flat = self.cpu_np_flat[cpu_i]
            self.stream.copy_flat_into(flat, start)

            filled_i = cpu_i
            cpu_i = (cpu_i + 1) % len(self.cpu_bufs)

            ev = torch.cuda.Event(enable_timing=False)
            with torch.cuda.stream(self.copy_stream):
                self.gpu_bufs[slot].copy_(self.cpu_bufs[filled_i], non_blocking=True)
                ev.record(self.copy_stream)

            self.ready.put((slot, ev, start))

    def next_gpu(self) -> Tuple[torch.Tensor, int, int, int]:
        slot, ev, start = self.ready.get()
        torch.cuda.current_stream().wait_event(ev)
        end = (int(start) + self.chunk) % self.stream.N
        return self.gpu_bufs[slot], int(slot), int(start), int(end)


# ==============================================================================
# Async writers
# ==============================================================================

class AsyncJSONWriter:
    def __init__(self, path: Path, mode: str):
        self.path = path
        self.mode = mode
        self.q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=256)
        self.stop_ev = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.t.start()

    def stop(self) -> None:
        self.stop_ev.set()
        try:
            self.q.put_nowait({"_stop": True})
        except Exception:
            pass
        self.t.join(timeout=2.0)

    def submit(self, obj: Dict[str, Any]) -> None:
        try:
            self.q.put_nowait(obj)
        except queue.Full:
            pass  # latest-wins

    def _run(self) -> None:
        while not self.stop_ev.is_set():
            obj = self.q.get()
            if obj.get("_stop"):
                break
            if self.mode == "atomic_json":
                atomic_write_json(self.path, obj)
            elif self.mode == "jsonl":
                append_jsonl(self.path, obj)
            else:
                raise ValueError(f"Unknown mode={self.mode}")


class AsyncCkptWriter:
    """I/O off the hot loop (state_dict still costs)."""
    def __init__(self):
        self.q: "queue.Queue[Tuple[Any, Path]]" = queue.Queue(maxsize=8)
        self.stop_ev = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.t.start()

    def stop(self) -> None:
        self.stop_ev.set()
        try:
            self.q.put_nowait(({"_stop": True}, Path("/dev/null")))
        except Exception:
            pass
        self.t.join(timeout=10.0)

    def submit(self, obj: Any, path: Path) -> None:
        try:
            self.q.put_nowait((obj, path))
        except queue.Full:
            pass  # latest-wins

    def _run(self) -> None:
        while not self.stop_ev.is_set():
            obj, path = self.q.get()
            if isinstance(obj, dict) and obj.get("_stop"):
                break
            try:
                safe_torch_save(obj, path)
            except Exception as e:
                append_jsonl(METRICS_PATH, {"ts": _now(), "event": "ckpt_error", "path": str(path), "err": str(e)})


# ==============================================================================
# LR schedule
# ==============================================================================

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


# ==============================================================================
# Autotune microbatch
# ==============================================================================

def autotune_microbatch(model: nn.Module, device: torch.device, vocab_size: int) -> int:
    cand_env = os.environ.get("ARDOR_B_CANDIDATES", "")
    if cand_env.strip():
        candidates = [int(x.strip()) for x in cand_env.split(",") if x.strip()]
    else:
        candidates = [8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256]

    best_B = candidates[0]
    best_toks = 0.0

    for B in candidates:
        try:
            x = torch.randint(0, vocab_size, (B, SEQ_LEN), device=device, dtype=torch.long)
            y = torch.randint(0, vocab_size, (B, SEQ_LEN), device=device, dtype=torch.long)

            for _ in range(AUTOTUNE_WARMUP):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                model.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(AUTOTUNE_ITERS):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            dt = time.time() - t0
            toks = (B * SEQ_LEN * AUTOTUNE_ITERS) / max(1e-9, dt)

            if toks >= best_toks * 1.01:
                best_toks = toks
                best_B = B
            else:
                if B > best_B and toks < best_toks * 1.005:
                    break

            print(f"[{_now()}] [autotune] B={B:<4d} tok/s={toks:,.0f} best_B={best_B} best_tok/s={best_toks:,.0f}")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{_now()}] [autotune] B={B} OOM → stop")
            break

    return int(best_B)


# ==============================================================================
# Main training
# ==============================================================================

def main() -> None:
    resume = "--resume" in sys.argv

    assert TOKENS_BIN.exists(), f"Missing {TOKENS_BIN}"

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA not available; this script is intended for a B200 GPU.")

    # Perf toggles
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass

    set_all_seeds(BASE_SEED)

    # Dataset stream
    stream = TokenStream(TOKENS_BIN)
    tokens_total = stream.N

    # One-time probe: eos/eot freq + vocab max
    rng = np.random.default_rng(BASE_SEED)
    probe_n = 8_000_000
    start = int(rng.integers(0, max(1, tokens_total - probe_n)))
    probe = np.asarray(stream.tokens_mm[start:start + probe_n], dtype=np.uint16)
    eos_freq = float((probe == EOS_ID).mean())
    eot_freq = float((probe == EOT_ID).mean())
    vocab_size = int(probe.max()) + 1

    # Build config (canonical)
    ArdorConfig = import_project_config()
    cfg = ArdorConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ff_mult=FF_MULT,
        max_len=SEQ_LEN,
        dropout=DROPOUT,
        attn_dropout=0.0,
        resid_dropout=0.0,
        use_rope=True,
        rope_theta=10000.0,
    )
    cfg.validate()

    repo_root = Path("/workspace/Ardor")
    git_commit = get_git_commit(repo_root)
    arch_fingerprint = _sha256_bytes(json.dumps(asdict(cfg), sort_keys=True).encode("utf-8"))
    tokenizer_path = repo_root / "tokenizer_v9.json"
    tokenizer_fingerprint = sha256_file(tokenizer_path) if tokenizer_path.exists() else None

    # Import and build the project decoder
    dec_cls = import_project_decoder()
    model = build_model(dec_cls, cfg).to(device)

    # Autotune B
    if AUTOTUNE:
        B = autotune_microbatch(model, device, vocab_size)
    else:
        B = int(os.environ.get("ARDOR_B", "128"))

    micro_tokens = B * SEQ_LEN
    grad_accum = int(math.ceil(TARGET_TOKENS_PER_OPT_STEP / max(1, micro_tokens)))
    grad_accum = max(1, min(MAX_GRAD_ACCUM, grad_accum))
    effective_tokens = micro_tokens * grad_accum

    epochs = float(os.environ.get("ARDOR_EPOCHS", "1.0"))
    train_tokens_target = int(tokens_total * epochs)
    opt_steps_total = int(math.ceil(train_tokens_target / max(1, effective_tokens)))

    warmup_steps = max(1, int(opt_steps_total * WARMUP_FRACTION))
    sched = WarmupCosine(BASE_LR, warmup_steps, opt_steps_total)

    # Optimizer (fused if available)
    fused_ok = False
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=BASE_LR,
            betas=(BETA1, BETA2),
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
            fused=True,
        )
        fused_ok = True
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=BASE_LR,
            betas=(BETA1, BETA2),
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )

    # Async writers
    run_state_writer = AsyncJSONWriter(RUN_STATE_PATH, mode="atomic_json")
    metrics_writer = AsyncJSONWriter(METRICS_PATH, mode="jsonl")
    run_state_writer.start()
    metrics_writer.start()
    ckpt_writer = AsyncCkptWriter()
    ckpt_writer.start()

    # State
    global_step = 0
    tokens_seen = 0
    token_cursor = 0

    # Resume (no RNG restore from run_state)
    if resume and RUN_STATE_PATH.exists():
        st = json.loads(RUN_STATE_PATH.read_text(encoding="utf-8"))
        ckpt_path = st.get("ckpt_last")
        if ckpt_path and Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=True)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            global_step = int(st.get("global_step", 0))
            tokens_seen = int(st.get("tokens_seen", 0))
            token_cursor = int(st.get("token_cursor", 0))
            print(f"[{_now()}] [resume] ckpt={ckpt_path} step={global_step} tokens_seen={tokens_seen:,} cursor={token_cursor:,}")
        else:
            print(f"[{_now()}] [resume] run_state found but ckpt missing; starting fresh")

    # Prefetcher (consumed cursor tracking)
    prefetch = Prefetcher(
        stream=stream,
        B=B,
        T=SEQ_LEN,
        ring=PINNED_RING,
        gpu_queue=GPU_QUEUE,
        device=device,
        start_cursor=token_cursor,
        token_batch_dtype=TOKEN_BATCH_DTYPE,
    )
    prefetch.start()

    # Loss fn (compile-friendly)
    def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                label_smoothing=LABEL_SMOOTHING,
            )

    if USE_COMPILE:
        try:
            loss_fn = torch.compile(loss_fn, mode=COMPILE_MODE, fullgraph=True)  # type: ignore
            print(f"[{_now()}] [compile] enabled mode={COMPILE_MODE}")
        except Exception as e:
            print(f"[{_now()}] [compile] failed; continuing without compile: {e}")

    # Boot header
    metrics_writer.submit({
        "ts": _now(),
        "event": "boot",
        "device": torch.cuda.get_device_name(0),
        "tokens_total": tokens_total,
        "vocab_size": vocab_size,
        "cfg": asdict(cfg),
        "arch_fingerprint": arch_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "git_commit": git_commit,
        "eos_freq_est": eos_freq,
        "eot_freq_est": eot_freq,
        "B": B,
        "grad_accum": grad_accum,
        "effective_tokens_per_opt_step": effective_tokens,
        "base_lr": BASE_LR,
        "wd": WEIGHT_DECAY,
        "fused_adamw": fused_ok,
        "label_smoothing": LABEL_SMOOTHING,
        "dropout": DROPOUT,
        "compile": USE_COMPILE,
        "opt_steps_total": opt_steps_total,
        "warmup_steps": warmup_steps,
        "token_batch_dtype": TOKEN_BATCH_DTYPE,
        "log_every": LOG_EVERY_STEPS,
        "weights_ckpt_every": WEIGHTS_CKPT_EVERY_STEPS,
        "full_ckpt_every": FULL_CKPT_EVERY_STEPS,
    })

    model.train()
    loss_accum = torch.zeros((), device=device)
    time_accum = 0.0
    t_step0 = time.time()

    try:
        while global_step < opt_steps_total:
            lr = sched.lr_at(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            for _ in range(grad_accum):
                gpu_batch, slot, _batch_start, batch_end = prefetch.next_gpu()

                # If batches are uint16, cast on GPU right before the model (fast path depends on bottleneck).
                if gpu_batch.dtype != torch.long:
                    x = gpu_batch[:, :-1].to(torch.long)
                    y = gpu_batch[:, 1:].to(torch.long)
                else:
                    x = gpu_batch[:, :-1]
                    y = gpu_batch[:, 1:]

                loss = loss_fn(x, y) / grad_accum
                loss.backward()

                prefetch.release(slot)
                token_cursor = batch_end
                prefetch.mark_consumed(batch_end)

                loss_accum = loss_accum + loss.detach()

            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            optimizer.step()

            global_step += 1
            tokens_seen += effective_tokens

            dt = time.time() - t_step0
            t_step0 = time.time()
            time_accum += dt

            if global_step % LOG_EVERY_STEPS == 0:
                if SYNC_LOG_EVERY > 0 and (global_step % SYNC_LOG_EVERY == 0):
                    torch.cuda.synchronize()
                # NOTE: .item() causes a sync; we make it rare (LOG_EVERY default=200)
                avg_loss = float(loss_accum.item()) / LOG_EVERY_STEPS
                tok_s = (effective_tokens * LOG_EVERY_STEPS) / max(1e-9, time_accum)
                metrics_writer.submit({
                    "ts": _now(),
                    "event": "train",
                    "step": global_step,
                    "loss": avg_loss,
                    "tok_s": tok_s,
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                    "cursor": token_cursor,
                })
                loss_accum.zero_()
                time_accum = 0.0

            if global_step % RUN_STATE_EVERY_STEPS == 0:
                run_state_writer.submit({
                    "arch_fingerprint": arch_fingerprint,
                    "tokenizer_fingerprint": tokenizer_fingerprint,
                    "git_commit": git_commit,
                    "global_step": global_step,
                    "tokens_seen": tokens_seen,
                    "token_cursor": token_cursor,
                    "B": B,
                    "grad_accum": grad_accum,
                    "seq_len": SEQ_LEN,
                    "effective_tokens_per_opt_step": effective_tokens,
                    "lr": lr,
                    "wd": WEIGHT_DECAY,
                    "label_smoothing": LABEL_SMOOTHING,
                    "dropout": DROPOUT,
                    "eos_id": EOS_ID,
                    "eot_id": EOT_ID,
                    "eos_freq_est": eos_freq,
                    "eot_freq_est": eot_freq,
                    "ckpt_last": str(CKPT_DIR / "ckpt_last.pt") if (CKPT_DIR / "ckpt_last.pt").exists() else None,
                })

            # checkpoints (controlled intervals; state_dict still costs)
            if global_step % WEIGHTS_CKPT_EVERY_STEPS == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "meta": {
                        "step": global_step,
                        "tokens_seen": tokens_seen,
                        "cursor": token_cursor,
                        "arch_fingerprint": arch_fingerprint,
                        "tokenizer_fingerprint": tokenizer_fingerprint,
                        "git_commit": git_commit,
                    },
                }
                ckpt_writer.submit(ckpt, CKPT_DIR / "ckpt_last.pt")

            if global_step % FULL_CKPT_EVERY_STEPS == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "meta": {
                        "step": global_step,
                        "tokens_seen": tokens_seen,
                        "cursor": token_cursor,
                        "arch_fingerprint": arch_fingerprint,
                        "tokenizer_fingerprint": tokenizer_fingerprint,
                        "git_commit": git_commit,
                        "rng_state": capture_rng_state_full(),
                    },
                }
                ckpt_writer.submit(ckpt, CKPT_DIR / f"ckpt_full_step_{global_step:07d}.pt")

        # final
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "meta": {
                "step": global_step,
                "tokens_seen": tokens_seen,
                "cursor": token_cursor,
                "arch_fingerprint": arch_fingerprint,
                "tokenizer_fingerprint": tokenizer_fingerprint,
                "git_commit": git_commit,
                "rng_state": capture_rng_state_full(),
            },
        }
        ckpt_writer.submit(ckpt, CKPT_DIR / "ckpt_full_final.pt")
        metrics_writer.submit({"ts": _now(), "event": "done", "step": global_step, "tokens_seen": tokens_seen})
        print(f"[{_now()}] [done] steps={global_step} tokens_seen={tokens_seen:,}")

    finally:
        prefetch.stop()
        ckpt_writer.stop()
        run_state_writer.stop()
        metrics_writer.stop()


if __name__ == "__main__":
    main()