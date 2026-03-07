# REM_consolidation_conv_fixed.py — Gentle, top-layer REM on conversations
# - Tokenizer↔shard fingerprint guard
# - Unfreeze only top-2 blocks + lm_head
# - No dropout; bf16 autocast without GradScaler
# - Random cropping; per-batch 80th percentile loss gating
# - Ultra-low LR; gradient clipping

import os, math, json, hashlib, random, time, sys
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

# ------------------ Paths ------------------
BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parents[1]
ARTIFACTS_REM_DIR = REPO_ROOT / "artifacts" / "rem"
ARTIFACTS_MODELS_DIR = REPO_ROOT / "artifacts" / "models"

TOKENIZER = Path(os.environ.get("ARDOR_TOKENIZER", REPO_ROOT / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v8.json"))
CONV_DIR  = Path(os.environ.get("ARDOR_CONV_DIR",  REPO_ROOT / "artifacts" / "datasets" / "Conversations"))
CHECKPOINT_IN = Path(os.environ.get("ARDOR_CKPT_IN", ARTIFACTS_MODELS_DIR / "Ardor_Final.pt"))
OUTPUT_MODEL  = Path(os.environ.get("ARDOR_REM_OUT", ARTIFACTS_MODELS_DIR / "Ardor_Ksai.pt"))
CKPT_DIR      = Path(os.environ.get("ARDOR_REM_CKPTS", ARTIFACTS_REM_DIR / "checkpoints"))
REM_STATUS_PATH = Path(os.environ.get("ARDOR_REM_STATUS", ARTIFACTS_REM_DIR / "rem_status.json"))

ARTIFACTS_REM_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

# ------------------ Hyperparams ------------------
EPOCHS = 3
SHARDS_PER_EPOCH = 32
BATCH_SIZE = 32
LR = 1e-6
CLIP_NORM = 0.5
TOP_UNFREEZE = 2

# ------------------ AMP dtype ------------------
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ------------------ Tokenizer & guard ------------------
tok = Tokenizer.from_file(str(TOKENIZER))
VOCAB_SIZE = tok.get_vocab_size()

def _fingerprint(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def _assert_tokenizer_compat(shard_root: Path, tokenizer_path: Path):
    meta = shard_root / "tokenizer_meta.json"
    if not shard_root.exists():
        raise RuntimeError(f"❌ Conversations dir not found: {shard_root}")
    if not meta.exists():
        raise RuntimeError(f"❌ Missing {meta}. Re-tokenize shards correctly and write meta.")
    saved = json.loads(meta.read_text())
    current = {"path": str(tokenizer_path), "sha256": _fingerprint(tokenizer_path)}
    if saved.get("sha256") != current["sha256"]:
        raise RuntimeError(
            f"❌ Tokenizer mismatch.\nShards expect: {saved}\nCurrent:  {current}\n"
            "→ Rebuild conversation shards with this tokenizer."
        )

_assert_tokenizer_compat(CONV_DIR, TOKENIZER)

# ------------------ Model ------------------
sys.path.append(str((REPO_ROOT / "Cerebrum" / "Cortex").resolve()))
from broca_decoder import ArdorDecoder
model = ArdorDecoder(VOCAB_SIZE, 384, 8, 6, dropout=0.0).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_IN, map_location=DEVICE))
print(f"✅ Loaded checkpoint: {CHECKPOINT_IN}")

# Freeze all, then unfreeze only top-2 blocks + head
for p in model.parameters(): p.requires_grad = False
for i, blk in enumerate(model.blocks):
    if i >= len(model.blocks) - TOP_UNFREEZE:
        for p in blk.parameters(): p.requires_grad = True
for p in model.lm_head.parameters(): p.requires_grad = True

# Ensure dropout is off
model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.0
model.train()

opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR, weight_decay=0.0)
criterion = nn.CrossEntropyLoss(reduction="none")

# ------------------ Data ------------------
def list_shards(path: Path) -> List[Path]:
    return sorted(path.glob("*.pt"))

def safe_load(p: Path) -> Optional[torch.Tensor]:
    try:
        t = torch.load(p, map_location="cpu")
        return t if isinstance(t, torch.Tensor) and t.ndim == 2 else None
    except Exception:
        return None

class RandomCropDataset(Dataset):
    def __init__(self, shards: List[Path], seq_len: int = 512):
        self.seq_len = seq_len
        self.samples: List[torch.Tensor] = []
        for s in shards:
            t = safe_load(s)
            if t is None: continue
            for row in t:
                if row.numel() > 2:
                    self.samples.append(row)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        T = row.size(0)
        L = min(self.seq_len + 1, T)
        start = 0 if L == T else torch.randint(0, T - L + 1, ()).item()
        chunk = row[start:start+L]
        return chunk[:-1], chunk[1:]

def make_loader(shards: List[Path], bs: int):
    ds = RandomCropDataset(shards, seq_len=512)
    if len(ds) == 0: return None
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, pin_memory=True)

all_shards = list_shards(CONV_DIR)
if not all_shards:
    raise RuntimeError("❌ No conversation shards found.")

# ------------------ REM loop ------------------


def _write_rem_status(epoch: int, total_epochs: int, progress: int, loss: float):
    payload = {
        "epoch": int(epoch),
        "total_epochs": int(total_epochs),
        "progress": int(progress),
        "loss": float(loss),
        "timestamp": time.time(),
    }
    REM_STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def save_ckpt(epoch: int):
    path = CKPT_DIR / f"REM_epoch{epoch}.pt"
    torch.save(model.state_dict(), path)
    return path

for epoch in range(1, EPOCHS + 1):
    sel = random.sample(all_shards, min(SHARDS_PER_EPOCH, len(all_shards)))
    loader = make_loader(sel, BATCH_SIZE)
    if loader is None:
        print("⚠️ No valid dataset this epoch; skipping.")
        continue

    total_kept, total_seen, running = 0, 0, 0.0
    for xb_cpu, yb_cpu in tqdm(loader, desc=f"🌙 REM {epoch}/{EPOCHS}"):
        xb, yb = xb_cpu.to(DEVICE, non_blocking=True), yb_cpu.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            logits = model(xb)
            tok_loss = criterion(logits.view(-1, VOCAB_SIZE), yb.view(-1))
            row_loss = tok_loss.view(xb.size(0), -1).mean(1)      # per-sequence loss

        # per-batch gating: keep best 80% (drop hardest 20% this batch)
        thresh = torch.quantile(row_loss.detach(), 0.80)
        keep = row_loss <= thresh
        total_seen += xb.size(0)
        kept = keep.sum().item()
        total_kept += kept
        if kept == 0:
            continue

        loss = row_loss[keep].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step(); opt.zero_grad()
        running += loss.item()

    avg = running / max(1, total_kept)
    _write_rem_status(epoch, EPOCHS, int((epoch / EPOCHS) * 100), avg)
    print(f"✅ REM epoch {epoch}: avg_kept_loss={avg:.4f}  kept/seen={total_kept}/{total_seen}")
    save_ckpt(epoch)

torch.save(model.state_dict(), OUTPUT_MODEL)
_write_rem_status(EPOCHS, EPOCHS, 100, 0.0)
print(f"\n🎉 REM consolidation complete → {OUTPUT_MODEL}")
