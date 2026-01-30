#!/usr/bin/env python3
# pooled_embeddings.py — build pooled features (X.npy) + labels (Y.npy) for GoEmotions, ED, MELD
# Usage example:
#   python pooled_embeddings.py \
#     --root /workspace/Ardor/Emotions \
#     --tokenizer /workspace/Ardor/ProjectTokenizer/ardor_tokenizer/tokenizer_v9.json \
#     --out-dir /workspace/Ardor/EmotionPooled \
#     --pool mean --max-len 256 --batch-size 256 --device cuda \
#     --encoder-hidden 384 --encoder-layers 8 --encoder-heads 6 \
#     --encoder-ckpt /workspace/Ardor/checkpoints/encoder.pt
from __future__ import annotations
import os, sys, json, math, argparse, glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

# --- optional libs for data i/o ---
HAS_DATASETS, HAS_PYARROW, HAS_PANDAS = True, True, True
try:
    from datasets import load_from_disk, DatasetDict, Dataset
except Exception:
    HAS_DATASETS = False

try:
    import pyarrow as pa
    import pyarrow.dataset as pads
    import pyarrow.csv as pacsv
except Exception:
    HAS_PYARROW = False

try:
    import pandas as pd
except Exception:
    HAS_PANDAS = False

from tokenizers import Tokenizer
# Your encoder
import sys
sys.path.append('../Cerebrum/Cortex')
from posterior_parietal_cortex import ArdorEncoder


# ---------------------------
# Helpers: dataset discovery
# ---------------------------
DS_HINTS = {
    "goe": ["goemotions", "goe", "go-emotions"],
    "ed":  ["empatheticdialogues", "empathetic_dialogues", "ed"],
    "meld":["meld"]
}
SPLIT_HINTS = ["train", "validation", "valid", "val", "test"]

def _find_datasets(root: Path) -> Dict[str, Path]:
    """Return {'goe': Path, 'ed': Path, 'meld': Path} for any present under root."""
    found = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        for key, aliases in DS_HINTS.items():
            if any(a in name for a in aliases):
                found[key] = child.resolve()
    return found

def _guess_columns(name: str, columns: List[str]) -> Tuple[str, str]:
    """Heuristically pick (text_col, label_col) per dataset."""
    cols = [c.lower() for c in columns]
    # text candidates
    txt_cands = ["text", "utterance", "sentence", "content", "comment_text", "Utterance".lower()]
    # label candidates (multi or single)
    if name == "goe":
        lab_cands = ["labels", "label_ids", "emotion_ids"]
    elif name == "ed":
        lab_cands = ["emotion", "label", "labels", "Emotion".lower()]
    else:  # meld
        lab_cands = ["emotion", "label", "labels", "Emotion".lower()]
    txt = next((c for c in cols if c in txt_cands), None)
    lab = next((c for c in cols if c in lab_cands), None)
    if txt is None or lab is None:
        raise ValueError(f"Could not infer text/label columns from: {columns}")
    # return original-case names
    txt_real = columns[cols.index(txt)]
    lab_real = columns[cols.index(lab)]
    return txt_real, lab_real

def _load_with_datasets_api(path: Path):
    if not HAS_DATASETS:
        raise RuntimeError("huggingface 'datasets' is not installed.")
    try:
        return load_from_disk(str(path))
    except Exception:
        # try to build from CSV files under directory
        data_files = {}
        for split in SPLIT_HINTS:
            files = list(path.glob(f"**/*{split}*.csv"))
            if files:
                # datasets needs a dict of lists or globs
                data_files[split if split != "valid" else "validation"] = [str(f) for f in files]
        if not data_files:
            raise
        from datasets import load_dataset
        return load_dataset("csv", data_files=data_files)

def _load_arrow_dir(path: Path):
    if not HAS_PYARROW:
        raise RuntimeError("pyarrow not installed and HF datasets loader failed.")
    # try common split directories
    d = {}
    for split in SPLIT_HINTS:
        sp = path / split
        if sp.exists():
            # any arrow files in subdir?
            arrs = list(sp.rglob("*.arrow"))
            if not arrs:
                continue
            # lazily read all shards
            tables = [pa.ipc.open_file(str(a)).read_all() for a in arrs]
            d[split if split != "valid" else "validation"] = pa.concat_tables(tables)
    if not d:
        # maybe flat arrow files with split in filename
        for split in SPLIT_HINTS:
            arrs = list(path.glob(f"*{split}*.arrow"))
            if arrs:
                tables = [pa.ipc.open_file(str(a)).read_all() for a in arrs]
                d[split if split != "valid" else "validation"] = pa.concat_tables(tables)
    if not d:
        raise RuntimeError(f"No Arrow data found in {path}")
    return d  # dict of split->pyarrow.Table

def _as_iterable_records(data, split: str):
    """Yield dict records for a split, independent of backend."""
    if HAS_DATASETS and isinstance(data, (dict,)):
        if split not in data: return []
        tbl = data[split]
        # HF Dataset or Arrow Table
        try:
            from datasets import Dataset
            if isinstance(tbl, Dataset):
                cols = tbl.column_names
                for r in tbl:
                    yield r, cols
                return
        except Exception:
            pass
        # maybe a pyarrow.Table
        if HAS_PYARROW and isinstance(tbl, pa.Table):
            cols = [c for c in tbl.column_names]
            for row in tbl.to_pylist():
                yield row, cols
            return
    # default: assume pyarrow.Table dict mapping
    if split in data:
        tbl = data[split]
        if HAS_PYARROW and isinstance(tbl, pa.Table):
            cols = [c for c in tbl.column_names]
            for row in tbl.to_pylist():
                yield row, cols
            return
    return []

# ---------------------------
# Tokenization utilities
# ---------------------------
def _build_tokenizer(tok_path: Path, max_len: int):
    tok = Tokenizer.from_file(str(tok_path))
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=max_len, pad_token="<pad>")
    return tok

def _tokenize_batch(tok: Tokenizer, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    encs = tok.encode_batch(texts)
    ids = [e.ids for e in encs]
    am  = [e.attention_mask for e in encs]
    input_ids = torch.tensor(ids, dtype=torch.long)
    attn_mask = torch.tensor(am, dtype=torch.long)   # 1 for real tokens, 0 for pad
    return input_ids, attn_mask

def _masked_mean(h: torch.Tensor, mask: torch.Tensor, start_index: int) -> torch.Tensor:
    # h: [B,S,H]; mask: [B,S] (1/0) of tokens-only positions (CLS already handled)
    m = mask[:, start_index:].float()  # exclude CLS slot if present
    denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (h[:, start_index:, :] * m.unsqueeze(-1)).sum(dim=1) / denom

# ---------------------------
# Label handling
# ---------------------------
def _finalize_goe_labels(col_values_lists: List[List[int]]) -> Tuple[np.ndarray, int]:
    # determine num classes from max label id across data
    max_id = 0
    for ids in col_values_lists:
        if ids:
            max_id = max(max_id, max(ids))
    C = int(max_id) + 1
    # Build multi-hot row-wise
    rows = []
    for ids in col_values_lists:
        row = np.zeros(C, dtype=np.float32)
        for i in ids:
            if 0 <= i < C:
                row[i] = 1.0
        rows.append(row)
    return np.stack(rows), C

def _finalize_single_labels(values: List) -> Tuple[np.ndarray, Dict[str, int]]:
    # Accept ints or strings; map strings to indices
    if len(values) == 0:
        return np.zeros((0,), dtype=np.int64), {}
    if isinstance(values[0], (int, np.integer)):
        y = np.array(values, dtype=np.int64)
        mapping = {}
    else:
        # string labels
        uniq = sorted({str(v) for v in values})
        to_idx = {name: i for i, name in enumerate(uniq)}
        y = np.array([to_idx[str(v)] for v in values], dtype=np.int64)
        mapping = to_idx
    return y, mapping

# ---------------------------
# Main build function
# ---------------------------
def build_for_dataset(
    name: str,
    ds_path: Path,
    tok: Tokenizer,
    encoder: ArdorEncoder,
    device: str,
    out_dir: Path,
    pool: str = "mean",
    batch_size: int = 256,
    max_len: int = 256,
):
    print(f"\n== {name.upper()} ==")
    # 1) Try HF datasets; fall back to Arrow; fall back to CSV->pandas if needed
    data = None
    if HAS_DATASETS:
        try:
            data = _load_with_datasets_api(ds_path)
        except Exception as e:
            print(f"[info] datasets.load_from_disk/csv failed: {e}")
    if data is None:
        try:
            data = _load_arrow_dir(ds_path)
        except Exception as e:
            print(f"[info] Arrow load failed: {e}")
            if not HAS_PANDAS:
                raise RuntimeError("No backend available to read dataset.")
            # last resort: CSV glob
            csvs = list(ds_path.glob("**/*.csv"))
            if not csvs:
                raise RuntimeError("No CSV files found either.")
            # emulate splits by filename
            data = {}
            for split in SPLIT_HINTS:
                files = [f for f in csvs if split in f.name.lower()]
                if not files: continue
                frames = [pd.read_csv(str(f)) for f in files]
                data[split if split != "valid" else "validation"] = pd.concat(frames, axis=0, ignore_index=True)

    # discover splits present
    splits = []
    for s in ["train", "validation", "test"]:
        # unify "valid" -> "validation"
        if s in data:
            splits.append(s)
        elif s == "validation" and "valid" in data:
            data["validation"] = data.pop("valid")
            splits.append("validation")
    if not splits:
        # maybe only train exists
        if "train" in data:
            splits = ["train"]

    print(f"[splits] detected: {splits}")

    # Probe columns on the first available split
    sample_iter = _as_iterable_records(data, splits[0])
    try:
        first_row, cols = next(iter(sample_iter))
    except StopIteration:
        raise RuntimeError("Dataset appears empty.")
    # If pandas dict-like row
    if hasattr(first_row, "keys"):
        columns = list(first_row.keys())
    else:
        # datasets row is already dict
        columns = list(first_row.keys())
    text_col, label_col = _guess_columns(name, columns)
    print(f"[columns] text='{text_col}'  labels='{label_col}'")

    # For multi-label (GoE), we need global class count across splits
    # Collect label arrays per split for later finalization
    label_cache = {}
    for s in splits:
        rows = []
        for row, _ in _as_iterable_records(data, s):
            rows.append(row[label_col])
        label_cache[s] = rows

    # finalize label targets per split
    if name == "goe":
        # build multi-hot per split with a shared C across splits
        all_lists = sum((label_cache[s] for s in splits), [])
        Y_all, C_goe = _finalize_goe_labels([[int(i) for i in (lst if isinstance(lst, (list, tuple)) else [])] for lst in all_lists])
        # split back
        offset = 0
        Y_split = {}
        for s in splits:
            n = len(label_cache[s])
            Y_split[s] = Y_all[offset:offset+n]
            offset += n
        label_maps = {}  # none needed for int labels
    else:
        # single label: indices; also return a mapping if strings
        Y_split, label_maps = {}, {}
        # determine mapping on the union to keep indices consistent across splits
        union_vals = []
        for s in splits:
            union_vals.extend(label_cache[s])
        y_all, mapping = _finalize_single_labels(union_vals)
        # split back
        offset = 0
        for s in splits:
            n = len(label_cache[s])
            Y_split[s] = y_all[offset:offset+n]
            offset += n
        if mapping:
            label_maps = mapping

    # process text -> pooled embeddings, per split, in batches
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "splits": {}, "pool": pool, "max_len": max_len}

    for s in splits:
        print(f"[encode] {name}:{s} …")
        texts = []
        for row, _ in _as_iterable_records(data, s):
            t = row[text_col]
            if t is None: t = ""
            texts.append(str(t))

        X_rows = []
        Y = Y_split[s]
        B = batch_size
        encoder.eval()
        with torch.no_grad():
            for i in range(0, len(texts), B):
                batch_texts = texts[i:i+B]
                ids, am = _tokenize_batch(tok, batch_texts)   # [b,T], [b,T]
                ids = ids.to(device)
                am  = am.to(device)
                # forward
                # we let the encoder handle CLS internally; we do masked mean ourselves to be robust
                h = encoder(ids, attn_mask=None, return_pooled=False)  # [b,S,H] (S = T or 1+T)
                if isinstance(h, tuple):
                    h = h[0]
                if pool == "cls" and getattr(encoder, "use_cls", False):
                    pooled = h[:, 0, :]
                else:
                    start = 1 if getattr(encoder, "use_cls", False) else 0
                    pooled = _masked_mean(h, am, start_index=start)  # [b,H]
                X_rows.append(pooled.detach().cpu().numpy())

        X = np.concatenate(X_rows, axis=0).astype(np.float32)
        # Save
        np.save(str(out_dir / f"{name}_{s}_X.npy"), X)
        np.save(str(out_dir / f"{name}_{s}_Y.npy"), Y)
        meta["splits"][s] = {"N": int(X.shape[0]), "H": int(X.shape[1])}

    # save label map if we built one (ED/MELD string classes)
    if name != "goe" and isinstance(label_maps, dict) and label_maps:
        with open(out_dir / f"{name}_label_map.json", "w", encoding="utf-8") as f:
            json.dump(label_maps, f, ensure_ascii=False, indent=2)

    # append per-dataset manifest
    with open(out_dir / f"{name}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ wrote {name.upper()} features/labels to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/Ardor/Emotions", help="Root directory with GoEmotions/ED/MELD (Arrow dirs or CSVs)")
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer_v9.json")
    ap.add_argument("--out-dir", default="/workspace/Ardor/EmotionPooled", help="Where to save *_X.npy / *_Y.npy")
    ap.add_argument("--pool", choices=["mean", "cls"], default="mean")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")

    # Encoder shape (match your current stack)
    ap.add_argument("--encoder-hidden", type=int, default=384)
    ap.add_argument("--encoder-layers", type=int, default=8)
    ap.add_argument("--encoder-heads", type=int, default=6)
    ap.add_argument("--encoder-dropout", type=float, default=0.10)
    ap.add_argument("--encoder-ckpt", default=None, help="Optional path to ArdorEncoder state_dict .pt")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    tok_path = Path(args.tokenizer).resolve()
    out_dir = Path(args.out_dir).resolve()

    # Tokenizer
    tok = _build_tokenizer(tok_path, max_len=args.max_len)

    # Encoder
    vocab_size = tok.get_vocab_size()
    use_cls = (args.pool == "cls")
    encoder = ArdorEncoder(
        vocab_size=vocab_size,
        hidden_dim=args.encoder_hidden,
        num_layers=args.encoder_layers,
        heads=args.encoder_heads,
        max_len=args.max_len,
        dropout=args.encoder_dropout,
        use_cls_token=use_cls,
        shared=None
    ).to(args.device)
    if args.encoder_ckpt and Path(args.encoder_ckpt).exists():
        sd = torch.load(args.encoder_ckpt, map_location=args.device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        print(f"[encoder] loaded ckpt; missing={len(missing)} unexpected={len(unexpected)}")

    # Find datasets
    ds = _find_datasets(root)
    if not ds:
        print(f"No recognized datasets under {root}. Expected folders matching {DS_HINTS}.", file=sys.stderr)
        sys.exit(1)

    for key in ["goe", "ed", "meld"]:
        if key not in ds:
            print(f"[warn] {key.upper()} not found under {root}; skipping.")
    for name, path in ds.items():
        build_for_dataset(
            name=name,
            ds_path=path,
            tok=tok,
            encoder=encoder,
            device=args.device,
            out_dir=out_dir,
            pool=args.pool,
            batch_size=args.batch_size,
            max_len=args.max_len,
        )

    # Global manifest
    manifest = {
        "root": str(root),
        "out_dir": str(out_dir),
        "tokenizer": str(tok_path),
        "pool": args.pool,
        "max_len": args.max_len,
        "encoder": {
            "hidden": args.encoder_hidden,
            "layers": args.encoder_layers,
            "heads":  args.encoder_heads,
            "dropout": args.encoder_dropout,
            "ckpt": args.encoder_ckpt or None
        }
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n✅ All done. Manifest saved to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
