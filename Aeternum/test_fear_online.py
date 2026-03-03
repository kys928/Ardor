# test_fear_online.py — probe Aeternum with pooled embeddings instead of dummy vectors

from Aeternum.AeternumCore import AeternumCore, AeternumConfig
from pathlib import Path
import numpy as np
import torch

# Where your pooled embeddings live
DATASET_DIR = Path(__file__).with_name("Dataset")  # Aeternum/Dataset

def load_any_pooled_embedding(row_idx: int = 0) -> torch.Tensor:
    """
    Load one pooled embedding from the *_X.npy files under Aeternum/Dataset.

    Assumes files like:
        goe_train_X.npy, goe_val_X.npy, meld_train_X.npy, ...
    and each has shape [N, 384].
    """
    # Find any *_X.npy file
    candidates = sorted(DATASET_DIR.glob("*_X.npy"))
    if not candidates:
        raise FileNotFoundError(f"No *_X.npy files found in {DATASET_DIR}")

    path = candidates[0]
    arr = np.load(path)  # shape [N, 384] (or similar)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")

    if row_idx < 0 or row_idx >= arr.shape[0]:
        raise IndexError(f"row_idx {row_idx} out of range for {path} with {arr.shape[0]} rows")

    vec = arr[row_idx]         # shape [384]
    emb = torch.from_numpy(vec).float()  # to torch
    return emb

cfg = AeternumConfig(device="cpu", hidden_dim=384, prefer_snn=True)
core = AeternumCore(cfg)

def probe(text: str, row_idx: int = 0):
    emb = load_any_pooled_embedding(row_idx=row_idx)
    dec = core.update(
        text=text,
        pooled_embedding=emb,
        is_new_turn=True,
    )
    st = dec.state
    print(f"\nTEXT: {text}")
    print("  anxiety :", st.anxiety)
    print("  surprise:", st.surprise)
    print("  stance  :", st.stance)

if __name__ == "__main__":
    # Row index is arbitrary here; you can play with it later
    probe("I keep having panic attacks and I feel like I'm going to die.", row_idx=0)
    probe("Weather is great, I'm just chilling today.", row_idx=1)
    probe("THERE IS A BOMB, EVERYONE RUN NOW!!!", row_idx=2)
