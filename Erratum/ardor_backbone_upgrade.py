#!/usr/bin/env python
# ardor_backbone_upgrade.py  (fixed FFN shapes)
import torch, re
from collections import OrderedDict

SRC = r"../Cerebrum/Models/Ardor/Ardor_IV_corefixed.pt"
DST = r"../Cerebrum/Models/Ardor/Ardor_V.pt"

H_OLD, H_NEW = 256, 384
L_OLD, L_NEW = 6, 8
NOISE_STD    = 0.02

print("🔑  loading", SRC)
old = torch.load(SRC, map_location="cpu")
if isinstance(old, dict) and "model_state_dict" in old:
    old = old["model_state_dict"]

# -------- helpers -------------------------------------------------
def resize_matrix(w):
    """Pad a (R,C) tensor where R = H_OLD*fac_r, C = H_OLD*fac_c."""
    R, C = w.shape
    fac_r = R // H_OLD
    fac_c = C // H_OLD
    Rn, Cn = fac_r * H_NEW, fac_c * H_NEW
    out = torch.zeros(Rn, Cn, dtype=w.dtype)
    out[:R, :C] = w
    if Cn > C:
        out[:, C:] = torch.randn(Rn, Cn - C) * NOISE_STD
    if Rn > R:
        out[R:, :] = torch.randn(Rn - R, Cn) * NOISE_STD
    return out

def resize_vector(v):
    """Pad a 1-D bias whose length = H_OLD*fac  → H_NEW*fac."""
    L = v.shape[0]
    fac = L // H_OLD            # 1 for attn/norm, 4 for FFN
    Ln = fac * H_NEW
    out = torch.zeros(Ln, dtype=v.dtype)
    out[:L] = v
    if Ln > L:
        out[L:] = torch.randn(Ln - L) * NOISE_STD
    return out

# -------- new state-dict -----------------------------------------
new = OrderedDict()

# 1. embeddings / lm_head
tok = old["token_embed.weight"]              # (V,256)
tok_384 = resize_matrix(tok)
new["token.weight"]   = tok_384
new["lm_head.weight"] = tok_384              # tied

# 2. RoPE placeholder
new["pos"] = torch.zeros(1, 2048, H_NEW)

# 3. map first 6 transformer layers
pat = re.compile(r"layers\.(\d+)\.(.+)")
for k, w in old.items():
    m = pat.match(k)
    if not m: continue
    idx, sub = int(m.group(1)), m.group(2)
    if idx >= L_OLD: continue

    tgt = f"blocks.{idx}."
    if sub.startswith("attention"):
        sub = sub.replace("attention.", "")
        tgt += "attn."
        for src, rep in [("query","q"),("key","k"),
                         ("value","v"),("out","out")]:
            if sub.startswith(src):
                tgt += sub.replace(src, rep); break
    else:
        tgt += sub  # norm.*, ff.*

    new[tgt] = resize_matrix(w) if w.ndim==2 else resize_vector(w)

# 4. init blocks 6 & 7
for blk in (6, 7):
    for k, v in list(new.items()):
        m = re.match(r"blocks\.5\.(.*)", k)
        if m:
            new[f"blocks.{blk}.{m.group(1)}"] = v + torch.randn_like(v)*0.01

# 5. final layernorm
new["norm.weight"] = resize_vector(old["norm.weight"])
new["norm.bias"]   = resize_vector(old["norm.bias"])

print(f"✅  converted tensors: {len(new)}")
torch.save(new, DST)
print("💾  saved to", DST)
