# Aeternum/emotion_heads_runtime.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn


class Trunk(nn.Module):
    """Same trunk as in train_emotion.py, but inference-only."""
    def __init__(self, in_dim: int, hidden: int = 512, drop: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmotionHeadsMTL(nn.Module):
    """
    Inference-only version of the multi-task head you trained in train_emotion.py.
    Outputs:
      - GoEmotions logits [B, C_goe]
      - ED logits         [B, C_ed]
      - MELD logits       [B, C_meld]
    """
    def __init__(self, in_dim: int, c_goe: int, c_ed: int, c_meld: int,
                 hidden: int = 512, drop: float = 0.10):
        super().__init__()
        self.trunk = Trunk(in_dim, hidden=hidden, drop=drop)
        self.goe_head = nn.Linear(hidden, c_goe)
        self.ed_head = nn.Linear(hidden, c_ed)
        self.meld_head = nn.Linear(hidden, c_meld)

        # Temperatures are buffers (loaded from ckpt)
        self.register_buffer("T_goe", torch.tensor(1.0))
        self.register_buffer("T_ed", torch.tensor(1.0))
        self.register_buffer("T_meld", torch.tensor(1.0))

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        lg_goe = self.goe_head(h)
        lg_ed = self.ed_head(h)
        lg_meld = self.meld_head(h)
        return lg_goe, lg_ed, lg_meld

    # ---- probability helpers (with temperature) ----
    def probs_goe(self, logits: torch.Tensor) -> torch.Tensor:
        T = self.T_goe.clamp(min=1e-6)
        return torch.sigmoid(logits / T)

    def probs_softmax(self, logits: torch.Tensor, head: str = "ed") -> torch.Tensor:
        if head == "ed":
            T = self.T_ed
        else:
            T = self.T_meld
        T = T.clamp(min=1e-6)
        return torch.softmax(logits / T, dim=-1)


def load_emotion_heads_mtl(
    device: torch.device | str = "cpu",
    ckpt_path: Optional[str] = None,
) -> Tuple[EmotionHeadsMTL, dict]:
    """
    Load the checkpoint you trained with train_emotion.py.

    Returns:
      model  - EmotionHeadsMTL on the requested device
      meta   - dict with dims, temps, args
    """
    root = Path(__file__).resolve().parent

    candidates = []
    if ckpt_path is not None:
        candidates.append(Path(ckpt_path))
    candidates.extend([
        root / "Models" / "emotion_heads_mtl.pt",
        root / "checkpoints" / "emotion_heads_mtl.pt",
    ])

    ckpt_file = None
    for c in candidates:
        if c.is_file():
            ckpt_file = c
            break

    if ckpt_file is None:
        raise FileNotFoundError(
            f"emotion_heads_mtl.pt not found. Tried: {', '.join(str(c) for c in candidates)}"
        )

    try:
        try:
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_file, map_location=device)
        dims = ckpt["dims"]
        H = int(dims["H"])
        C_goe = int(dims["C_goe"])
        C_ed = int(dims["C_ed"])
        C_meld = int(dims["C_meld"])
        args = ckpt.get("args", {})
        hidden = int(args.get("hidden", 512))
        drop = float(args.get("dropout", 0.10))
        temps = ckpt.get("temps", {})
    except Exception as e:
        print(f"[emotion_heads_runtime] WARNING: failed to load {ckpt_file}: {e}. Using random-init fallback head.")
        ckpt = None
        H, C_goe, C_ed, C_meld = 384, 28, 32, 7
        args = {"hidden": 512, "dropout": 0.10}
        hidden = 512
        drop = 0.10
        temps = {}
    dims = {"H": H, "C_goe": C_goe, "C_ed": C_ed, "C_meld": C_meld}

    model = EmotionHeadsMTL(
        in_dim=H,
        c_goe=C_goe,
        c_ed=C_ed,
        c_meld=C_meld,
        hidden=hidden,
        drop=drop,
    )
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        if "T_goe" in temps:
            model.T_goe.fill_(float(temps["T_goe"]))
        if "T_ed" in temps:
            model.T_ed.fill_(float(temps["T_ed"]))
        if "T_meld" in temps:
            model.T_meld.fill_(float(temps["T_meld"]))

    model.to(device)
    model.eval()
    return model, {"dims": dims, "temps": temps, "args": args}
