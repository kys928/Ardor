# Aeternum/mtl_emotion_classifier.py
from __future__ import annotations
from dataclasses import replace
from typing import Optional

import numpy as np
import torch

from .protocols import EmotionState, AeternumObservation, AeternumModule, clamp01
from .emotion_heads_runtime import load_emotion_heads_mtl


class MTLEmotionClassifier(AeternumModule):
    """
    Temporal pole-style emotion classifier backed by the MTL head:
    - GoEmotions (multi-label)
    - EmpatheticDialogues (single-label)
    - MELD (single-label)

    For now it mainly drives arousal + surprise using:
      - "intensity" ~ mean GoE activation
      - "surprise/uncertainty" ~ entropy of ED distribution

    Later you can bolt on a proper label->VAD mapping and let this
    also steer valence/dominance.
    """

    def __init__(
        self,
        hidden_dim: int,
        device: str | torch.device = "cpu",
        ckpt_path: Optional[str] = None,
        alpha_arousal: float = 0.15,
        alpha_surprise: float = 0.20,
        decay: float = 0.85,
    ):
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.alpha_arousal = float(alpha_arousal)
        self.alpha_surprise = float(alpha_surprise)
        self.decay = float(decay)

        self.model, meta = load_emotion_heads_mtl(device=self.device, ckpt_path=ckpt_path)
        dims = meta["dims"]
        H = int(dims["H"])

        if H != hidden_dim:
            # We can still run, but it's worth warning in logs.
            print(f"[MTLEmotionClassifier] Warning: pooled dim={hidden_dim}, "
                  f"MTL head expects H={H}. Using H={H} and trusting caller.")

        self._last_x: Optional[torch.Tensor] = None

    # ---------------- core API ----------------
    def reset(self) -> None:
        self._last_x = None

    def observe(self, obs: AeternumObservation, state: EmotionState) -> None:
        if obs.pooled_embedding is None:
            return
        x = torch.as_tensor(obs.pooled_embedding, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1,H]
        self._last_x = x

    def step(self, state: EmotionState) -> EmotionState:
        if self._last_x is None:
            return state

        with torch.no_grad():
            lg_goe, lg_ed, lg_meld = self.model(self._last_x)  # [1,C*]

            # ---- probabilities ----
            p_goe = self.model.probs_goe(lg_goe)[0].cpu().numpy()    # [C_goe]
            p_ed = self.model.probs_softmax(lg_ed, head="ed")[0].cpu().numpy()     # [C_ed]
            # p_meld currently unused, but available:
            # p_meld = self.model.probs_softmax(lg_meld, head="meld")[0].cpu().numpy()

        # ---------- crude but safe mapping to Aeternum ----------

        # 1) "Intensity" of emotion: how many GoE labels are lit up on average
        #    (neutral/flat utterances tend to have very low mean activation).
        intensity = float(p_goe.mean())  # ~ [0, 1]

        # 2) Uncertainty / surprise from ED entropy
        #    entropy is max when distribution is flat, min when peaked
        eps = 1e-8
        ent = float(-(p_ed * np.log(p_ed + eps)).sum())
        ent_max = np.log(len(p_ed) + eps)
        # normalized 0..1 (0 = perfectly certain, 1 = maximally uncertain)
        ent_norm = float(ent / (ent_max + eps))
        # Use "surprise" as "how peaky" => 1 - entropy
        surprise_raw = 1.0 - ent_norm

        # Smoothly update arousal & surprise; do NOT smash valence yet
        new_arousal = clamp01(self.decay * state.arousal + self.alpha_arousal * intensity)
        new_surprise = clamp01(self.decay * state.surprise + self.alpha_surprise * surprise_raw)

        # Keep valence/dominance as other modules define them for now.
        return replace(state, arousal=new_arousal, surprise=new_surprise)
