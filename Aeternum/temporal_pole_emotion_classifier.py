from __future__ import annotations
import torch, torch.nn as nn
from .protocols import EmotionState, AeternumObservation, AeternumModule

class EmotionHead(nn.Module):
    def __init__(self, hidden_dim=384):
        super().__init__()
        h = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, 2*h), nn.GELU(),
            nn.Linear(2*h, h), nn.GELU(),
            nn.Linear(h, 4),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1: x = x.unsqueeze(0)
        return self.net(x).mean(dim=0)

class TemporalPoleEmotionClassifier(AeternumModule):
    def __init__(self, hidden_dim=384, device="cpu"):
        self.head = EmotionHead(hidden_dim).to(device)
        self.device = device
        self.last = None

    def reset(self): self.last = None

    def observe(self, obs: AeternumObservation, state: EmotionState):
        if obs.pooled_embedding is None:
            return
        x = obs.pooled_embedding
        if hasattr(x, "to"): x = x.to(self.device)
        with torch.no_grad():
            out = self.head(x)
        v, a, d, s = out.tolist()
        v = max(-1.0, min(1.0, v))
        a = max(0.0, min(1.0, a))
        d = max(0.0, min(1.0, d))
        s = max(0.0, min(1.0, s))
        self.last = (v, a, d, s)

    def step(self, state: EmotionState) -> EmotionState:
        st = state.copy()
        if self.last is None:
            return st
        v, a, d, s = self.last
        st.valence    = 0.85*st.valence    + 0.15*v
        st.arousal    = 0.85*st.arousal    + 0.15*a
        st.dominance  = 0.90*st.dominance  + 0.10*d
        st.surprise   = 0.80*st.surprise   + 0.20*s
        return st
