from __future__ import annotations
import math, torch
from .protocols import EmotionState, AeternumObservation, AeternumModule, clamp01

def _entropy_from_logits(logits: torch.Tensor) -> float:
    if logits.ndim == 2: logits = logits[0]
    p = torch.softmax(logits.detach(), dim=-1)
    h = -(p * torch.where(p>0, p.log(), torch.zeros_like(p))).sum().item()
    return h / max(1e-6, math.log(max(2, p.numel())))

class LocusCoeruleus(AeternumModule):
    def __init__(self, k=0.9):
        self.k = float(k)
        self.last_uncert = 0.0

    def reset(self): self.last_uncert = 0.0

    def observe(self, obs: AeternumObservation, state: EmotionState):
        if obs.last_logits is not None:
            try:
                self.last_uncert = clamp01(_entropy_from_logits(obs.last_logits))
            except Exception:
                self.last_uncert = 0.0

    def step(self, state: EmotionState) -> EmotionState:
        st = state.copy()
        st.uncertainty = clamp01(0.8*st.uncertainty + 0.2*self.last_uncert)
        st.alertness = clamp01(self.k*st.uncertainty + 0.1*st.surprise + 0.1*st.anxiety)
        return st
