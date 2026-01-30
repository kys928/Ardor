from __future__ import annotations
from .protocols import EmotionState, AeternumModule, clamp01

class BNST(AeternumModule):
    def __init__(self, rise=0.08, decay=0.96):
        self.rise = float(rise); self.decay = float(decay)
        self.drive = 0.0

    def reset(self): self.drive = 0.0

    def observe(self, obs, state):
        self.drive = 0.8*self.drive + 0.2*max(state.uncertainty, state.surprise*0.5)

    def step(self, state: EmotionState) -> EmotionState:
        st = state.copy()
        st.anxiety = clamp01(self.decay*st.anxiety + self.rise*self.drive)
        return st
