from __future__ import annotations
from .protocols import EmotionState, AeternumModule

class AnteriorCingulate(AeternumModule):
    def __init__(self, threshold=1.25):
        self.th = float(threshold)

    def reset(self): pass
    def observe(self, obs, state): pass

    def step(self, state: EmotionState) -> EmotionState:
        st = state.copy()
        risk = st.anxiety + st.uncertainty + 0.5*st.surprise
        st.safe_mode = risk >= self.th
        return st
