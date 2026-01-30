from __future__ import annotations
import re
from .protocols import EmotionState, AeternumObservation, AeternumModule, clamp01

CAPS_RE = re.compile(r"[A-Z]{4,}")
EXCL_RE = re.compile(r"!{2,}")
URGENT = {"urgent","help","asap","now","immediately","emergency","please","panic"}

class AnteriorInsula(AeternumModule):
    def __init__(self, weight_caps=0.15, weight_excl=0.10, weight_urgent=0.20):
        self.wc, self.we, self.wu = float(weight_caps), float(weight_excl), float(weight_urgent)
        self.last_signal = 0.0

    def reset(self): self.last_signal = 0.0

    def observe(self, obs: AeternumObservation, state: EmotionState):
        t = obs.text or ""
        caps = 1.0 if CAPS_RE.search(t) else 0.0
        excl = 1.0 if EXCL_RE.search(t) else 0.0
        urg = 1.0 if any(w in t.lower() for w in URGENT) else 0.0
        s = self.wc*caps + self.we*excl + self.wu*urg
        self.last_signal = clamp01(s)

    def step(self, state: EmotionState) -> EmotionState:
        st = state.copy()
        st.arousal = clamp01(0.85*st.arousal + 0.15*self.last_signal)
        return st
