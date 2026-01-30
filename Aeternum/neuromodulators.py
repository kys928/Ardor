from __future__ import annotations
from typing import Dict
from .protocols import EmotionState

class Neuromodulators:
    def compute_decoding_scales(self, st: EmotionState) -> Dict[str, float]:
        temp = max(0.85, 1.0 - 0.20*st.alertness - 0.10*st.anxiety)
        top_p = max(0.85, 1.0 - 0.20*st.uncertainty)
        rep = max(0.90, 1.0 - 0.10*max(0.0, st.reward_tone))
        return {"temperature_scale": temp, "top_p_scale": top_p, "rep_penalty_scale": rep}
