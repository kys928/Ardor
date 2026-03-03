from __future__ import annotations
from .protocols import EmotionState, AeternumModule

class VentroMedialPFC(AeternumModule):
    def reset(self):
        pass

    def observe(self, obs, state):
        pass

    def step(self, state: EmotionState) -> EmotionState:
        st = state.copy()

        # 1) Hard safety: high anxiety + uncertainty => stay cautious
        if st.anxiety + st.uncertainty > 0.9:
            st.stance = "cautious"
            return st

        # 2) Positive, low-arousal => supportive
        if st.valence > 0.2 and st.arousal < 0.5:
            st.stance = "supportive"
            return st

        # 3) Negative but controlled => analytical
        if st.valence < -0.2 and st.dominance > 0.6:
            st.stance = "analytical"
            return st

        # 4) Calm baseline: if anxiety is low, stop being cautious
        if st.anxiety < 0.2 and st.uncertainty < 0.4:
            st.stance = "balanced"

        return st
