from __future__ import annotations
from .protocols import EmotionState, AeternumObservation, AeternumModule, clamp

class VentralStriatum(AeternumModule):
    def __init__(self):
        self.running_reward = 0.0

    def reset(self): self.running_reward = 0.0

    def observe(self, obs: AeternumObservation, state: EmotionState):
        if obs.user_feedback is not None:
            self.running_reward = 0.9*self.running_reward + 0.1*float(obs.user_feedback)

    def step(self, state: EmotionState) -> EmotionState:
        st = state.copy()
        st.reward_tone = clamp(self.running_reward, -1.0, +1.0)
        return st
