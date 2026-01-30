from __future__ import annotations
from .protocols import EmotionState, AeternumModule, clamp01
import torch

class HomeostaticPlasticity(AeternumModule):
    def __init__(self, target_arousal=0.35, target_anxiety=0.20,
                 adapt_thresholds=True, lr=0.01,
                 theta_bounds=(0.5, 3.0), thr_bounds=(0.30, 0.80)):
        self.ta = float(target_arousal)
        self.tx = float(target_anxiety)
        self.adapt_thresholds = bool(adapt_thresholds)
        self.lr = float(lr)
        self.theta_lo, self.theta_hi = theta_bounds
        self.thr_lo, self.thr_hi = thr_bounds
        self._amyg = None
        self._alarm_ema = 0.0
        self._uncert_ema = 0.0
        self._reward_ema = 0.0

    def bind_amygdala(self, amyg):
        self._amyg = amyg

    def reset(self):
        self._alarm_ema = 0.0; self._uncert_ema = 0.0; self._reward_ema = 0.0

    def observe(self, obs, state):
        # track uncertainty & reward drift for adaptation heuristics
        self._uncert_ema = 0.99*self._uncert_ema + 0.01*state.uncertainty
        self._reward_ema = 0.99*self._reward_ema + 0.01*state.reward_tone
        if self._amyg is not None:
            self._alarm_ema = 0.99*self._alarm_ema + 0.01*float(self._amyg.last_spike)

    def step(self, state: EmotionState) -> EmotionState:
        st = state.copy()
        # vanilla affect centering
        st.arousal = clamp01( st.arousal + 0.05*(self.ta - st.arousal) )
        st.anxiety = clamp01( st.anxiety + 0.05*(self.tx - st.anxiety) )

        # threshold adaptation (slow)
        if self.adapt_thresholds and (self._amyg is not None):
            with torch.no_grad():
                # over-sensitive: many alarms and negative reward -> increase thresholds
                if self._alarm_ema > 0.25 and self._reward_ema < -0.1:
                    k = 1.0 + self.lr
                    if hasattr(self._amyg, "theta1"): self._amyg.theta1.data.mul_(k).clamp_(self.theta_lo, self.theta_hi)
                    if hasattr(self._amyg, "theta2"): self._amyg.theta2.data.mul_(k).clamp_(self.theta_lo, self.theta_hi)
                    if hasattr(self._amyg, "thr_rate"):
                        self._amyg.thr_rate = float(min(self.thr_hi, self._amyg.thr_rate + 0.01))
                # under-sensitive: few alarms, high uncertainty, negative reward -> lower thresholds
                if self._alarm_ema < 0.05 and self._uncert_ema > 0.5 and self._reward_ema < -0.05:
                    k = 1.0 - self.lr
                    if hasattr(self._amyg, "theta1"): self._amyg.theta1.data.mul_(k).clamp_(self.theta_lo, self.theta_hi)
                    if hasattr(self._amyg, "theta2"): self._amyg.theta2.data.mul_(k).clamp_(self.theta_lo, self.theta_hi)
                    if hasattr(self._amyg, "thr_rate"):
                        self._amyg.thr_rate = float(max(self.thr_lo, self._amyg.thr_rate - 0.01))

        return st
