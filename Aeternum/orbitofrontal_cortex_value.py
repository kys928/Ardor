from __future__ import annotations
from .protocols import EmotionState, AeternumModule

class OrbitofrontalValue(AeternumModule):
    def reset(self): pass
    def observe(self, obs, state): pass

    def step(self, state: EmotionState) -> EmotionState:
        st = state.copy()
        u_ask   = 0.3 + 0.4*st.uncertainty + 0.2*st.alertness
        u_steps = 0.4 + 0.2*st.dominance - 0.2*st.uncertainty
        u_ans   = 0.5 + 0.2*st.valence - 0.2*st.anxiety
        u_decl  = 0.2 + 0.4*st.anxiety + 0.3*st.uncertainty
        plan = max([("ask",u_ask), ("steps",u_steps), ("answer",u_ans), ("decline",u_decl)], key=lambda x: x[1])[0]
        st.plan = plan
        return st
