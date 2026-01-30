from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import time, heapq
from .protocols import EmotionState, AeternumObservation, AeternumModule

@dataclass
class Episode:
    ts: float
    text: str
    valence: float
    arousal: float
    surprise: float
    score: float

class Hippocampus(AeternumModule):
    def __init__(self, capacity=256):
        self.capacity = int(capacity)
        self.buf: List[Tuple[float, Episode]] = []

    def reset(self): self.buf.clear()

    def observe(self, obs: AeternumObservation, state: EmotionState):
        if not obs.text: return
        sc = state.arousal*(0.6+0.4*abs(state.valence)) + 0.5*state.surprise
        ep = Episode(time.time(), obs.text[:300], state.valence, state.arousal, state.surprise, sc)
        if len(self.buf) < self.capacity:
            heapq.heappush(self.buf, (sc, ep))
        else:
            if sc > self.buf[0][0]:
                heapq.heapreplace(self.buf, (sc, ep))

    def step(self, state: EmotionState) -> EmotionState:
        return state

    def sample_for_rem(self, k=8) -> List[Episode]:
        return [e for _, e in sorted(self.buf, key=lambda x: -x[0])[:k]]
