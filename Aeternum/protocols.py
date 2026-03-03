from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Optional, Dict, Any

def clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else x

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

@dataclass
class EmotionState:
    valence: float = 0.0      # [-1, +1]
    arousal: float = 0.20     # [0, 1]
    dominance: float = 0.50   # [0, 1]
    surprise: float = 0.0     # [0, 1]
    uncertainty: float = 0.0  # [0, 1]
    anxiety: float = 0.0      # [0, 1]
    alertness: float = 0.20   # [0, 1]
    reward_tone: float = 0.0  # [-1, +1]
    safe_mode: bool = False
    stance: str = "balanced"  # supportive|cautious|analytical|balanced
    plan: str = "answer"      # answer|ask|steps|decline

    def copy(self) -> "EmotionState":
        return EmotionState(**self.__dict__)

@dataclass
class AeternumObservation:
    text: str = ""
    pooled_embedding: Optional[Any] = None  # torch.Tensor [B,H] or [H]
    last_logits: Optional[Any] = None       # torch.Tensor [V] or [1,V]
    user_feedback: Optional[float] = None   # [-1, +1]
    is_new_turn: bool = True

@dataclass
class AeternumDecision:
    state: EmotionState
    token_bias: Dict[int, float] = field(default_factory=dict)
    temperature_scale: float = 1.0
    top_p_scale: float = 1.0
    rep_penalty_scale: float = 1.0

class AeternumModule(Protocol):
    def reset(self) -> None: ...
    def observe(self, obs: AeternumObservation, state: EmotionState) -> None: ...
    def step(self, state: EmotionState) -> EmotionState: ...
    def export_bias(self, tokenizer) -> Dict[int, float]:
        return {}
