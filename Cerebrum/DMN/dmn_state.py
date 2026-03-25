from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class DMNMode(str, Enum):
    DISABLED = "DISABLED"
    IDLE = "IDLE"
    ACTIVE_SUMMARY = "ACTIVE_SUMMARY"
    ERROR = "ERROR"


@dataclass
class NarrativeTakeaway:
    theme: str = ""
    summary: str = ""
    salience_score: float = 0.0
    confidence: float = 0.0
    memory_candidates: List[str] = field(default_factory=list)
    source_mode: str = DMNMode.DISABLED.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DMNEvent:
    event_type: str
    timestamp: float = field(default_factory=lambda: time.time())
    mode: str = DMNMode.DISABLED.value
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DMNState:
    mode: DMNMode = DMNMode.DISABLED
    last_seed: str = ""
    last_retrieved_traces: List[str] = field(default_factory=list)
    current_narrative_takeaway: Optional[NarrativeTakeaway] = None
    self_model_state: Dict[str, Any] = field(default_factory=dict)
    salience_info: Dict[str, Any] = field(default_factory=dict)
    cycle_count: int = 0
    last_updated: float = field(default_factory=lambda: time.time())
    last_error: Optional[str] = None
    is_running: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["mode"] = self.mode.value if isinstance(self.mode, DMNMode) else str(self.mode)
        return out


def to_plain_data(obj: Any) -> Any:
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return obj.to_dict()
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_plain_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain_data(v) for v in obj]
    return obj
