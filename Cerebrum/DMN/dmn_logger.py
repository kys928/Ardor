from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .dmn_state import DMNEvent, DMNState, NarrativeTakeaway, to_plain_data


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class DMNLogger:
    """Append-only JSONL logger for DMN activity."""

    def __init__(self, log_path: Optional[str] = None):
        default_path = _repo_root() / 'artifacts' / 'logs' / 'dmn_events.jsonl'
        self.log_path = Path(log_path or os.environ.get('ARDOR_DMN_LOG', str(default_path)))
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _append(self, payload: Dict[str, Any]) -> bool:
        try:
            with self.log_path.open('a', encoding='utf-8') as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + '\n')
            return True
        except Exception:
            return False

    def log_event(self, event: DMNEvent) -> bool:
        payload = event.to_dict() if hasattr(event, 'to_dict') else to_plain_data(event)
        payload.setdefault('logged_at', time.time())
        return self._append(payload)

    def log_summary(self, state: DMNState, takeaway: NarrativeTakeaway, event_type: str) -> bool:
        payload = {
            'event_type': event_type,
            'timestamp': time.time(),
            'mode': state.mode.value if hasattr(state.mode, 'value') else str(state.mode),
            'state': state.to_dict() if hasattr(state, 'to_dict') else to_plain_data(state),
            'takeaway': takeaway.to_dict() if hasattr(takeaway, 'to_dict') else to_plain_data(takeaway),
        }
        return self._append(payload)
