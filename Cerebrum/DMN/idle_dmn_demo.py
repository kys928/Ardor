from __future__ import annotations

import json
from typing import Any


def render_dmn_state(state: Any) -> str:
    if hasattr(state, 'to_dict'):
        state = state.to_dict()
    return json.dumps(state, ensure_ascii=False, indent=2)
