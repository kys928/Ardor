from __future__ import annotations

from typing import Dict, Sequence, Tuple


class TemporalParietalJunction:
    """Optional perspective enrichment for clearly social/dialogic contexts."""

    def enrich(self, recent_turns: Sequence[Tuple[str, str]] | None = None, prompt: str = "") -> Dict[str, str]:
        recent_turns = list(recent_turns or [])
        text = ' '.join(str(t[1]) for t in recent_turns[-4:] if len(t) > 1)
        text = f"{text} {prompt}".lower()
        if any(word in text for word in ('you ', 'they ', 'he ', 'she ', 'friend', 'person', 'people')):
            return {'perspective_note': 'social context detected'}
        return {}
