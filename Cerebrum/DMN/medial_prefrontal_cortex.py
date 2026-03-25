from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence


_STOP = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'so', 'because', 'as', 'of', 'in', 'on', 'for', 'to',
    'from', 'by', 'with', 'about', 'into', 'over', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it',
    'this', 'that', 'i', 'you', 'we', 'they', 'he', 'she', 'them', 'my', 'your', 'our'
}


def _kw(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (text or '').lower())
    return [t for t in toks if t not in _STOP]


class MedialPrefrontalCortex:
    """Maintains a modest structured self-context for the DMN."""

    def update_state(
        self,
        previous_state: Dict[str, Any] | None,
        *,
        seed: str,
        retrieved_traces: Sequence[str] | None = None,
        recent_texts: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        previous_state = dict(previous_state or {})
        retrieved_traces = list(retrieved_traces or [])
        recent_texts = list(recent_texts or [])

        bag: List[str] = []
        bag.extend(_kw(seed))
        for txt in retrieved_traces[:3]:
            bag.extend(_kw(txt))
        for txt in recent_texts[-3:]:
            bag.extend(_kw(txt))

        themes = [w for w, _ in Counter(bag).most_common(4)]
        continuity = themes[:2]
        unresolved = previous_state.get('unresolved_topics') or []
        if seed:
            unresolved = list(dict.fromkeys(list(unresolved) + [seed[:96]]))[-3:]

        confidence = min(0.95, 0.25 + 0.15 * len(retrieved_traces) + 0.05 * len(continuity))
        uncertainty = max(0.0, 1.0 - confidence)

        return {
            'current_themes': themes,
            'continuity_anchors': continuity,
            'unresolved_topics': unresolved,
            'confidence': round(confidence, 3),
            'uncertainty': round(uncertainty, 3),
            'narrative_style_preference': 'reflective-compact',
        }
