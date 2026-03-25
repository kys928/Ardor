from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List


_STOP = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'so', 'because', 'as', 'of', 'in', 'on', 'for', 'to',
    'from', 'by', 'with', 'about', 'into', 'over', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it',
    'this', 'that', 'i', 'you', 'we', 'they', 'he', 'she', 'them', 'my', 'your', 'our'
}


def _keywords(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (text or '').lower())
    return [t for t in toks if t not in _STOP]


class QuietMind:
    """Builds a low-noise baseline from recent text only."""

    def build_baseline(self, recent_texts: List[str] | None = None, self_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        recent_texts = recent_texts or []
        self_state = self_state or {}
        bag: List[str] = []
        for txt in recent_texts[-6:]:
            bag.extend(_keywords(txt))
        common = [w for w, _ in Counter(bag).most_common(3)]
        baseline_theme = ', '.join(common) if common else 'recent context'
        unresolved = list(self_state.get('unresolved_topics') or [])
        return {
            'baseline_theme_prior': baseline_theme,
            'recent_topic_prior': common,
            'uncertainty_marker': 0.15 if unresolved else 0.05,
            'narrative_noise': min(1.0, len(set(bag)) / max(1, len(bag))) if bag else 0.0,
        }
