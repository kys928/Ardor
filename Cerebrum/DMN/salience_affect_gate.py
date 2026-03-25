from __future__ import annotations

from typing import Any, Dict, Sequence


class SalienceAffectGate:
    """Produces simple salience weights from available context and optional Aeternum state."""

    def score(
        self,
        *,
        seed: str,
        self_state: Dict[str, Any] | None = None,
        retrieved_traces: Sequence[str] | None = None,
        aet_state: Any = None,
    ) -> Dict[str, float]:
        self_state = self_state or {}
        retrieved_traces = list(retrieved_traces or [])

        confidence = float(self_state.get('confidence', 0.4) or 0.4)
        uncertainty = float(self_state.get('uncertainty', 0.6) or 0.6)
        novelty = 0.2 if not retrieved_traces else min(0.9, 0.25 + 0.1 * len(retrieved_traces))
        continuity = min(1.0, 0.35 + 0.15 * len(self_state.get('continuity_anchors') or []))
        emotional_activation = 0.0

        if aet_state is not None:
            for attr in ('arousal', 'activation', 'intensity', 'salience'):
                try:
                    val = getattr(aet_state, attr, None)
                    if val is not None:
                        emotional_activation = max(emotional_activation, abs(float(val)))
                except Exception:
                    pass
            if isinstance(aet_state, dict):
                for key in ('arousal', 'activation', 'intensity', 'salience'):
                    try:
                        if key in aet_state:
                            emotional_activation = max(emotional_activation, abs(float(aet_state[key])))
                    except Exception:
                        pass

        self_relevance = min(1.0, 0.25 + 0.05 * len(seed.split()) + 0.1 * len(retrieved_traces))
        return {
            'self_relevance': round(self_relevance, 3),
            'uncertainty': round(uncertainty, 3),
            'emotional_activation': round(min(1.0, emotional_activation), 3),
            'novelty': round(novelty, 3),
            'continuity_importance': round(continuity, 3),
            'confidence': round(confidence, 3),
        }
