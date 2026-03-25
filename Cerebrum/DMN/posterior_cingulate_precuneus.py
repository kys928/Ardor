from __future__ import annotations

from typing import Any, Dict, Sequence

from .dmn_state import DMNMode, NarrativeTakeaway


class PosteriorCingulatePrecuneus:
    """Builds the compact DMN narrative takeaway."""

    def integrate(
        self,
        *,
        mode: DMNMode,
        quiet_state: Dict[str, Any],
        seed: str,
        retrieved_traces: Sequence[str],
        self_state: Dict[str, Any],
        salience: Dict[str, float],
        tpj_context: Dict[str, str] | None = None,
    ) -> NarrativeTakeaway:
        if self_state.get('current_themes'):
            theme = ', '.join(self_state['current_themes'][:2])
        elif quiet_state.get('baseline_theme_prior'):
            theme = str(quiet_state.get('baseline_theme_prior'))
        else:
            theme = 'current context'

        trace_hint = ''
        if retrieved_traces:
            trace_hint = f" Recalled: {retrieved_traces[0][:96]}."
        perspective = ''
        if tpj_context and tpj_context.get('perspective_note'):
            perspective = f" {tpj_context['perspective_note']}."

        summary = f"Theme: {theme}. Seed: {seed}{trace_hint}{perspective}".strip()
        summary = summary[:320].rstrip()

        confidence = float(self_state.get('confidence', salience.get('confidence', 0.4)))
        salience_score = max(
            float(salience.get('self_relevance', 0.0)),
            float(salience.get('continuity_importance', 0.0)),
            float(salience.get('emotional_activation', 0.0)),
        )

        memory_candidates = []
        if confidence >= 0.45:
            memory_candidates = [seed[:120]]
            if retrieved_traces:
                memory_candidates.append(retrieved_traces[0][:120])

        return NarrativeTakeaway(
            theme=theme,
            summary=summary,
            salience_score=round(min(1.0, salience_score), 3),
            confidence=round(min(1.0, confidence), 3),
            memory_candidates=memory_candidates[:2],
            source_mode=mode.value,
        )
