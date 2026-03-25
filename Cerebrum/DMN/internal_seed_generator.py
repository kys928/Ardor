from __future__ import annotations

from typing import Any, Dict, Sequence


class InternalSeedGenerator:
    """Generates a short reflective seed without becoming a second chatbot."""

    def generate(
        self,
        unresolved_themes: Sequence[str] | None = None,
        recent_turns: Sequence[Any] | None = None,
        recent_texts: Sequence[str] | None = None,
        system_state: Dict[str, Any] | None = None,
    ) -> str:
        unresolved_themes = [str(x).strip() for x in (unresolved_themes or []) if str(x).strip()]
        recent_texts = [str(x).strip() for x in (recent_texts or []) if str(x).strip()]
        system_state = system_state or {}
        baseline = str(system_state.get('baseline_theme_prior') or '').strip()

        if unresolved_themes:
            return f"Revisit unresolved topic: {unresolved_themes[0]}."
        if baseline:
            return f"Stay centered on {baseline}."
        if recent_texts:
            last = recent_texts[-1].strip().replace('\n', ' ')
            last = last[:80].rstrip(' .,;:')
            return f"Summarize the thread around: {last}."
        if recent_turns:
            return "Reflect on the most recent exchange."
        return "Maintain a quiet summary of the current line of thought."
