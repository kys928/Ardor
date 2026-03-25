from __future__ import annotations

from typing import Any, List, Sequence


class MedialTemporalLobe:
    """Thin wrapper over Ardor's existing retrieval path."""

    def retrieve(
        self,
        prompt: str,
        *,
        parietal: Any = None,
        retrieval_enabled: bool = False,
        recent_turns: Sequence[Any] | None = None,
        recent_texts: Sequence[str] | None = None,
        k: int = 3,
    ) -> List[str]:
        traces: List[str] = []

        if retrieval_enabled and parietal is not None:
            try:
                if hasattr(parietal, 'encode') and hasattr(parietal, 'topk_from_vec'):
                    q_vec = parietal.encode(prompt)
                    hits = parietal.topk_from_vec(q_vec, k=int(k))
                    traces = [str(t).strip() for t, _score in hits if str(t).strip()]
            except Exception:
                traces = []

        if not traces and recent_turns:
            for role, text in list(recent_turns)[-max(1, int(k)):]:
                txt = str(text).strip()
                if txt:
                    traces.append(f"{role}: {txt[:180]}")

        if not traces and recent_texts:
            traces = [str(t).strip()[:180] for t in list(recent_texts)[-max(1, int(k)): ] if str(t).strip()]

        seen = set()
        out: List[str] = []
        for t in traces:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out[:max(1, int(k))]
