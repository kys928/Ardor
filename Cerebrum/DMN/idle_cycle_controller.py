from __future__ import annotations

import time
from typing import Any, Tuple

from .dmn_state import DMNEvent, DMNMode, DMNState, NarrativeTakeaway


class IdleCycleController:
    """Owns the DMN stage order and one-cycle execution flow."""

    def __init__(self, quiet_mind, seed_generator, memory, self_model, salience_gate, integrator, tpj=None):
        self.quiet_mind = quiet_mind
        self.seed_generator = seed_generator
        self.memory = memory
        self.self_model = self_model
        self.salience_gate = salience_gate
        self.integrator = integrator
        self.tpj = tpj

    def run_cycle(
        self,
        state: DMNState,
        *,
        mode: DMNMode,
        prompt: str,
        recent_turns=None,
        recent_texts=None,
        parietal=None,
        retrieval_enabled: bool = False,
        aet_state: Any = None,
        retrieved_memory_summary=None,
    ) -> Tuple[DMNState, NarrativeTakeaway, DMNEvent]:
        quiet_state = self.quiet_mind.build_baseline(recent_texts=list(recent_texts or []), self_state=state.self_model_state)
        unresolved = list((state.self_model_state or {}).get('unresolved_topics') or [])
        seed = self.seed_generator.generate(
            unresolved_themes=unresolved,
            recent_turns=recent_turns,
            recent_texts=recent_texts,
            system_state=quiet_state,
        )

        if retrieved_memory_summary is not None:
            retrieved_traces = [str(x).strip() for x in list(retrieved_memory_summary) if str(x).strip()]
        else:
            retrieved_traces = self.memory.retrieve(
                prompt=prompt or seed,
                parietal=parietal,
                retrieval_enabled=retrieval_enabled,
                recent_turns=recent_turns,
                recent_texts=recent_texts,
                k=3,
            )

        self_state = self.self_model.update_state(
            state.self_model_state,
            seed=seed,
            retrieved_traces=retrieved_traces,
            recent_texts=recent_texts,
        )

        salience = self.salience_gate.score(
            seed=seed,
            self_state=self_state,
            retrieved_traces=retrieved_traces,
            aet_state=aet_state,
        )

        tpj_context = self.tpj.enrich(recent_turns=recent_turns, prompt=prompt) if self.tpj is not None else {}
        takeaway = self.integrator.integrate(
            mode=mode,
            quiet_state=quiet_state,
            seed=seed,
            retrieved_traces=retrieved_traces,
            self_state=self_state,
            salience=salience,
            tpj_context=tpj_context,
        )

        state.mode = mode
        state.last_seed = seed
        state.last_retrieved_traces = list(retrieved_traces[:3])
        state.current_narrative_takeaway = takeaway
        state.self_model_state = dict(self_state)
        state.salience_info = dict(salience)
        state.last_updated = time.time()
        state.cycle_count += 1
        state.last_error = None

        event = DMNEvent(
            event_type='idle_cycle' if mode == DMNMode.IDLE else 'active_summary',
            mode=mode.value,
            payload={
                'seed': seed,
                'retrieved_traces': list(retrieved_traces[:3]),
                'self_model_state': self_state,
                'salience': salience,
                'takeaway': takeaway.to_dict(),
            },
        )
        return state, takeaway, event
