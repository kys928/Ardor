from __future__ import annotations

from typing import Any, Optional

from .dmn_logger import DMNLogger
from .dmn_state import DMNEvent, DMNMode, DMNState, NarrativeTakeaway
from .idle_cycle_controller import IdleCycleController
from .internal_seed_generator import InternalSeedGenerator
from .medial_prefrontal_cortex import MedialPrefrontalCortex
from .medial_temporal_lobe import MedialTemporalLobe
from .posterior_cingulate_precuneus import PosteriorCingulatePrecuneus
from .quiet_mind import QuietMind
from .salience_affect_gate import SalienceAffectGate
from .temporoparietal_junction import TemporalParietalJunction


class InsideASyntheticThought:
    """Facade for the DMN subsystem. The rest of Ardor should call only this class."""

    def __init__(self, *, logger: Optional[DMNLogger] = None):
        self.logger = logger or DMNLogger()
        self.state = DMNState(mode=DMNMode.IDLE)
        self.controller = IdleCycleController(
            quiet_mind=QuietMind(),
            seed_generator=InternalSeedGenerator(),
            memory=MedialTemporalLobe(),
            self_model=MedialPrefrontalCortex(),
            salience_gate=SalienceAffectGate(),
            integrator=PosteriorCingulatePrecuneus(),
            tpj=TemporalParietalJunction(),
        )

    def start_idle_cycle(self, **_kwargs) -> DMNState:
        self.state.is_running = True
        self.state.mode = DMNMode.IDLE
        self.state.last_error = None
        return self.state

    def step_idle_cycle(
        self,
        *,
        prompt: str = '',
        recent_turns=None,
        recent_texts=None,
        parietal=None,
        retrieval_enabled: bool = False,
        aet_state: Any = None,
    ) -> NarrativeTakeaway:
        try:
            self.start_idle_cycle()
            self.state, takeaway, event = self.controller.run_cycle(
                self.state,
                mode=DMNMode.IDLE,
                prompt=prompt,
                recent_turns=recent_turns,
                recent_texts=recent_texts,
                parietal=parietal,
                retrieval_enabled=retrieval_enabled,
                aet_state=aet_state,
            )
            self.logger.log_event(event)
            self.logger.log_summary(self.state, takeaway, 'idle_cycle')
            return takeaway
        except Exception as e:
            self.state.mode = DMNMode.ERROR
            self.state.last_error = str(e)
            self.logger.log_event(DMNEvent(event_type='idle_cycle_error', mode=DMNMode.ERROR.value, payload={'error': str(e)}))
            raise
        finally:
            self.state.is_running = False

    def summarize_active_context(
        self,
        *,
        prompt: str,
        response: str,
        recent_turns=None,
        recent_texts=None,
        retrieved_memory_summary=None,
        aet_state: Any = None,
        parietal=None,
        retrieval_enabled: bool = False,
    ) -> NarrativeTakeaway:
        try:
            working_recent_texts = list(recent_texts or []) + [prompt, response]
            self.state, takeaway, event = self.controller.run_cycle(
                self.state,
                mode=DMNMode.ACTIVE_SUMMARY,
                prompt=prompt,
                recent_turns=recent_turns,
                recent_texts=working_recent_texts,
                parietal=parietal,
                retrieval_enabled=retrieval_enabled,
                aet_state=aet_state,
                retrieved_memory_summary=retrieved_memory_summary,
            )
            self.logger.log_event(event)
            self.logger.log_summary(self.state, takeaway, 'active_summary')
            return takeaway
        except Exception as e:
            self.state.mode = DMNMode.ERROR
            self.state.last_error = str(e)
            self.logger.log_event(DMNEvent(event_type='active_summary_error', mode=DMNMode.ERROR.value, payload={'error': str(e)}))
            raise

    def get_state(self) -> DMNState:
        return self.state

    def stop(self) -> DMNState:
        self.state.is_running = False
        self.state.mode = DMNMode.DISABLED
        return self.state
