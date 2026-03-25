#!/usr/bin/env python3
"""
Aeternum/aeternum_core.py  (PATCHED)

Fixes (stability-first):
1) ✅ Guard against missing pooled embeddings:
   Your PFC patch disables retrieval, so GUI sometimes calls AeternumCore.update(pooled_embedding=None).
   Some downstream Aeternum modules try to "pool" by calling an encoder/_encode path.
   If that path is disabled (None), you get:
     [Aeternum] pooling failed: 'NoneType' object has no attribute '_encode'

   This file now guarantees obs.pooled_embedding is ALWAYS a valid tensor of shape [1, hidden_dim].
   If None is provided, we inject a zero embedding (neutral / no signal).

2) ✅ Fix Amygdala SNN interface mismatch:
   Raw Amygdala.step() expects a 384-d tensor, but AeternumCore.update calls m.step(state) for ALL modules.
   Passing EmotionState into Amygdala.step caused (1x8)@(384x64).
   We wrap Amygdala in CorticoAmygdalarRelay that consumes obs.pooled_embedding and conforms to AeternumModule.

PY38-COMPAT: uses from __future__ annotations and avoids PEP 695 features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
from pathlib import Path
import torch

from .protocols import EmotionState, AeternumObservation, AeternumDecision, AeternumModule
from .anterior_insula import AnteriorInsula
from .bnst import BNST
from .locus_coeruleus_noradrenaline import LocusCoeruleus
from .ventromedial_prefrontal_cortex import VentroMedialPFC
from .orbitofrontal_cortex_value import OrbitofrontalValue
from .anterior_cingulate_reflex import AnteriorCingulate
from .ventral_striatum import VentralStriatum
from .neuromodulators import Neuromodulators
from .Hippocampus import Hippocampus
from .homeostatic_plasticity import HomeostaticPlasticity
from .orbitofrontal_lexical_bias import lexical_bias, lexical_bias_from_vad, load_vad_csv
from .temporal_pole_emotion_classifier import TemporalPoleEmotionClassifier
from .mtl_emotion_classifier import MTLEmotionClassifier

HAS_SNN = True
try:
    from .Amygdala import Amygdala
except Exception:
    HAS_SNN = False


@dataclass
class AeternumConfig:
    hidden_dim: int = 384
    device: str = "cpu"
    vad_csv_path: Optional[str] = None
    state_path: Optional[str] = None
    prefer_snn: bool = True


class CorticoAmygdalarRelay(AeternumModule):
    """Cortico-amygdalar relay to make the Amygdala SNN behave like an AeternumModule.

    AeternumCore.update() calls:
      - m.observe(obs, state)
      - m.step(state)

    The raw Amygdala SNN does NOT follow that interface; it expects a tensor input.
    This relay stores obs.pooled_embedding (cortical pooled embedding) and feeds it into amygdala.step(x).
    """

    def __init__(self, amygdala: "Amygdala", hidden_dim: int, device: str):
        self.amyg = amygdala
        self.hidden_dim = int(hidden_dim)
        self.device = device
        self._x: Optional[torch.Tensor] = None

    def reset(self):
        self._x = None

    def observe(self, obs: AeternumObservation, st: EmotionState):
        x = getattr(obs, "pooled_embedding", None)
        if x is None or (not isinstance(x, torch.Tensor)):
            self._x = None
            return

        # Accept shapes: [H], [1,H], [1,T,H]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            x = x.mean(dim=1)

        if x.ndim != 2 or x.shape[-1] != self.hidden_dim:
            # Hard guard: never feed wrong dims into the SNN.
            self._x = None
            return

        self._x = x.to(self.device)

    def step(self, st: EmotionState) -> EmotionState:
        if self._x is None:
            return st

        try:
            out = self.amyg.step(self._x)
        except Exception as e:
            print(f"[AeternumCore] ERROR in Amygdala.step: {e}")
            self._x = None
            return st

        # Robustly map amygdala output into EmotionState (only if fields exist).
        fear = None
        if isinstance(out, torch.Tensor):
            try:
                fear = float(out.detach().mean().clamp(0, 1).cpu().item())
            except Exception:
                fear = None
        elif isinstance(out, (float, int)):
            fear = float(out)
        elif isinstance(out, dict):
            if "fear" in out:
                try:
                    fear = float(out["fear"])
                except Exception:
                    fear = None

        if fear is not None:
            fear = max(0.0, min(1.0, float(fear)))
            if hasattr(st, "fear"):
                st.fear = fear
            # bump arousal slightly if fear rises
            if hasattr(st, "arousal"):
                try:
                    st.arousal = max(0.0, min(1.0, float(st.arousal) + 0.15 * (fear - 0.5)))
                except Exception:
                    pass
            # safe_mode if fear is very high
            if hasattr(st, "safe_mode"):
                st.safe_mode = bool(fear >= 0.85)

        return st


class AeternumCore:
    def __init__(self, cfg: Optional[AeternumConfig] = None):
        self.cfg = cfg or AeternumConfig()
        self.state = EmotionState()

        self.modules: List[AeternumModule] = [
            TemporalPoleEmotionClassifier(self.cfg.hidden_dim, self.cfg.device),
            AnteriorInsula(),
        ]

        self.mtl_temporal_pole = MTLEmotionClassifier(
            hidden_dim=self.cfg.hidden_dim,
            device=self.cfg.device,
            ckpt_path=None,  # auto-discover emotion_heads_mtl.pt
        )
        self.modules.append(self.mtl_temporal_pole)

        self.amyg: Optional["Amygdala"] = None
        if self.cfg.prefer_snn and HAS_SNN:
            self.amyg = Amygdala(
                in_feats=self.cfg.hidden_dim,  # MUST be 384-d pooled embedding
                h1=64,
                h2=32,
                Ts=6,
                rate_encode=True,
                use_gru=False,
                combine_latency=True,
                alpha_rate=0.7,
                device=self.cfg.device,
            )

            # ✅ Wrap with adapter so update() never passes EmotionState into the SNN.
            self.modules.append(CorticoAmygdalarRelay(self.amyg, self.cfg.hidden_dim, self.cfg.device))

            # Load trained fear SNN (aversive_v3)
            try:
                ckpt_path = Path(__file__).with_name("Models") / "amygdala_fear_snn_aversive_v3.pt"
                if ckpt_path.exists():
                    try:
                        sd = torch.load(ckpt_path, map_location=self.cfg.device, weights_only=False)
                    except TypeError:
                        sd = torch.load(ckpt_path, map_location=self.cfg.device)
                    filtered = {k: v for k, v in sd.items() if not (k.endswith("syn.y") or k.endswith("syn.z"))}
                    missing, unexpected = self.amyg.load_state_dict(filtered, strict=False)
                    print(f"[AeternumCore] loaded Amygdala SNN from {ckpt_path}")
                    if missing or unexpected:
                        print(f"  missing: {missing}")
                        print(f"  unexpected: {unexpected}")
                else:
                    print(f"[AeternumCore] WARNING: no SNN checkpoint at {ckpt_path}, using random init")
            except Exception as e:
                print(f"[AeternumCore] WARNING: failed to load Amygdala SNN checkpoint: {e}")

        # downstream modules
        self.modules += [
            BNST(),
            LocusCoeruleus(),
            VentroMedialPFC(),
            OrbitofrontalValue(),
            AnteriorCingulate(),
            VentralStriatum(),
        ]

        self.neuromod = Neuromodulators()
        self.hippo = Hippocampus()

        # Homeostasis at the end; bind amygdala for threshold adaptation
        hp = HomeostaticPlasticity(adapt_thresholds=True)
        if self.amyg is not None:
            hp.bind_amygdala(self.amyg)
        self.modules.append(hp)

        self.vad_lex: Dict[str, tuple] = {}
        if self.cfg.vad_csv_path:
            try:
                self.vad_lex = load_vad_csv(self.cfg.vad_csv_path)
            except Exception:
                self.vad_lex = {}

        if self.cfg.state_path:
            self.load_state(self.cfg.state_path)

    def reset(self):
        self.state = EmotionState()
        for m in self.modules:
            m.reset()
        self.hippo.reset()

    # ───────────────────────── pooled embedding guard ─────────────────────────
    def _ensure_pooled_embedding(self, pooled_embedding: Any, text: str = "") -> torch.Tensor:
        """Always return a valid pooled embedding tensor [1, hidden_dim].

        Why:
          - When retrieval/encoder utilities are disabled in PFC, pooled_embedding can be None.
          - Some Aeternum modules try to compute pooling via an encoder/_encode path and crash if it's None.
          - We keep Aeternum stable by providing a neutral embedding instead of triggering that path.
        """
        H = int(self.cfg.hidden_dim)
        dev = self.cfg.device

        # None -> neutral (zero) pooled embedding
        if pooled_embedding is None:
            return torch.zeros(1, H, device=dev, dtype=torch.float32)

        # If it's already a tensor, normalize shapes.
        if isinstance(pooled_embedding, torch.Tensor):
            x = pooled_embedding
            # [H] -> [1,H]
            if x.ndim == 1:
                x = x.unsqueeze(0)
            # [1,T,H] -> [1,H]
            elif x.ndim == 3:
                x = x.mean(dim=1)
            # Anything else -> neutral
            if x.ndim != 2 or x.shape[-1] != H:
                return torch.zeros(1, H, device=dev, dtype=torch.float32)
            return x.to(dev)

        # Unknown type -> neutral
        return torch.zeros(1, H, device=dev, dtype=torch.float32)

    def save_state(self, path: str):
        st = self.state.__dict__.copy()
        try:
            eps = [
                {
                    "ts": e.ts,
                    "text": e.text,
                    "valence": e.valence,
                    "arousal": e.arousal,
                    "surprise": e.surprise,
                    "score": e.score,
                }
                for e in self.hippo.sample_for_rem(k=64)
            ]
        except Exception:
            eps = []
        payload = {"state": st, "episodes": eps}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load_state(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.state = EmotionState(**payload.get("state", {}))
        self.hippo.reset()
        from .Hippocampus import Episode

        for ep in payload.get("episodes", []):
            try:
                self.hippo.buf.append((ep["score"], Episode(**ep)))
            except Exception:
                pass

    def update(
        self,
        *,
        text: str = "",
        pooled_embedding=None,
        last_logits=None,
        user_feedback: Optional[float] = None,
        is_new_turn: bool = True,
    ) -> AeternumDecision:
        # ✅ CRITICAL: guarantee a valid pooled embedding tensor.
        # This prevents downstream modules from trying to call encoder/_encode paths when they’re disabled.
        pooled_embedding = self._ensure_pooled_embedding(pooled_embedding, text=text)

        obs = AeternumObservation(
            text=text,
            pooled_embedding=pooled_embedding,
            last_logits=last_logits,
            user_feedback=user_feedback,
            is_new_turn=is_new_turn,
        )

        for m in self.modules:
            try:
                m.observe(obs, self.state)
            except Exception as e:
                print(f"[AeternumCore] ERROR in {m.__class__.__name__}.observe: {e}")
                raise

        st = self.state
        for m in self.modules:
            try:
                st = m.step(st)
            except Exception as e:
                print(f"[AeternumCore] ERROR in {m.__class__.__name__}.step: {e}")
                raise

        try:
            self.hippo.observe(obs, st)
        except Exception:
            pass

        scales = self.neuromod.compute_decoding_scales(st)
        self.state = st
        return AeternumDecision(
            state=st,
            token_bias={},
            temperature_scale=scales["temperature_scale"],
            top_p_scale=scales["top_p_scale"],
            rep_penalty_scale=scales["rep_penalty_scale"],
        )

    def apply_bias(self, tokenizer, logits_1d: torch.Tensor) -> torch.Tensor:
        if logits_1d.ndim != 1:
            logits_1d = logits_1d.view(-1)

        st = self.state

        # lexical bias
        try:
            lb = lexical_bias(tokenizer, st)
            if self.vad_lex:
                from_vad = lexical_bias_from_vad(tokenizer, st, self.vad_lex, scale=0.15)
                for k, v in from_vad.items():
                    lb[k] = lb.get(k, 0.0) + v
            if lb:
                idx = torch.tensor(
                    [i for i in lb.keys() if 0 <= i < logits_1d.shape[0]],
                    dtype=torch.long,
                    device=logits_1d.device,
                )
                if idx.numel() > 0:
                    delta = torch.tensor(
                        [lb[int(i)] for i in idx.tolist()],
                        dtype=logits_1d.dtype,
                        device=logits_1d.device,
                    )
                    logits_1d.index_add_(0, idx, delta)
        except Exception:
            pass

        # safe_mode punctuation taming
        try:
            q_id = tokenizer.encode("?", add_special_tokens=False).ids
            e_id = tokenizer.encode("!", add_special_tokens=False).ids
            if getattr(st, "safe_mode", False):
                for arr, pen in ((q_id, 0.25), (e_id, 0.50)):
                    if arr:
                        t = arr[0]
                        if 0 <= t < logits_1d.shape[0]:
                            logits_1d[t] -= pen
        except Exception:
            pass

        return logits_1d
