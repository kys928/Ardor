from __future__ import annotations
import torch, torch.nn as nn
from typing import Optional, Tuple
from .protocols import EmotionState, AeternumObservation, AeternumModule, clamp01

HAS_SNNTORCH = True
try:
    import snntorch as snn
    from snntorch import surrogate
except Exception:
    HAS_SNNTORCH = False

def _hazard_features(txt: str) -> torch.Tensor:
    t = (txt or "").lower()
    HI = ["suicide", "kill myself", "overdose", "bomb", "explosive"]
    MID = ["panic", "urgent", "immediately"]
    LO = ["asap", "now"]

    hi = 1.0 if any(w in t for w in HI) else 0.0
    mid = 0.6 if any(w in t for w in MID) else 0.0
    lo  = 0.3 if any(w in t for w in LO) else 0.0

    return torch.tensor([
        hi,               # lethal/self-harm/violence
        mid,              # strong distress/urgency
        lo,               # mild urgency
        1.0 if "!" in t else 0.0,
        1.0 if ("prescrib" in t or "diagnos" in t) else 0.0,
        1.0 if t.isupper() and len(t) > 6 else 0.0,
        min(1.0, t.count("!")/3.0),
        min(1.0, t.count("?")/3.0),
    ], dtype=torch.float32)


class BiExpSynapse(nn.Module):
    def __init__(self, tau_rise=2.0, tau_decay=8.0):
        super().__init__()
        self.tau_rise  = nn.Parameter(torch.tensor(float(tau_rise)))
        self.tau_decay = nn.Parameter(torch.tensor(float(tau_decay)))
        self.register_buffer("y", torch.tensor(0.0))
        self.register_buffer("z", torch.tensor(0.0))
    def reset_state(self, batch_size: int, feat_dim: int, device: str):
        self.y = torch.zeros(batch_size, feat_dim, device=device)
        self.z = torch.zeros(batch_size, feat_dim, device=device)
    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        a_r = torch.exp(-1.0 / self.tau_rise.clamp(min=1e-2))
        a_d = torch.exp(-1.0 / self.tau_decay.clamp(min=1e-2))
        self.z = self.z * a_r + (1.0 - a_r) * x_t
        self.y = self.y * a_d + self.z
        return self.y

class RefractoryGate(nn.Module):
    def __init__(self, ref_steps: int):
        super().__init__()
        self.ref_steps = int(ref_steps)
        self.register_buffer("tleft", None)
    def reset_state(self, shape: Tuple[int,int], device: str):
        B,H = shape
        self.tleft = torch.zeros(B, H, device=device, dtype=torch.int64)
    def forward(self, spk: torch.Tensor) -> torch.Tensor:
        if self.tleft is None:
            self.reset_state(spk.shape, spk.device)
        allow = (self.tleft <= 0).float()
        spk_out = spk * allow
        self.tleft = torch.where(self.tleft > 0, self.tleft - 1, self.tleft)
        self.tleft = torch.where(spk_out > 0, torch.full_like(self.tleft, self.ref_steps), self.tleft)
        return spk_out

class Amygdala(AeternumModule, nn.Module):
    def __init__(self,
        in_feats=8, h1=64, h2=32,
        Ts:int=6, rate_encode: bool=False,
        use_gru: bool=False, gru_hidden:int=32,
        tau_rise: float=2.0, tau_decay: float=8.0,
        beta1: float=0.9, beta2: float=0.9,
        refrac_steps: int=2,
        combine_latency: bool=True, alpha_rate: float=0.7,
        device: str="cpu"
    ):
        print("[Amygdala] __init__ in", __file__)

        nn.Module.__init__(self)
        self.device = device
        self.in_feats = int(in_feats); self.h1 = int(h1); self.h2 = int(h2)
        self.Ts = int(Ts); self.rate_encode = bool(rate_encode)
        self.use_gru = bool(use_gru); self.gru_hidden = int(gru_hidden)
        self.combine_latency = bool(combine_latency)
        self.alpha_rate = float(alpha_rate)
        self._last_text = ""; self._last_rate = 0.0; self._last_spike = 0.0
        self._last_latency = 0.0
        self._last_embed = None
        self.use_snn_online = True
        self.thr_rate = 0.55  # decision threshold on combined score

        # Learnable "thresholds": scale currents (effective spike thresholds)
        self.theta1 = nn.Parameter(torch.tensor(1.0))
        self.theta2 = nn.Parameter(torch.tensor(1.0))

        # Optional GRU
        if self.use_gru:
            self.gru = nn.GRU(input_size=in_feats, hidden_size=gru_hidden, num_layers=1, batch_first=False).to(device)
            in_to_fc1 = gru_hidden
        else:
            self.gru = None
            in_to_fc1 = in_feats

        self.syn = BiExpSynapse(tau_rise=tau_rise, tau_decay=tau_decay).to(device)

        if not HAS_SNNTORCH:
            self.fc1 = nn.Linear(in_to_fc1, h1).to(device)
            self.fc2 = nn.Linear(h1, h2).to(device)
            self.fc_out = nn.Linear(h2, 1).to(device)
            self.lif1 = nn.Identity(); self.lif2 = nn.Identity()
            self.ref_gate = None; self.surr = None
            return

        self.fc1 = nn.Linear(in_to_fc1, h1).to(device)
        self.fc2 = nn.Linear(h1, h2).to(device)
        self.fc_out = nn.Linear(h2, 1).to(device)

        self.surr = surrogate.fast_sigmoid()
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=self.surr).to(device)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=self.surr).to(device)
        self.ref_gate = RefractoryGate(refrac_steps)

    def reset(self):
        self._last_text = ""; self._last_rate = 0.0; self._last_latency = 0.0; self._last_spike = 0.0

    def observe(self, obs: AeternumObservation, state: EmotionState):
        # keep text for hazard-feature fast path
        self._last_text = obs.text or ""
        self._last_embed = None

        pe = getattr(obs, "pooled_embedding", None)
        if pe is not None:
            if isinstance(pe, torch.Tensor):
                emb = pe.detach().to(self.device).float()
            else:
                emb = torch.tensor(pe, device=self.device, dtype=torch.float32)

            if emb.ndim == 1:
                emb = emb.unsqueeze(0)   # [1, F]

            self._last_embed = emb


    def _encode_seq(self, x: torch.Tensor) -> torch.Tensor:
        if self.Ts <= 1: return x.unsqueeze(0)
        if not self.rate_encode: return x.unsqueeze(0).repeat(self.Ts, 1, 1)
        x_clamp = x.clamp(0.0, 1.0)
        return torch.stack([torch.bernoulli(x_clamp) for _ in range(self.Ts)], dim=0)

    def _forward_T(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T,B,F = x_seq.shape
        self.syn.reset_state(batch_size=B, feat_dim=(self.in_feats if not self.use_gru else self.gru_hidden), device=x_seq.device)

        if not HAS_SNNTORCH:
            x_mean = x_seq.mean(dim=0)
            h = torch.relu(self.fc1(x_mean)); h = torch.relu(self.fc2(h))
            p = torch.sigmoid(self.fc_out(h)).squeeze(-1)
            latency = torch.zeros_like(p)
            return p, p, latency

        mem1 = self.lif1.init_leaky(); mem2 = self.lif2.init_leaky()
        spike_count = torch.zeros(B, self.h2, device=x_seq.device)
        self.ref_gate.reset_state((B, self.h2), x_seq.device)
        first_t = torch.full((B,), fill_value=T+1, dtype=torch.long, device=x_seq.device)

        for t in range(T):
            xt = x_seq[t]
            xt = self.syn(xt if not self.use_gru else xt[:, :self.in_feats])
            if self.use_gru:
                g_out, _ = self.gru(xt.unsqueeze(0))     # [1,B,gru_hidden]
                cur1 = self.fc1(g_out.squeeze(0))
            else:
                cur1 = self.fc1(xt)
            cur1 = cur1 / self.theta1.abs().clamp(min=1e-3)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1) / self.theta2.abs().clamp(min=1e-3)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.ref_gate(spk2)
            spike_count += spk2
            fired = (spk2.sum(dim=1) > 0) & (first_t == T+1)
            first_t = torch.where(fired, torch.full_like(first_t, t+1), first_t)

        rate = (spike_count / max(1.0, T)).mean(dim=1)        # [B]
        latency = torch.where(first_t <= T, 1.0 - (first_t.float()-1.0)/T, torch.zeros_like(rate))
        prob = rate.clamp(0,1)
        return rate, prob, latency

    def forward_from_embedding(self, emb: torch.Tensor):
        """
        emb: [F] or [B, F] pooled embeddings (same 384-d space used in training).
        Returns (rate, prob, latency) for the batch.
        """
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)  # [1, F]

        x_seq = self._encode_seq(emb)   # [T, B, F] using Ts/rate_encode
        rate, prob, latency = self._forward_T(x_seq)
        return rate, prob, latency

    def step(self, state: EmotionState) -> EmotionState:
        st = state.detach().clone()

        # 1) Hazard fast path (text)
        haz_vec = _hazard_features(self._last_text).to(self.device)
        haz_score = float(haz_vec.max().item())

        # 2) SNN score from pooled embedding, if present
        snn_score = 0.0
        if self.use_snn_online and (self._last_embed is not None):
            rate, prob, latency = self.forward_from_embedding(self._last_embed)
            r = float(rate[0].item())
            L = float(latency[0].item())
            self._last_rate = r
            self._last_latency = L
            snn_score = self.alpha_rate * r + (1.0 - self.alpha_rate) * L if self.combine_latency else r
        else:
            # fallback: run SNN on hazard features
            x = haz_vec.unsqueeze(0)  # [1, F]
            x_seq = self._encode_seq(x)
            rate, prob, latency = self._forward_T(x_seq)
            r = float(rate[0].item())
            L = float(latency[0].item())
            self._last_rate = r
            self._last_latency = L
            snn_score = r

        # 3) Fuse: **max** so hazard overrides lazy SNN
        danger = max(haz_score, snn_score)
        danger = float(max(0.0, min(1.0, danger)))
        self._last_spike = danger

        # 4) Update state
        st.surprise = clamp01(0.7 * st.surprise + 0.3 * danger)
        st.anxiety = clamp01(0.85 * st.anxiety + 0.15 * danger)

        # --- 4) Update stance with hysteresis ---
        HIGH = 0.75  # only very strong danger makes it cautious
        LOW = 0.30  # if it calms down below this, we can relax stance

        if danger >= HIGH:
            st.stance = "cautious"
        elif danger < LOW and st.stance == "cautious":
            # we've calmed down enough, let vmPFC homeostasis move us back
            st.stance = "open"

        print(
            f"[Amygdala] text='{self._last_text[:50]}...' "
            f"haz={haz_score:.3f} snn={snn_score:.3f} danger={danger:.3f}"
        )

        return st

    def forward_rate_prob(self, feats_TBF: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if feats_TBF.ndim == 2: feats_TBF = feats_TBF.unsqueeze(0)
        rate, prob, _ = self._forward_T(feats_TBF)
        return rate, prob

    def parameters(self):
        params = list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc_out.parameters())
        params += [self.theta1, self.theta2] + list(self.syn.parameters())
        if self.use_gru: params += list(self.gru.parameters())
        return params

    # convenience getters for homeostasis
    @property
    def last_spike(self) -> float: return self._last_spike
    @property
    def last_rate(self) -> float: return self._last_rate
    @property
    def last_latency(self) -> float: return self._last_latency
