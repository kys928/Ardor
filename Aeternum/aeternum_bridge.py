from __future__ import annotations
import torch
from Aeternum import AeternumCore, AeternumConfig

def init_aeternum(device="cpu", vad_csv_path=None, state_path=None):
    cfg = AeternumConfig(device=device, vad_csv_path=vad_csv_path, state_path=state_path)
    return AeternumCore(cfg)

def mean_pool_hidden(last_hidden: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
    # last_hidden: [B, T, H]; attn_mask: [B, T] (1 for tokens to include)
    if last_hidden.ndim != 3:
        return None
    if attn_mask is None:
        return last_hidden.mean(dim=1)  # [B,H]
    mask = attn_mask.float().unsqueeze(-1)  # [B,T,1]
    s = (last_hidden * mask).sum(dim=1)
    n = mask.sum(dim=1).clamp(min=1.0)
    return s / n  # [B,H]

def affect_step(aet, tokenizer, logits_1d, *, text_for_obs="", pooled=None, feedback=None, is_new_turn=True,
                temperature=1.0, top_p=0.9, rep_penalty=1.1):
    # Update state from current context and logits
    decision = aet.update(text=text_for_obs, pooled_embedding=pooled, last_logits=logits_1d,
                          user_feedback=feedback, is_new_turn=is_new_turn)
    # Token-level bias
    logits_1d = aet.apply_bias(tokenizer, logits_1d)
    # Scale sampler controls
    temperature *= decision.temperature_scale
    top_p       *= decision.top_p_scale
    rep_penalty *= decision.rep_penalty_scale
    return logits_1d, temperature, top_p, rep_penalty, decision.state
