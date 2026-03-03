from __future__ import annotations

import math
import torch
import torch.nn as nn

from Cerebrum.Cortex.ardor_config import ArdorConfig


def _build_rope_cache(max_len: int, head_dim: int, theta: float, device, dtype):
    # inv_freq: [head_dim/2]
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_len, head_dim/2]
    cos = torch.cos(freqs).to(dtype=dtype)
    sin = torch.sin(freqs).to(dtype=dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: int = 0) -> torch.Tensor:
    """
    x: [B, heads, T, head_dim]
    cos/sin: [max_len, head_dim/2]
    """
    B, H, T, D = x.shape
    cos_t = cos[pos:pos+T].unsqueeze(0).unsqueeze(0)  # [1,1,T,D/2]
    sin_t = sin[pos:pos+T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    xr1 = x1 * cos_t - x2 * sin_t
    xr2 = x1 * sin_t + x2 * cos_t
    out = torch.empty_like(x)
    out[..., 0::2] = xr1
    out[..., 1::2] = xr2
    return out


class SelfAttention(nn.Module):
    def __init__(self, config: ArdorConfig):
        super().__init__()
        config.validate()
        self.cfg = config
        self.heads = config.n_heads
        self.head_dim = config.head_dim

        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.attn_drop = nn.Dropout(config.attn_dropout)
        self.proj_drop = nn.Dropout(config.resid_dropout)

        self.register_buffer("_rope_cos", None, persistent=False)
        self.register_buffer("_rope_sin", None, persistent=False)
        self._rope_cached_len = 0

    def _ensure_rope_cache(self, T: int, device, dtype):
        if not self.cfg.use_rope:
            return
        need = max(int(self.cfg.max_len), int(T))
        if self._rope_cos is None or self._rope_sin is None or self._rope_cached_len < need or self._rope_cos.device != device:
            cos, sin = _build_rope_cache(need, self.head_dim, float(self.cfg.rope_theta), device=device, dtype=dtype)
            self._rope_cos = cos
            self._rope_sin = sin
            self._rope_cached_len = need

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)  # [B,h,T,d]
        k = self.k(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        if self.cfg.use_rope:
            self._ensure_rope_cache(T, device=x.device, dtype=q.dtype)
            q = _apply_rope(q, self._rope_cos, self._rope_sin)
            k = _apply_rope(k, self._rope_cos, self._rope_sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,h,T,T]
        if mask is not None:
            scores = scores.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        ctx = weights @ v  # [B,h,T,d]
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.out(ctx))
