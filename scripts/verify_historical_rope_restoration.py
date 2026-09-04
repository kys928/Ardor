#!/usr/bin/env python3
"""Prove current SelfAttention matches the verified historical April 2026 RoPE path."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORTEX_ROOT = REPO_ROOT / "Cerebrum" / "Cortex"
for p in (str(CORTEX_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
import torch.nn.functional as F

from dorsolateral_prefrontal_cortex import SelfAttention


class HistoricalSelfAttention(nn.Module):
    """Exact functional shape of the verified 2026-04-11 Network Volume implementation."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        self.heads = heads
        self.head_dim = hidden_dim // heads
        assert self.head_dim % 2 == 0
        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.attn_dropout_p = float(attn_dropout)
        self.register_buffer("_rope_inv_freq", self._build_inv_freq(), persistent=False)

    def _build_inv_freq(self) -> torch.Tensor:
        half_dim = self.head_dim // 2
        freq_seq = torch.arange(0, half_dim, dtype=torch.float32)
        return 1.0 / (self.rope_theta ** (freq_seq / half_dim))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _rope_cos_sin(self, T: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self._rope_inv_freq.to(device=device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return (
            emb.cos().to(dtype=dtype)[None, None, :, :],
            emb.sin().to(dtype=dtype)[None, None, :, :],
        )

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor):
        T = q.size(-2)
        cos, sin = self._rope_cos_sin(T, q.device, q.dtype)
        return (
            (q * cos) + (self._rotate_half(q) * sin),
            (k * cos) + (self._rotate_half(k) * sin),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q, k = self._apply_rope(q, k)
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, :, :T, :T].to(dtype=torch.bool, device=x.device)
        ctx = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=(attn_mask is None),
        )
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.out(ctx))


def prove(use_rope: bool) -> None:
    torch.manual_seed(928)
    ref = HistoricalSelfAttention(32, 4, use_rope=use_rope, rope_theta=10000.0).eval()
    cur = SelfAttention(32, 4, use_rope=use_rope, rope_theta=10000.0).eval()
    cur.load_state_dict(ref.state_dict(), strict=True)

    if set(cur.state_dict()) != set(ref.state_dict()):
        raise AssertionError("state_dict keys changed")
    if any("rope" in key for key in cur.state_dict()):
        raise AssertionError("derived RoPE state leaked into persistent state_dict")

    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        y_ref = ref(x)
        y_cur = cur(x)
    torch.testing.assert_close(y_cur, y_ref, rtol=0.0, atol=0.0)


def main() -> int:
    prove(True)
    prove(False)
    print("historical_rope_restoration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
