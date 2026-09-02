import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
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
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        self.heads = heads
        self.head_dim = hidden_dim // heads
        assert self.head_dim % 2 == 0, "RoPE requires even head_dim"

        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)

        # names must match checkpoint: ...attn.{q,k,v,out}.*
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.attn_dropout_p = float(attn_dropout)

        # Derived, non-learned state: intentionally absent from state_dict.
        self.register_buffer("_rope_inv_freq", self._build_inv_freq(), persistent=False)

    def _build_inv_freq(self) -> torch.Tensor:
        half_dim = self.head_dim // 2
        freq_seq = torch.arange(0, half_dim, dtype=torch.float32)
        return 1.0 / (self.rope_theta ** (freq_seq / half_dim))

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1)
        return x_rot.flatten(-2)

    def _rope_cos_sin(
        self,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self._rope_inv_freq.to(device=device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = q.size(-2)
        cos, sin = self._rope_cos_sin(T, q.device, q.dtype)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = self._apply_rope(q, k)

        attn_mask = None
        if mask is not None:
            # incoming mask expected [1,1,T,T], True means keep
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
