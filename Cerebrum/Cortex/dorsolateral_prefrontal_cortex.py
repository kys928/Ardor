import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = hidden_dim // heads

        # names must match checkpoint: ...attn.{q,k,v,out}.*
        self.q   = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k   = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v   = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # mask shape [1,1,T,T]; allow True (1)
            scores = scores.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        ctx = weights @ v
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.out(ctx))
