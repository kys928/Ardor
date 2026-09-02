import torch.nn as nn
from dorsolateral_prefrontal_cortex import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        heads,
        ff_hidden_mult=4,
        dropout=0.1,
        use_rope=True,
        rope_theta=10000.0,
    ):
        super().__init__()
        self.attn = SelfAttention(
            hidden_dim,
            heads,
            attn_dropout=dropout,
            proj_dropout=dropout,
            use_rope=use_rope,
            rope_theta=rope_theta,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden_mult * hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_mult * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *, mask=None):
        a = self.attn(x, mask=mask)
        x = self.norm1(x + self.dropout(a))
        f = self.ff(x)
        x = self.norm2(x + self.dropout(f))
        return x
