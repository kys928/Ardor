from __future__ import annotations

import torch.nn as nn

from Cerebrum.Cortex.ardor_config import ArdorConfig
from Cerebrum.Cortex.dorsolateral_prefrontal_cortex import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, config: ArdorConfig):
        super().__init__()
        config.validate()
        self.cfg = config
        self.attn = SelfAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)

        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.hidden_size),
        )
        self.resid_drop = nn.Dropout(config.resid_dropout)

    def forward(self, x, *, mask=None):
        a = self.attn(x, mask=mask)
        x = self.norm1(x + self.resid_drop(a))
        f = self.ff(x)
        x = self.norm2(x + self.resid_drop(f))
        return x
