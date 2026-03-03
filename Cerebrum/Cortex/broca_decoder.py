from __future__ import annotations

import torch
import torch.nn as nn

from Cerebrum.Cortex.ardor_config import ArdorConfig
from Cerebrum.Cortex.fronto_parietal_loop import TransformerBlock
from Cerebrum.Cortex.thalamic_utils import causal_mask as _causal_mask


class ArdorDecoder(nn.Module):
    def __init__(self, config: ArdorConfig):
        super().__init__()
        config.validate()
        self.cfg = config

        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)

        # Learned absolute positions are forbidden when RoPE is active.
        self.position_embed = None
        if not config.use_rope:
            self.position_embed = nn.Embedding(config.max_len, config.hidden_size)

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)

        # weight tying
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.token_embed(idx)

        if self.position_embed is not None:
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
            x = x + self.position_embed(pos)

        x = self.drop(x)
        mask = _causal_mask(T, device=x.device, dtype=torch.bool)
        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.norm(x)
        return self.lm_head(x)
