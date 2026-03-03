from __future__ import annotations

from typing import Optional, Literal

import torch
import torch.nn as nn

from Cerebrum.Cortex.ardor_config import ArdorConfig
from Cerebrum.Cortex.fronto_parietal_loop import TransformerBlock

PoolType = Literal["mean", "cls"]


class ArdorEncoder(nn.Module):
    """
    Lightweight bidirectional encoder for retrieval/indexing.

    Note: This encoder is intentionally allowed to be 384-d (parietal space) even when the
    decoder is 1536-d. It can still be driven by an ArdorConfig-like object, but you may pass
    a separate config tailored to parietal_dim if desired.
    """
    def __init__(
        self,
        config: ArdorConfig,
        *,
        use_cls_token: bool = True,
        shared: Optional[object] = None,
    ):
        super().__init__()
        config.validate()
        self.cfg = config
        self.hidden_dim = int(config.hidden_size)
        self.max_len = int(config.max_len)
        self.use_cls = bool(use_cls_token)

        # token embeddings (optional share)
        if shared is not None and hasattr(shared, "token_embed"):
            self.token_embed = shared.token_embed
        else:
            self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)

        # Under RoPE, do not add learned absolute positions.
        self.position_embed = None
        if not config.use_rope:
            pos_rows = config.max_len + (1 if self.use_cls else 0)
            if shared is not None and hasattr(shared, "position_embed") and shared.position_embed is not None:
                pe = shared.position_embed
                if getattr(pe, "num_embeddings", 0) >= pos_rows and pe.embedding_dim == config.hidden_size:
                    self.position_embed = pe
            if self.position_embed is None:
                self.position_embed = nn.Embedding(pos_rows, config.hidden_size)

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.register_parameter("cls_token", None)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.drop = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)

    def forward(
        self,
        x: torch.Tensor,  # [B,T]
        attn_mask: Optional[torch.Tensor] = None,
        return_pooled: bool = False,
        pool: PoolType = "mean",
    ):
        B, T = x.shape
        h = self.token_embed(x)

        if self.position_embed is not None:
            pos_tok = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            h = h + self.position_embed(pos_tok)

        if self.use_cls:
            cls = self.cls_token.expand(B, 1, self.hidden_dim)
            if self.position_embed is not None:
                pos_cls = torch.full((B, 1), self.max_len, device=x.device, dtype=torch.long)
                cls = cls + self.position_embed(pos_cls)
            h = torch.cat([cls, h], dim=1)

        h = self.drop(h)
        for layer in self.layers:
            h = layer(h, mask=attn_mask)
        h = self.norm(h)

        if not return_pooled:
            return h

        if pool == "cls" and self.use_cls:
            pooled = h[:, 0, :]
        else:
            start = 1 if self.use_cls else 0
            pooled = h[:, start:, :].mean(dim=1)
        return h, pooled
