
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Literal
from fronto_parietal_loop import TransformerBlock
from ardor_config import ArdorConfig

PoolType = Literal["mean", "cls"]


class ArdorEncoder(nn.Module):
    def __init__(self,
                 vocab_size_or_config,
                 hidden_dim: int = 384,
                 num_layers: int = 8,
                 heads: int = 6,
                 max_len: int = 1024,
                 dropout: float = 0.10,
                 use_cls_token: bool = True,
                 shared: Optional[object] = None):
        super().__init__()

        if isinstance(vocab_size_or_config, ArdorConfig):
            config = vocab_size_or_config
            vocab_size = int(config.vocab_size)
            hidden_dim = int(config.hidden_size)
            num_layers = int(config.n_layers)
            heads = int(config.n_heads)
            max_len = int(config.max_len)
            dropout = float(config.dropout)
        else:
            vocab_size = int(vocab_size_or_config)
            config = ArdorConfig(
                vocab_size=vocab_size,
                hidden_size=int(hidden_dim),
                n_layers=int(num_layers),
                n_heads=int(heads),
                max_len=int(max_len),
                dropout=float(dropout),
                attn_dropout=float(dropout),
                resid_dropout=float(dropout),
            )
        config.validate()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        self.cfg = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.hidden_size)
        self.hidden_dim = int(config.hidden_size)
        self.num_layers = int(config.n_layers)
        self.n_heads = int(config.n_heads)
        self.max_len = int(config.max_len)
        self.ff_mult = int(config.ff_mult)
        self.use_rope = bool(config.use_rope)
        self.rope_theta = float(config.rope_theta)
        self.use_cls = bool(use_cls_token)

        if shared is not None and hasattr(shared, "token_embed"):
            self.token_embed = shared.token_embed
        else:
            self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        pos_rows = max_len + (1 if self.use_cls else 0)
        if shared is not None and hasattr(shared, "position_embed") \
           and getattr(shared.position_embed, "num_embeddings", 0) >= pos_rows \
           and shared.position_embed.embedding_dim == hidden_dim:
            self.position_embed = shared.position_embed
        else:
            self.position_embed = nn.Embedding(pos_rows, hidden_dim)

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        else:
            self.register_parameter("cls_token", None)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, heads, ff_hidden_mult=self.ff_mult, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def model_config(self) -> dict:
        return {
            "arch": "ArdorEncoder",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "hidden": self.hidden_size,
            "n_layers": self.num_layers,
            "layers": self.num_layers,
            "n_heads": self.n_heads,
            "heads": self.n_heads,
            "ff_mult": self.ff_mult,
            "max_len": self.max_len,
            "use_rope": self.use_rope,
            "rope_theta": self.rope_theta,
            "use_cls_token": self.use_cls,
            "positional_encoding": "rope" if self.use_rope else "learned_absolute",
        }

    def tie_from_broca(self, broca) -> None:
        be = getattr(broca, "token_embed", None)
        if isinstance(be, nn.Embedding) and be.weight.shape == self.token_embed.weight.shape:
            self.token_embed.weight = be.weight

    def attach_shared_embeddings(self, shared: object) -> None:
        if hasattr(shared, "token_embed") and isinstance(shared.token_embed, nn.Embedding):
            if shared.token_embed.weight.shape == self.token_embed.weight.shape:
                self.token_embed = shared.token_embed
        if hasattr(shared, "position_embed") and isinstance(shared.position_embed, nn.Embedding):
            if shared.position_embed.weight.shape == self.position_embed.weight.shape:
                self.position_embed = shared.position_embed

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                return_pooled: bool = False,
                pool: PoolType = "mean"):
        B, T = x.shape
        device = x.device

        pos_tok = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        tok = self.token_embed(x) + self.position_embed(pos_tok)

        if self.use_cls:
            cls = self.cls_token.expand(B, 1, self.hidden_dim)
            pos_cls = torch.full((B, 1), self.max_len, device=device, dtype=torch.long)
            cls = cls + self.position_embed(pos_cls)
            h = torch.cat([cls, tok], dim=1)
        else:
            h = tok

        h = self.dropout(h)

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
