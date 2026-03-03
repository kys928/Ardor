
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Literal
from fronto_parietal_loop import TransformerBlock

PoolType = Literal["mean", "cls"]


class ArdorEncoder(nn.Module):
    """
    Bidirectional text encoder (parietal integration) for:
      - retrieval embeddings
      - classification/routing heads
      - cross-modal fusion (future)
      - decoder prefix/context compression

    Args:
        vocab_size: tokenizer vocab size
        hidden_dim: model width (must match decoder for best tying)
        num_layers: transformer depth
        heads: attention heads (hidden_dim % heads == 0)
        max_len: max sequence length for position embeddings (tokens only, CLS uses +1)
        dropout: dropout prob for attention/MLP/residuals (passed through TransformerBlock)
        use_cls_token: if True, prepends a learned [CLS]-like token at position index = max_len
        shared: optional module with attributes `token_embed` and/or `position_embed` to share
                embeddings across modules (see hippocampal_embeddings.SharedEmbeddings)
    """
    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int = 384,
                 num_layers: int = 8,
                 heads: int = 6,
                 max_len: int = 1024,
                 dropout: float = 0.10,
                 use_cls_token: bool = True,
                 shared: Optional[object] = None):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        self.hidden_dim = int(hidden_dim)
        self.max_len = int(max_len)
        self.use_cls = bool(use_cls_token)

        # --- token embeddings (optionally shared) ---
        if shared is not None and hasattr(shared, "token_embed"):
            self.token_embed = shared.token_embed  # share the module
        else:
            self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        # --- position embeddings (optionally shared) ---
        pos_rows = max_len + (1 if self.use_cls else 0)  # reserve extra index for CLS if used
        if shared is not None and hasattr(shared, "position_embed") \
           and getattr(shared.position_embed, "num_embeddings", 0) >= pos_rows \
           and shared.position_embed.embedding_dim == hidden_dim:
            self.position_embed = shared.position_embed  # share if capacity is sufficient
        else:
            self.position_embed = nn.Embedding(pos_rows, hidden_dim)

        # --- CLS token parameter (only for content; position index is provided by position_embed) ---
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        else:
            self.register_parameter("cls_token", None)

        # --- transformer stack ---
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    # ─────────────────────────────────────────────────────────────────────
    # Sharing / tying helpers
    # ─────────────────────────────────────────────────────────────────────
    def tie_from_broca(self, broca) -> None:
        """
        Tie token embeddings with the decoder (Broca) to keep a shared lexical space.
        Safe to call after decoder construction or load.
        """
        be = getattr(broca, "token_embed", None)
        if isinstance(be, nn.Embedding) and be.weight.shape == self.token_embed.weight.shape:
            self.token_embed.weight = be.weight  # weight-tying (shared parameter tensor)

    def attach_shared_embeddings(self, shared: object) -> None:
        """
        Attach a central SharedEmbeddings module at runtime (optional).
        """
        if hasattr(shared, "token_embed") and isinstance(shared.token_embed, nn.Embedding):
            if shared.token_embed.weight.shape == self.token_embed.weight.shape:
                self.token_embed = shared.token_embed
        if hasattr(shared, "position_embed") and isinstance(shared.position_embed, nn.Embedding):
            if shared.position_embed.weight.shape == self.position_embed.weight.shape:
                self.position_embed = shared.position_embed

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────
    def forward(self,
                x: torch.Tensor,                         # [B, T] token ids
                attn_mask: Optional[torch.Tensor] = None,# [1,1,S,S] boolean (True=keep), or None
                return_pooled: bool = False,
                pool: PoolType = "mean"):
        """
        Returns:
            if return_pooled=False: h      : [B, S, H]   (S=T or 1+T if CLS)
            if return_pooled=True : (h, p) : ([B, S, H], [B, H])
        """
        B, T = x.shape
        device = x.device

        # positions for tokens [0..T-1]
        pos_tok = torch.arange(T, device=device).unsqueeze(0).expand(B, T)   # [B, T]
        tok = self.token_embed(x) + self.position_embed(pos_tok)             # [B, T, H]

        if self.use_cls:
            # CLS content + its own position index (last row == self.max_len)
            cls = self.cls_token.expand(B, 1, self.hidden_dim)
            pos_cls = torch.full((B, 1), self.max_len, device=device, dtype=torch.long)
            cls = cls + self.position_embed(pos_cls)
            h = torch.cat([cls, tok], dim=1)  # [B, 1+T, H]
        else:
            h = tok  # [B, T, H]

        h = self.dropout(h)

        # pass through encoder blocks (mask expected as [1,1,S,S] if provided)
        for layer in self.layers:
            h = layer(h, mask=attn_mask)

        h = self.norm(h)

        if not return_pooled:
            return h

        # pooled representation
        if pool == "cls" and self.use_cls:
            pooled = h[:, 0, :]
        else:
            start = 1 if self.use_cls else 0
            pooled = h[:, start:, :].mean(dim=1)

        return h, pooled
