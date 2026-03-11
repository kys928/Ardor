import math
import torch
import torch.nn as nn
from fronto_parietal_loop import TransformerBlock
from ardor_config import ArdorConfig


class ArdorDecoder(nn.Module):
    def __init__(self, config: ArdorConfig):
        super().__init__()
        config.validate()
        self.cfg = config

        self.vocab_size = int(config.vocab_size)
        self.hidden = int(config.hidden_size)
        self.hidden_size = int(config.hidden_size)
        self.num_layers = int(config.n_layers)
        self.n_layers = int(config.n_layers)
        self.heads = int(config.n_heads)
        self.n_heads = int(config.n_heads)
        self.max_len = int(config.max_len)
        self.ff_mult = int(config.ff_mult)
        self.dropout_p = float(config.dropout)
        self.dropout = float(config.dropout)
        self.attn_dropout = float(config.attn_dropout)
        self.resid_dropout = float(config.resid_dropout)
        self.layernorm_eps = float(config.layernorm_eps)
        self.use_rope = bool(config.use_rope)
        self.rope_theta = float(config.rope_theta)

        self.token_embed = nn.Embedding(self.vocab_size, self.hidden_size)

        self.position_embed = None
        if not self.use_rope:
            self.position_embed = nn.Embedding(self.max_len, self.hidden_size)

        self.drop = nn.Dropout(self.dropout_p)

        self.blocks = nn.ModuleList([
            TransformerBlock(self.hidden_size, self.n_heads, ff_hidden_mult=self.ff_mult, dropout=self.dropout_p)
            for _ in range(self.num_layers)
        ])
        self.layers = self.blocks

        self.norm = nn.LayerNorm(self.hidden_size, eps=self.layernorm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        scale = 1.0 / math.sqrt(2.0 * self.num_layers)
        for block in self.blocks:
            block.attn.out.weight.data.mul_(scale)
            block.ff[2].weight.data.mul_(scale)

    def model_config(self) -> dict:
        return {
            "arch": "ArdorDecoder",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "hidden": self.hidden_size,
            "n_layers": self.num_layers,
            "layers": self.num_layers,
            "n_heads": self.n_heads,
            "heads": self.n_heads,
            "ff_mult": self.ff_mult,
            "max_len": self.max_len,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "resid_dropout": self.resid_dropout,
            "layernorm_eps": self.layernorm_eps,
            "use_rope": self.use_rope,
            "rope_theta": self.rope_theta,
            "positional_encoding": "rope" if self.use_rope else "learned_absolute",
        }

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.token_embed(idx)

        if self.position_embed is not None:
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
            x = x + self.position_embed(pos)

        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, mask=None)
        x = self.norm(x)
        return self.lm_head(x)