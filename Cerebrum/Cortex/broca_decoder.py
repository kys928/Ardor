import torch
import torch.nn as nn
from fronto_parietal_loop import TransformerBlock
from thalamic_utils import causal_mask as _causal_mask

class ArdorDecoder(nn.Module):
   
    def __init__(self, vocab_size, hidden=384, layers=8, heads=6,
                 max_len=2048, dropout=0.15):
        super().__init__()
        self.token_embed   = nn.Embedding(vocab_size, hidden)
        self.position_embed = nn.Embedding(max_len, hidden)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, heads, dropout=dropout)
            for _ in range(layers)
        ])

        self.norm    = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # tie

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_embed(idx) + self.position_embed(pos)
        x = self.dropout(x)
        mask = _causal_mask(T, device=x.device)
        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.norm(x)
        return self.lm_head(x)
