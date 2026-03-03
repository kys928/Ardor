from __future__ import annotations

import torch.nn as nn
from Cerebrum.Cortex.ardor_config import ArdorConfig


class SharedEmbeddings(nn.Module):
    """
    Central embedding hub.

    Under use_rope=True:
      - token_embed only (no learned position embeddings)
    Under use_rope=False:
      - token_embed + position_embed (legacy)
    """
    def __init__(self, config: ArdorConfig):
        super().__init__()
        config.validate()
        self.cfg = config
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embed = None
        if not config.use_rope:
            self.position_embed = nn.Embedding(config.max_len, config.hidden_size)
