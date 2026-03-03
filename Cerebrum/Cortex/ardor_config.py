from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class ArdorConfig:
    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int
    ff_mult: int
    max_len: int
    dropout: float = 0.1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    layernorm_eps: float = 1e-5
    use_rope: bool = True
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    @property
    def ffn_dim(self) -> int:
        return self.ff_mult * self.hidden_size

    def validate(self) -> None:
        assert self.hidden_size % self.n_heads == 0, "hidden_size must be divisible by n_heads"
        # 1B target invariant (can be relaxed by env override upstream if you want)
        assert self.head_dim == 64, f"head_dim must be 64, got {self.head_dim}"
        assert self.ffn_dim == 6144, f"ffn_dim must be 6144, got {self.ffn_dim}"
        assert self.hidden_size == 1536
        assert self.n_heads == 24
        assert self.n_layers == 33
        assert self.max_len >= 2048
        assert 48000 <= self.vocab_size <= 60000
