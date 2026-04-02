from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class ArdorConfig:
    vocab_size: int
    hidden_size: int = 1536
    n_layers: int = 33
    n_heads: int = 24
    max_len: int = 2048
    ff_mult: int = 4
    dropout: float = 0.15
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    layernorm_eps: float = 1e-5
    use_rope: bool = False
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return int(self.hidden_size) // int(self.n_heads)

    @property
    def ffn_dim(self) -> int:
        return int(self.hidden_size) * int(self.ff_mult)

    def validate(self) -> None:
        ints = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "max_len": self.max_len,
            "ff_mult": self.ff_mult,
        }
        for name, val in ints.items():
            if int(val) <= 0:
                raise ValueError(f"{name} must be > 0 (got {val})")

        if int(self.hidden_size) % int(self.n_heads) != 0:
            raise ValueError("hidden_size must be divisible by n_heads")

        for name in ("dropout", "attn_dropout", "resid_dropout"):
            val = float(getattr(self, name))
            if not (0.0 <= val < 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0) (got {val})")

        if float(self.layernorm_eps) <= 0:
            raise ValueError("layernorm_eps must be > 0")

        if bool(self.use_rope) and (self.head_dim % 2 != 0):
            raise ValueError("RoPE requires even head_dim")

    def to_dict(self) -> dict:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArdorConfig":
        raw = dict(d or {})
        if "hidden_size" not in raw and "hidden" in raw:
            raw["hidden_size"] = raw["hidden"]
        if "n_layers" not in raw and "layers" in raw:
            raw["n_layers"] = raw["layers"]
        if "n_heads" not in raw and "heads" in raw:
            raw["n_heads"] = raw["heads"]

        cfg = cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw.get("hidden_size", 384)),
            n_layers=int(raw.get("n_layers", 8)),
            n_heads=int(raw.get("n_heads", 6)),
            max_len=int(raw.get("max_len", 2048)),
            ff_mult=int(raw.get("ff_mult", 4)),
            dropout=float(raw.get("dropout", 0.15)),
            attn_dropout=float(raw.get("attn_dropout", raw.get("dropout", 0.0))),
            resid_dropout=float(raw.get("resid_dropout", raw.get("dropout", 0.0))),
            layernorm_eps=float(raw.get("layernorm_eps", 1e-5)),
            use_rope=bool(raw.get("use_rope", False)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
        )
        cfg.validate()
        return cfg
