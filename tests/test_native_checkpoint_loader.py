import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CORTEX = ROOT / "Cerebrum" / "Cortex"
for path in (str(ROOT), str(CORTEX)):
    if path not in sys.path:
        sys.path.insert(0, path)

from Cerebrum.Cortex.ardor_config import ArdorConfig
import Cerebrum.Cortex.loaders.native_checkpoint as native_checkpoint


class _DummyDecoder(torch.nn.Module):
    def __init__(self, cfg: ArdorConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(cfg.vocab_size)
        self.hidden = int(cfg.hidden_size)
        self.hidden_size = int(cfg.hidden_size)
        self.num_layers = int(cfg.n_layers)
        self.n_layers = int(cfg.n_layers)
        self.heads = int(cfg.n_heads)
        self.n_heads = int(cfg.n_heads)
        self.max_len = int(cfg.max_len)

    def state_dict(self, *args, **kwargs):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        return [], []

    def model_config(self):
        return self.cfg.to_dict()


def _write_fallback_checkpoint(path: Path, *, hidden_size: int, meta: dict | None = None) -> None:
    torch.save(
        {
            "state_dict": {
                "token_embed.weight": torch.zeros(8, hidden_size),
            },
            "config": dict(meta or {}),
        },
        path,
    )


def test_load_native_decoder_prefers_checkpoint_n_heads_over_best_heads(tmp_path, monkeypatch):
    monkeypatch.setattr(native_checkpoint, "ArdorDecoder", _DummyDecoder)
    ckpt = tmp_path / "native.pt"
    _write_fallback_checkpoint(
        ckpt,
        hidden_size=1536,
        meta={"hidden_size": 1536, "n_heads": 24, "n_layers": 33, "max_len": 2048, "vocab_size": 8},
    )

    model, schema, want_vocab, meta = native_checkpoint.load_native_decoder(str(ckpt), "cpu")

    assert native_checkpoint.best_heads(1536) == 6
    assert model.cfg.n_heads == 24
    assert schema["heads"] == 24
    assert want_vocab == 8
    assert meta["n_heads"] == 24


def test_load_native_decoder_without_n_heads_falls_back_to_best_heads(tmp_path, monkeypatch):
    monkeypatch.setattr(native_checkpoint, "ArdorDecoder", _DummyDecoder)
    ckpt = tmp_path / "native.pt"
    _write_fallback_checkpoint(ckpt, hidden_size=1536, meta={"hidden_size": 1536, "vocab_size": 8})

    model, schema, _, meta = native_checkpoint.load_native_decoder(str(ckpt), "cpu")

    assert "n_heads" not in meta
    assert model.cfg.n_heads == native_checkpoint.best_heads(1536)
    assert schema["heads"] == 6


def test_load_native_decoder_invalid_checkpoint_n_heads_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(native_checkpoint, "ArdorDecoder", _DummyDecoder)
    ckpt = tmp_path / "native.pt"
    _write_fallback_checkpoint(ckpt, hidden_size=1536, meta={"hidden_size": 1536, "n_heads": 25, "vocab_size": 8})

    with pytest.raises(ValueError, match="Invalid checkpoint head count: hidden_size=1536 is not divisible by n_heads=25"):
        native_checkpoint.load_native_decoder(str(ckpt), "cpu")
