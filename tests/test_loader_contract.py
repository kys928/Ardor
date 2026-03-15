import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORTEX = ROOT / "Cerebrum" / "Cortex"
LANG = ROOT / "Cerebrum" / "LanguageProcessing"
for path in (str(ROOT), str(CORTEX), str(LANG)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

from Cerebrum.Cortex.ardor_config import ArdorConfig
from Cerebrum.Cortex.broca_decoder import ArdorDecoder
import Cerebrum.Cortex.prefrontal_cortex as pfc


def make_cfg() -> ArdorConfig:
    return ArdorConfig(
        vocab_size=128,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        max_len=32,
        dropout=0.1,
    )


def test_read_checkpoint_meta_from_dict():
    raw = {
        "arch": "ArdorDecoder",
        "config": {
            "vocab_size": 128,
            "hidden_size": 64,
            "n_layers": 2,
            "n_heads": 4,
            "max_len": 32,
        },
    }

    meta = pfc._read_checkpoint_meta(raw)

    assert meta["arch"] == "ArdorDecoder"
    assert meta["vocab_size"] == 128
    assert meta["hidden_size"] == 64
    assert meta["n_layers"] == 2
    assert meta["n_heads"] == 4


def test_config_from_meta():
    meta = {
        "vocab_size": 128,
        "hidden_size": 64,
        "n_layers": 2,
        "n_heads": 4,
        "max_len": 32,
    }

    cfg = pfc._config_from_meta(meta)

    assert cfg.vocab_size == 128
    assert cfg.hidden_size == 64
    assert cfg.n_layers == 2
    assert cfg.n_heads == 4
    assert cfg.max_len == 32


def test_infer_dims_from_state():
    model = ArdorDecoder(make_cfg())
    sd = model.state_dict()

    vocab_size, hidden_size, _, _ = pfc._infer_dims_from_state(sd)

    assert vocab_size == 128
    assert hidden_size == 64
