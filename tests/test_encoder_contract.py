import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORTEX = ROOT / "Cerebrum" / "Cortex"
for path in (str(ROOT), str(CORTEX)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

from Cerebrum.Cortex.ardor_config import ArdorConfig
from Cerebrum.Cortex.posterior_parietal_cortex import ArdorEncoder


def make_cfg() -> ArdorConfig:
    return ArdorConfig(
        vocab_size=128,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        max_len=32,
        dropout=0.1,
    )


def test_encoder_forward_shape():
    model = ArdorEncoder(make_cfg(), use_cls_token=True)
    model.eval()

    x = torch.randint(0, 128, (2, 8), dtype=torch.long)
    with torch.no_grad():
        h = model(x)

    assert h.shape[0] == 2
    assert h.shape[2] == 64


def test_encoder_pooled_mean_shape():
    model = ArdorEncoder(make_cfg(), use_cls_token=True)
    model.eval()

    x = torch.randint(0, 128, (2, 8), dtype=torch.long)
    with torch.no_grad():
        _, pooled = model(x, return_pooled=True, pool="mean")

    assert tuple(pooled.shape) == (2, 64)


def test_encoder_pooled_cls_shape():
    model = ArdorEncoder(make_cfg(), use_cls_token=True)
    model.eval()

    x = torch.randint(0, 128, (2, 8), dtype=torch.long)
    with torch.no_grad():
        _, pooled = model(x, return_pooled=True, pool="cls")

    assert tuple(pooled.shape) == (2, 64)


def test_encoder_model_config_core_keys():
    model = ArdorEncoder(make_cfg(), use_cls_token=True)
    cfg = model.model_config()

    assert cfg["arch"] == "ArdorEncoder"
    assert cfg["vocab_size"] == 128
    assert cfg["hidden_size"] == 64
    assert cfg["n_layers"] == 2
    assert cfg["n_heads"] == 4
    assert cfg["max_len"] == 32
    assert cfg["use_cls_token"] is True
