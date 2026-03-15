import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORTEX = ROOT / "Cerebrum" / "Cortex"
for path in (str(ROOT), str(CORTEX)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

from Cerebrum.Cortex.ardor_config import ArdorConfig
from Cerebrum.Cortex.broca_decoder import ArdorDecoder


def make_cfg() -> ArdorConfig:
    return ArdorConfig(
        vocab_size=128,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        max_len=32,
        dropout=0.1,
    )


def test_decoder_forward_shape():
    model = ArdorDecoder(make_cfg())
    model.eval()

    x = torch.randint(0, 128, (2, 8), dtype=torch.long)
    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (2, 8, 128)


def test_decoder_weight_tying():
    model = ArdorDecoder(make_cfg())
    assert model.lm_head.weight.data_ptr() == model.token_embed.weight.data_ptr()


def test_decoder_model_config_core_keys():
    model = ArdorDecoder(make_cfg())
    cfg = model.model_config()

    assert cfg["arch"] == "ArdorDecoder"
    assert cfg["vocab_size"] == 128
    assert cfg["hidden_size"] == 64
    assert cfg["n_layers"] == 2
    assert cfg["n_heads"] == 4
    assert cfg["max_len"] == 32


def test_decoder_layers_alias():
    model = ArdorDecoder(make_cfg())
    assert model.layers is model.blocks
    assert len(model.blocks) == 2
