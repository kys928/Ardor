from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CORTEX_DIR = ROOT / "Cerebrum" / "Cortex"
CORTICAL_INTEGRATION_DIR = ROOT / "Cerebrum" / "CorticalIntegration"
PRAETOR_DIR = ROOT / "Praetor"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CORTEX_DIR))
sys.path.insert(0, str(CORTICAL_INTEGRATION_DIR))
sys.path.insert(0, str(PRAETOR_DIR))


def require(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Missing expected file: {path}")


def assert_shape(name: str, got: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if got != expected:
        raise AssertionError(f"{name}: expected {expected}, got {got}")


def smoke_repo_layout() -> None:
    require(CORTEX_DIR / "broca_decoder.py")
    require(CORTEX_DIR / "posterior_parietal_cortex.py")
    require(CORTEX_DIR / "prefrontal_cortex.py")
    require(CORTEX_DIR / "neural_plasticity_training.py")
    require(CORTICAL_INTEGRATION_DIR / "REM.py")
    require(PRAETOR_DIR / "GUI_Cortex.py")
    print("repo layout smoke passed")


def smoke_decoder() -> None:
    from ardor_config import ArdorConfig
    from broca_decoder import ArdorDecoder

    vocab_size = 128
    hidden_dim = 64
    num_layers = 2
    heads = 4
    max_len = 32
    batch = 2
    seq = 8

    cfg = ArdorConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_dim,
        n_layers=num_layers,
        n_heads=heads,
        max_len=max_len,
        dropout=0.1,
    )

    model = ArdorDecoder(cfg)
    model.eval()

    x = torch.randint(0, vocab_size, (batch, seq), dtype=torch.long)
    with torch.no_grad():
        logits = model(x)

    assert_shape("decoder logits", tuple(logits.shape), (batch, seq, vocab_size))
    print("decoder smoke passed")


def smoke_encoder() -> None:
    from ardor_config import ArdorConfig
    from posterior_parietal_cortex import ArdorEncoder

    vocab_size = 128
    hidden_dim = 64
    num_layers = 2
    heads = 4
    max_len = 32
    batch = 2
    seq = 8

    cfg = ArdorConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_dim,
        n_layers=num_layers,
        n_heads=heads,
        max_len=max_len,
        dropout=0.1,
    )

    model = ArdorEncoder(cfg, use_cls_token=True)
    model.eval()

    x = torch.randint(0, vocab_size, (batch, seq), dtype=torch.long)
    with torch.no_grad():
        h = model(x)
        h2, pooled = model(x, return_pooled=True, pool="mean")

    if tuple(h.shape)[0] != batch or tuple(h.shape)[2] != hidden_dim:
        raise AssertionError(f"encoder hidden shape looks wrong: {tuple(h.shape)}")
    assert_shape("encoder pooled", tuple(pooled.shape), (batch, hidden_dim))
    print("encoder smoke passed")


def smoke_text_presence() -> None:
    training_source = (CORTEX_DIR / "neural_plasticity_training.py").read_text(encoding="utf-8")
    rem_source = (CORTICAL_INTEGRATION_DIR / "REM.py").read_text(encoding="utf-8")

    if "ArdorDecoder" not in training_source:
        raise AssertionError("training script missing ArdorDecoder reference")
    if "ArdorDecoder" not in rem_source:
        raise AssertionError("REM script missing ArdorDecoder reference")

    print("text presence smoke passed")


def main() -> None:
    print("Running Ardor CI smoke test...")
    smoke_repo_layout()
    smoke_decoder()
    smoke_encoder()
    smoke_text_presence()
    print("Ardor CI smoke test passed")


if __name__ == "__main__":
    main()
