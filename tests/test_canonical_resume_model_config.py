from types import SimpleNamespace

import pytest

from Cerebrum.Cortex.ardor_config import ArdorConfig
from Cerebrum.Cortex.neural_plasticity_training import STAGE_PRESETS, resolve_model_config


ARGS = SimpleNamespace(
    hidden_size=1536,
    n_layers=33,
    n_heads=24,
    ff_mult=4,
    ctx=2048,
)
SPECIAL_IDS = {
    "pad_id": 0,
    "unk_id": 1,
    "bos_id": 2,
    "eos_id": 3,
    "user_id": 4,
    "assistant_id": 5,
    "system_id": 6,
    "eot_id": 7,
}
V14A2_MODEL_CONFIG = {
    "vocab_size": 52224,
    "hidden_size": 1536,
    "n_layers": 33,
    "n_heads": 24,
    "max_len": 2048,
    "ff_mult": 4,
    "dropout": 0.0,
    "attn_dropout": 0.0,
    "resid_dropout": 0.0,
    "layernorm_eps": 1e-5,
    "use_rope": True,
    "rope_theta": 10000.0,
}


def _resolve(*, args=ARGS, vocab_size=52224, special_ids=SPECIAL_IDS, meta=None):
    return resolve_model_config(
        ArdorConfig,
        args=args,
        cfg_stage=STAGE_PRESETS["sft"],
        vocab_size=vocab_size,
        special_ids=special_ids,
        resume_meta=meta or {},
    )


def test_v14a2_resume_model_config_overrides_sft_dropout():
    cfg = _resolve(
        meta={
            "model_config": V14A2_MODEL_CONFIG,
            "special_ids": SPECIAL_IDS,
        }
    )
    assert STAGE_PRESETS["sft"].dropout == 0.10
    assert cfg.dropout == 0.0
    assert cfg.attn_dropout == 0.0
    assert cfg.resid_dropout == 0.0
    assert cfg.use_rope is True
    assert cfg.rope_theta == 10000.0
    assert cfg.layernorm_eps == 1e-5
    assert cfg.to_dict() == V14A2_MODEL_CONFIG


def test_resume_rejects_checkpoint_architecture_conflict():
    conflicting_args = SimpleNamespace(
        hidden_size=1024,
        n_layers=33,
        n_heads=24,
        ff_mult=4,
        ctx=2048,
    )
    with pytest.raises(SystemExit, match="conflicts with canonical architecture CLI"):
        _resolve(
            args=conflicting_args,
            meta={
                "model_config": V14A2_MODEL_CONFIG,
                "special_ids": SPECIAL_IDS,
            },
        )


def test_resume_rejects_tokenizer_vocab_conflict():
    with pytest.raises(SystemExit, match="does not match tokenizer vocab_size"):
        _resolve(
            vocab_size=32000,
            meta={
                "model_config": V14A2_MODEL_CONFIG,
                "special_ids": SPECIAL_IDS,
            },
        )


def test_resume_rejects_special_token_conflict():
    wrong_special_ids = dict(SPECIAL_IDS)
    wrong_special_ids["assistant_id"] = 9
    with pytest.raises(SystemExit, match="special-token ids do not match"):
        _resolve(
            special_ids=wrong_special_ids,
            meta={
                "model_config": V14A2_MODEL_CONFIG,
                "special_ids": SPECIAL_IDS,
            },
        )


def test_legacy_resume_without_model_config_keeps_stage_fallback():
    cfg = _resolve(meta={})
    assert cfg.dropout == STAGE_PRESETS["sft"].dropout == 0.10
    assert cfg.attn_dropout == 0.10
    assert cfg.resid_dropout == 0.10
    assert cfg.use_rope is True
    assert cfg.rope_theta == 10000.0
