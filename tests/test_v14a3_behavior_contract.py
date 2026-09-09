from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from Erratum.build_v14a3_behavior_data import COLLISIONS, ROUTES, build_rows, validate
from Erratum.canonical_contract_v14a2 import (
    CANONICAL_CHECKPOINT_SHA256,
    EXPECTED_SPECIAL_IDS,
    EXPECTED_VOCAB_SIZE,
    MODEL_CONFIG,
    validate_static_contract,
)
from Erratum.ardor_v14a3_behavior_first_trainer import ROUTE_SEQUENCE


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONTRACT = ROOT / "Erratum/v14a3_behavior_first_training_contract_20260909.json"


def test_canonical_v14a2_contract_is_exact_33_layer_tokenizer_v9_contract():
    validate_static_contract()
    assert MODEL_CONFIG["n_layers"] == 33
    assert MODEL_CONFIG["n_heads"] == 24
    assert MODEL_CONFIG["hidden_size"] == 1536
    assert EXPECTED_VOCAB_SIZE == 52224
    assert EXPECTED_SPECIAL_IDS == {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<|user|>": 4,
        "<|assistant|>": 5,
        "<|system|>": 6,
        "<|eot|>": 7,
    }
    assert CANONICAL_CHECKPOINT_SHA256 == "500252cead0b6ff9825ff9c4ef8e878c00403f866f23911133c26fff56999816"


def test_v14a3_training_contract_changes_signal_not_architecture_tricks():
    contract = json.loads(TRAINING_CONTRACT.read_text(encoding="utf-8"))
    assert contract["objective"] == {
        "primary": "chosen-answer continuation cross entropy with prompt tokens masked from loss",
        "continuation_ce_weight": 1.0,
        "local_margin_weight": 0.0,
        "geometry_loss_weight": 0.0,
        "rationale": contract["objective"]["rationale"],
    }
    assert contract["optimizer"]["lr"] == 2e-8
    assert contract["optimizer"]["max_updates"] == 600
    assert contract["optimizer"]["batch_rows"] == 4
    assert contract["optimizer"]["warmup_updates"] == 20
    assert contract["optimizer"]["grad_clip"] == 0.5
    assert contract["optimizer"]["weight_decay"] == 0.0
    assert contract["freeze_policy"]["v14a3"].startswith("none by default")
    assert contract["promotion"]["raw_eight_target_nll_classifier_is_gating"] is False


def test_v14a3_route_pressure_cycle_matches_contract():
    contract = json.loads(TRAINING_CONTRACT.read_text(encoding="utf-8"))
    expected = {k: int(v) for k, v in contract["route_schedule_per_16_rows"].items()}
    assert len(ROUTE_SEQUENCE) == 16
    assert dict(Counter(ROUTE_SEQUENCE)) == expected


def test_behavior_first_dataset_is_varied_balanced_and_collision_complete():
    rows = build_rows()
    summary = validate(rows, holdout=[])
    assert summary["rows"] == len(rows)
    assert set(summary["route_counts"]) == set(ROUTES)
    assert summary["holdout_exact_prompt_overlap"] == 0
    assert summary["chosen_duplicate_fraction"] <= 0.15
    assert summary["unique_chosen"] >= int(0.85 * len(rows))

    directed = {
        (str(row["route"]), str(row["hard_negative"]))
        for row in rows
        if row.get("hard_negative")
    }
    for left, right in COLLISIONS:
        assert (left, right) in directed
        assert (right, left) in directed
