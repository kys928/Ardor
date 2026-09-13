from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from Erratum.build_v14a4_family_data import (
    FAMILIES,
    PRIORITY_ROUTES,
    ROUTES,
    TARGET_TRAIN_COUNTS,
    build_holdout_rows,
    build_train_rows,
    norm,
    validate,
)
from Erratum.canonical_contract_v14a2 import CANONICAL_CHECKPOINT_SHA256
from Erratum.ardor_v14a4_family_balanced_trainer import (
    FamilyBalancedSampler,
    normalize_behavior_rows,
)
from Erratum.v14a4_diagnostics import family_sem_check

CONTRACT_PATH = Path("Erratum/v14a4_family_balanced_training_contract_20260913.json")


def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_v14a4_contract_keeps_canonical_parent_and_training_mechanics_fixed():
    value = contract()
    assert value["designation"] == "v14a4_family_balanced_semantic_landing"
    assert value["canonical_parent"]["checkpoint_sha256"] == CANONICAL_CHECKPOINT_SHA256
    assert value["canonical_parent"]["checkpoint_sha256"] == "500252cead0b6ff9825ff9c4ef8e878c00403f866f23911133c26fff56999816"
    assert value["canonical_parent"]["tokenizer_sha256"] == "f4a6bb3e1f9cabf0ecf7a0a728bc78401f2cda2400033902df6cd7b5e6257514"
    assert value["objective"] == {
        "primary": "chosen-answer continuation cross entropy with prompt tokens masked from loss",
        "continuation_ce_weight": 1.0,
        "local_margin_weight": 0.0,
        "geometry_loss_weight": 0.0,
    }
    assert value["optimizer"]["lr"] == 2e-8
    assert value["optimizer"]["max_updates"] == 600
    assert value["optimizer"]["batch_rows"] == 4
    assert value["freeze_policy"] == "none"
    assert value["scientific_change"]["architecture_change"] is False
    assert value["scientific_change"]["optimizer_change"] is False
    assert value["scientific_change"]["loss_change"] is False


def test_v14a4_dataset_schema_counts_hashes_and_diversity_are_pinned():
    train = build_train_rows()
    holdout = build_holdout_rows()
    summary = validate(train, holdout)
    value = contract()

    assert len(train) == 1175
    assert len(holdout) == 256
    assert summary["train_sha256"] == value["data"]["behavior_train"]["sha256"]
    assert summary["holdout_sha256"] == value["data"]["clean_family_holdout"]["sha256"]
    assert summary["train_route_counts"] == {route: TARGET_TRAIN_COUNTS[route] for route in ROUTES}
    assert all(count == 32 for count in summary["holdout_route_counts"].values())
    assert summary["exact_train_clean_holdout_prompt_overlap"] == 0

    required = {
        "id", "split", "route", "semantic_family", "prompt_style", "answer_style",
        "question_class", "collision_route", "prompt", "chosen",
    }
    assert all(required <= set(row) for row in train + holdout)
    assert all(summary["unique_chosen_ratio_by_route"][route] >= 0.65 for route in PRIORITY_ROUTES)
    assert summary["unique_chosen_ratio_by_route"]["gradient_clipping"] > 0.89
    assert summary["unique_chosen_ratio_by_route"]["dropout"] > 0.94
    assert summary["unique_chosen_ratio_by_route"]["rag"] == 1.0
    assert summary["unique_chosen_ratio_by_route"]["tokenizer"] == 1.0


def test_collision_text_is_explicit_and_natural_not_boilerplate_in_every_answer():
    train = build_train_rows()
    assert all("specifically contrasted with" not in norm(row["chosen"]) for row in train)
    for route in ROUTES:
        rows = [row for row in train if row["route"] == route and row["question_class"] != "false_premise_correction"]
        collision_fraction = sum(row["question_class"] == "collision_comparison" for row in rows) / len(rows)
        assert 0.08 <= collision_fraction <= 0.16


def test_false_premise_rows_are_separate_difficulty_class():
    train = build_train_rows()
    false_rows = [row for row in train if row["question_class"] == "false_premise_correction"]
    assert false_rows
    assert {row["route"] for row in false_rows} == {"gradient_clipping", "dropout"}
    assert all(row["collision_route"] is None for row in false_rows)
    assert all(row["answer_style"] == "correction_first" for row in false_rows)


def _synthetic_replay_rows() -> list[dict]:
    rows: list[dict] = []
    for route in ROUTES:
        for i in range(600):
            rows.append({
                "id": f"balanced_{route}_{i}",
                "route": route,
                "semantic_family": "legacy_generic",
                "prompt": f"balanced {route} {i}\n-",
                "chosen": f"balanced chosen {route} {i}",
                "source": "balanced",
            })
        if route not in {"dropout", "correlation"}:
            for i in range(600):
                rows.append({
                    "id": f"targeted_{route}_{i}",
                    "route": route,
                    "semantic_family": "legacy_generic",
                    "prompt": f"targeted {route} {i}\n-",
                    "chosen": f"targeted chosen {route} {i}",
                    "source": "targeted",
                })
    return rows


def test_u600_sampler_consumes_every_behavior_row_exactly_once_and_balances_families():
    behavior = normalize_behavior_rows(build_train_rows())
    sampler = FamilyBalancedSampler(behavior + _synthetic_replay_rows(), seed=1029)

    selected: list[dict] = []
    for _ in range(600):
        selected.extend(sampler.next_batch(4))

    assert len(selected) == 2400
    route_counts = Counter(row["route"] for row in selected)
    assert route_counts == Counter({
        "gradient_clipping": 750,
        "dropout": 450,
        "rag": 300,
        "tokenizer": 300,
        "checkpoint": 150,
        "correlation": 150,
        "direct_answer": 150,
        "overfitting": 150,
    })

    behavior_rows = [row for row in selected if row["source"] == "behavior"]
    behavior_counts = Counter(row["route"] for row in behavior_rows)
    assert behavior_counts == Counter(TARGET_TRAIN_COUNTS)
    assert len(behavior_rows) == 1175
    assert len({row["id"] for row in behavior_rows}) == 1175

    expected_family_counts = Counter(
        f"{row['route']}/{row['semantic_family']}" for row in build_train_rows()
    )
    observed_family_counts = Counter(
        f"{row['route']}/{row['semantic_family']}" for row in behavior_rows
    )
    assert observed_family_counts == expected_family_counts

    # Targeted replay has no dropout/correlation rows; fallback must go to balanced replay,
    # not consume extra behavior rows.
    assert not any(row["source"] == "targeted" and row["route"] in {"dropout", "correlation"} for row in selected)

    # The next GC behavior slot after the pinned u600 schedule must fail closed rather than recycle.
    with pytest.raises(RuntimeError, match="forbids behavior recycling"):
        sampler.next_batch(4)


def test_family_semantic_checker_requires_family_signal_and_false_premise_correction():
    gc_false = {
        "route": "gradient_clipping",
        "semantic_family": "false_premise_correction",
        "question_class": "false_premise_correction",
    }
    good = "They do not. Exploding gradients destabilize training; gradient clipping bounds the gradient before the optimizer update."
    bad = "Exploding gradients help training stability by creating large updates."
    assert family_sem_check(gc_false, good)["passed"] is True
    assert family_sem_check(gc_false, bad)["passed"] is False
    assert "false_premise_not_corrected" in family_sem_check(gc_false, bad)["failures"]

    tok_compat = {
        "route": "tokenizer",
        "semantic_family": "model_compatibility",
        "question_class": "normal",
    }
    text = "Tokenizer compatibility matters because token IDs and the model vocabulary must use the same tokenizer mapping."
    assert family_sem_check(tok_compat, text)["passed"] is True


def test_clean_holdout_has_at_least_four_unseen_paraphrases_per_family():
    holdout = build_holdout_rows()
    counts = Counter(f"{row['route']}/{row['semantic_family']}" for row in holdout)
    for route in ROUTES:
        for family in FAMILIES[route]:
            assert counts[f"{route}/{family}"] >= 4
