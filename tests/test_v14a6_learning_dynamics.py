from __future__ import annotations

from collections import Counter

from Erratum import ardor_v14a6_learning_dynamics_probe as v14a6
from Erratum.build_v14a4_family_data import FAMILIES


def synthetic_behavior_rows(per_family: int = 8):
    rows = []
    for route, families in FAMILIES.items():
        for family in families:
            for index in range(per_family):
                rows.append(
                    {
                        "id": f"{route}-{family}-{index}",
                        "source": "behavior",
                        "route": route,
                        "semantic_family": family,
                        "prompt": "p",
                        "chosen": "c",
                    }
                )
    return rows


def test_target_selection_is_gradient_clipping_family_balanced():
    rows = synthetic_behavior_rows()
    selected = v14a6.select_target_rows(rows, seed=928, rows_per_family=4)
    assert len(selected) == 4 * len(FAMILIES["gradient_clipping"])
    counts = Counter(row["semantic_family"] for row in selected)
    assert set(counts) == set(FAMILIES["gradient_clipping"])
    assert all(value == 4 for value in counts.values())
    assert all(row["route"] == "gradient_clipping" for row in selected)


def test_pulse_sampler_recycles_only_target_and_stays_family_balanced():
    rows = v14a6.select_target_rows(
        synthetic_behavior_rows(), seed=928, rows_per_family=4
    )
    sampler = v14a6.TargetPulseSampler(rows, seed=100)
    sampled = [row for _ in range(20) for row in sampler.next_batch(4)]
    counts = Counter(row["semantic_family"] for row in sampled)
    assert all(row["route"] == "gradient_clipping" for row in sampled)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert len(sampled) > len({row["id"] for row in sampled})


def test_chase_sampler_excludes_gradient_clipping_and_does_not_recycle():
    sampler = v14a6.InterferenceChaseSampler(
        synthetic_behavior_rows(per_family=12), seed=200
    )
    sampled = [row for _ in range(20) for row in sampler.next_batch(4)]
    assert all(row["route"] != "gradient_clipping" for row in sampled)
    ids = [row["id"] for row in sampled]
    assert len(ids) == len(set(ids))


def snapshot(nll: float, top1: float, semantic: float):
    by_family = {
        family: {
            "rows": 4,
            "tokens": 20,
            "nll": nll,
            "top1_rate": top1,
        }
        for family in FAMILIES["gradient_clipping"]
    }
    family_success = {
        f"gradient_clipping/{family}": semantic
        for family in FAMILIES["gradient_clipping"]
    }
    return {
        "exact": {
            "nll": nll,
            "top1_rate": top1,
            "by_family": by_family,
        },
        "semantic": {
            "family_macro_success_rate": semantic,
            "family_success_rate": family_success,
        },
    }


def test_decision_detects_target_token_learning_failure():
    decision = v14a6.build_decision(
        snapshot(5.0, 0.20, 0.0),
        snapshot(4.99, 0.205, 0.0),
        snapshot(4.98, 0.21, 0.0),
    )
    assert decision["primary_interpretation"] == "target_token_learning_failure_supported"


def test_decision_detects_interference_after_successful_pulse():
    decision = v14a6.build_decision(
        snapshot(5.0, 0.20, 0.0),
        snapshot(4.0, 0.40, 0.25),
        snapshot(4.7, 0.25, 0.0),
    )
    assert decision["primary_interpretation"] == "interference_or_overwrite_supported"


def test_decision_detects_token_learning_without_semantic_transfer():
    decision = v14a6.build_decision(
        snapshot(5.0, 0.20, 0.0),
        snapshot(4.0, 0.40, 0.0),
        snapshot(4.05, 0.39, 0.0),
    )
    assert (
        decision["primary_interpretation"]
        == "token_learning_without_semantic_generalization_supported"
    )
