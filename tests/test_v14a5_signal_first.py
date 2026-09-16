from __future__ import annotations

from collections import Counter

from Erratum import ardor_v14a4_family_balanced_trainer as base
from Erratum import ardor_v14a5_signal_first_trainer as v14a5
from Erratum.build_v14a4_family_data import (
    FAMILIES,
    FAMILY_SIGNALS,
    ROUTE_SIGNALS,
    build_train_rows,
    norm,
)


def has_any(text: str, signals: list[str]) -> bool:
    low = norm(text)
    return any(norm(signal) in low for signal in signals)


def test_repairs_only_dropout_overfitting_relationship_supervision():
    raw = build_train_rows()
    original = base.normalize_behavior_rows(raw)
    base._V14A5_ORIGINAL_NORMALIZE_BEHAVIOR = base.normalize_behavior_rows
    try:
        repaired = v14a5.normalize_behavior_rows(raw)
    finally:
        delattr(base, "_V14A5_ORIGINAL_NORMALIZE_BEHAVIOR")

    before = {row["id"]: row for row in original}
    changed = []
    for row in repaired:
        target = row["route"] == "dropout" and row["semantic_family"] == "overfitting_relationship"
        if target:
            assert has_any(row["chosen"], ROUTE_SIGNALS["dropout"])
            assert has_any(
                row["chosen"], FAMILY_SIGNALS[("dropout", "overfitting_relationship")]
            )
            if row["chosen"] != before[row["id"]]["chosen"]:
                changed.append(row["id"])
                assert row["supervision_repair"] == "v14a5_dropout_overfitting_relationship"
        else:
            assert row["chosen"] == before[row["id"]]["chosen"]

    assert changed, "The v14a5 repair should fix at least one defective target row"


def test_replay_family_mapping_is_conservative_and_answer_only():
    family, _ = v14a5.infer_replay_family(
        "dropout", "Dropout improves generalization."
    )
    assert family == "legacy_generic"

    family, _ = v14a5.infer_replay_family(
        "dropout", "Dropout can reduce overfitting and improve generalization."
    )
    assert family == "overfitting_relationship"

    family, _ = v14a5.infer_replay_family(
        "dropout", "Overfitting and generalization can move in opposite directions."
    )
    assert family == "legacy_generic"


def synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for route, families in FAMILIES.items():
        for family in families:
            for index in range(30):
                rows.append(
                    {
                        "id": f"behavior-{route}-{family}-{index}",
                        "source": "behavior",
                        "route": route,
                        "semantic_family": family,
                        "prompt": "p",
                        "chosen": "c",
                    }
                )
            for source in ("balanced", "targeted"):
                rows.append(
                    {
                        "id": f"{source}-{route}-{family}",
                        "source": source,
                        "route": route,
                        "semantic_family": family,
                        "prompt": "p",
                        "chosen": "c",
                    }
                )
    rows.append(
        {
            "id": "generic-replay-must-not-sample",
            "source": "balanced",
            "route": "gradient_clipping",
            "semantic_family": "legacy_generic",
            "prompt": "p",
            "chosen": "c",
        }
    )
    return rows


def test_u100_sampler_prioritizes_behavior_and_never_spends_generic_replay():
    sampler = v14a5.SignalFirstSampler(synthetic_rows(), seed=928)
    sampled = [sampler.next_row() for _ in range(400)]

    sources = Counter(str(row["source"]) for row in sampled)
    behavior_share = sources["behavior"] / len(sampled)
    assert behavior_share >= 0.80
    assert all(row["semantic_family"] != "legacy_generic" for row in sampled)

    replay_ids = [str(row["id"]) for row in sampled if row["source"] != "behavior"]
    assert len(replay_ids) == len(set(replay_ids)), "Mapped replay must not recycle in u100"

    behavior_exposure = Counter(
        (str(row["route"]), str(row["semantic_family"]))
        for row in sampled
        if row["source"] == "behavior"
    )
    for route, families in FAMILIES.items():
        for family in families:
            minimum = 10 if route == "gradient_clipping" else 4
            assert behavior_exposure[(route, family)] >= minimum


def test_signal_first_probe_is_hard_capped_at_u100():
    parser = v14a5.build_argparser()
    args = parser.parse_args(["--max-updates", "101"])
    try:
        v14a5.run(args)
    except ValueError as exc:
        assert "capped at 100 updates" in str(exc)
    else:
        raise AssertionError("v14a5 must reject runs beyond u100")
