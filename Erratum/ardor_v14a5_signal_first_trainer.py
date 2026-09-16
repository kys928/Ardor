#!/usr/bin/env python3
"""v14a5 signal-first semantic-landing probe.

This keeps the v14a4 canonical parent, tokenizer, objective, optimizer, and evaluators fixed.
It changes only supervision quality/exposure:
- repair dropout/overfitting_relationship chosen continuations when they omit the family relation;
- map legacy replay to a semantic family only when the chosen answer uniquely supports one family;
- spend four of every five early semantic-landing slots on authored behavior, and use mapped replay
  only for the currently targeted family. Generic replay is retained but not sampled in this u100 probe.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch

from Erratum import ardor_v14a4_family_balanced_trainer as base
from Erratum.build_v14a4_family_data import (
    FAMILIES,
    FAMILY_SIGNALS,
    ROUTE_SIGNALS,
    norm,
)

OUTPUT_DIR = Path(
    "/workspace/Ardor/training/runs/"
    "sft_v14a5_signal_first_semantic_landing_probe_u100"
)
POLICY_CONTRACT_PATH = Path(
    "/opt/Ardor/Erratum/v14a5_signal_first_training_contract_20260916.json"
)
SOURCE_CYCLE = ("behavior", "behavior", "behavior", "behavior", "replay")
REPLAY_SOURCES = ("balanced", "targeted")
MAX_PROBE_UPDATES = 100
GC_MIN_BEHAVIOR_EXPOSURE = 10
OTHER_MIN_BEHAVIOR_EXPOSURE = 4

_REPAIR_VARIANTS = (
    "Dropout's training-time masks can reduce overfitting by making memorized feature dependencies less reliable, which supports generalization beyond the training set.",
    "Dropout reduces overfitting pressure by changing the training-time mask, so features must remain useful across masks and generalize beyond memorized training patterns.",
    "Because dropout applies changing training-time masks, a rigid memorized pathway is less dependable; this can reduce overfitting and improve generalization.",
    "Training-time dropout masks make fixed feature dependencies unreliable, which is why dropout can act against overfitting and support generalization.",
)


def _has_any(text: str, signals: Sequence[str]) -> bool:
    low = norm(text)
    return any(norm(signal) in low for signal in signals)


def _stable_variant(row_id: str) -> str:
    index = sum((i + 1) * ord(ch) for i, ch in enumerate(str(row_id))) % len(_REPAIR_VARIANTS)
    return _REPAIR_VARIANTS[index]


def normalize_behavior_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize v14a4 behavior rows and repair only dropout/overfitting_relationship."""
    normalized = base._V14A5_ORIGINAL_NORMALIZE_BEHAVIOR(rows)
    target = ("dropout", "overfitting_relationship")
    for row in normalized:
        if (str(row["route"]), str(row["semantic_family"])) != target:
            continue
        chosen = str(row["chosen"])
        route_ok = _has_any(chosen, ROUTE_SIGNALS["dropout"])
        family_ok = _has_any(chosen, FAMILY_SIGNALS[target])
        if not (route_ok and family_ok):
            row["chosen"] = chosen.rstrip() + " " + _stable_variant(str(row["id"]))
            row["supervision_repair"] = "v14a5_dropout_overfitting_relationship"
        if not _has_any(str(row["chosen"]), ROUTE_SIGNALS["dropout"]):
            raise RuntimeError(f"v14a5 repair still lacks dropout route signal: {row['id']}")
        if not _has_any(str(row["chosen"]), FAMILY_SIGNALS[target]):
            raise RuntimeError(f"v14a5 repair still lacks overfitting-family signal: {row['id']}")
    return normalized


def infer_replay_family(route: str, chosen: str) -> tuple[str, dict[str, int]]:
    """Map replay by chosen-answer semantics only; ambiguous/broad replay remains generic."""
    if route not in FAMILIES or not _has_any(chosen, ROUTE_SIGNALS[route]):
        return "legacy_generic", {}
    scores: dict[str, int] = {}
    low = norm(chosen)
    for family in FAMILIES[route]:
        score = sum(1 for signal in FAMILY_SIGNALS[(route, family)] if norm(signal) in low)
        scores[family] = score
    best = max(scores.values(), default=0)
    winners = sorted(family for family, score in scores.items() if score == best and score > 0)
    if best <= 0 or len(winners) != 1:
        return "legacy_generic", scores
    return winners[0], scores


def normalize_replay_rows(
    rows: Sequence[dict[str, Any]],
    source: str,
    *,
    required_edges: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    normalized = base._V14A5_ORIGINAL_NORMALIZE_REPLAY(
        rows, source, required_edges=required_edges
    )
    for row in normalized:
        family, scores = infer_replay_family(str(row["route"]), str(row["chosen"]))
        row["semantic_family"] = family
        row["family_map_method"] = "chosen_unique_family_signal"
        row["family_map_scores"] = scores
    return normalized


class SignalFirstSampler:
    """Family-first u100 sampler with authored behavior as the early landing signal."""

    def __init__(self, rows: Sequence[dict[str, Any]], seed: int):
        self.rng = random.Random(seed)
        self.behavior: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.replay: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            route = str(row["route"])
            family = str(row["semantic_family"])
            source = str(row["source"])
            if source == "behavior":
                self.behavior[(route, family)].append(dict(row))
            elif family != "legacy_generic":
                self.replay[(source, route, family)].append(dict(row))

        for pool in [*self.behavior.values(), *self.replay.values()]:
            self.rng.shuffle(pool)

        self.route_cursor = 0
        self.family_cursor: Counter[str] = Counter()
        self.source_cursor: Counter[str] = Counter()
        self.replay_source_cursor: Counter[str] = Counter()

    def _pop_behavior(self, route: str, family: str) -> dict[str, Any]:
        pool = self.behavior.get((route, family), [])
        if not pool:
            raise RuntimeError(
                f"v14a5 exhausted authored behavior for {route}/{family}; "
                "the u100 signal-first contract forbids behavior recycling"
            )
        return pool.pop()

    def _pop_replay(self, route: str, family: str) -> dict[str, Any] | None:
        start = self.replay_source_cursor[route] % len(REPLAY_SOURCES)
        self.replay_source_cursor[route] += 1
        for offset in range(len(REPLAY_SOURCES)):
            source = REPLAY_SOURCES[(start + offset) % len(REPLAY_SOURCES)]
            pool = self.replay.get((source, route, family), [])
            if pool:
                return pool.pop()
        return None

    def next_row(self) -> dict[str, Any]:
        route = base.ROUTE_SEQUENCE[self.route_cursor % len(base.ROUTE_SEQUENCE)]
        self.route_cursor += 1

        families = FAMILIES[route]
        family = families[self.family_cursor[route] % len(families)]
        self.family_cursor[route] += 1

        source_slot = SOURCE_CYCLE[self.source_cursor[route] % len(SOURCE_CYCLE)]
        self.source_cursor[route] += 1
        if source_slot == "replay":
            replay = self._pop_replay(route, family)
            if replay is not None:
                if replay["semantic_family"] == "legacy_generic":
                    raise RuntimeError("v14a5 sampled legacy_generic replay")
                return replay
        return self._pop_behavior(route, family)

    def next_batch(self, batch_rows: int) -> list[dict[str, Any]]:
        return [self.next_row() for _ in range(batch_rows)]


def _save_checkpoint_v14a5(
    path: Path,
    model,
    optimizer,
    *,
    update: int,
    args,
    dataset_manifest: dict[str, Any],
    gate: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "meta": {
                "stage": "v14a5_signal_first_semantic_landing_probe_u100",
                "step": int(update),
                "trainer": "Erratum/ardor_v14a5_signal_first_trainer.py",
                "model_config": dict(base.MODEL_CONFIG),
                "canonical_parent": base.canonical_parent_reference(),
                "tokenizer_contract": {
                    "path": str(base.TOKENIZER_PATH),
                    "vocab_size": base.EXPECTED_VOCAB_SIZE,
                    "special_ids": dict(base.EXPECTED_SPECIAL_IDS),
                },
                "objective": {
                    "continuation_ce_weight": 1.0,
                    "local_margin_weight": 0.0,
                    "geometry_loss_weight": 0.0,
                    "prompt_tokens_masked": True,
                },
                "freeze_policy": "none",
                "sampling_policy": (
                    "family-first u100; 4/5 authored behavior slots; mapped replay only "
                    "for current family; generic replay retained but not sampled"
                ),
                "parent_data_contract": str(base.CONTRACT_PATH),
                "policy_contract": str(POLICY_CONTRACT_PATH),
                "dataset_manifest": dataset_manifest,
                "promotion_gate": gate,
                "diagnostics": diagnostics,
                "args": vars(args),
                "created_at_unix": time.time(),
            },
        },
        str(path),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_decision(output_dir: Path, checkpoint: str) -> dict[str, Any]:
    summary = _read_json(output_dir / "run_summary.json")
    sampled = _read_jsonl(output_dir / "sampled_rows.jsonl")
    source_counts = Counter(str(row["source"]) for row in sampled)
    behavior_exposure: Counter[str] = Counter()
    replay_exposure: Counter[str] = Counter()
    for row in sampled:
        key = f"{row['route']}/{row['semantic_family']}"
        if row["source"] == "behavior":
            behavior_exposure[key] += 1
        else:
            replay_exposure[key] += 1

    final_family = summary["final"]["clean_family"]["family_success_rate"]
    baseline_family = summary["baseline"]["clean_family"]["family_success_rate"]
    families: dict[str, Any] = {}
    for key in sorted(final_family):
        families[key] = {
            "behavior_exposure": behavior_exposure[key],
            "mapped_replay_exposure": replay_exposure[key],
            "total_exposure": behavior_exposure[key] + replay_exposure[key],
            "baseline_success_rate": float(baseline_family[key]),
            "final_success_rate": float(final_family[key]),
        }

    gc_keys = sorted(key for key in final_family if key.startswith("gradient_clipping/"))
    other_keys = sorted(key for key in final_family if not key.startswith("gradient_clipping/"))
    gc_min = min((behavior_exposure[key] for key in gc_keys), default=0)
    other_min = min((behavior_exposure[key] for key in other_keys), default=0)
    generic_sampled = sum(
        1 for row in sampled if str(row["semantic_family"]) == "legacy_generic"
    )
    behavior_share = source_counts["behavior"] / max(1, len(sampled))
    replay_share = 1.0 - behavior_share

    signal_precondition = {
        "gradient_clipping_min_behavior_exposure": gc_min,
        "gradient_clipping_required_min": GC_MIN_BEHAVIOR_EXPOSURE,
        "other_family_min_behavior_exposure": other_min,
        "other_family_required_min": OTHER_MIN_BEHAVIOR_EXPOSURE,
        "behavior_share": behavior_share,
        "required_behavior_share": 0.80,
        "generic_replay_sampled": generic_sampled,
        "passed": (
            gc_min >= GC_MIN_BEHAVIOR_EXPOSURE
            and other_min >= OTHER_MIN_BEHAVIOR_EXPOSURE
            and behavior_share >= 0.80
            and generic_sampled == 0
        ),
    }

    gc_failures = {
        key: float(final_family[key])
        for key in gc_keys
        if float(final_family[key]) < 1.0
    }
    if not signal_precondition["passed"]:
        interpretation = "signal_not_yet_saturated"
    elif gc_failures:
        interpretation = "optimization_or_learning_limit_supported_after_signal_control"
    else:
        interpretation = "gradient_clipping_clean_family_landed"

    return {
        "experiment": "v14a5_signal_first_semantic_landing_probe_u100",
        "checkpoint": checkpoint,
        "rows_sampled": len(sampled),
        "source_counts": dict(source_counts),
        "behavior_share": behavior_share,
        "replay_share": replay_share,
        "signal_precondition": signal_precondition,
        "gradient_clipping_remaining_failures": gc_failures,
        "interpretation": interpretation,
        "interpretation_strength": (
            "strong evidence, not causal proof"
            if interpretation == "optimization_or_learning_limit_supported_after_signal_control"
            else "diagnostic"
        ),
        "families": families,
    }


def _postprocess_outputs(output_dir: Path, result_checkpoint: str, diagnostic_only: bool) -> None:
    run_config = _read_json(output_dir / "run_config.json")
    run_config.update(
        {
            "experiment": "v14a5_signal_first_semantic_landing_probe_u100",
            "trainer": "Erratum/ardor_v14a5_signal_first_trainer.py",
            "parent_data_contract": str(base.CONTRACT_PATH),
            "policy_contract": str(POLICY_CONTRACT_PATH),
            "sampling_policy": (
                "family-first u100; 4/5 authored behavior slots; mapped replay only "
                "for current family; generic replay retained but not sampled"
            ),
            "max_probe_updates": MAX_PROBE_UPDATES,
        }
    )
    _write_json(output_dir / "run_config.json", run_config)

    summary = _read_json(output_dir / "run_summary.json")
    summary["trainer"] = "ardor_v14a5_signal_first_trainer"
    summary["experiment"] = "v14a5_signal_first_semantic_landing_probe_u100"
    summary["parent_data_contract"] = str(base.CONTRACT_PATH)
    summary["policy_contract"] = str(POLICY_CONTRACT_PATH)
    summary["result_checkpoint"] = result_checkpoint
    summary["result_checkpoint_diagnostic_only"] = diagnostic_only
    _write_json(output_dir / "run_summary.json", summary)

    decision = build_decision(output_dir, result_checkpoint)
    _write_json(output_dir / "v14a5_signal_landing_decision.json", decision)


def run(args) -> None:
    if args.max_updates > MAX_PROBE_UPDATES:
        raise ValueError(
            f"v14a5 is an early semantic-landing diagnostic capped at {MAX_PROBE_UPDATES} updates"
        )

    output_dir = Path(args.output_dir)
    model_box: dict[str, Any] = {}
    original_load = base._load_canonical_model

    def capture_load(device: str):
        model, strict_load, meta = original_load(device)
        model_box["model"] = model
        return model, strict_load, meta

    base._V14A5_ORIGINAL_NORMALIZE_BEHAVIOR = base.normalize_behavior_rows
    base._V14A5_ORIGINAL_NORMALIZE_REPLAY = base.normalize_replay_rows
    original_sampler = base.FamilyBalancedSampler
    original_save = base.save_checkpoint
    base.normalize_behavior_rows = normalize_behavior_rows
    base.normalize_replay_rows = normalize_replay_rows
    base.FamilyBalancedSampler = SignalFirstSampler
    base.save_checkpoint = _save_checkpoint_v14a5
    base._load_canonical_model = capture_load

    try:
        base.run(args)

        summary = _read_json(output_dir / "run_summary.json")
        best_checkpoint = summary.get("best_checkpoint")
        diagnostic_only = False
        if best_checkpoint:
            result_checkpoint = str(best_checkpoint)
        else:
            diagnostic_only = True
            diagnostic_path = output_dir / "checkpoints" / "diagnostic_u100_model.pt"
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            model = model_box.get("model")
            if model is None:
                raise RuntimeError("v14a5 could not capture final model for diagnostic checkpoint")
            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": {
                        "stage": "v14a5_signal_first_semantic_landing_probe_u100",
                        "step": int(args.max_updates),
                        "trainer": "Erratum/ardor_v14a5_signal_first_trainer.py",
                        "canonical_parent": base.canonical_parent_reference(),
                        "tokenizer_contract": {
                            "path": str(base.TOKENIZER_PATH),
                            "vocab_size": base.EXPECTED_VOCAB_SIZE,
                            "special_ids": dict(base.EXPECTED_SPECIAL_IDS),
                        },
                        "model_config": dict(base.MODEL_CONFIG),
                        "diagnostic_only": True,
                        "promotion_gate_passed": False,
                        "objective_unchanged_from_v14a4": True,
                        "parent_data_contract": str(base.CONTRACT_PATH),
                        "policy_contract": str(POLICY_CONTRACT_PATH),
                        "created_at_unix": time.time(),
                    },
                },
                str(diagnostic_path),
            )
            result_checkpoint = str(diagnostic_path)

        _postprocess_outputs(output_dir, result_checkpoint, diagnostic_only)
    finally:
        base.normalize_behavior_rows = base._V14A5_ORIGINAL_NORMALIZE_BEHAVIOR
        base.normalize_replay_rows = base._V14A5_ORIGINAL_NORMALIZE_REPLAY
        base.FamilyBalancedSampler = original_sampler
        base.save_checkpoint = original_save
        base._load_canonical_model = original_load
        delattr(base, "_V14A5_ORIGINAL_NORMALIZE_BEHAVIOR")
        delattr(base, "_V14A5_ORIGINAL_NORMALIZE_REPLAY")


def build_argparser() -> argparse.ArgumentParser:
    parser = base.build_argparser()
    parser.set_defaults(
        output_dir=str(OUTPUT_DIR),
        max_updates=MAX_PROBE_UPDATES,
        eval_every=25,
        eval_updates=[1, 10],
        stop_on_regression=False,
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
