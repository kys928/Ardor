#!/usr/bin/env python3
"""Behavior-first diagnostics and promotion gate for v14a3.

This module deliberately reuses the canonical v14a2 semantic checker, generation format,
route targets, representation capture, and chosen-answer scorer. The only new diagnostic is
prior-calibrated fixed-target ranking; raw eight-target NLL ranking is retained for inspection
but is never a promotion gate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
from statistics import mean
from typing import Any, Sequence

import torch

from Erratum.canonical_analysis_v14a2 import (
    LayerCapture,
    ROUTES,
    SEED,
    anchor_of,
    collect_reps,
    geometry_stats,
    score_actual_chosen,
    score_target_batch,
    stratified,
)

PRIORITY_ROUTES = ["gradient_clipping", "dropout", "rag", "tokenizer"]
GUARD_ROUTES = [r for r in ROUTES if r not in PRIORITY_ROUTES]


def compact_behavior(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k != "outputs"}


@torch.inference_mode()
def evaluate_behavior(
    model,
    tok,
    holdout: Sequence[dict[str, Any]],
    device: torch.device,
    history,
    args,
    special: dict[str, list[int]],
    *,
    name: str,
) -> dict[str, Any]:
    """Run the exact historical v14a2 generation/semantic checks with route-wise missing-signal accounting."""
    model.eval()
    rows = list(holdout[: int(args.eval_samples)])
    route_total: Counter[str] = Counter()
    route_bad: Counter[str] = Counter()
    route_missing: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    repetitions: list[float] = []
    loops = 0
    empties = 0
    outputs: list[dict[str, Any]] = []

    for row in rows:
        route = str(row["route"])
        generated = history.generate(model, tok, row["prompt"], device, args, special)
        check = history.sem_check(route, generated["text"])
        failures = list(check["failures"])
        if generated["empty"]:
            failures.append("empty_generation")
        if generated["first_special"]:
            failures.append("first_special")
        if generated["word_count"] < int(args.min_eval_words):
            failures.append("too_short_generation")

        bad = bool(failures)
        missing_key = f"{route}_missing_signal"
        route_total[route] += 1
        route_bad[route] += int(bad)
        route_missing[route] += int(missing_key in failures)
        failure_counts.update(failures)
        repetitions.append(float(generated["repetition_rate"]))
        loops += int(check["model_loop"]["has_model_loop"])
        empties += int(generated["empty"])
        outputs.append({
            "id": str(row["id"]),
            "route": route,
            "generation": generated["text"],
            "bad": bad,
            "failures": failures,
        })

    n = max(1, len(outputs))
    return {
        "name": name,
        "rows": len(outputs),
        "bad_count": sum(route_bad.values()),
        "bad_rate": sum(route_bad.values()) / n,
        "success_rate": 1.0 - (sum(route_bad.values()) / n),
        "model_loop_count": loops,
        "model_loop_rate": loops / n,
        "empty_count": empties,
        "mean_repetition_rate": sum(repetitions) / max(1, len(repetitions)),
        "failure_counts": dict(failure_counts),
        "route_counts": dict(route_total),
        "route_bad_rate": {
            route: route_bad[route] / max(1, route_total[route]) for route in sorted(route_total)
        },
        "route_missing_signal_rate": {
            route: route_missing[route] / max(1, route_total[route]) for route in sorted(route_total)
        },
        "outputs": outputs,
    }


@torch.inference_mode()
def evaluate_balanced_chosen(
    model,
    tok,
    balanced_rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    per_route: int = 32,
    seed: int = SEED + 103,
) -> dict[str, Any]:
    rows = stratified(balanced_rows, per_route, seed)
    scored: list[dict[str, Any]] = []
    for row in rows:
        score = score_actual_chosen(model, tok, anchor_of(row), str(row["chosen"]), device)
        if score.get("tokens", 0):
            scored.append({"route": str(row["route"]), **score})

    def agg(values: list[float]) -> float | None:
        return mean(values) if values else None

    by_route: dict[str, dict[str, Any]] = {}
    for route in ROUTES:
        rs = [r for r in scored if r["route"] == route]
        by_route[route] = {
            "n": len(rs),
            "mean_chosen_nll": agg([float(r["mean_nll"]) for r in rs]),
            "mean_chosen_top1_rate": agg([float(r["top1_rate"]) for r in rs]),
            "mean_chosen_token_rank": agg([float(r["mean_rank"]) for r in rs]),
        }
    return {
        "rows": len(scored),
        "overall_nll": agg([float(r["mean_nll"]) for r in scored]),
        "overall_top1_rate": agg([float(r["top1_rate"]) for r in scored]),
        "overall_mean_rank": agg([float(r["mean_rank"]) for r in scored]),
        "by_route": by_route,
    }


@torch.inference_mode()
def calibrated_target_ranking(
    model,
    tok,
    balanced_rows: Sequence[dict[str, Any]],
    holdout: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    calibration_per_route: int = 8,
    seed: int = SEED + 107,
) -> dict[str, Any]:
    """Prior-calibrate the historical fixed-target NLL diagnostic before route ranking."""
    calibration = stratified(balanced_rows, calibration_per_route, seed)
    priors: dict[str, list[float]] = defaultdict(list)
    for row in calibration:
        score = score_target_batch(model, tok, anchor_of(row), device)
        for target_route, details in score["route_scores"].items():
            priors[target_route].append(float(details["mean_target_nll"]))
    offsets = {route: mean(priors[route]) for route in ROUTES}

    raw_correct = 0
    calibrated_correct = 0
    by_route_total: Counter[str] = Counter()
    by_route_raw: Counter[str] = Counter()
    by_route_calibrated: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for row in holdout:
        route = str(row["route"])
        score = score_target_batch(model, tok, str(row["prompt"]), device)
        raw_pred = str(score["predicted_route"])
        calibrated_scores = {
            target_route: float(details["mean_target_nll"]) - offsets[target_route]
            for target_route, details in score["route_scores"].items()
        }
        calibrated_pred = min(ROUTES, key=lambda r: calibrated_scores[r])
        raw_ok = raw_pred == route
        calibrated_ok = calibrated_pred == route
        raw_correct += int(raw_ok)
        calibrated_correct += int(calibrated_ok)
        by_route_total[route] += 1
        by_route_raw[route] += int(raw_ok)
        by_route_calibrated[route] += int(calibrated_ok)
        if len(examples) < 32:
            examples.append({
                "id": str(row["id"]),
                "route": route,
                "raw_predicted_route": raw_pred,
                "calibrated_predicted_route": calibrated_pred,
                "calibrated_scores": calibrated_scores,
            })

    n = max(1, sum(by_route_total.values()))
    return {
        "calibration_rows": len(calibration),
        "target_nll_offsets": offsets,
        "raw_accuracy": raw_correct / n,
        "calibrated_accuracy": calibrated_correct / n,
        "by_route": {
            route: {
                "n": by_route_total[route],
                "raw_accuracy": by_route_raw[route] / max(1, by_route_total[route]),
                "calibrated_accuracy": by_route_calibrated[route] / max(1, by_route_total[route]),
            }
            for route in ROUTES
        },
        "raw_classifier_is_promotion_gate": False,
        "examples": examples,
    }


@torch.inference_mode()
def evaluate_final_geometry(
    model,
    tok,
    balanced_rows: Sequence[dict[str, Any]],
    holdout: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    prototype_per_route: int = 64,
    seed: int = SEED + 101,
) -> dict[str, Any]:
    proto_rows = stratified(balanced_rows, prototype_per_route, seed)
    capture = LayerCapture(model)
    train_reps, train_labels, _ = collect_reps(model, tok, capture, proto_rows, device)
    holdout_reps, holdout_labels, _ = collect_reps(model, tok, capture, holdout, device)
    capture.close()
    stats = geometry_stats(
        train_reps["final_norm"], train_labels, holdout_reps["final_norm"], holdout_labels
    )
    return {k: v for k, v in stats.items() if k not in {"predictions", "correct", "gaps"}}


def _priority_mean(mapping: dict[str, float]) -> float:
    return mean(float(mapping[r]) for r in PRIORITY_ROUTES)


def primary_promotion_gate(
    baseline_behavior: dict[str, Any],
    current_behavior: dict[str, Any],
    baseline_chosen: dict[str, Any],
    current_chosen: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["overall_behavior_bad_rate_improved"] = (
        float(current_behavior["bad_rate"]) < float(baseline_behavior["bad_rate"])
    )
    base_priority = _priority_mean(baseline_behavior["route_bad_rate"])
    cur_priority = _priority_mean(current_behavior["route_bad_rate"])
    checks["priority_route_mean_bad_rate_improved"] = cur_priority < base_priority
    for route in PRIORITY_ROUTES:
        checks[f"{route}_bad_rate_not_worse"] = (
            float(current_behavior["route_bad_rate"][route])
            <= float(baseline_behavior["route_bad_rate"][route])
        )
        checks[f"{route}_missing_signal_not_worse"] = (
            float(current_behavior["route_missing_signal_rate"][route])
            <= float(baseline_behavior["route_missing_signal_rate"][route])
        )
    for route in GUARD_ROUTES:
        checks[f"{route}_guard_bad_rate"] = (
            float(current_behavior["route_bad_rate"][route])
            <= float(baseline_behavior["route_bad_rate"][route]) + 0.04
        )
    checks["chosen_nll_improved"] = (
        float(current_chosen["overall_nll"]) < float(baseline_chosen["overall_nll"])
    )
    checks["chosen_top1_not_worse"] = (
        float(current_chosen["overall_top1_rate"]) >= float(baseline_chosen["overall_top1_rate"])
    )
    checks["chosen_mean_rank_not_worse"] = (
        float(current_chosen["overall_mean_rank"]) <= float(baseline_chosen["overall_mean_rank"])
    )
    checks["loop_regression_guard"] = (
        float(current_behavior["model_loop_rate"])
        <= float(baseline_behavior["model_loop_rate"]) + 0.008
    )
    checks["repetition_regression_guard"] = (
        float(current_behavior["mean_repetition_rate"])
        <= float(baseline_behavior["mean_repetition_rate"]) + 0.01
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "priority_mean_bad_rate": {
            "baseline": base_priority,
            "current": cur_priority,
            "delta": cur_priority - base_priority,
        },
    }


def secondary_promotion_gate(
    baseline_calibrated: dict[str, Any],
    current_calibrated: dict[str, Any],
    baseline_geometry: dict[str, Any],
    current_geometry: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "calibrated_target_ranking_not_worse": (
            float(current_calibrated["calibrated_accuracy"])
            >= float(baseline_calibrated["calibrated_accuracy"])
        ),
        "final_geometry_prototype_accuracy_not_worse": (
            float(current_geometry["prototype_accuracy"])
            >= float(baseline_geometry["prototype_accuracy"])
        ),
    }
    for route in PRIORITY_ROUTES:
        checks[f"{route}_final_geometry_gap_guard"] = (
            float(current_geometry["by_route"][route]["mean_own_vs_best_wrong_gap"])
            >= float(baseline_geometry["by_route"][route]["mean_own_vs_best_wrong_gap"]) - 0.02
        )
    return {"passed": all(checks.values()), "checks": checks}


def promotion_key(
    behavior: dict[str, Any],
    chosen: dict[str, Any],
    calibrated: dict[str, Any],
    geometry: dict[str, Any],
) -> tuple[float, ...]:
    """Lexicographic ordering follows the declared promotion priority rather than one mixed score."""
    priority_success = 1.0 - _priority_mean(behavior["route_bad_rate"])
    return (
        float(behavior["success_rate"]),
        priority_success,
        -float(chosen["overall_nll"]),
        float(chosen["overall_top1_rate"]),
        -float(chosen["overall_mean_rank"]),
        float(calibrated["calibrated_accuracy"]),
        float(geometry["prototype_accuracy"]),
        -float(behavior["model_loop_rate"]),
        -float(behavior["mean_repetition_rate"]),
    )
