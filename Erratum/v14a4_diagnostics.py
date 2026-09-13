#!/usr/bin/env python3
"""v14a4 family-aware diagnostics and promotion gates.

Primary evaluation is the clean route x semantic_family holdout. The frozen historical holdout
remains a secondary comparability gate. Expensive calibrated ranking/geometry remain tertiary.
"""
from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Sequence

import torch

from Erratum.build_v14a4_family_data import (
    FAMILY_SIGNALS,
    FALSE_PREMISE_FAMILIES,
    PRIORITY_ROUTES,
    ROUTE_SIGNALS,
    norm,
)
from Erratum.v14a3_diagnostics import (
    calibrated_target_ranking,
    compact_behavior,
    evaluate_balanced_chosen,
    evaluate_behavior,
    evaluate_final_geometry,
)

CORRECTION_MARKERS = [
    "do not", "does not", "don't", "doesn't", "not beneficial", "not the normal",
    "premise is backwards", "premise is wrong", "incorrect", "rather", "instead",
    "not from", "not by", "no—", "no -", "no,", "it does not", "they do not",
]


def _has_any(text: str, signals: Sequence[str]) -> bool:
    low = norm(text)
    return any(norm(signal) in low for signal in signals)


def family_sem_check(row: dict[str, Any], text: str) -> dict[str, Any]:
    route = str(row["route"])
    family = str(row["semantic_family"])
    qclass = str(row.get("question_class", "normal"))
    low = norm(text)
    failures: list[str] = []
    if not low:
        failures.append("empty")
        return {"passed": False, "failures": failures}

    if not _has_any(low, ROUTE_SIGNALS[route]):
        failures.append(f"{route}_missing_route_signal")
    if not _has_any(low, FAMILY_SIGNALS[(route, family)]):
        failures.append(f"{route}/{family}_missing_family_signal")

    if qclass == "false_premise_correction":
        if (route, family) not in FALSE_PREMISE_FAMILIES:
            failures.append("undeclared_false_premise_family")
        if not _has_any(low, CORRECTION_MARKERS):
            failures.append("false_premise_not_corrected")
        if route == "gradient_clipping":
            harmful = (
                ("exploding gradients help" in low)
                or ("exploding gradient helps" in low)
                or ("large gradients help" in low)
            )
            if harmful and not _has_any(low, ["do not", "does not", "don't", "doesn't", "not"]):
                failures.append("false_premise_affirmed")
        if route == "dropout":
            harmful = (
                ("evaluation" in low or "inference" in low)
                and ("adds" in low or "uses" in low or "keeps" in low)
                and ("dropout" in low or "noise" in low or "mask" in low)
            )
            if harmful and not _has_any(low, ["do not", "does not", "don't", "doesn't", "disabled", "not"]):
                failures.append("false_premise_affirmed")

    return {"passed": not failures, "failures": failures}


@torch.inference_mode()
def evaluate_clean_family(
    model,
    tok,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    history,
    args,
    special: dict[str, list[int]],
    *,
    name: str,
) -> dict[str, Any]:
    model.eval()
    route_total: Counter[str] = Counter()
    route_pass: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    family_pass: Counter[str] = Counter()
    class_total: Counter[str] = Counter()
    class_pass: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    repetitions: list[float] = []
    loops = 0
    empties = 0
    outputs: list[dict[str, Any]] = []

    for row in rows:
        route = str(row["route"])
        family = str(row["semantic_family"])
        key = f"{route}/{family}"
        qclass = str(row.get("question_class", "normal"))
        generated = history.generate(model, tok, str(row["prompt"]), device, args, special)
        historical = history.sem_check(route, generated["text"])
        family_check = family_sem_check(row, generated["text"])
        failures = list(historical["failures"]) + [
            failure for failure in family_check["failures"] if failure not in historical["failures"]
        ]
        if generated["empty"] and "empty_generation" not in failures:
            failures.append("empty_generation")
        if generated["first_special"]:
            failures.append("first_special")
        if generated["word_count"] < int(args.min_eval_words):
            failures.append("too_short_generation")

        passed = not failures
        route_total[route] += 1
        route_pass[route] += int(passed)
        family_total[key] += 1
        family_pass[key] += int(passed)
        class_total[qclass] += 1
        class_pass[qclass] += int(passed)
        failure_counts.update(failures)
        repetitions.append(float(generated["repetition_rate"]))
        loops += int(historical["model_loop"]["has_model_loop"])
        empties += int(generated["empty"])
        outputs.append({
            "id": str(row["id"]),
            "route": route,
            "semantic_family": family,
            "question_class": qclass,
            "prompt_style": str(row["prompt_style"]),
            "generation": generated["text"],
            "passed": passed,
            "failures": failures,
        })

    n = max(1, len(outputs))
    family_success = {
        key: family_pass[key] / max(1, family_total[key]) for key in sorted(family_total)
    }
    priority_keys = [
        key for key in family_success if key.split("/", 1)[0] in PRIORITY_ROUTES
    ]
    family_values = list(family_success.values())
    priority_values = [family_success[key] for key in priority_keys]
    all_pass_groups = sum(value >= 1.0 for value in family_values)
    pass_75_groups = sum(value >= 0.75 for value in family_values)
    return {
        "name": name,
        "rows": len(outputs),
        "success_count": sum(route_pass.values()),
        "success_rate": sum(route_pass.values()) / n,
        "bad_rate": 1.0 - (sum(route_pass.values()) / n),
        "route_success_rate": {
            route: route_pass[route] / max(1, route_total[route]) for route in sorted(route_total)
        },
        "family_success_rate": family_success,
        "family_macro_success_rate": mean(family_values) if family_values else 0.0,
        "priority_family_macro_success_rate": mean(priority_values) if priority_values else 0.0,
        "min_priority_family_success_rate": min(priority_values) if priority_values else 0.0,
        "question_class_success_rate": {
            cls: class_pass[cls] / max(1, class_total[cls]) for cls in sorted(class_total)
        },
        "paraphrase_all_pass_rate": all_pass_groups / max(1, len(family_values)),
        "paraphrase_75pct_pass_rate": pass_75_groups / max(1, len(family_values)),
        "model_loop_count": loops,
        "model_loop_rate": loops / n,
        "empty_count": empties,
        "mean_repetition_rate": sum(repetitions) / max(1, len(repetitions)),
        "failure_counts": dict(failure_counts),
        "outputs": outputs,
    }


def compact_clean(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "outputs"}


def clean_primary_gate(base: dict[str, Any], cur: dict[str, Any], *, max_family_regression: float = 0.05) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["clean_overall_success_improved"] = float(cur["success_rate"]) > float(base["success_rate"])
    checks["priority_family_macro_improved"] = (
        float(cur["priority_family_macro_success_rate"]) >
        float(base["priority_family_macro_success_rate"])
    )

    family_regressions: dict[str, float] = {}
    for key, base_value in base["family_success_rate"].items():
        if key.split("/", 1)[0] not in PRIORITY_ROUTES:
            continue
        delta = float(cur["family_success_rate"][key]) - float(base_value)
        family_regressions[key] = delta
    checks["no_priority_family_regression"] = all(
        delta >= -max_family_regression for delta in family_regressions.values()
    )

    base_false = float(base["question_class_success_rate"].get("false_premise_correction", 0.0))
    cur_false = float(cur["question_class_success_rate"].get("false_premise_correction", 0.0))
    checks["false_premise_not_worse"] = cur_false >= base_false
    checks["paraphrase_75pct_not_worse"] = (
        float(cur["paraphrase_75pct_pass_rate"]) >= float(base["paraphrase_75pct_pass_rate"])
    )
    checks["loop_guard"] = float(cur["model_loop_rate"]) <= float(base["model_loop_rate"]) + 0.008
    checks["repetition_guard"] = (
        float(cur["mean_repetition_rate"]) <= float(base["mean_repetition_rate"]) + 0.01
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "family_regressions": family_regressions,
        "metrics": {
            "base_success": float(base["success_rate"]),
            "current_success": float(cur["success_rate"]),
            "base_priority_family_macro": float(base["priority_family_macro_success_rate"]),
            "current_priority_family_macro": float(cur["priority_family_macro_success_rate"]),
            "base_false_premise": base_false,
            "current_false_premise": cur_false,
        },
    }


def secondary_gate(
    base_historical: dict[str, Any],
    cur_historical: dict[str, Any],
    base_chosen: dict[str, Any],
    cur_chosen: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "historical_bad_rate_not_worse": (
            float(cur_historical["bad_rate"]) <= float(base_historical["bad_rate"])
        ),
        "historical_loop_guard": (
            float(cur_historical["model_loop_rate"]) <= float(base_historical["model_loop_rate"]) + 0.008
        ),
        "historical_repetition_guard": (
            float(cur_historical["mean_repetition_rate"]) <=
            float(base_historical["mean_repetition_rate"]) + 0.01
        ),
        "chosen_nll_improved": float(cur_chosen["overall_nll"]) < float(base_chosen["overall_nll"]),
        "chosen_top1_not_worse": (
            float(cur_chosen["overall_top1_rate"]) >= float(base_chosen["overall_top1_rate"])
        ),
        "chosen_rank_not_worse": (
            float(cur_chosen["overall_mean_rank"]) <= float(base_chosen["overall_mean_rank"])
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def tertiary_gate(
    base_calibrated: dict[str, Any],
    cur_calibrated: dict[str, Any],
    base_geometry: dict[str, Any],
    cur_geometry: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "calibrated_target_ranking_not_worse": (
            float(cur_calibrated["calibrated_accuracy"]) >=
            float(base_calibrated["calibrated_accuracy"])
        ),
        "final_geometry_prototype_accuracy_not_worse": (
            float(cur_geometry["prototype_accuracy"]) >=
            float(base_geometry["prototype_accuracy"])
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def promotion_key(
    clean: dict[str, Any],
    historical: dict[str, Any],
    chosen: dict[str, Any],
    calibrated: dict[str, Any],
    geometry: dict[str, Any],
) -> tuple[float, ...]:
    return (
        float(clean["success_rate"]),
        float(clean["priority_family_macro_success_rate"]),
        float(clean["paraphrase_all_pass_rate"]),
        -float(historical["bad_rate"]),
        -float(chosen["overall_nll"]),
        float(chosen["overall_top1_rate"]),
        float(calibrated["calibrated_accuracy"]),
        float(geometry["prototype_accuracy"]),
    )


__all__ = [
    "calibrated_target_ranking",
    "compact_behavior",
    "evaluate_balanced_chosen",
    "evaluate_behavior",
    "evaluate_final_geometry",
    "family_sem_check",
    "evaluate_clean_family",
    "compact_clean",
    "clean_primary_gate",
    "secondary_gate",
    "tertiary_gate",
    "promotion_key",
]
