#!/usr/bin/env python3
"""Forensic signal-quality audit for v14a4.

This audit performs no model training and no model loading. It joins:
- the deterministic 256-row v14a4 clean holdout,
- canonical-v14a2 training-matched generations from the completed serialization audit,
- the deterministic v14a4 authored behavior set,
- normalized balanced/targeted legacy replay,
- the exact rows sampled by the completed u100 v14a4 probe,
- the existing 32-row manual generation review and balanced evaluator calibration.

For every evaluator failure, plus any manually confirmed evaluator false positive, it retrieves
same-route/same-family supervision, exact chosen continuations, nearest authored examples, and
actual u100 exposure. It assigns one primary diagnosis:
coverage_failure, supervision_failure, collision_failure, or model_learning_failure.

The diagnosis is explicitly heuristic and evidence-carrying. It is intended to decide what data
repair to attempt next, not to replace manual review or claim causal proof from the canonical
parent (which never received v14a4 supervision).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any, Iterable, Sequence

from Erratum.build_v14a4_family_data import (
    COLLISIONS,
    FAMILY_SIGNALS,
    FAMILIES,
    ROUTE_SIGNALS,
    ROUTES,
    build_holdout_rows,
    build_train_rows,
)

REPO_ROOT = Path("/workspace/Ardor")
BALANCED_PATH = REPO_ROOT / "training/data/v14_v3/dataset_v3b_route_contrastive_balanced.jsonl"
TARGETED_PATH = REPO_ROOT / "training/data/v14_v3/dataset_v14b3_targeted_route_collisions.jsonl"
PROBE_DIR = REPO_ROOT / "training/runs/sft_v14a4_family_balanced_semantic_landing_probe_u100"
CANONICAL_AUDIT_DIR = REPO_ROOT / "training/runs/v14a4_exact_runtime_serialization_audit"
OUTPUT_DIR = REPO_ROOT / "training/runs/v14a4_signal_quality_audit"

CANONICAL_OUTPUTS_PATH = CANONICAL_AUDIT_DIR / "training_matched_outputs.json"
MANUAL_AUDIT_PATH = CANONICAL_AUDIT_DIR / "manual_audit_32_generation_pairs.json"
CALIBRATION_PATH = CANONICAL_AUDIT_DIR / "balanced_evaluator_calibration.json"
SAMPLED_ROWS_PATH = PROBE_DIR / "sampled_rows.jsonl"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "can", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "then", "this", "to", "what", "when", "where", "which", "why", "with",
    "would", "you", "your", "simply", "explain", "describe", "give", "start", "definition",
}
GENERIC_PATTERNS = (
    "the answer is", "the answer should", "the model is", "the model should", "this is correct",
    "it is correct", "not directly answerable", "not a perfect fit", "in this example",
)

DIAGNOSIS_ORDER = (
    "coverage_failure",
    "supervision_failure",
    "collision_failure",
    "model_learning_failure",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def norm(text: str) -> str:
    value = re.sub(r"[-‐‑‒–—]+", " ", str(text).strip().lower())
    value = re.sub(r"[^a-z0-9<>|]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokens(text: str) -> set[str]:
    return {token for token in norm(text).split() if token not in STOPWORDS and len(token) > 1}


def similarity(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    ta, tb = tokens(na), tokens(nb)
    union = ta | tb
    jaccard = len(ta & tb) / len(union) if union else 0.0
    seq = SequenceMatcher(None, na, nb).ratio()
    return 0.65 * jaccard + 0.35 * seq


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text)))


def ngram_repetition(text: str, n: int = 3) -> float:
    values = norm(text).split()
    if len(values) < n:
        return 0.0
    grams = [tuple(values[i:i+n]) for i in range(len(values) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def has_any(text: str, signals: Iterable[str]) -> bool:
    low = norm(text)
    return any(norm(signal) in low for signal in signals)


def route_of(row: dict[str, Any]) -> str:
    return str(row.get("route") or row.get("positive_group") or "unknown")


def chosen_of(row: dict[str, Any]) -> str:
    return str(row.get("chosen") or row.get("answer") or row.get("response") or "").strip()


def prompt_of(row: dict[str, Any]) -> str:
    text = str(row.get("anchor_context") or row.get("prompt") or row.get("text") or "").strip()
    if text and not text.endswith("\n-"):
        text += "\n-"
    return text


def parse_negatives(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
        return [item.strip() for item in text.split(",") if item.strip()]
    return [str(value)]


def normalize_behavior_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "id": str(row["id"]),
        "route": str(row["route"]),
        "semantic_family": str(row["semantic_family"]),
        "prompt": str(row["prompt"]).rstrip() + "\n-",
        "chosen": str(row["chosen"]).strip(),
        "source": "behavior",
        "question_class": str(row.get("question_class", "normal")),
        "prompt_style": str(row.get("prompt_style", "")),
        "answer_style": str(row.get("answer_style", "")),
        "collision_route": row.get("collision_route"),
    } for row in rows]


def required_directed_edges() -> set[tuple[str, str]]:
    return {(route, neg) for route, negatives in COLLISIONS.items() for neg in negatives}


def normalize_replay_rows(
    rows: Sequence[dict[str, Any]], source: str, *, required_edges: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        route = route_of(row)
        prompt = prompt_of(row)
        chosen = chosen_of(row)
        if route not in ROUTES or not prompt or not chosen:
            continue
        if source == "targeted":
            negatives = parse_negatives(row.get("hard_negatives")) + parse_negatives(row.get("hard_negative"))
            if required_edges is not None and not any((route, neg) in required_edges for neg in negatives):
                continue
        output.append({
            "id": str(row.get("id", f"{source}_{index:06d}")),
            "route": route,
            "semantic_family": "legacy_generic",
            "prompt": prompt,
            "chosen": chosen,
            "source": source,
            "collision_route": None,
        })
    return output


def generic_answer(text: str) -> bool:
    low = norm(text)
    if any(norm(pattern) in low for pattern in GENERIC_PATTERNS):
        return True
    return word_count(text) < 5


def answer_quality(row: dict[str, Any], route: str, family: str) -> dict[str, Any]:
    chosen = str(row["chosen"])
    route_ok = has_any(chosen, ROUTE_SIGNALS[route])
    family_ok = has_any(chosen, FAMILY_SIGNALS[(route, family)])
    generic = generic_answer(chosen)
    repetition = ngram_repetition(chosen)
    return {
        "route_signal": route_ok,
        "family_signal": family_ok,
        "generic": generic,
        "repetition": repetition,
        "word_count": word_count(chosen),
    }


def family_profile(
    route: str,
    family: str,
    behavior_rows: Sequence[dict[str, Any]],
    holdout_rows: Sequence[dict[str, Any]],
    sampled_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    authored = [row for row in behavior_rows if row["route"] == route and row["semantic_family"] == family]
    sampled_behavior = [row for row in sampled_rows if row["source"] == "behavior" and row["route"] == route and row["semantic_family"] == family]
    sampled_replay = [row for row in sampled_rows if row["source"] != "behavior" and row["route"] == route]
    qualities = [answer_quality(row, route, family) for row in authored]

    unique_prompts = len({norm(row["prompt"]) for row in authored})
    unique_chosen = len({norm(row["chosen"]) for row in authored})
    count = max(1, len(authored))
    route_signal_rate = mean([float(item["route_signal"]) for item in qualities]) if qualities else 0.0
    family_signal_rate = mean([float(item["family_signal"]) for item in qualities]) if qualities else 0.0
    generic_rate = mean([float(item["generic"]) for item in qualities]) if qualities else 1.0
    repetition_rate = mean([float(item["repetition"]) for item in qualities]) if qualities else 1.0
    chosen_diversity = unique_chosen / count
    prompt_diversity = unique_prompts / count

    family_signals = FAMILY_SIGNALS[(route, family)]
    replay_family_support = [has_any(row["chosen"], family_signals) for row in sampled_replay]
    replay_route_support = [has_any(row["chosen"], ROUTE_SIGNALS[route]) for row in sampled_replay]
    competitor_routes = COLLISIONS.get(route, [])
    replay_competitor = [
        any(has_any(row["chosen"], ROUTE_SIGNALS[competitor]) for competitor in competitor_routes)
        for row in sampled_replay
    ]
    replay_conflict = [
        competitor and not route_ok
        for competitor, route_ok in zip(replay_competitor, replay_route_support)
    ]

    reference_support = []
    for holdout in holdout_rows:
        if holdout["route"] != route or holdout["semantic_family"] != family:
            continue
        reference_support.append(max((similarity(holdout["chosen"], row["chosen"]) for row in authored), default=0.0))

    quality_score = mean([
        route_signal_rate,
        family_signal_rate,
        min(1.0, chosen_diversity / 0.70),
        1.0 - generic_rate,
        1.0 - min(1.0, repetition_rate / 0.25),
    ])
    quality_label = "high" if quality_score >= 0.80 else "medium" if quality_score >= 0.62 else "low"

    replay_n = len(sampled_replay)
    family_support_rate = sum(replay_family_support) / replay_n if replay_n else 0.0
    route_support_rate = sum(replay_route_support) / replay_n if replay_n else 0.0
    competitor_rate = sum(replay_competitor) / replay_n if replay_n else 0.0
    conflict_rate = sum(replay_conflict) / replay_n if replay_n else 0.0
    dilution_score = max(0.0, route_support_rate - family_support_rate)

    flags: list[str] = []
    if len(sampled_behavior) < 4:
        flags.append("low_actual_probe_family_exposure")
    if route_signal_rate < 0.90:
        flags.append("authored_answers_often_miss_route_signal")
    if family_signal_rate < 0.80:
        flags.append("authored_answers_often_miss_family_signal")
    if chosen_diversity < 0.60:
        flags.append("low_chosen_answer_diversity")
    if generic_rate > 0.10:
        flags.append("generic_authored_answers")
    if repetition_rate > 0.12:
        flags.append("repetitive_authored_answers")
    if family_support_rate < 0.20 and replay_n >= 4:
        flags.append("legacy_replay_rarely_supports_family_signal")
    if conflict_rate >= 0.15 and replay_n >= 4:
        flags.append("legacy_replay_contains_competing_route_without_route_anchor")
    if dilution_score >= 0.45 and replay_n >= 4:
        flags.append("legacy_replay_preserves_route_but_dilutes_family_specificity")

    return {
        "route": route,
        "semantic_family": family,
        "authored_behavior_rows": len(authored),
        "holdout_rows": sum(1 for row in holdout_rows if row["route"] == route and row["semantic_family"] == family),
        "actual_u100_behavior_exposures": len(sampled_behavior),
        "actual_u100_replay_exposures_same_route": replay_n,
        "prompt_diversity": prompt_diversity,
        "chosen_diversity": chosen_diversity,
        "route_signal_rate": route_signal_rate,
        "family_signal_rate": family_signal_rate,
        "generic_answer_rate": generic_rate,
        "mean_answer_repetition": repetition_rate,
        "mean_reference_to_authored_answer_similarity": mean(reference_support) if reference_support else 0.0,
        "behavior_supervision_quality_score": quality_score,
        "behavior_supervision_quality": quality_label,
        "legacy_replay_route_support_rate": route_support_rate,
        "legacy_replay_family_support_rate": family_support_rate,
        "legacy_replay_competitor_signal_rate": competitor_rate,
        "legacy_replay_conflict_rate": conflict_rate,
        "legacy_replay_family_dilution_score": dilution_score,
        "flags": flags,
    }


def top_matches(
    holdout: dict[str, Any], candidates: Sequence[dict[str, Any]], *, limit: int = 5,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, float, float, dict[str, Any]]] = []
    for row in candidates:
        prompt_score = similarity(str(holdout["prompt"]), str(row["prompt"]))
        answer_score = similarity(str(holdout["chosen"]), str(row["chosen"]))
        score = 0.70 * prompt_score + 0.30 * answer_score
        scored.append((score, prompt_score, answer_score, row))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [{
        "score": score,
        "prompt_similarity": prompt_score,
        "reference_answer_similarity": answer_score,
        "id": str(row["id"]),
        "source": str(row["source"]),
        "route": str(row["route"]),
        "semantic_family": str(row["semantic_family"]),
        "prompt": str(row["prompt"]),
        "chosen": str(row["chosen"]),
        "collision_route": row.get("collision_route"),
    } for score, prompt_score, answer_score, row in scored[:limit]]


def diagnosis_for_failure(
    profile: dict[str, Any], matches: Sequence[dict[str, Any]], seen_matches: Sequence[dict[str, Any]],
) -> tuple[str, dict[str, float], list[str]]:
    best_prompt = max((item["prompt_similarity"] for item in matches), default=0.0)
    best_reference_answer = max((item["reference_answer_similarity"] for item in matches), default=0.0)
    actual_exposure = float(profile["actual_u100_behavior_exposures"])

    coverage_score = 0.0
    coverage_score += 0.40 if actual_exposure < 4 else 0.0
    coverage_score += 0.35 if best_prompt < 0.30 else 0.0
    coverage_score += 0.25 if best_reference_answer < 0.28 else 0.0

    supervision_score = 0.0
    supervision_score += 0.35 if profile["family_signal_rate"] < 0.80 else 0.0
    supervision_score += 0.25 if profile["route_signal_rate"] < 0.90 else 0.0
    supervision_score += 0.20 if profile["chosen_diversity"] < 0.60 else 0.0
    supervision_score += 0.20 if profile["generic_answer_rate"] > 0.10 else 0.0

    collision_score = 0.0
    if profile["actual_u100_replay_exposures_same_route"] >= 4:
        collision_score += min(0.55, profile["legacy_replay_conflict_rate"] * 1.8)
        collision_score += min(0.30, profile["legacy_replay_family_dilution_score"] * 0.65)
    collision_score += 0.15 if any(item.get("collision_route") for item in seen_matches[:3]) else 0.0

    evidence_scores = {
        "coverage_failure": coverage_score,
        "supervision_failure": supervision_score,
        "collision_failure": collision_score,
    }
    strongest_non_learning = max(evidence_scores, key=evidence_scores.get)
    strongest_score = evidence_scores[strongest_non_learning]
    if strongest_score >= 0.55:
        diagnosis = strongest_non_learning
    else:
        diagnosis = "model_learning_failure"

    reasons: list[str] = []
    if actual_exposure < 4:
        reasons.append(f"only {int(actual_exposure)} actual u100 behavior exposures for this family")
    if best_prompt < 0.30:
        reasons.append(f"best authored prompt similarity is low ({best_prompt:.3f})")
    if best_reference_answer < 0.28:
        reasons.append(f"best authored answer/reference similarity is low ({best_reference_answer:.3f})")
    if profile["family_signal_rate"] < 0.80:
        reasons.append(f"family-signal rate is {profile['family_signal_rate']:.3f}")
    if profile["route_signal_rate"] < 0.90:
        reasons.append(f"route-signal rate is {profile['route_signal_rate']:.3f}")
    if profile["chosen_diversity"] < 0.60:
        reasons.append(f"chosen-answer diversity is {profile['chosen_diversity']:.3f}")
    if profile["legacy_replay_conflict_rate"] >= 0.15:
        reasons.append(f"legacy replay conflict rate is {profile['legacy_replay_conflict_rate']:.3f}")
    if profile["legacy_replay_family_dilution_score"] >= 0.45:
        reasons.append(f"legacy replay family dilution score is {profile['legacy_replay_family_dilution_score']:.3f}")
    if diagnosis == "model_learning_failure" and not reasons:
        reasons.append("same-family supervision is present, diverse, and semantically anchored; no stronger data defect crossed the audit thresholds")
    elif diagnosis == "model_learning_failure":
        reasons.append("observed data defects did not cross the predeclared 0.55 evidence threshold")

    all_scores = {**evidence_scores, "model_learning_failure": 1.0 - min(1.0, strongest_score)}
    return diagnosis, all_scores, reasons


def evaluator_disagreement(
    output: dict[str, Any], holdout: dict[str, Any], manual_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row_id = str(holdout["id"])
    heuristic_passed = bool(output.get("passed", False))
    manual = manual_by_id.get(row_id)
    if manual is not None:
        human_label = bool(manual.get("human_generation_label"))
        if heuristic_passed != human_label:
            kind = "confirmed_false_positive" if heuristic_passed else "confirmed_false_negative"
            return {"status": kind, "human_label": human_label, "source": "manual_32"}
        return {"status": "manual_agreement", "human_label": human_label, "source": "manual_32"}

    generation = str(output.get("generation", ""))
    ref_similarity = similarity(generation, str(holdout["chosen"]))
    generic = generic_answer(generation)
    repeated = ngram_repetition(generation) > 0.45
    if heuristic_passed and (generic or repeated or ref_similarity < 0.10):
        return {
            "status": "suspected_false_positive",
            "source": "automatic_screen",
            "reference_answer_similarity": ref_similarity,
            "generic": generic,
            "high_repetition": repeated,
        }
    if (not heuristic_passed) and ref_similarity >= 0.52 and not generic:
        return {
            "status": "suspected_false_negative",
            "source": "automatic_screen",
            "reference_answer_similarity": ref_similarity,
        }
    return {"status": "no_disagreement_flag", "source": "automatic_screen", "reference_answer_similarity": ref_similarity}


def main() -> int:
    required = [CANONICAL_OUTPUTS_PATH, SAMPLED_ROWS_PATH, BALANCED_PATH, TARGETED_PATH]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required signal-audit inputs are missing: {missing}")

    holdout = build_holdout_rows()
    behavior_raw = build_train_rows()
    behavior = normalize_behavior_rows(behavior_raw)
    balanced = normalize_replay_rows(read_jsonl(BALANCED_PATH), "balanced")
    targeted = normalize_replay_rows(
        read_jsonl(TARGETED_PATH), "targeted", required_edges=required_directed_edges()
    )
    all_training = behavior + balanced + targeted

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_keys: list[tuple[str, str]] = []
    for row in all_training:
        key = (str(row["source"]), str(row["id"]))
        if key in lookup:
            duplicate_keys.append(key)
        lookup[key] = row
    if duplicate_keys:
        raise RuntimeError(f"Duplicate normalized training keys prevent exact sampled-row join: {duplicate_keys[:10]}")

    sampled_log = read_jsonl(SAMPLED_ROWS_PATH)
    sampled: list[dict[str, Any]] = []
    unresolved_samples: list[dict[str, Any]] = []
    for item in sampled_log:
        key = (str(item["source"]), str(item["id"]))
        row = lookup.get(key)
        if row is None:
            unresolved_samples.append(item)
            continue
        sampled.append({**row, "update": int(item["update"])})
    if unresolved_samples:
        raise RuntimeError(f"Could not resolve sampled rows to exact prompt/chosen text: {unresolved_samples[:10]}")

    canonical_outputs = read_json(CANONICAL_OUTPUTS_PATH)
    if len(canonical_outputs) != 256:
        raise RuntimeError(f"Expected 256 canonical outputs, got {len(canonical_outputs)}")
    output_by_id = {str(row["id"]): row for row in canonical_outputs}
    holdout_by_id = {str(row["id"]): row for row in holdout}
    if set(output_by_id) != set(holdout_by_id):
        raise RuntimeError("Canonical training-matched output IDs do not match deterministic clean holdout IDs")

    manual_rows = read_json(MANUAL_AUDIT_PATH) if MANUAL_AUDIT_PATH.is_file() else []
    manual_by_id = {str(row["id"]): row for row in manual_rows}
    calibration = read_json(CALIBRATION_PATH) if CALIBRATION_PATH.is_file() else None

    profiles: dict[tuple[str, str], dict[str, Any]] = {}
    for route in ROUTES:
        for family in FAMILIES[route]:
            profiles[(route, family)] = family_profile(route, family, behavior, holdout, sampled)

    failures: list[dict[str, Any]] = []
    diagnosis_counts: Counter[str] = Counter()
    disagreement_counts: Counter[str] = Counter()
    family_failure_counts: Counter[tuple[str, str]] = Counter()

    for holdout_row in holdout:
        row_id = str(holdout_row["id"])
        output = output_by_id[row_id]
        disagreement = evaluator_disagreement(output, holdout_row, manual_by_id)
        disagreement_counts[disagreement["status"]] += 1
        heuristic_failed = not bool(output.get("passed", False))
        manual_false_positive = disagreement["status"] == "confirmed_false_positive"
        suspected_false_positive = disagreement["status"] == "suspected_false_positive"
        if not (heuristic_failed or manual_false_positive or suspected_false_positive):
            continue

        route = str(holdout_row["route"])
        family = str(holdout_row["semantic_family"])
        profile = profiles[(route, family)]
        same_family = [row for row in behavior if row["route"] == route and row["semantic_family"] == family]
        actually_seen_same_family = [
            row for row in sampled if row["source"] == "behavior" and row["route"] == route and row["semantic_family"] == family
        ]
        matches = top_matches(holdout_row, same_family)
        seen_matches = top_matches(holdout_row, actually_seen_same_family)
        diagnosis, scores, reasons = diagnosis_for_failure(profile, matches, seen_matches)
        diagnosis_counts[diagnosis] += 1
        family_failure_counts[(route, family)] += 1

        failures.append({
            "id": row_id,
            "route": route,
            "semantic_family": family,
            "question_class": str(holdout_row.get("question_class", "normal")),
            "prompt": str(holdout_row["prompt"]),
            "reference_chosen": str(holdout_row["chosen"]),
            "canonical_generation": str(output.get("generation", "")),
            "heuristic_passed": bool(output.get("passed", False)),
            "heuristic_failures": list(output.get("failures", [])),
            "evaluator_disagreement": disagreement,
            "diagnosis": diagnosis,
            "diagnosis_scores": scores,
            "diagnosis_reasons": reasons,
            "family_profile": profile,
            "nearest_authored_same_family_rows": matches,
            "nearest_actual_u100_seen_same_family_rows": seen_matches,
        })

    family_rows: list[dict[str, Any]] = []
    for route in ROUTES:
        for family in FAMILIES[route]:
            profile = dict(profiles[(route, family)])
            family_failures = [item for item in failures if item["route"] == route and item["semantic_family"] == family]
            family_diag = Counter(item["diagnosis"] for item in family_failures)
            dominant = family_diag.most_common(1)[0][0] if family_diag else "no_failure_candidates"
            authored = [row for row in behavior if row["route"] == route and row["semantic_family"] == family]
            exemplars = []
            for holdout_row in [row for row in holdout if row["route"] == route and row["semantic_family"] == family][:2]:
                exemplars.extend(top_matches(holdout_row, authored, limit=1))
            profile.update({
                "failure_candidates": len(family_failures),
                "heuristic_failures": sum(1 for item in family_failures if not item["heuristic_passed"]),
                "evaluator_disagreement_flags": sum(1 for item in family_failures if item["evaluator_disagreement"]["status"] not in {"no_disagreement_flag", "manual_agreement"}),
                "diagnosis_counts": dict(family_diag),
                "dominant_diagnosis": dominant,
                "nearest_training_examples": exemplars[:2],
            })
            family_rows.append(profile)

    source_counts = Counter(row["source"] for row in sampled)
    route_source_counts = Counter((row["route"], row["source"]) for row in sampled)
    replay_rows = [row for row in sampled if row["source"] != "behavior"]
    replay_route_anchor = sum(has_any(row["chosen"], ROUTE_SIGNALS[row["route"]]) for row in replay_rows)
    replay_competitor_only = 0
    for row in replay_rows:
        route = row["route"]
        has_route = has_any(row["chosen"], ROUTE_SIGNALS[route])
        has_competitor = any(has_any(row["chosen"], ROUTE_SIGNALS[other]) for other in COLLISIONS.get(route, []))
        replay_competitor_only += int(has_competitor and not has_route)

    replay_family_support_values = [profile["legacy_replay_family_support_rate"] for profile in profiles.values()]
    replay_dilution_values = [profile["legacy_replay_family_dilution_score"] for profile in profiles.values()]
    replay_conflict_values = [profile["legacy_replay_conflict_rate"] for profile in profiles.values()]

    summary = {
        "schema_version": 1,
        "purpose": "v14a4 signal-quality audit: canonical failures vs authored and actually sampled supervision",
        "causal_scope_note": (
            "Canonical-v14a2 outputs identify capability gaps; canonical-v14a2 itself did not receive v14a4 rows. "
            "The audit therefore evaluates whether v14a4 supervision adequately targets those gaps and uses the "
            "actual u100 sampled stream to measure exposure. Diagnosis labels are heuristic evidence buckets, not causal proof."
        ),
        "inputs": {
            "canonical_outputs": str(CANONICAL_OUTPUTS_PATH),
            "sampled_rows": str(SAMPLED_ROWS_PATH),
            "balanced_replay": str(BALANCED_PATH),
            "targeted_replay": str(TARGETED_PATH),
            "manual_audit": str(MANUAL_AUDIT_PATH) if MANUAL_AUDIT_PATH.is_file() else None,
            "calibration": str(CALIBRATION_PATH) if CALIBRATION_PATH.is_file() else None,
        },
        "counts": {
            "holdout_rows": len(holdout),
            "canonical_outputs": len(canonical_outputs),
            "authored_behavior_rows": len(behavior),
            "normalized_balanced_rows": len(balanced),
            "normalized_targeted_rows": len(targeted),
            "actual_u100_sampled_rows": len(sampled),
            "actual_u100_sampled_by_source": dict(source_counts),
            "failure_candidates": len(failures),
            "diagnosis_counts": dict(diagnosis_counts),
            "evaluator_disagreement_counts": dict(disagreement_counts),
        },
        "evaluator_calibration": ({
            "confusion_matrix": calibration.get("confusion_matrix"),
            "accuracy": calibration.get("accuracy"),
            "precision": calibration.get("precision"),
            "recall": calibration.get("recall"),
            "false_positive_rate": calibration.get("false_positive_rate"),
            "false_negative_rate": calibration.get("false_negative_rate"),
        } if calibration else None),
        "diagnosis_thresholds": {
            "coverage": "score >= 0.55; components: <4 actual family exposures, best prompt sim <0.30, best reference-answer sim <0.28",
            "supervision": "score >= 0.55; components: family signal <0.80, route signal <0.90, chosen diversity <0.60, generic rate >0.10",
            "collision": "score >= 0.55; components: sampled legacy conflict/dilution plus nearby authored collision examples",
            "model_learning": "default when no data-defect evidence bucket reaches 0.55",
        },
        "legacy_replay": {
            "actual_u100_replay_rows": len(replay_rows),
            "actual_u100_replay_route_anchor_rate": replay_route_anchor / max(1, len(replay_rows)),
            "actual_u100_replay_competitor_without_route_anchor_rate": replay_competitor_only / max(1, len(replay_rows)),
            "mean_family_support_rate_across_route_families": mean(replay_family_support_values) if replay_family_support_values else 0.0,
            "mean_family_dilution_score_across_route_families": mean(replay_dilution_values) if replay_dilution_values else 0.0,
            "mean_conflict_rate_across_route_families": mean(replay_conflict_values) if replay_conflict_values else 0.0,
            "actual_u100_by_route_source": {
                f"{route}/{source}": count for (route, source), count in sorted(route_source_counts.items())
            },
        },
        "outputs": {
            "failure_training_matches": str(OUTPUT_DIR / "failure_training_matches.json"),
            "route_family_diagnosis": str(OUTPUT_DIR / "route_family_diagnosis.csv"),
            "supervision_quality_flags": str(OUTPUT_DIR / "supervision_quality_flags.json"),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "signal_audit_summary.json", summary)
    write_json(OUTPUT_DIR / "failure_training_matches.json", failures)
    write_json(OUTPUT_DIR / "supervision_quality_flags.json", {
        "families": family_rows,
        "legacy_replay": summary["legacy_replay"],
    })

    csv_path = OUTPUT_DIR / "route_family_diagnosis.csv"
    columns = [
        "route", "semantic_family", "authored_behavior_rows", "actual_u100_behavior_exposures",
        "actual_u100_replay_exposures_same_route", "holdout_rows", "failure_candidates", "heuristic_failures",
        "evaluator_disagreement_flags", "behavior_supervision_quality", "behavior_supervision_quality_score",
        "chosen_diversity", "family_signal_rate", "route_signal_rate", "legacy_replay_route_support_rate",
        "legacy_replay_family_support_rate", "legacy_replay_conflict_rate", "legacy_replay_family_dilution_score",
        "dominant_diagnosis", "diagnosis_counts", "nearest_training_examples", "flags",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in family_rows:
            record = {key: row.get(key) for key in columns}
            for key in ("diagnosis_counts", "nearest_training_examples", "flags"):
                record[key] = json.dumps(record[key], ensure_ascii=False, sort_keys=True)
            writer.writerow(record)

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
