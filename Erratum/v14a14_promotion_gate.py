#!/usr/bin/env python3
"""v14a14 CPU-only blinded review and evidence-gated research promotion.

PREPARE: create 56 anonymous generation tasks, never expose arms to reviewer.
ASSESS: resolve reviewer scores against the frozen 28-case panel, reject missing
or altered generations, enforce precommitted scientific thresholds, and
(optionally) promote a SHA-verified research checkpoint. No GPU import here.
This does not overwrite or relabel the canonical v14a2 parent.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from Erratum.v14a14_controlled_eos_train import PIN, PROMOTION_CRITERIA
from Erratum.v14a14_signal_dataset import digest
from Erratum.v14a14_blind_panel import build_holdout

ARMS = ("original_80", "causal_80")
FINAL = 40
REVIEW_SEED = "v14a14-frozen-blind-20261025"


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(piece)
    return hasher.hexdigest()


def _load(root: Path):
    cases, panel = build_holdout()
    if panel["holdout_rows_and_references_sha256"] != PIN["holdout_rows_and_references_sha256"]:
        raise RuntimeError("Holdout changed")
    case_by_id = {r["id"]: r for r in cases}
    generations = {}
    metrics = {}
    for arm in ARMS:
        d = root / arm / "explicit_eos"
        gen = json.loads((d / "gradient_clipping_outputs_u0040.json").read_text())
        if not isinstance(gen, list) or len(gen) != 28:
            raise RuntimeError(f"Expected exactly 28 generated answers from {arm}")
        by_id = {}
        for row in gen:
            rid = str(row["id"])
            if rid not in case_by_id or rid in by_id:
                raise RuntimeError(f"{arm}: invalid/duplicate generated ID {rid}")
            if str(row["semantic_family"]) != str(case_by_id[rid]["semantic_family"]):
                raise RuntimeError("Generated family diverged from frozen panel")
            by_id[rid] = str(row["generation"])
        if set(by_id) != set(case_by_id):
            raise RuntimeError(f"{arm}: incomplete blind evaluation generations")
        generations[arm] = by_id
        metrics[arm] = json.loads(
            (d / "generalization_diagnostic_u0040.json").read_text()
        )
        if metrics[arm]["update"] != FINAL:
            raise RuntimeError("Not the frozen update-40 evaluation")
    result_file = root / "v14a14_posttraining_diagnostic.json"
    run_info = json.loads(result_file.read_text())
    if not run_info["matched_batch_order"] or not run_info["candidate_retained"]:
        raise RuntimeError("Run must have matched exposure and retained causal checkpoint")
    if run_info.get("promotion_attempted") is not False:
        raise RuntimeError("Cannot score a run that previously attempted promotion")
    return cases, generations, metrics, run_info


def _sealed(cases: list[dict], generations: dict) -> list[dict]:
    pairs = []
    for arm in ARMS:
        for case in cases:
            rid = case["id"]
            generated = generations[arm][rid]
            key = hashlib.sha256(
                f"{REVIEW_SEED}|{arm}|{rid}|{generated}".encode("utf-8")
            ).hexdigest()
            pairs.append({
                "blind_id": "H" + key[:16],
                "case_id": rid,
                "arm": arm,
                "family": case["semantic_family"],
                "generation_sha256": hashlib.sha256(generated.encode()).hexdigest(),
            })
    if len({p["blind_id"] for p in pairs}) != 56:
        raise RuntimeError("Blind key collision")
    return sorted(pairs, key=lambda p: hashlib.sha256(
        (REVIEW_SEED + "|" + p["blind_id"]).encode()).hexdigest())


def prepare(root: Path, review_dir: Path) -> dict:
    cases, gens, _, _ = _load(root)
    key = _sealed(cases, gens)
    case_by_id = {row["id"]: row for row in cases}
    packet = [
        "# v14a14 — blinded factual adequacy review of generated outputs",
        "",
        "56 model responses, drawn from two arms across the same 28 frozen",
        "questions. Origin is intentionally masked. Judge the complete",
        "requested causal mechanism, factual correctness, question-specific",
        "completion, and any contradiction. Short correct answers may pass.",
        "Scores: adequate / inadequate / uncertain; include reasons for failures.",
        "Do NOT view the sealed mapping or compare other scores until review is locked.",
        "",
    ]
    blank = []
    for i, p in enumerate(key, 1):
        q = case_by_id[p["case_id"]]
        response = gens[p["arm"]][p["case_id"]]
        packet.extend([
            f"## {i:02d} — {p['blind_id']}", "",
            f"**Family:** {p['family']}", "",
            "**Question:**", "", q["prompt"], "",
            "**Generation:**", "", response, "",
            "Verdict: [ ] adequate  [ ] inadequate  [ ] uncertain", "",
            "Reason: __________________________________", "",
        ])
        blank.append({"blind_id": p["blind_id"], "verdict": "unreviewed", "reason": ""})
    review_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        ("review_packet_blinded.md", "\n".join(packet)),
        ("blank_adjudication.json", json.dumps({"reviews": blank}, indent=2)),
        ("sealed_arm_mapping.json", json.dumps({"mapping": key}, indent=2)),
        ("review_freeze.json", json.dumps({
            "generation_pairing_sha256": digest(key),
            "frozen_holdout_sha256": PIN["holdout_rows_and_references_sha256"],
            "n": 56,
            "research_policy": "promotion_eligible_if_frozen_gates_pass",
            "source_references_are_not_human_verdicts": True,
        }, indent=2)),
    ]
    for name, value in paths:
        path = review_dir / name
        if path.exists():
            raise RuntimeError(f"Refusing to replace an existing blind packet: {path}")
        path.write_text(value + "\n", encoding="utf-8")
    return {"review_packet": str(review_dir / paths[0][0]),
            "score_template": str(review_dir / paths[1][0]),
            "sealed_key": str(review_dir / paths[2][0]), "cases": 56}


def assess(root: Path, review_dir: Path, score_path: Path, apply: bool = False) -> dict:
    cases, gens, metrics, run_info = _load(root)
    source_pairs = _sealed(cases, gens)
    stored = json.loads((review_dir / "sealed_arm_mapping.json").read_text())["mapping"]
    if stored != source_pairs:
        raise RuntimeError("Sealed mapping or generated answers have changed")
    freeze = json.loads((review_dir / "review_freeze.json").read_text())
    if freeze["generation_pairing_sha256"] != digest(source_pairs):
        raise RuntimeError("Blind review mapping fingerprint mismatch")
    submitted = json.loads(score_path.read_text())
    grades = submitted["reviews"]
    if not isinstance(grades, list) or len(grades) != 56:
        raise RuntimeError("Exactly 56 human grade records required")
    scores = {}
    for row in grades:
        name = str(row.get("blind_id"))
        verdict = str(row.get("verdict")).lower()
        if name in scores or verdict not in {"adequate", "inadequate", "uncertain"}:
            raise RuntimeError("Duplicated blind ID or an invalid/missing verdict")
        if verdict != "adequate" and not str(row.get("reason", "")).strip():
            raise RuntimeError("Every inadequate/uncertain judgment requires a reason")
        scores[name] = verdict
    if set(scores) != {r["blind_id"] for r in source_pairs}:
        raise RuntimeError("Human review must cover all blinded outputs")
    count = Counter()
    family_count = Counter()
    for row in source_pairs:
        if scores[row["blind_id"]] == "adequate":
            count[row["arm"]] += 1
            family_count[(row["arm"], row["family"])] += 1
    old_rep = metrics["original_80"]["first_sentence_and_generation_repetition"]["high_repetition_count"]
    new_rep = metrics["causal_80"]["first_sentence_and_generation_repetition"]["high_repetition_count"]
    eos = metrics["causal_80"]["eos_eot_probability_at_reference_answer_boundary"][
        "mean_p_either_termination_after_reference"
    ]
    weak = ("backprop_stability", "learning_rate_comparison", "false_premise_correction")
    gates = {
        "factual_gain_6_of_28": count["causal_80"] - count["original_80"] >= 6,
        "all_three_weak_families_improve": all(
            family_count[("causal_80", f)] > family_count[("original_80", f)]
            for f in weak
        ),
        "false_premise_at_least_2_of_4": family_count[
            ("causal_80", "false_premise_correction")
        ] >= 2,
        "high_repetition_at_most_2_and_not_worse": new_rep <= min(2, old_rep),
        "eos_reference_boundary_at_least_0_80": eos >= 0.80,
    }
    passed = all(gates.values())
    candidate = run_info["arms"]["causal_80"].get("candidate_checkpoint")
    if not candidate:
        raise RuntimeError("No retained causal-arm candidate metadata")
    candidate_path = Path(candidate["path"]).resolve()
    if root.resolve() not in candidate_path.parents:
        raise RuntimeError("Candidate must remain within the experiment output directory")
    if candidate_path.stat().st_size != candidate["size_bytes"]:
        raise RuntimeError("Candidate file size changed")
    if _file_sha256(candidate_path) != candidate["sha256"]:
        raise RuntimeError("Candidate checkpoint content changed")
    report = {
        "status": "GATES_PASS_PROMOTION_ELIGIBLE" if passed else "GATES_FAIL_NOT_ELIGIBLE",
        "gate_results": gates,
        "human_review_blinded_56": True,
        "factual_adequate": dict(count),
        "weak_family_adequate": {
            f: {arm: family_count[(arm, f)] for arm in ARMS} for f in weak
        },
        "repetition": {"original_high": old_rep, "causal_high": new_rep},
        "reference_boundary_eos": eos,
        "reference_nll": {
            arm: metrics[arm]["holdout_reference_teacher_forcing"]["reference_continuation_nll"]
            for arm in ARMS
        },
        "reference_top1": {
            arm: metrics[arm]["holdout_reference_teacher_forcing"]["reference_continuation_top1"]
            for arm in ARMS
        },
        "candidate_sha256": candidate["sha256"],
        "candidate_path": str(candidate_path),
        "dataset_sha256": PIN["revised_sha256"],
        "holdout_sha256": PIN["holdout_rows_and_references_sha256"],
        "review_sha256": _file_sha256(score_path),
        "promotion_scope": "research_checkpoint_not_canonical_parent_replacement",
        "automatic_promotion": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    decision = root / "v14a14_frozen_promotion_decision.json"
    if decision.exists():
        old = json.loads(decision.read_text())
        if old != report:
            raise RuntimeError("Existing decision differs; never overwrite evidence")
    else:
        decision.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if apply:
        if not passed:
            raise RuntimeError("Promotion denied by frozen scientific thresholds")
        dest = root / "promoted" / "v14a14_causal_u0040_promoted.pt"
        if dest.exists():
            raise RuntimeError("Promotion destination exists; refuse overwrite")
        dest.parent.mkdir(parents=True, exist_ok=True)
        pending = dest.with_suffix(".pt.pending")
        if pending.exists():
            raise RuntimeError("Stale incomplete promotion staging file")
        shutil.copyfile(candidate_path, pending)
        if _file_sha256(pending) != candidate["sha256"]:
            pending.unlink(missing_ok=True)
            raise RuntimeError("Promoted artifact hash mismatch")
        pending.replace(dest)
        (dest.with_suffix(".decision.json")).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "assess"))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--review-dir", type=Path)
    p.add_argument("--scores", type=Path)
    p.add_argument("--apply", action="store_true", help="Copy validated candidate into promoted research checkpoint")
    args = p.parse_args()
    rd = args.review_dir or (args.run_dir / "blind_human_review")
    if args.action == "prepare":
        if args.apply or args.scores:
            raise ValueError("prepare cannot apply promotion or consume scores")
        print(json.dumps(prepare(args.run_dir, rd), indent=2))
    else:
        if not args.scores:
            raise ValueError("assess requires --scores, independently human-graded")
        print(json.dumps(assess(args.run_dir, rd, args.scores, args.apply), indent=2))


if __name__ == "__main__":
    main()
