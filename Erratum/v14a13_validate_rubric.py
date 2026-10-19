#!/usr/bin/env python3
"""Run external-reference-grounded rubric calibration on CPU, no model loading.

This report measures policy behavior against manually authored expectations.
The gold annotations are not independent human judgments, so this is
NOT evidence of factual-grader accuracy on real model outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Erratum.v14a13_rubric_challenges import CHALLENGES, annotations_for_challenge
from Erratum.v14a13_proposition_evaluator import evaluate

REFS = [
    "https://docs.pytorch.org/docs/main/generated/torch.nn.utils.clip_grad_norm_.html",
    "https://docs.pytorch.org/docs/main/generated/torch.nn.utils.clip_grads_with_norm_.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html",
]


def run_validation() -> dict:
    cases = []
    for sample in CHALLENGES:
        case, annotation = annotations_for_challenge(sample)
        result = evaluate(case, annotation)
        expected = sample["gold"]
        cases.append({
            "id": sample["id"], "family": sample["family"],
            "predeclared_adequate": expected,
            "policy_adequate": result["adequate"],
            "matches_predeclared": expected == result["adequate"],
            "failure_rationale": None if expected else sample["reason"],
        })
    tp = sum(x["policy_adequate"] and x["predeclared_adequate"] for x in cases)
    tn = sum(not x["policy_adequate"] and not x["predeclared_adequate"] for x in cases)
    fp = sum(x["policy_adequate"] and not x["predeclared_adequate"] for x in cases)
    fn = sum(not x["policy_adequate"] and x["predeclared_adequate"] for x in cases)
    return {
        "version": "v14a13-rubric-challenges-1",
        "calibration_kind": "source-grounded-hand-authored-policy-consistency-test",
        "reference_documents": REFS,
        "independent_external_human_validation": "NOT_PERFORMED",
        "gold_created_by": "single assistant author, independent from training examples",
        "automatic_truth_assessment": False,
        "challenge_count": len(cases),
        "positive_gold": sum(x["predeclared_adequate"] for x in cases),
        "negative_gold": sum(not x["predeclared_adequate"] for x in cases),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "cases": cases,
        "limitation": (
            "A reviewer can mislabel or omit contradictions. Structural validation "
            "and two distinct reviewer IDs do not establish reviewer independence "
            "or correctness; use blinded independent expert labels on real outputs."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("v14a13_audit_output"))
    args = p.parse_args()
    report = run_validation()
    if report["FP"] or report["FN"]:
        raise RuntimeError("Predeclared challenge policy did not match gold outcomes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v14a13_rubric_calibration.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "challenge_count": report["challenge_count"],
        "TP": report["TP"], "TN": report["TN"],
        "FP": report["FP"], "FN": report["FN"],
        "human_validation": report["independent_external_human_validation"],
    }))


if __name__ == "__main__":
    main()
