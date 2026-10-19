#!/usr/bin/env python3
"""v14a13 proposition-based factual adjudication contract (CPU only).

Deliberately NOT a keyword classifier or an LLM-as-judge. A reviewer labels
atomic factual propositions and supplies verbatim supporting/conflicting spans.
The program validates those spans and deterministically applies strict scoring.
One reviewer remains exploratory; require blinded independent review for claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Each proposition expresses a necessary, factually testable part of an answer.
# Applicable requirements are derived from the case's semantic family/class.
PROPOSITIONS: dict[str, dict[str, str]] = {
    "definition_mechanism": {
        "gradient_clipping_identified": "Identifies gradient clipping rather than another mechanism",
        "post_backprop_pre_optimizer": "Clipping applies to computed gradients before the optimizer update",
        "large_gradient_bounded": "Explains limiting/rescaling an unusually large gradient",
    },
    "exploding_gradients": {
        "explosion_harms_stability": "Oversized/exploding gradients can destabilize optimization",
        "clipping_mitigates_spike": "Clipping reduces an excessive gradient before the optimizer uses it",
    },
    "norm_threshold": {
        "norm_is_compared_to_limit": "Gradient vector norm is compared to a maximum/threshold",
        "conditional_rescaling": "Only gradients exceeding the limit are scaled down; smaller ones are unchanged",
        "optimizer_follows_clipping": "Adjusted gradient is used for the optimizer update",
    },
    "optimizer_stability": {
        "spike_precedes_update": "A spike in computed gradient can create pressure for unstable updates",
        "gradient_bounded_before_optimizer": "Clipping bounds gradients before they are supplied to the optimizer",
        "mitigates_not_guarantees": "Reduces risk of unstable update without claiming a universal stability guarantee",
    },
    "backprop_stability": {
        "backprop_computes_gradients": "Backpropagation first computes gradients",
        "clipping_after_backward": "Clipping acts on computed gradients after backward and before optimizer",
        "next_update_mitigation": "Primary protection is for subsequent updates, not a claim to fix all backprop errors",
    },
    "learning_rate_comparison": {
        "clipping_conditional_gradient_control": "Clipping conditionally reduces a gradient exceeding a threshold",
        "learning_rate_controls_step_scale": "Learning rate controls how aggressively the optimizer updates across steps",
        "mechanisms_distinguished": "Does not equate gradient clipping with changing the learning rate",
    },
    "false_premise_correction": {
        "explicitly_reject_false_premise": "Explicitly rejects claim that exploding/unbounded gradients protect training",
        "explosion_is_risk": "Explains exploding gradients can produce unsafe or unstable updates",
        "clipping_is_protection": "Explains clipping, not exploding gradients, mitigates oversized update-driving signal",
    },
}
ADDITIONAL = {
    "collision_comparison": {
        "contrasted_collision": "Correctly distinguishes named competing concept from gradient clipping"
    },
}
STATUSES = {"supported", "missing", "contradicted", "unclear"}


def applicable(case: dict) -> dict[str, str]:
    family = case["family"]
    if family not in PROPOSITIONS:
        raise ValueError(f"Unknown semantic family: {family}")
    specs = dict(PROPOSITIONS[family])
    if case.get("question_class") == "collision_comparison":
        specs.update(ADDITIONAL["collision_comparison"])
    return specs


def _verbatim_span(text: str, span: str) -> bool:
    return isinstance(span, str) and bool(span.strip()) and span in text


def evaluate(case: dict, annotation: dict) -> dict:
    """Validate human annotation and strictly score one output.

    Format:
      case {id,family,question_class,...}
      annotation {case_id,reviewer,response,propositions:{
        id:{status:'supported|missing|contradicted|unclear',evidence:'...'}
      }, contradictions:[{evidence:'...',description:'...'}]}
    A supported proposition REQUIRES its literal evidence span in response.
    Evidence text alone is not a determination of correctness; reviewer owns
    that judgment. Any wrong, missing or unclear required proposition is failure.
    """
    if annotation.get("case_id") != case["id"]:
        raise ValueError("Case ID mismatch")
    if not annotation.get("reviewer"):
        raise ValueError("Adjudication requires named reviewer ID")
    text = annotation.get("response")
    if not isinstance(text, str):
        raise ValueError("response must be a string")
    specs = applicable(case)
    claims = annotation.get("propositions")
    if not isinstance(claims, dict) or set(claims) != set(specs):
        raise ValueError("Required proposition IDs must match exactly")
    failed, statuses = [], {}
    for key in specs:
        item = claims[key]
        status = item.get("status")
        if status not in STATUSES:
            raise ValueError(f"Invalid annotation status for {key}")
        evidence = item.get("evidence", "")
        if status in {"supported", "contradicted"}:
            if not _verbatim_span(text, evidence):
                raise ValueError(f"Unsupported evidence span: {key}")
        elif evidence:
            raise ValueError(f"Missing/unclear claim must not fabricate an evidence span: {key}")
        statuses[key] = status
        if status != "supported":
            failed.append(key)
    contradictions = annotation.get("contradictions", [])
    if not isinstance(contradictions, list):
        raise ValueError("contradictions must be a list")
    for issue in contradictions:
        if not isinstance(issue, dict) or not issue.get("description"):
            raise ValueError("Each contradiction needs a reviewer explanation")
        if not _verbatim_span(text, issue.get("evidence", "")):
            raise ValueError("Contradiction requires a verbatim quoted span")
    return {
        "case_id": case["id"],
        "family": case["family"],
        "reviewer": annotation["reviewer"],
        "adequate": not failed and not contradictions,
        "failed_propositions": failed,
        "statuses": statuses,
        "contradiction_count": len(contradictions),
        "non_blind_exploratory_only": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, required=True,
                   help="Output from v14a13_supervision_audit.py")
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    raw = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = {r["id"]: r for r in raw["cases"]}
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(annotations, list):
        raise ValueError("Expected JSON list of adjudicated outputs")
    scores = []
    for a in annotations:
        case = cases.get(a["case_id"])
        if case is None:
            raise ValueError(f"Unknown case: {a['case_id']}")
        scores.append(evaluate(case, a))
    args.output.write_text(
        json.dumps({"scores": scores, "reviewer_status": "exploratory_not_blind",
                    "adequate": sum(x["adequate"] for x in scores),
                    "total": len(scores)}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
