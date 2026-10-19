#!/usr/bin/env python3
"""Curate v14a13 training examples without mutating frozen v14a12 evidence.

CPU-only *candidate*, not wired to any trainer. Inputs to the curated set are
first rebuilt from the exact v14a12 chosen-answer digests (hard fail on drift).
Only the explicitly enumerated fields below are changed. There are still four
rows per gradient-clipping family, but every prompt is distinct and each task
has a clearly matched response.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from Erratum.v14a13_supervision_audit import (
    rebuild, digest, normalized_prompt, flags,
)

# ID -> prompt, answer, metadata and human-readable reason.
# A prompt is always stored in the training wrapper format: "...?\n-".
OVERRIDES = {
    "v14a4_train_gradient_clipping_0085": {
        "prompt": (
            "A backpropagated gradient becomes so large that using it directly could "
            "destabilize the next parameter update. Why is that dangerous, and how "
            "does gradient clipping intervene?\n-"
        ),
        "chosen": (
            "An exploding gradient can supply an excessively large signal to the "
            "optimizer, risking an unstable update. After backpropagation, gradient "
            "clipping reduces the magnitude of an oversized computed gradient "
            "before the optimizer uses it."
        ),
        "reason": "Replace repeated exploding-gradient question with a causal update-risk question.",
    },
    "v14a4_train_gradient_clipping_0086": {
        "prompt": (
            "The computed global gradient norm is already below the allowed maximum. "
            "Should norm clipping change the gradient? State the rule for both sides "
            "of the threshold.\n-"
        ),
        "chosen": (
            "No. If the computed norm is at or below the configured maximum, "
            "global-norm clipping leaves the gradient unchanged. If it exceeds "
            "that maximum, clipping scales down the gradient before the "
            "optimizer step."
        ),
        "reason": "Replace duplicate norm prompt with explicit below-threshold counterexample.",
    },
    "v14a4_train_gradient_clipping_0152": {
        "prompt": (
            "A colleague claims that clipping an unusually large gradient is the "
            "same as lowering the learning rate for every update. Explain why "
            "these are different controls.\n-"
        ),
        "chosen": (
            "They are different controls. Norm clipping conditionally rescales a "
            "computed gradient when its norm exceeds a bound. Lowering the "
            "learning rate changes the optimizer's update scale more generally, "
            "including steps without a gradient spike. Clipping does not itself "
            "change the configured learning rate."
        ),
        "question_class": "normal",
        "collision_route": None,
        "prompt_style": "v14a13_learning_rate_distinction",
        "answer_style": "v14a13_causal_contrast",
        "reason": "Remove the mismatched checkpoint contrast; explicitly distinguish gradient vs LR.",
    },
    "v14a4_train_gradient_clipping_0376": {
        "prompt": (
            "One training step has a rare gradient spike. Compare imposing a maximum "
            "gradient norm for that step with reducing the learning rate across "
            "all steps. Which method is conditional, and what does each change?\n-"
        ),
        "chosen": (
            "Maximum-norm clipping conditionally scales down computed gradients "
            "whose norm exceeds the threshold before the optimizer step. Lowering "
            "the learning rate reduces the update scale used by the optimizer "
            "more generally. Clipping changes the gradient input; it does not "
            "lower the learning-rate setting."
        ),
        "question_class": "normal",
        "collision_route": None,
        "prompt_style": "v14a13_learning_rate_distinction",
        "answer_style": "v14a13_causal_contrast",
        "reason": "Give the second former checkpoint-collision row a distinct LR-vs-clipping task.",
    },
    "v14a4_train_gradient_clipping_0391": {
        "prompt": (
            "A student argues that unbounded exploding gradients protect the "
            "optimizer because they let it learn faster. Correct the premise, "
            "then explain the protective operation.\n-"
        ),
        "chosen": (
            "That premise is false: exploding gradients can destabilize optimizer "
            "updates, not reliably protect or accelerate them. The protective "
            "mechanism is gradient clipping, which conditionally reduces an "
            "oversized computed gradient before the optimizer processes it."
        ),
        "reason": "Replace duplicate false-premise prompt with a distinct explicit speedup claim.",
    },
}

ALLOWED_CHANGES = {
    "prompt", "chosen", "question_class", "collision_route",
    "prompt_style", "answer_style",
}


def build_curated() -> tuple[list[dict], dict]:
    original, corrected = rebuild()  # frozen digests checked on every run
    if set(OVERRIDES) - {r["id"] for r in corrected}:
        raise RuntimeError("Curated overrides include unknown IDs")
    # Preserve unmodified rows byte-for-byte.
    result = []
    for row in corrected:
        new = dict(row)
        if row["id"] in OVERRIDES:
            spec = OVERRIDES[row["id"]]
            if set(spec) - ALLOWED_CHANGES - {"reason"}:
                raise RuntimeError(f"Unsafe edit fields for {row['id']}")
            new.update({k: v for k, v in spec.items() if k != "reason"})
        if not new["prompt"].endswith("\n-"):
            raise RuntimeError(f"Incorrect frozen prompt wrapper: {row['id']}")
        result.append(new)
    if len(result) != 28 or len(set(r["id"] for r in result)) != 28:
        raise RuntimeError("v14a13 must retain exactly 28 IDs")
    if Counter(r["semantic_family"] for r in result) != Counter(
        {family: 4 for family in {r["semantic_family"] for r in corrected}}
    ):
        raise RuntimeError("v14a13 must retain the original seven family counts")
    if len({normalized_prompt(r["prompt"]) for r in result}) != 28:
        raise RuntimeError("v14a13 has duplicate prompts")
    if any(r["semantic_family"] == "learning_rate_comparison" and
           (r["collision_route"] or "checkpoint" in r["prompt"].lower())
           for r in result):
        raise RuntimeError("Mixed learning-rate and checkpoint questions remain")
    if [(r["id"], r["route"], r["semantic_family"], r["source"])
        for r in result] != [(r["id"], r["route"], r["semantic_family"], r["source"])
                              for r in corrected]:
        raise RuntimeError("Changed frozen ID, route, family, source or ordering")
    edits = []
    for old, new in zip(corrected, result):
        fields = [k for k in ALLOWED_CHANGES if old[k] != new[k]]
        if fields:
            if old["id"] not in OVERRIDES:
                raise RuntimeError("Unapproved edit")
            edits.append({
                "id": old["id"], "family": old["semantic_family"],
                "reason": OVERRIDES[old["id"]]["reason"],
                "changed_fields": sorted(fields),
                "old_prompt": old["prompt"], "new_prompt": new["prompt"],
                "old_chosen": old["chosen"], "new_chosen": new["chosen"],
            })
    if len(edits) != 5:
        raise RuntimeError("Expected five curated row edits")
    digest_payload = [
        {"id": r["id"], "prompt": r["prompt"], "chosen": r["chosen"],
         "question_class": r["question_class"], "collision_route": r["collision_route"]}
        for r in result
    ]
    candidate_sha256 = hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    report = {
        "version": "v14a13-curation-1",
        "status": "candidate_not_trainer_wired",
        "frozen_v14a12_control_chosen_sha256": digest(original),
        "frozen_v14a12_corrected_chosen_sha256": digest(corrected),
        "curated_prompt_answer_metadata_sha256": candidate_sha256,
        "source_rows": 28, "candidate_rows": 28,
        "unique_prompts_before": len({normalized_prompt(r["prompt"]) for r in corrected}),
        "unique_prompts_after": len({normalized_prompt(r["prompt"]) for r in result}),
        "edited_rows": edits,
        "triage_flags_after": {
            r["id"]: flags(r, result) for r in result if flags(r, result)
        },
        "independent_human_factual_review": "pending",
    }
    return result, report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("v14a13_audit_output"))
    args = p.parse_args()
    rows, report = build_curated()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v14a13_curated_training_candidate.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    (args.output_dir / "v14a13_curated_edit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(rows),
        "unique_prompts": report["unique_prompts_after"],
        "edited": len(report["edited_rows"]),
        "candidate_digest": report["curated_prompt_answer_metadata_sha256"],
    }))


if __name__ == "__main__":
    main()
