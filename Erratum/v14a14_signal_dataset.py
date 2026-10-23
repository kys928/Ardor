#!/usr/bin/env python3
"""v14a14 stronger supervision: controlled same-prompt original vs causal targets.

CPU-only dataset contract. Reads the verified v14a4 train generator, excludes
all frozen v14a12 diagnostic training IDs, then selects 80 *distinct* actual
gradient-clipping prompts. Both arms use those 80 EXACT prompts in the EXACT
same order. The control retains the original v14a4 chosen text; the
intervention teaches proposition-audited causal mechanisms.

Answers comprise researcher-authored propositions plus a cue-specific worked
application. Structural evidence-span verification is NOT independent factual
validation. Neither training nor repository promotion happens here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from Erratum.build_v14a4_family_data import (
    FAMILIES, TRAIN_CUES, build_train_rows,
)
from Erratum.v14a13_supervision_audit import IDS as FROZEN_V14A12_IDS
from Erratum.v14a13_proposition_evaluator import PROPOSITIONS

ROUTE = "gradient_clipping"
FAMILIES_TARGET = {
    "definition_mechanism": 8,
    "exploding_gradients": 8,
    "norm_threshold": 8,
    "optimizer_stability": 8,
    "backprop_stability": 16,
    "learning_rate_comparison": 16,
    "false_premise_correction": 16,
}
assert sum(FAMILIES_TARGET.values()) == 80
FAMILY_ORDER = FAMILIES[ROUTE]

# Each numbered claim is a required atomic proposition for that family.
# The three columns correspond EXACTLY to evaluator proposition IDs in order.
# Four authored wording variants reduce pure memorization of one reference.
CLAIMS = {
"definition_mechanism": [
("Gradient clipping is the operation at work.", "It acts on computed gradients after backpropagation and before the optimizer updates parameters.", "It reduces an unusually large gradient magnitude instead of passing the full spike onward."),
("The described intervention is gradient clipping.", "Clipping modifies the computed backward-pass gradient prior to the weight update.", "An oversized vector is scaled down rather than sent unchanged to optimization."),
("This is gradient clipping, not a change to the model's objective.", "The calculation follows gradient computation but precedes the optimizer step.", "It places a bound on unusually large update-driving gradients."),
("The mechanism is gradient clipping.", "It processes gradients that have already been obtained by backpropagation, before the optimizer consumes them.", "Excessive gradient magnitude is reduced while the remaining signal can still drive learning."),
],
"exploding_gradients": [
("Exploding gradients can destabilize training because their magnitude becomes excessive.", "Gradient clipping mitigates this by reducing a too-large gradient after backward computation and before the optimizer update."),
("An exploding gradient can feed the optimizer an extreme update-driving signal and destabilize learning.", "Clipping conditionally scales down that computed signal before the optimizer uses it."),
("A very large backpropagated gradient is a stability risk, not evidence of safer learning.", "Gradient clipping constrains that extreme gradient magnitude before parameters are updated."),
("Exploding gradients can make subsequent optimization steps unstable.", "Clipping reduces an oversized computed gradient instead of letting its full magnitude reach the optimizer."),
],
"norm_threshold": [
("Norm clipping compares the total gradient norm with a configured maximum.", "Only when the observed norm exceeds the bound are gradients scaled down; a smaller norm is left unchanged.", "The optimizer then uses the resulting computed gradients."),
("The algorithm first measures a norm of the computed gradient vector and compares it to a threshold.", "A value above the threshold triggers rescaling; values at or below it are not amplified.", "The optimizer receives the clipped or unchanged gradient for its update."),
("Global-norm clipping checks whether the combined gradient norm violates its upper limit.", "It rescales gradients only on a violation and never boosts subthreshold gradients toward the cap.", "The adjusted gradient is used at the following optimizer step."),
("The norm is measured and tested against a maximum allowed magnitude.", "The scale is bounded above by one, so only oversized gradients shrink and smaller gradients pass through.", "Optimization proceeds using this conditionally bounded gradient."),
],
"optimizer_stability": [
("A gradient spike can create an unusually risky optimizer update.", "Clipping limits that computed gradient before it is passed to the optimizer.", "The reduced input can mitigate a destabilizing parameter change but cannot guarantee perfect stability."),
("An extreme computed gradient can cause an unstable weight update.", "Gradient clipping bounds the incoming gradient prior to the optimizer step.", "That reduces the risk from this particular spike rather than ensuring every update is safe."),
("The backward gradient sometimes spikes enough to threaten update stability.", "Clipping rescales that extreme gradient before the optimizer uses it.", "This can temper the following update without claiming a universal safety guarantee."),
("A large gradient can destabilize a subsequent parameter update.", "Clipping modifies the gradient before optimization instead of acting after the update.", "The optimizer works with a reduced spike, lowering the associated instability risk."),
],
"backprop_stability": [
("Backpropagation first computes the parameter gradients.", "Gradient clipping acts on those computed gradients after the backward pass and before the optimizer step.", "Reducing an oversized result can limit the instability risk of the subsequent weight update."),
("The backward pass produces gradients; it is not itself made error-free by clipping.", "Clipping then constrains excessive gradient magnitude before the optimizer receives it.", "The next parameter update is less exposed to the extreme backward-pass signal."),
("Backpropagation supplies the gradient that will drive an update.", "After this gradient has been calculated, norm clipping can scale it down before the optimizer applies an update.", "That controls the next step's exposure to a spike, not every numerical event during backpropagation."),
("Parameter gradients are computed during backpropagation.", "Clipping follows gradient computation and precedes the optimizer's use of those values.", "By reducing a dangerously large computed gradient, it mitigates risk in the following optimization step."),
],
"learning_rate_comparison": [
("Norm clipping selectively reduces computed gradients that exceed a specified magnitude threshold.", "The learning rate instead determines the optimizer's update scale more generally.", "The first changes an excessive gradient, whereas the second changes the step-size setting; clipping does not itself lower that setting."),
("Clipping conditionally rescales a too-large gradient vector.", "Changing the learning rate alters the general scale of optimizer parameter updates.", "Clipping and lowering the learning rate are distinct operations with different conditions and inputs."),
("When the gradient norm passes the maximum, clipping reduces that gradient before optimization.", "The learning rate controls how aggressively the optimizer updates across steps.", "Clipping does not edit the learning-rate parameter and a lower learning rate does not selectively cap a gradient."),
("Gradient-norm clipping bounds a computed gradient only when its norm is excessive.", "The learning rate is a separate optimizer control affecting update scale generally.", "Reducing the optimizer learning rate is not the same causal operation as clipping the gradient."),
],
"false_premise_correction": [
("No: exploding gradients are not a reliable source of safe learning.", "Very large gradient signals can destabilize optimizer updates.", "Gradient clipping is the protective operation: it reduces excessive computed gradients before the optimizer step."),
("That claim is false: exploding gradients do not safeguard optimization.", "An unbounded gradient can cause an unstable parameter update.", "Clipping limits the extreme gradient prior to the update and thereby mitigates that risk."),
("The premise is backwards; oversized gradients can be harmful rather than protective.", "Exploding backward signals can destabilize the next optimizer update.", "The corrective mechanism is clipping, which scales down computed gradients above a norm limit before optimization."),
("No. A very large gradient is not automatically beneficial or stabilizing.", "When gradients explode, their magnitude can make parameter updates unstable.", "Clipping reduces the oversized gradient before the optimizer acts; exploding it is not the remedy."),
],
}

# These four explicitly authored cue interpretations connect a particular
# source example to the mechanism, instead of tacking a generic final phrase
# onto every explanation. They are keyed to TRAIN_CUES ordering, *not* to IDs.
APPLICATION = {
"definition_mechanism": [
"Here the stated cap occurs on the gradient rather than on the parameter value itself.",
"The unusually large backward signal is reduced only after the backward computation has produced it.",
"The optimizer receives the controlled gradient instead of the raw oversized one.",
"Weight changes come after, not before, this gradient-magnitude intervention.",
],
"exploding_gradients": [
"A spike of this kind is a risk specifically because it can distort the next optimizer update.",
"The gradient is generated in the backward pass; clipping does not mean the entire backward pass becomes numerically safe.",
"The threat is the scale of the update-driving input, not simply that a derivative exists.",
"Rescaling the exploding signal before optimization targets the dangerous magnitude.",
],
"norm_threshold": [
"The comparison is conditional: gradients already inside the bound are not boosted to match it.",
"Having a maximum does not mean every update must have exactly that norm.",
"The statement 'only when it exceeds the limit' rules out routine rescaling of small gradients.",
"The magnitude threshold is enforced on the computed gradient before the update, not after it.",
],
"optimizer_stability": [
"Here it is the occasional large computed gradient that clipping reduces, not every normal training step.",
"AdamW will see the altered computed gradient, although its learning-rate setting is not changed by clipping.",
"Bounding the incoming gradient addresses this source of update instability without guaranteeing stability overall.",
"A more controlled gradient can lower spike-related update risk, but the optimizer may still behave badly for other reasons.",
],
"backprop_stability": [
"The dangerous signal has already been computed by backpropagation when clipping acts.",
"Clipping follows the gradient-producing backward pass; it does not intercept gradients before they exist.",
"Training stability is supported through the next update, not a promise that backpropagation can never fail.",
"The rescaling occurs between obtaining the backward signal and using it in the weight update.",
],
"learning_rate_comparison": [
"The cited difference is the quantity being modified: a computed gradient versus a general optimizer setting.",
"A gradient cap reacts to a threshold breach; a configured learning rate applies as part of the optimizer step even without one.",
"An exceptional gradient does not require lowering the learning rate everywhere to be reduced by clipping.",
"A gradient-norm rule leaves subthreshold gradients alone, whereas the learning rate still participates in normal optimizer updates.",
],
"false_premise_correction": [
"The mistaken claim confuses a dangerous extreme gradient with the stabilizing intervention that limits it.",
"The teammate's suggested protection comes from clipping the gradient, not letting its magnitude grow.",
"Faster learning is not established by an exploding gradient; excessive updates can instead destabilize optimization.",
"An unbounded backward signal threatens rather than protects the optimizer step that follows.",
],
}


def normalize(value: str) -> str:
    return " ".join(str(value).casefold().split())


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def select_rows() -> list[dict]:
    raw = build_train_rows()
    pools = {name: [] for name in FAMILIES_TARGET}
    seen = set()
    frozen = set(FROZEN_V14A12_IDS)
    for row in raw:
        family = row["semantic_family"]
        if row["route"] != ROUTE or family not in pools or row["id"] in frozen:
            continue
        if row["collision_route"] or row["question_class"] == "collision_comparison":
            continue
        p = normalize(row["prompt"])
        if p in seen:
            continue
        seen.add(p)
        pools[family].append(row)
    chosen = {}
    for family, desired in FAMILIES_TARGET.items():
        # Hash-sorted stratification: deterministic and independent of model outcomes.
        pool = sorted(pools[family],
                      key=lambda r: (hashlib.sha256(("v14a14|" + r["id"]).encode()).hexdigest(), r["id"]))
        if len(pool) < desired:
            raise RuntimeError(f"Insufficient unique prompts for {family}: {len(pool)} < {desired}")
        chosen[family] = pool[:desired]
    rows = []
    for family in FAMILY_ORDER:
        rows.extend(chosen[family])
    if len(rows) != 80 or len({normalize(r["prompt"]) for r in rows}) != 80:
        raise RuntimeError("80-row candidate failed unique prompt contract")
    return rows


def build_pairs() -> tuple[list[dict], list[dict], dict]:
    raw = select_rows()
    control, revised, evidence = [], [], []
    for family in FAMILY_ORDER:
        family_rows = [x for x in raw if x["semantic_family"] == family]
        for idx, original in enumerate(family_rows):
            prompt = original["prompt"].strip() + "\n-"
            base_row = {
                "id": original["id"], "route": ROUTE, "source": "behavior",
                "semantic_family": family,
                "question_class": original["question_class"],
                "collision_route": None, "prompt": prompt,
                "prompt_style": original["prompt_style"], "answer_style": original["answer_style"],
            }
            if original["question_class"] != ("false_premise_correction" if family == "false_premise_correction" else "normal"):
                raise RuntimeError("Question class mismatch")
            cues = TRAIN_CUES[(ROUTE, family)]
            cue_matches = [i for i, cue in enumerate(cues) if cue.casefold() in original["prompt"].casefold()]
            if len(cue_matches) != 1:
                raise RuntimeError(f"Could not uniquely identify source cue: {original['id']}")
            cue_idx = cue_matches[0]
            claim_variant = (idx // 4 + cue_idx + idx) % len(CLAIMS[family])
            clauses = CLAIMS[family][claim_variant]
            keys = tuple(PROPOSITIONS[family])
            if len(keys) != len(clauses):
                raise RuntimeError(f"Missing factual proposition for {family}")
            application = APPLICATION[family][cue_idx]
            new_answer = " ".join([*clauses, application])
            control.append({**base_row, "chosen": str(original["chosen"]).strip()})
            revised.append({**base_row, "chosen": new_answer})
            evidence.append({
                "id": original["id"], "family": family, "source_cue": cues[cue_idx],
                "requirements": [
                    {"proposition_id": key, "status": "supported",
                     "evidence": clause, "source": "author_reviewed_pyTorch_mechanism"}
                    for key, clause in zip(keys, clauses)
                ],
                "contextual_application": application,
                "unsupported_claim_or_contradiction_review": "author_checked_pending_independent_review",
                "reviewer_independence": False,
                "approval": "PENDING_HUMAN_REVIEW",
            })
    for old, new in zip(control, revised):
        assert old["id"] == new["id"] and old["prompt"] == new["prompt"]
    if len({r["chosen"] for r in revised}) < 55:
        raise RuntimeError("Revised examples lack sufficient answer diversity")
    for new, rec in zip(revised, evidence):
        if not all(part["evidence"] in new["chosen"] for part in rec["requirements"]):
            raise RuntimeError("Incomplete proposition evidence")
        if any(not part["evidence"].strip() for part in rec["requirements"]):
            raise RuntimeError("Empty factual proposition")
    summary = {
        "version": "v14a14-signal-1",
        "status": "DRAFT_NOT_TRAINING_APPROVED",
        "data_source": "verified v14a4 train generator; v14a12 selected IDs excluded",
        "train_cases": len(revised),
        "per_family": dict(Counter(r["semantic_family"] for r in revised)),
        "distinct_prompts": len({normalize(r["prompt"]) for r in revised}),
        "distinct_revised_answers": len({normalize(r["chosen"]) for r in revised}),
        "original_control_paired_prompt_sha256": digest([(r["id"], r["prompt"]) for r in control]),
        "revised_paired_prompt_sha256": digest([(r["id"], r["prompt"]) for r in revised]),
        "control_sha256": digest(control),
        "revised_sha256": digest(revised),
        "proposition_audit_sha256": digest(evidence),
        "assurance": "required proposition spans present; truth and contradictions require independent review",
        "training_eligible": False,
        "known_confound": "answer lengths differ between arms and original legacy answers may contain noise",
    }
    if summary["original_control_paired_prompt_sha256"] != summary["revised_paired_prompt_sha256"]:
        raise RuntimeError("Paired prompt sequences differ")
    return control, revised, {"manifest": summary, "evidence": evidence}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("v14a14_signal_output"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a, b, audit = build_pairs()
    for filename, obj in (
        ("original_80.jsonl", a), ("causal_80.jsonl", b),
        ("proposition_evidence_80.jsonl", audit["evidence"]),
    ):
        (args.output_dir / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in obj),
            encoding="utf-8",
        )
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(audit["manifest"], ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["manifest"], sort_keys=True))


if __name__ == "__main__":
    main()
