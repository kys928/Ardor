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


# First-pass human adjudication of all 80 originally generated answers.
# Exactly ten original proposals were judged inadequate; all remaining
# proposals were judged adequate. This is ONE human review, NOT independent
# two-reviewer approval. The 10 revised examples require fresh adjudication.
REVIEW_ROUND_1_INADEQUATE = {
    "v14a4_train_gradient_clipping_0050": {
        "case": 9, "mode": "append",
        "reason": "First-principles request omitted the gradient-to-parameter-update causal link.",
        "text": (
            "In basic gradient descent, the optimizer forms a parameter change "
            "from the gradient and learning rate. At a fixed learning rate, a "
            "larger raw gradient can cause an excessively large parameter step "
            "that overshoots; clipping reduces that signal before the update."
        ),
    },
    "v14a4_train_gradient_clipping_0071": {
        "case": 14, "mode": "append",
        "reason": "Large gradient was not causally linked to the optimizer's parameter update.",
        "text": (
            "For ordinary gradient descent, the update is minus learning rate "
            "times the gradient, so an enormous gradient at fixed learning rate "
            "can drive an enormous parameter step. Adaptive optimizers can "
            "change that scaling, but clipping still constrains the incoming "
            "computed gradient."
        ),
    },
    "v14a4_train_gradient_clipping_0092": {
        "case": 15, "mode": "append",
        "reason": "First-principles answer missed how computed derivatives influence the update.",
        "text": (
            "Backpropagation computes the derivatives used by the optimizer. "
            "If their magnitudes become extreme, they can contribute to an "
            "unstable subsequent parameter update; clipping scales down the "
            "computed gradient before that update is performed."
        ),
    },
    "v14a4_train_gradient_clipping_0051": {
        "case": 24, "mode": "prepend",
        "reason": "Prompt requested that the concept be named explicitly.",
        "text": "The concept is global gradient-norm clipping.",
    },
    "v14a4_train_gradient_clipping_0109": {
        "case": 33, "mode": "prepend",
        "reason": "Prompt required leading with the named concept.",
        "text": "The relevant concept is gradient clipping.",
    },
    "v14a4_train_gradient_clipping_0067": {
        "case": 35, "mode": "prepend",
        "reason": "Prompt required leading with the named concept.",
        "text": "The relevant concept is gradient clipping.",
    },
    "v14a4_train_gradient_clipping_0004": {
        "case": 36, "mode": "prepend",
        "reason": "Prompt required leading with the named concept.",
        "text": "The relevant concept is gradient clipping.",
    },
    "v14a4_train_gradient_clipping_0046": {
        "case": 44, "mode": "prepend",
        "reason": "Prompt required leading with the named concept.",
        "text": "The relevant concept is gradient clipping.",
    },
    "v14a14_train_gradient_clipping_backprop_stability_01": {
        "case": 45, "mode": "append",
        "reason": "Prompt asked specifically what clipping cannot repair retroactively.",
        "text": (
            "Because clipping happens only after gradient computation, it "
            "cannot retroactively undo numerical overflow, non-finite "
            "intermediate values, or a failed calculation that already "
            "occurred inside the backward pass."
        ),
    },
    "v14a14_train_gradient_clipping_backprop_stability_03": {
        "case": 47, "mode": "append",
        "reason": "Gradient accumulation required an accumulation-specific sequence.",
        "text": (
            "With gradient accumulation, backpropagation runs for each "
            "microbatch and the gradients accumulate. The usual procedure "
            "is to clip the accumulated gradient before the optimizer step, "
            "not to treat clipping as a substitute for backward computation."
        ),
    },
}
REVIEW_ROUND_1_INADEQUATE_CASES = (9, 14, 15, 24, 33, 35, 36, 44, 45, 47)
if sorted(item["case"] for item in REVIEW_ROUND_1_INADEQUATE.values()) != sorted(REVIEW_ROUND_1_INADEQUATE_CASES):
    raise RuntimeError("Human adjudication case-to-ID mapping drift")


def apply_human_review_repair(row_id: str, original_answer: str) -> str:
    amendment = REVIEW_ROUND_1_INADEQUATE.get(row_id)
    if amendment is None:
        return original_answer
    if amendment["mode"] == "prepend":
        return amendment["text"] + " " + original_answer
    if amendment["mode"] == "append":
        return original_answer + " " + amendment["text"]
    raise RuntimeError(f"Unknown review-amendment mode for {row_id}")

# Human review round 2: nine of the ten first-round repairs were accepted.
# Case 15 remained inadequate. Independently, the reviewer requested less
# template reuse in 33/35/36/44/45/47 despite their adequate factual scores.
# These seven fresh rewrites are NEW candidates, not approved by that review.
REVIEW_ROUND_2_ADEQUATE_CASES = (9, 14, 24, 33, 35, 36, 44, 45, 47)
REVIEW_ROUND_2_INADEQUATE_CASES = (15,)

# Explicit individual responses, authored according to the *particular*
# source question; unlike the generic family CLAIMS and APPLICATION pool.
# Each proposition span is deliberately short and verbatim within the answer.
ROUND_3_TARGETED_REWRITES = {
    "v14a4_train_gradient_clipping_0092": {
        "case": 15,
        "reason": "First-principles explanation must connect gradient magnitude to the parameter-step equation.",
        "chosen": (
            "Backpropagation computes the loss gradient with respect to model parameters. "
            "For ordinary gradient descent, the update is Δθ = −η∇θL: at a fixed "
            "learning rate η, an exploding gradient can produce an excessively "
            "large parameter change that overshoots or destabilizes training. "
            "Gradient clipping reduces an oversized computed gradient before "
            "the optimizer uses it, mitigating that update risk."
        ),
        "support": {
            "explosion_harms_stability": (
                "an exploding gradient can produce an excessively "
                "large parameter change that overshoots or destabilizes training"
            ),
            "clipping_mitigates_spike": (
                "Gradient clipping reduces an oversized computed gradient "
                "before the optimizer uses it"
            ),
        },
        "application": "For ordinary gradient descent, the update is Δθ = −η∇θL",
    },
    "v14a4_train_gradient_clipping_0109": {
        "case": 33,
        "reason": "Lead with mechanism and adapt to rescaling an already-computed oversized backward signal.",
        "chosen": (
            "Gradient clipping is the intervention: when backpropagation has "
            "computed an excessively large gradient, it rescales that gradient "
            "before the optimizer step. Because the optimizer receives a reduced "
            "spike instead of the original magnitude, the subsequent parameter "
            "update is less exposed to instability."
        ),
        "support": {
            "backprop_computes_gradients": "backpropagation has computed an excessively large gradient",
            "clipping_after_backward": "it rescales that gradient before the optimizer step",
            "next_update_mitigation": "the subsequent parameter update is less exposed to instability",
        },
        "application": "the optimizer receives a reduced spike instead of the original magnitude",
    },
    "v14a4_train_gradient_clipping_0067": {
        "case": 35,
        "reason": "Explain how the observed spike arises, rather than reusing a generic stabilization answer.",
        "chosen": (
            "Gradient clipping handles the spike produced by backpropagation. "
            "The backward pass first computes parameter gradients; if their "
            "norm exceeds a chosen cap, clipping scales them down before "
            "the optimizer uses them. This reduces exposure to an excessive "
            "update-driving signal without preventing the spike itself "
            "from arising during backward computation."
        ),
        "support": {
            "backprop_computes_gradients": "The backward pass first computes parameter gradients",
            "clipping_after_backward": (
                "clipping scales them down before the optimizer uses them"
            ),
            "next_update_mitigation": "This reduces exposure to an excessive update-driving signal",
        },
        "application": "without preventing the spike itself from arising during backward computation",
    },
    "v14a4_train_gradient_clipping_0004": {
        "case": 36,
        "reason": "Emphasize the boundary between computed gradient and optimizer input.",
        "chosen": (
            "Gradient clipping bounds a large signal after the backward pass "
            "has computed the gradient. For norm clipping, an above-threshold "
            "gradient is scaled down, while smaller gradients are unchanged. "
            "The optimizer then uses this bounded result for its parameter "
            "update, avoiding direct use of the full outlier magnitude."
        ),
        "support": {
            "backprop_computes_gradients": "the backward pass has computed the gradient",
            "clipping_after_backward": (
                "Gradient clipping bounds a large signal after the backward pass "
                "has computed the gradient"
            ),
            "next_update_mitigation": (
                "avoiding direct use of the full outlier magnitude"
            ),
        },
        "application": "The optimizer then uses this bounded result for its parameter update",
    },
    "v14a4_train_gradient_clipping_0046": {
        "case": 44,
        "reason": "Explain next-update stabilization without guaranteeing backward-pass safety.",
        "chosen": (
            "Gradient clipping reduces the risk from an oversized gradient "
            "produced by backpropagation. Once the backward pass has calculated "
            "parameter derivatives, clipping limits excessive gradient "
            "magnitude before the optimizer updates weights. This can protect "
            "the next update from a spike; it does not make numerical "
            "operations inside the backward pass infallible."
        ),
        "support": {
            "backprop_computes_gradients": "the backward pass has calculated parameter derivatives",
            "clipping_after_backward": (
                "clipping limits excessive gradient magnitude before the optimizer updates weights"
            ),
            "next_update_mitigation": "This can protect the next update from a spike",
        },
        "application": "it does not make numerical operations inside the backward pass infallible",
    },
    "v14a14_train_gradient_clipping_backprop_stability_01": {
        "case": 45,
        "reason": "Put retroactive failure limitation at the center of the answer.",
        "chosen": (
            "Autograd has already run backpropagation and produced gradient "
            "tensors. Norm clipping can now rescale an oversized computed "
            "gradient before the optimizer step, lowering risk in the "
            "following parameter update. It cannot retroactively repair "
            "numerical overflow, non-finite intermediate values, or a "
            "failed computation that happened during the backward pass."
        ),
        "support": {
            "backprop_computes_gradients": (
                "Autograd has already run backpropagation and produced gradient tensors"
            ),
            "clipping_after_backward": (
                "Norm clipping can now rescale an oversized computed "
                "gradient before the optimizer step"
            ),
            "next_update_mitigation": "lowering risk in the following parameter update",
        },
        "application": (
            "It cannot retroactively repair numerical overflow, non-finite "
            "intermediate values, or a failed computation that happened during the backward pass"
        ),
    },
    "v14a14_train_gradient_clipping_backprop_stability_03": {
        "case": 47,
        "reason": "Center the gradient-accumulation sequence and the question's contrast.",
        "chosen": (
            "Clipping protects the next optimizer update, not the backward "
            "computations themselves. With gradient accumulation, each "
            "microbatch's backward pass adds to parameter gradient buffers. "
            "After the intended accumulation steps, global-norm clipping "
            "can limit the accumulated gradients before the optimizer "
            "step. That reduces the spike reaching the update but cannot "
            "undo a numerical error in any backward pass."
        ),
        "support": {
            "backprop_computes_gradients": (
                "each microbatch's backward pass adds to parameter gradient buffers"
            ),
            "clipping_after_backward": (
                "After the intended accumulation steps, global-norm clipping "
                "can limit the accumulated gradients before the optimizer step"
            ),
            "next_update_mitigation": (
                "That reduces the spike reaching the update"
            ),
        },
        "application": "Clipping protects the next optimizer update, not the backward computations themselves",
    },
}
assert sorted(item["case"] for item in ROUND_3_TARGETED_REWRITES.values()) == [15, 33, 35, 36, 44, 45, 47]


def normalize(value: str) -> str:
    return " ".join(str(value).casefold().split())


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()



# New concept-specific questions fill the verified shortfall in the v14a4
# source: 12 distinct non-collision prompts per ordinary family but only 4
# false-premise prompts. They are researcher-written, not random paraphrases.
# Tuples: (cue_index, distinct question frame). Each embeds the exact source
# cue to ensure original-family answer compatibility is checkable.
EXTENSION_CASES = {
"backprop_stability": [
(0, "An autograd pass has completed and {cue}. What has already happened, and what cannot be fixed retroactively by clipping?"),
(1, "A training engineer is debugging this situation: {cue}. Place clipping correctly between derivative computation and AdamW."),
(2, "During accumulation, an intern notices that {cue}. Does clipping avert the backward computation itself, or the next update?"),
(3, "Investigate the order of operations when {cue}. Name the data produced by backward computation before any clipping."),
],
"learning_rate_comparison": [
(0, "A new practitioner mistakes a single-batch norm cap for changing every optimizer step: {cue}. Correct that identification."),
(1, "During a training review, someone combines two different controls: {cue}. Explain which control acts conditionally."),
(2, "A loss spike occurs but the configured optimizer step size should remain unchanged: {cue}. Contrast the two interventions."),
(3, "The training trace shows that {cue}. Which quantity is altered, and which optimizer setting remains separate?"),
],
"false_premise_correction": [
(0, "Correct this risky misconception from an optimization lecture: {cue}. Why does the stated causal direction fail?"),
(0, "A report describes the following as a safety guarantee: {cue}. Reject or accept the premise, then identify the real protection."),
(0, "A student writes this unqualified conclusion: {cue}. What is the actual effect on update stability?"),
(1, "A teammate misunderstands a sudden outlier: {cue}. Explain why the statement is not a valid optimizer-safety principle."),
(1, "A reviewer endorses this claim: {cue}. Which operation, if any, genuinely mitigates update instability?"),
(1, "Would the following be a valid training rule: {cue}? Correct it with a causal description of gradient clipping."),
(2, "A group suggests leaving spikes untouched for faster convergence: {cue}. What dangerous assumption is hidden here?"),
(2, "During a code review, an engineer repeats: {cue}. What does the optimizer actually risk?"),
(2, "In a training retrospective, the claim is: {cue}. Explain whether more magnitude is necessarily beneficial."),
(3, "A workshop slide offers this explanation: {cue}. Diagnose what it reverses."),
(3, "A teammate says the following prevents overshooting: {cue}. What genuinely limits the incoming signal?"),
(3, "Correct this high-confidence but false statement: {cue}. State the right sequence after backward computation."),
],
}


def _authored_extensions(family: str, rows: list[dict]) -> list[dict]:
    cases = EXTENSION_CASES.get(family, [])
    if not cases:
        return []
    cues = TRAIN_CUES[(ROUTE, family)]
    from_source = {}
    for r in rows:
        for idx, cue in enumerate(cues):
            if cue.casefold() in r["prompt"].casefold() and idx not in from_source:
                from_source[idx] = r
    if len(from_source) < 4:
        raise RuntimeError(f"Missing control cue sources for {family}")
    extra = []
    for i, (cue_idx, template) in enumerate(cases, 1):
        donor = from_source[cue_idx]
        item = dict(donor)
        item.update({
            "id": f"v14a14_train_gradient_clipping_{family}_{i:02d}",
            "prompt": template.format(cue=cues[cue_idx]),
            "prompt_style": "v14a14_authored_scenario",
            "data_origin": "new_prompt_with_legacy_cue_matched_control",
        })
        extra.append(item)
    return extra


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
            pool = pool + _authored_extensions(family, pool)
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
    unrepaired_answers = {}
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
                "data_origin": original.get("data_origin", "unaltered_v14a4_source"),
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
            original_proposed_answer = " ".join([*clauses, application])
            unrepaired_answers[original["id"]] = original_proposed_answer
            new_answer = apply_human_review_repair(original["id"], original_proposed_answer)
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
                "review_round_1_case_number": len(evidence) + 1,
                "review_round_1_original_judgment": (
                    "inadequate" if original["id"] in REVIEW_ROUND_1_INADEQUATE else "adequate"
                ),
                "review_round_1_repair": (
                    {"reason": REVIEW_ROUND_1_INADEQUATE[original["id"]]["reason"],
                     "mode": REVIEW_ROUND_1_INADEQUATE[original["id"]]["mode"],
                     "added_text": REVIEW_ROUND_1_INADEQUATE[original["id"]]["text"]}
                    if original["id"] in REVIEW_ROUND_1_INADEQUATE else None
                ),
                "review_round_2_status": "PENDING_REVIEW",
            })
    for old, new in zip(control, revised):
        assert old["id"] == new["id"] and old["prompt"] == new["prompt"]
    # Programmatically reconstruct the original, already-reviewed 80-target
    # artifact. This verifies precisely ten edits, with all 70 previously
    # adequate responses byte-for-byte unchanged.
    prior_candidate = [
        {**r, "chosen": unrepaired_answers[r["id"]]} for r in revised
    ]
    prior_digest = digest(prior_candidate)
    if prior_digest != "4b3afb700aa1cd7f880010949860a33aa127842db2148b2b1d6060accb70e3a9":
        raise RuntimeError("Pre-adjudication 80-target source no longer reproduces")
    modified_ids = {
        r["id"] for r in revised if r["chosen"] != unrepaired_answers[r["id"]]
    }
    if modified_ids != set(REVIEW_ROUND_1_INADEQUATE):
        raise RuntimeError("Only ten human-rejected target answers may change")
    for old, new in zip(prior_candidate, revised):
        if old["id"] not in modified_ids and old != new:
            raise RuntimeError("Accepted target was mutated")
    if len({r["chosen"] for r in revised}) < 55:
        raise RuntimeError("Revised examples lack sufficient answer diversity")
    if {x["id"] for x in revised if x["id"] in REVIEW_ROUND_1_INADEQUATE} != set(REVIEW_ROUND_1_INADEQUATE):
        raise RuntimeError("Some adjudicated row IDs were not in the deterministic selection")
    if sum(r["review_round_1_original_judgment"] == "inadequate" for r in evidence) != 10:
        raise RuntimeError("Incorrect review outcome count")
    # Before replacing any round-two answer, verify its exact frozen dataset
    # AND per-proposition audit. Later edits must not overwrite past evidence.
    round_2_digest = digest(revised)
    round_2_evidence_digest = digest(evidence)
    if round_2_digest != "80d7716b29d06469d7bf4dc3ef91cbb4a0ff69490d93bb9845888c1d21451ab3":
        raise RuntimeError("Human round-2 reviewed dataset drift")
    if round_2_evidence_digest != "9682546efc3b561845a71d858455434d7f8389081221a6e38dfb123aa7319899":
        raise RuntimeError("Human round-2 evidence drift")
    seen_rewrites = set()
    for row, ev in zip(revised, evidence):
        previous_answer = row["chosen"]
        if row["id"] in REVIEW_ROUND_1_INADEQUATE:
            case = ev["review_round_1_case_number"]
            ev["review_round_2_judgment"] = (
                "ADEQUATE" if case in REVIEW_ROUND_2_ADEQUATE_CASES
                else "INADEQUATE"
            )
            ev["review_round_2_submitted_answer"] = previous_answer
        if row["id"] not in ROUND_3_TARGETED_REWRITES:
            continue
        repair = ROUND_3_TARGETED_REWRITES[row["id"]]
        if repair["case"] != ev["review_round_1_case_number"]:
            raise RuntimeError("Round-three revised wrong case")
        if row["semantic_family"] not in PROPOSITIONS:
            raise RuntimeError("Unknown family")
        if set(repair["support"]) != set(PROPOSITIONS[row["semantic_family"]]):
            raise RuntimeError("A round-three repair omitted necessary propositions")
        if any(value not in repair["chosen"] for value in [*repair["support"].values(), repair["application"]]):
            raise RuntimeError("Round-three source span not in chosen answer")
        if previous_answer == repair["chosen"]:
            raise RuntimeError("Round-three answer not actually changed")
        row["chosen"] = repair["chosen"]
        ev["review_round_3_prior_answer"] = previous_answer
        ev["review_round_3_reason"] = repair["reason"]
        ev["requirements"] = [
            {
                "proposition_id": key,
                "status": "supported",
                "evidence": repair["support"][key],
                "source": "round_3_authored_revision_pending_independent_factual_review",
            }
            for key in PROPOSITIONS[row["semantic_family"]]
        ]
        ev["contextual_application"] = repair["application"]
        ev["review_round_3_status"] = "PENDING_RE_ADJUDICATION"
        seen_rewrites.add(row["id"])
    if seen_rewrites != set(ROUND_3_TARGETED_REWRITES):
        raise RuntimeError("Incomplete round-three rewrite coverage")
    # The source snapshots guarantee only the targeted seven answers changed.
    # Previous reviewer judgments must not be extrapolated to their rewrites.
    current_by_id = {r["id"]: r["chosen"] for r in revised}
    previous_by_id = {
        r["id"]: (
            ev["review_round_3_prior_answer"]
            if ev.get("review_round_3_status") else r["chosen"]
        )
        for r, ev in zip(revised, evidence)
    }
    if {rid for rid in current_by_id if current_by_id[rid] != previous_by_id[rid]} != seen_rewrites:
        raise RuntimeError("Round-three revisions modified unexpected cases")
    for new, rec in zip(revised, evidence):
        if not all(part["evidence"] in new["chosen"] for part in rec["requirements"]):
            raise RuntimeError("Incomplete proposition evidence")
        if any(not part["evidence"].strip() for part in rec["requirements"]):
            raise RuntimeError("Empty factual proposition")
    summary = {
        "version": "v14a14-signal-1",
        "status": "DRAFT_NOT_TRAINING_APPROVED",
        "data_source": "verified v14a4 train generator plus 20 explicit cue-matched authored prompts; v14a12 IDs excluded",
        "train_cases": len(revised),
        "review_round_1_summary": {
            "previous_80_causal_sha256": prior_digest,
            "amended_case_count": len(modified_ids),
            "untouched_adequate_text_count": len(revised) - len(modified_ids),
            "original_cases": 80, "adequate_original": 70,
            "inadequate_original": 10, "uncertain_original": 0,
            "repaired_case_numbers": list(REVIEW_ROUND_1_INADEQUATE_CASES),
            "repair_status": "revised_10_pending_readjudication",
            "independent_second_review_status": "PENDING",
        },
        "review_round_2_summary": {
            "adjudicated": 10, "adequate": 9, "inadequate": 1,
            "inadequate_cases": list(REVIEW_ROUND_2_INADEQUATE_CASES),
            "seven_rewrites_pending_re_adjudication": sorted(
                x["case"] for x in ROUND_3_TARGETED_REWRITES.values()
            ),
            "round_2_causal_sha256": round_2_digest,
            "round_2_evidence_sha256": round_2_evidence_digest,
            "second_independent_reviewer": "NOT_PERFORMED",
        },
        "per_family": dict(Counter(r["semantic_family"] for r in revised)),
        "origin_counts": dict(Counter(r["data_origin"] for r in revised)),
        "distinct_prompts": len({normalize(r["prompt"]) for r in revised}),
        "distinct_revised_answers": len({normalize(r["chosen"]) for r in revised}),
        "original_control_paired_prompt_sha256": digest([(r["id"], r["prompt"]) for r in control]),
        "revised_paired_prompt_sha256": digest([(r["id"], r["prompt"]) for r in revised]),
        "control_sha256": digest(control),
        "revised_sha256": digest(revised),
        "proposition_audit_sha256": digest(evidence),
        "assurance": "required proposition spans present; truth and contradictions require independent review",
        "training_eligible": False,
        "known_confound": "answer lengths differ; 20 authored prompts pair with cue-matched original-family legacy answers, not historical examples",
    }
    if summary["original_control_paired_prompt_sha256"] != summary["revised_paired_prompt_sha256"]:
        raise RuntimeError("Paired prompt sequences differ")
    return control, revised, {"manifest": summary, "evidence": evidence}



def human_review_packet(rows: list[dict], evidence: list[dict]) -> str:
    lines = [
        "# v14a14 — independent reviewer training-data packet",
        "",
        "80 matched-source questions. Review ONLY intervention answers here.",
        "First-pass results: 70 adequate, 10 inadequate; the 10 amended answers",
        "and the overall corpus are PENDING new review and second reviewer.",
        "Prior adequate judgments were for unchanged answer text only.",
        "Judge all required causal propositions and scan the FULL answer for",
        "contradictions, unsupported guarantees, question mismatch and ambiguity.",
        "Automatic spans establish traceability, NOT semantic truth.",
        "Do not consult the model or holdout to choose a passing answer.",
        "",
        "Decision per example: ADEQUATE, INADEQUATE, or UNCERTAIN.",
        "Any missing essential claim or material contradiction is inadequate.",
        "",
    ]
    for i, (r, e) in enumerate(zip(rows, evidence), 1):
        lines.extend([
            f"## {i:02d} — {r['id']} ({r['semantic_family']})",
            "",
            "First-pass original judgment: " + e["review_round_1_original_judgment"].upper(),
            ("Repair reason: " + e["review_round_1_repair"]["reason"]) if e["review_round_1_repair"] else "Original answer preserved exactly.",
            "",
            "**Prompt**",
            "",
            r["prompt"],
            "",
            "**Proposed supervision**",
            "",
            r["chosen"],
            "",
            "**Required proposition evidence (verify semantically)**",
            "",
        ])
        for q in e["requirements"]:
            lines.append(f"- {q['proposition_id']}: {q['evidence']}")
        lines.extend([
            "",
            "Verdict: [ ] Adequate  [ ] Inadequate  [ ] Uncertain",
            "",
            "Contradiction or missing claim: ________________________",
            "",
        ])
    return "\n".join(lines)



def review_only_amended_ten(rows: list[dict], evidence: list[dict]) -> str:
    """Follow-up packet: the 10 human-rejected supervision answers only."""
    lines = [
        "# v14a14 — ten corrected answers, second-pass review",
        "",
        "Human round 1 judged 70/80 adequate and 10/80 inadequate.",
        "The other 70 proposed answers were preserved without changes.",
        "These ten revisions are unapproved and require renewed review.",
        "This packet is not an independent second reviewer attestation.",
        "",
    ]
    for row, ev in zip(rows, evidence):
        amendment = ev["review_round_1_repair"]
        if amendment is None:
            continue
        new = row["chosen"]
        added = amendment["added_text"]
        if amendment["mode"] == "prepend":
            if not new.startswith(added + " "):
                raise RuntimeError(f"Cannot reverse prepended repair: {row['id']}")
            old = new[len(added) + 1:]
        elif amendment["mode"] == "append":
            if not new.endswith(" " + added):
                raise RuntimeError(f"Cannot reverse appended repair: {row['id']}")
            old = new[:-(len(added) + 1)]
        else:
            raise RuntimeError("Unknown repair mode")
        lines.extend([
            f"## Case {ev['review_round_1_case_number']:02d} — {row['id']}",
            "",
            "Reason for original rejection: " + amendment["reason"],
            "",
            "**Prompt**",
            "",
            row["prompt"],
            "",
            "**Original inadequate answer**",
            "",
            old,
            "",
            "**Revised proposed answer**",
            "",
            new,
            "",
            "**Required proposition evidence**",
            "",
        ])
        for item in ev["requirements"]:
            lines.append(f"- {item['proposition_id']}: {item['evidence']}")
        lines.extend([
            "",
            "Re-adjudication: [ ] Adequate  [ ] Inadequate  [ ] Uncertain",
            "",
            "Reason / unresolved issue: __________________________________",
            "",
        ])
    if sum(ev["review_round_1_repair"] is not None for ev in evidence) != 10:
        raise RuntimeError("Expected ten revised examples")
    return "\n".join(lines)


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
    (args.output_dir / "v14a14_human_training_review.md").write_text(
        human_review_packet(b, audit["evidence"]), encoding="utf-8",
    )
    (args.output_dir / "v14a14_round2_ten_corrections.md").write_text(
        review_only_amended_ten(b, audit["evidence"]), encoding="utf-8",
    )
    (args.output_dir / "review_attestation_template.json").write_text(
        json.dumps({
            "proposition_audit_sha256": audit["manifest"]["proposition_audit_sha256"],
            "train_prompt_sha256": audit["manifest"]["revised_paired_prompt_sha256"],
            "reviewers": [
                {"reviewer_id": "REVIEWER_1_NOT_YET_ASSIGNED",
                 "status": "pending", "approved_row_ids": [],
                 "contradictions_found_unresolved": True},
                {"reviewer_id": "REVIEWER_2_NOT_YET_ASSIGNED",
                 "status": "pending", "approved_row_ids": [],
                 "contradictions_found_unresolved": True},
            ],
            "resolved_disagreements": False,
            "round_1_original_adequate": 70,
            "round_1_original_inadequate": 10,
            "round_1_revised_case_numbers_pending_recheck": list(REVIEW_ROUND_1_INADEQUATE_CASES),
            "round_3_revised_case_numbers_pending_recheck": sorted(x["case"] for x in ROUND_3_TARGETED_REWRITES.values()),
            "revised_10_independent_review_pending": True,
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(audit["manifest"], ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["manifest"], sort_keys=True))


if __name__ == "__main__":
    main()
