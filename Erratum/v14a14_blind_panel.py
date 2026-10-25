#!/usr/bin/env python3
"""v14a14 independently drafted, frozen evaluation-panel source (CPU-only).

28 new scenario questions, four per semantic family; none are drawn from the
v14a4 TRAIN_CUES/HOLDOUT_CUES or a training prompt-style list. References were
authored separately from training answers. The panel is *model-unseen*, not
independent-human-blind. Freeze its SHA BEFORE any training launch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from Erratum.v14a14_signal_dataset import (
    FAMILY_ORDER, build_pairs, digest, normalize,
)
from Erratum.v14a13_proposition_evaluator import PROPOSITIONS, applicable

# Manually authored. A question requires the causal mechanism, not keywords.
# Questions intentionally use situations and negations absent from train cues.
PANELS = {
"definition_mechanism": [
("A log shows the backward computation finished, followed by a norm adjustment before AdamW stepped. Identify that operation and what it adjusted.",
 "That operation is gradient clipping. Once backward computation has produced the gradient, clipping reduces excessive gradient magnitude before the optimizer updates weights."),
("A training graph contains a gradient-processing stage between computing derivatives and updating parameters. What stage could constrain exceptionally large vectors?",
 "Gradient clipping operates in that intermediate stage. It scales down overly large computed gradients before the optimizer uses them for the parameter update."),
("An engineer says the model weights were clamped because a derivative was huge. Explain what gradient clipping actually changes at this point.",
 "Norm-based gradient clipping changes the computed gradient, not the weights directly. It acts after backpropagation and before the optimizer step by reducing a gradient whose norm exceeds the cap."),
("A learner confuses gradient clipping with overfitting. Compare the two given that the symptom is an oversized derivative before a parameter step.",
 "Gradient clipping is the operation that reduces the oversized computed gradient before the optimizer update. Overfitting concerns poor generalization to unseen data; it is not a gradient-limiting operation."),
],
"exploding_gradients": [
("In a recurrent network, derivative magnitudes grow across backward steps and the next optimizer step becomes erratic. State the danger and mitigation.",
 "Growing backward derivatives can produce exploding gradients that destabilize updates. Clipping reduces overly large computed gradients before passing them to the optimizer."),
("A training loss suddenly diverges after an unusually large backward signal. Why might imposing a gradient bound help without proving the cause?",
 "An exploding gradient can increase update instability. A gradient-norm bound clips exceptional computed gradients before the optimizer uses them, potentially reducing the risk without guaranteeing it caused the divergence."),
("A colleague thinks a very large gradient means the optimization signal must be especially reliable. Explain the failure mode.",
 "Very large gradients can be unstable update-driving signals rather than inherently reliable ones. Gradient clipping reduces their magnitude before optimizer updates are applied."),
("During fine-tuning, one batch produces a derivative norm far above typical values. Explain the risk of using it unchanged.",
 "An extremely large computed gradient can provoke an unstable subsequent parameter update. Gradient clipping reduces that oversized signal before the optimizer step."),
],
"norm_threshold": [
("Suppose a gradient norm is 0.3 and the configured maximum is 0.5. For global-norm clipping, should the gradient be enlarged to 0.5? Explain.",
 "No. Since 0.3 is below the 0.5 maximum, norm clipping leaves the gradient unchanged. The optimizer uses this unchanged gradient; the bound is a ceiling, not a target."),
("A computed global norm is 2.0 while max_norm is 0.5. Describe which operation should occur before the optimizer step and its approximate scale.",
 "Global-norm clipping measures the norm and scales the gradient down because 2.0 exceeds 0.5. Ignoring the small numerical epsilon, the multiplier is 0.5 / 2.0 = 0.25; the optimizer then receives the reduced gradient."),
("A student proposes scaling every gradient to exactly max_norm. Is that the correct norm-clipping algorithm?",
 "No. Norm clipping compares the current norm with the maximum, scales down only when that maximum is exceeded, and otherwise keeps the gradient unchanged before optimization."),
("Why is max_norm not a request to normalize every gradient vector to the same length?",
 "It is an upper bound. Norm clipping compares the computed norm with the threshold, shrinks excessive gradients but does not amplify smaller ones, and the optimizer then uses the resulting gradient."),
],
"optimizer_stability": [
("A single minibatch yields a massive derivative spike. Why can gradient clipping help the next AdamW step, and what can it not guarantee?",
 "The spike can contribute to an unstable optimizer update. Gradient clipping bounds the computed gradient before AdamW uses it, mitigating that source of instability; it cannot guarantee every parameter update is safe."),
("An engineer says clipping can prevent any training divergence whatsoever. Assess that claim using update mechanics.",
 "The claim is too strong. Exceptionally large gradients can cause destabilizing updates, and clipping reduces such computed gradients before the optimizer step, lowering that risk without preventing all divergence."),
("A rare backward-pass outlier threatens to dominate the next step. What precisely does clipping constrain?",
 "A gradient spike may drive an unstable update. Clipping constrains the computed gradient magnitude before the optimizer uses it, reducing exposure to that outlier rather than replacing the optimizer."),
("Someone reports that gradient clipping is applied only after the optimizer has already updated the weights. Explain why this timing defeats its intended protection.",
 "An excessive computed gradient threatens the next optimizer update. Clipping must bound the gradient before the optimizer step to mitigate this danger; acting afterward would not protect that completed step."),
],
"backprop_stability": [
("A teammate claims norm clipping runs before the backward pass and prevents large gradients from ever being computed. Correct the sequence.",
 "Backpropagation computes parameter gradients first. Norm clipping can then reduce an excessively large computed gradient before the optimizer step; this mitigates update risk rather than preventing gradient computation."),
("Training crashes inside the backward computation, before gradient tensors are available. Would ordinary post-backward clipping necessarily repair that failure?",
 "No. Backpropagation must produce gradients before ordinary gradient clipping acts. Clipping is applied to computed gradients ahead of the optimizer step, reducing update risk but not guaranteeing that a failed backward pass is repaired."),
("Describe the handoff between backpropagation, norm clipping, and AdamW when the derivative magnitude becomes excessive.",
 "Backpropagation produces the gradient. Norm clipping then reduces the computed gradient if it is too large, and AdamW uses the adjusted gradient to perform a less spike-prone parameter update."),
("A new researcher says clipping makes the chain rule numerically safe throughout reverse-mode differentiation. What does it actually safeguard?",
 "Backpropagation first generates the parameter derivatives. Gradient clipping subsequently bounds excessive computed gradients prior to optimization, helping protect the following update, not every internal numerical operation of differentiation."),
],
"learning_rate_comparison": [
("A model experiences one isolated gradient spike. Compare a conditional norm cap with permanently halving AdamW's learning rate.",
 "Clipping rescales the computed gradient only when it exceeds a bound. Halving AdamW's learning rate changes the optimizer update scale more generally, including normal steps; clipping does not itself change that learning-rate setting."),
("An intern claims max_norm=1.0 is interchangeable with optimizer learning_rate=1.0. Explain the type of quantity each controls.",
 "Max-norm clipping tests and potentially scales computed gradient magnitude; it acts conditionally. The learning rate controls the update scale within the optimizer. They are different parameters and are not interchangeable."),
("A clipping threshold is never exceeded across several updates. How can learning-rate adjustments still matter?",
 "Norm clipping leaves those subthreshold computed gradients unchanged. Changing the learning rate still alters optimizer update scaling more generally, so the two controls differ even when clipping does nothing."),
("In AdamW, if a computed gradient is scaled because its norm is huge, has the optimizer's configured learning-rate hyperparameter necessarily changed?",
 "No. Gradient clipping conditionally scales down the computed gradient before AdamW uses it; AdamW's learning rate separately sets the update scale and is not changed by clipping."),
],
"false_premise_correction": [
("A classmate argues that allowing derivatives to explode protects the next weight update from instability. Correct them.",
 "That is false. Exploding gradients can destabilize parameter updates. Clipping is the protection: it reduces excessively large computed gradients before the optimizer step."),
("A lab note says unbounded gradients are an optimizer safety feature because large values force faster convergence. What is wrong?",
 "Unbounded exploding gradients do not guarantee faster or safer training; they may destabilize updates. Clipping constrains excessive gradient magnitude before optimization and can mitigate that risk."),
("An assistant asserts that the training fix for exploding gradients is to amplify them further. Correct this causal direction.",
 "No. Excessively large gradients can cause unstable optimizer updates, and further amplification is not the protective remedy. Gradient clipping instead reduces such computed gradients before the optimizer step."),
("A student writes, 'If gradients explode, clipping makes them even bigger to stabilize learning.' Give a correction-first answer.",
 "That statement reverses the mechanism. Exploding gradients threaten update stability; norm clipping reduces oversized computed gradients before the optimizer uses them, rather than amplifying them."),
],
}

def build_holdout() -> tuple[list[dict], dict]:
    control, revised, train_meta = build_pairs()
    seen = {normalize(r["prompt"]) for r in revised}
    rows = []
    for family in FAMILY_ORDER:
        if len(PANELS[family]) != 4:
            raise RuntimeError("Exactly four independently drafted holdout prompts per family")
        for idx, (question, reference) in enumerate(PANELS[family], 1):
            prompt = question.rstrip() + "\n-"
            key = normalize(prompt)
            if key in seen:
                raise RuntimeError("Exact train/holdout prompt overlap")
            seen.add(key)
            rows.append({
                "id": f"v14a14_blind_{family}_{idx:02d}",
                "route": "gradient_clipping",
                "semantic_family": family,
                "source": "blind_eval",
                "split": "holdout",
                "question_class": "false_premise_correction" if family == "false_premise_correction" else (
                    "collision_comparison" if family == "definition_mechanism" and idx == 4 else "normal"
                ),
                "collision_route": "overfitting" if family == "definition_mechanism" and idx == 4 else None,
                "prompt_style": "independently_authored_scenario",
                "answer_style": "independently_authored_reference",
                "prompt": prompt,
                "chosen": reference,
                "required_propositions": list(applicable({
                    "family": family,
                    "question_class": "collision_comparison" if family == "definition_mechanism" and idx == 4 else (
                        "false_premise_correction" if family == "false_premise_correction" else "normal"
                    ),
                })),
            })
    if len(rows) != 28 or len(seen) != 108:
        raise RuntimeError("Holdout failed unique prompt count")
    if set(r["id"] for r in rows) & set(r["id"] for r in revised):
        raise RuntimeError("ID overlap between train and holdout")
    # Prefix uniqueness is an additional guard against old cue/prompt reuse.
    all_prompts = [normalize(r["prompt"]) for r in revised]
    for h in rows:
        h_tokens = normalize(h["prompt"]).split()
        if any(len(h_tokens) >= 7 and " ".join(h_tokens[:7]) in p for p in all_prompts):
            raise RuntimeError("Suspicious holdout prefix overlap")
    fingerprint = {
        "version": "v14a14-blind-panel-1",
        "status": "FROZEN_EVALUATION_CANDIDATE_NO_MODEL_EXPOSURE",
        "holdout_n": len(rows),
        "holdout_per_family": dict(Counter(r["semantic_family"] for r in rows)),
        "train_prompt_count": len(revised),
        "train_prompt_sha256": train_meta["manifest"]["revised_paired_prompt_sha256"],
        "holdout_prompt_sha256": digest([(r["id"], r["prompt"]) for r in rows]),
        "holdout_rows_and_references_sha256": digest(rows),
        "cross_split_overlap": 0,
        "reference_authorship": "separately authored from training targets",
        "not_independent_human_validation": True,
        "not_seen_by_model_or_optimizer": True,
        "rubric_version": "Erratum/v14a13_proposition_evaluator.py on frozen v14a14 branch",
        "warning": "Reference NLL is not equivalent to factual adequacy",
    }
    return rows, fingerprint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("v14a14_signal_output"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, stamp = build_holdout()
    (args.output_dir / "blind_holdout_28.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    (args.output_dir / "blind_panel_manifest.json").write_text(
        json.dumps(stamp, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stamp, sort_keys=True))


if __name__ == "__main__":
    main()
