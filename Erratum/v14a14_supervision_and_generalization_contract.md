# v14a14: expanded supervision, frozen holdout and controlled EOS diagnostic

Status: CPU dataset and hash preflight verified. **No training permitted until independent factual review.**

## Live-source implementation
Derived from main commit 00528e34f294538d84b047892f991c2e1c146075 and stacked on v14a13 PR #32. Reuses the verified v14a4 data generator, v14a10 EOS training runner and v14a13 proposition schema. No tokenizer or architecture modifications.

## Paired training intervention
80 matched prompt IDs in both original_80 and causal_80. The only training difference within this matched comparison is chosen-answer content. The four stronger families have 8 examples each; backprop_stability, learning_rate_comparison and false_premise_correction have 16 each. Total 80, all prompt strings unique; causal answers contain 62 distinct full strings.

Verified source limitation: v14a4 has at most 12 unique non-collision questions per family and only 4 unique false-premise questions. Therefore 60 prompts are unchanged v14a4 source prompts; 20 are explicitly authored new questions that match existing family cues. On those 20 questions the control uses cue-matched legacy-family answers, not historical exact prompt-answer examples. Do not claim this two-arm experiment compares historical 28 vs new 80 samples. It is an 80-versus-80 matched *target-quality* ablation. The 20 added control answers are a confound to disclose. The historical round-robin sampler does not allocate extra update slots to weak families; those families gain distinct examples, not a higher update frequency.

SHA-256 pinned input fingerprints:
- Paired prompts: b94c802089c466d3fc64f1b0ba0274d4fa032c44da3c8fc4a9b5a56a41422627
- Original 80: 09341553fb2b689c244e52ec47c6cdc736c964ca68d8ea61843cf5eb7f92745d
- Revised 80: 80d7716b29d06469d7bf4dc3ef91cbb4a0ff69490d93bb9845888c1d21451ab3
- 80-row proposition evidence: 9682546efc3b561845a71d858455434d7f8389081221a6e38dfb123aa7319899

Each revised target records verbatim evidence for its family's required causal propositions plus a cue-specific application. This is a source-traceability audit, **not** an automated truth or contradiction check. All 80 must receive substantive independent human adjudication before any GPU run.

## Fresh holdout
28 separately authored scenario prompts (four/family), never supplied to optimizer training. Includes numeric below/above norm thresholds, false premise correction, causal timing, comparison with learning-rate adjustment and an overfitting distinction. Training prompts and holdout questions have no exact overlap.

- Holdout prompt fingerprint: cb1b0a03e139039fc1a4e0301f6e62f1a6b298e9920f7e9a944d7745a5c4a589
- Holdout rows and authored references: 9e1b57a09804cea5c3d4ee6fb505c68c220c6925868939e19814b44d3ee81220

This is model-unseen and separated from train data, but not an external independent-human-blind evaluation. Preserve the frozen panel and do not revise it after inspecting outputs.

## Mandatory review and experiment gate
A CPU-only workflow exports v14a14_human_training_review.md, containing 80 prompts, complete proposed answers and every required proposition. Two genuine independent reviewers must judge causal mechanism, operation order, question completeness, false-premise rejection, factual contradictions and unsupported guarantees. A review attestation requires two distinct reviewer IDs, all 80 approved IDs per reviewer, resolved disagreements, and the exact proposition/prompt digests. The generated review_attestation_template.json remains PENDING, never auto-approved.

CPU preflight: python -m Erratum.v14a14_controlled_eos_train --preflight-only

Only after substantive review, invoke the CUDA runner with a reviewed attestation and --verify-parent-sha256. It verifies fixed IDs, parent and tokenizer/special-token contracts, then calls the unmodified v14a10 40-update explicit-EOS routine on both arms. Same seed 928, batch rows 4, sequence max 768, LR peak 2e-5, gradient clipping 0.5, one EOS label per answer and matched batch ID sequence. Fresh optimizer/model load per arm. No checkpoints are saved or promoted.

## Precommitted decision protocol
- Primary: causal arm must obtain at least 6 additional strictly adequate outputs out of 28, with at least one extra adequate output in each of backprop_stability, learning_rate_comparison and false_premise_correction.
- False-premise correction: at least two of four strictly adequate.
- Repetition: causal arm <=2/28 high-trigram-repetition outputs and no more than control.
- Mean teacher-forced EOS/EOT boundary probability >=0.80; do not confuse this with natural stopping.
- Report independently: blind human factual adequacy, per-family scores, material contradiction count, reference NLL/top1, own-target NLL/top1, repetition, EOS, sampled training exposure and prompt/answer token counts.
- A single-seed 40-update diagnostic cannot on its own justify promotion; the runner NEVER saves weights even if all criteria pass.

## Independent-source checks
PyTorch implementation and documentation:
https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/clip_grad.py
https://docs.pytorch.org/docs/main/generated/torch.nn.utils.clip_grad_norm_.html
https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html

Research stop condition: genuine human adjudication of all 80 target answers is outstanding, so GPU training remains blocked.

## First independent-of-author human adjudication — 23 October 2026

A user reviewer assessed the 80 proposed intervention answers under the strict factual-adequacy standard:

- 70 adequate; 10 inadequate; 0 uncertain.
- Failing cases 09, 14, 15: requests for first-principles explanation need an explicit causal connection from exploding gradients to optimizer parameter changes.
- Case 24: requested concept name was missing.
- Cases 33, 35, 36 and 44: instructions explicitly said to lead with the relevant concept; original answer led with generic backpropagation wording.
- Case 45: no direct explanation that clipping cannot undo errors that happened during backpropagation.
- Case 47: no accumulation-specific treatment of clipping the accumulated gradient before the optimizer step.

The first-pass review is human feedback, **not** two independent human approvals. Only those ten targeted answers were amended. The dataset builder programmatically reconstructs and verifies the exact prior causal-arm SHA-256 (4b3afb700aa1cd7f880010949860a33aa127842db2148b2b1d6060accb70e3a9) and asserts that exactly the ten rejected IDs changed. All **70 accepted answer texts remain byte-for-byte unchanged**; the unchanged baseline and frozen holdout are preserved. The 10 revised answers are now **PENDING RE-ADJUDICATION**, and a second independent review is required for training.

A focused second-pass report is generated as v14a14_round2_ten_corrections.md. It lists each rejected answer, the corrected target, original prompt, original reviewer rationale, and proposition evidence for rechecking.

Updated SHA-256:
- Revised-arm 80-row records: 80d7716b29d06469d7bf4dc3ef91cbb4a0ff69490d93bb9845888c1d21451ab3
- Proposition audit with human-review metadata: 9682546efc3b561845a71d858455434d7f8389081221a6e38dfb123aa7319899

The training preflight pins both new digests and retains the exact original control, 80-prompt sequence, and 28-case unseen holdout fingerprints. The approval attestation template remains unsigned. **No CUDA training or checkpoint promotion has been authorized or performed.**


## Second-pass adjudication and prompt-sensitive rewrites — 23 October 2026

The human reviewer re-adjudicated the ten round-one repairs, recording **9 adequate, 1 inadequate (Case 15)**, with **0 uncertain**. The reviewer also noted that previously adequate Cases 33, 35, 36, 44, 45 and 47 reused too much family-level boilerplate and requested better adaptation to each prompt. That critique is a *supervision-diversity concern*, not a retroactive factual failure judgment.

Exactly **seven** targeted current answers were replaced:
- **15:** explicit first-principles gradient-descent update relationship Δθ = −η∇θL, explaining why a large gradient can produce a large parameter step at fixed learning rate.
- **33:** lead with clipping and emphasize rescaling an already-computed backward signal.
- **35:** emphasize spike creation during backward computation and conditional norm clipping before optimizer use.
- **36:** emphasize the computed-gradient/optimizer-input boundary and unchanged subthreshold gradients.
- **44:** center stabilization of the subsequent update without claiming backward-pass numerical guarantees.
- **45:** center the limitation that clipping cannot retroactively repair numerical problems during backward computation.
- **47:** center gradient accumulation across microbatches, then clipping accumulated gradients before the optimizer step.

Rewrites for six cases previously marked adequate require **new review of the changed texts**. Case 15's new revision also requires re-adjudication. Earlier review approvals apply only to the previously submitted answer strings. All seven remain marked **PENDING_RE_ADJUDICATION**.

Strong source-integrity requirements:
1. Prior round-one 80-answer data digest remains independently reconstructible and unchanged: 4b3afb700aa1cd7f880010949860a33aa127842db2148b2b1d6060accb70e3a9. Exactly ten first-round inadequate rows were changed in round two; all 70 first-round accepted answer strings stayed byte-for-byte unchanged.
2. Prior round-two submitted data digest remains independently reproducible: 80d7716b29d06469d7bf4dc3ef91cbb4a0ff69490d93bb9845888c1d21451ab3. Its frozen proposition-audit digest remains 9682546efc3b561845a71d858455434d7f8389081221a6e38dfb123aa7319899.
3. **Active round-three** revised-arm data digest is **70796738b3023c243caf5bc678de12f826086008c8a2d6c91d80f91465dfbd7c**, with active proposition-audit digest **c0c8f20aa5dc787931e165475b090c1f0223d4cb3dc08fa4cc7c67660680a6ab**.
4. The original control, all 80 prompt texts, and all 28 blind evaluation references remain unchanged.

The third-pass targeted review packet is **v14a14_round3_seven_rewrites.md**. The round-two review packet remains an archival snapshot showing the exact responses judged (not the currently rewritten answers). The export code, pinned CUDA preflight and all 10 CPU tests passed. **No actual CUDA training or checkpoint promotion has occurred.** Two genuine independent reviewers and final adjudication remain necessary before training is enabled.

## Human review paused following the Case 35 final adjustment — 24 October 2026

The third human round adjudicated Cases 15, 33, 35, 36, 44, 45, 47: **six adequate, one inadequate (Case 35)**, with no uncertainty and no identified material factual contradictions.

Case 35 was incomplete because its prompt called for the origin of a backpropagated spike, while the answer had focused primarily on the clipping intervention. The latest authored correction explicitly explains chain-rule multiplication of local derivatives/Jacobians through layers or unrolled time steps, noting amplification is conditional, and explains global-norm clipping after backpropagation and before the optimizer update.

The builder verifies the exact frozen third-review training-data digest `70796738b3023c243caf5bc678de12f826086008c8a2d6c91d80f91465dfbd7c` and evidence digest `c0c8f20aa5dc787931e165475b090c1f0223d4cb3dc08fa4cc7c67660680a6ab` before modifying only the Case 35 target. It preserves all prior human scores attached to the exact reviewed answer texts and does not mark the new Case 35 text human-approved.

Current fourth-pass candidate SHA-256: `08f95290f2a49d0f8bd01fdd45389c9a3dd53639a6c6120b7938878b726423d6`; current proposition evidence SHA-256: `825fee6e29fbbb2fa57539f0d0afb4bea72bb3ab0d136884e048cc437e961821`. All 80 prompts, original control, and the 28 frozen holdout examples retain the same fingerprints.

**Human review is paused at the user's request.** The current corpus contains 79 answers accepted by the first human reviewer in the exact current versions, and one author-corrected answer (Case 35) not yet re-adjudicated. A second genuinely independent reviewer has not signed off. CPU tests and the authoring provenance are not substitutes for this missing review. The GPU training gate and checkpoint-promotion gate remain closed.
