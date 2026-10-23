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
- Revised 80: 4b3afb700aa1cd7f880010949860a33aa127842db2148b2b1d6060accb70e3a9
- 80-row proposition evidence: 53f151386d07411d7b8ee3ab4aecddc35e067e7db965505a98da4cd9f3602820

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