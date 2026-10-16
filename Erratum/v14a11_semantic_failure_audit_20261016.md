# v14a11 — Semantic Failure Audit (v14a10 u40)

**Date:** 2026-10-16  
**Scope:** All 32 held-out `gradient_clipping` generations in **both** `no_eos` and `explicit_eos` arms at update 40. This is a **read-only artifact audit**: no model training, checkpoint promotion, or changes to the v14a10 experiment.  
**Interpretation:** This is one researcher's qualitative classification of generated text, not an independently adjudicated human panel. Do not equate the qualitative 6/32 with a predeclared benchmark.

## Evidence and provenance

- S3 base: `s3://ahhwttai5o/Ardor/training/runs/v14a10_eos_supervision_ablation_u40/`
- Decision (S3 read): [workflow 38935047513](https://github.com/kys928/Ardor/actions/runs/38935047513).
- `no_eos/gradient_clipping_outputs_u0040.json`: [workflow 38941799710](https://github.com/kys928/Ardor/actions/runs/38941799710).
- `explicit_eos/gradient_clipping_outputs_u0040.json`: [workflow 38935165596](https://github.com/kys928/Ardor/actions/runs/38935165596).
- `no_eos/generalization_diagnostic_u0040.json`: [workflow 38941815205](https://github.com/kys928/Ardor/actions/runs/38941815205).
- `explicit_eos/generalization_diagnostic_u0040.json`: [workflow 38942138278](https://github.com/kys928/Ardor/actions/runs/38942138278).
- Verified implementation: [EOS experiment](https://github.com/kys928/Ardor/blob/main/Erratum/ardor_v14a10_eos_supervision_ablation.py), [v14a9 evaluator](https://github.com/kys928/Ardor/blob/main/Erratum/ardor_v14a9_holdout_generalization_diagnostic.py), [data builder](https://github.com/kys928/Ardor/blob/main/Erratum/build_v14a4_family_data.py), and [v14a5 normalization](https://github.com/kys928/Ardor/blob/main/Erratum/ardor_v14a5_signal_first_trainer.py).

## Frozen v14a10 result

| u40 metric | No EOS | Explicit EOS |
|---|---:|---:|
| Heldout continuation NLL | 1.257918 | 1.256342 |
| Heldout token top-1 | 77.934% | 78.145% |
| Mean P(EOS or EOT) after *teacher-forced reference answer* | 0.00003536% | 92.2003% |
| High trigram repetition | 26/32 | 1/32 |
| Mean trigram repetition | 0.622301 | 0.079439 |
| First-sentence keyword-rubric pass | 4/32 | 6/32 |
| Full-generation automated rubric pass | 3/32 | 13/32 |

All three predeclared gates in v14a10 passed, as the decision JSON reports. The termination probability is **not** a measurement of autonomous stop-event frequency. One route, one seed, no checkpoint saved or promoted.

## Independent qualitative review of explicit-EOS outputs

**Evaluation rule:** A minimally adequate response must address the actual requested mechanism accurately and at least enough to distinguish it from adjacent concepts for the given prompt. A named keyword alone, a meta instruction about answers, or an uncorrected false premise is insufficient. Classification is mutually exclusive, one *primary* diagnosis per case; several examples have multiple defects.

- **6/32** minimally adequate (18.75%; provisional human-style judgement, not a reproducible model metric).
- **11/32** materially incorrect / contradictory (34.38%).
- **14/32** incomplete or semantically evasive (43.75%).
- **1/32** residual conspicuous looping (3.13%).
- Of **13** full-answer automatic rubric passes, only **5** meet this qualitative minimum; **8** are qualitative false positives. One minimally adequate response fails the automatic rubric. These precision/recall labels depend on this single reviewer's stated threshold.

### By semantic family

| Family | Cases | Min. adequate | Incorrect | Incomplete | Loop | Automated pass |
|---|---:|---:|---:|---:|---:|---:|
| `definition_mechanism` | 5 | 0 | 4 | 1 | 0 | 4 |
| `exploding_gradients` | 5 | 1 | 2 | 2 | 0 | 0 |
| `norm_threshold` | 5 | 0 | 2 | 3 | 0 | 1 |
| `optimizer_stability` | 5 | 3 | 1 | 1 | 0 | 5 |
| `backprop_stability` | 4 | 2 | 1 | 1 | 0 | 2 |
| `learning_rate_comparison` | 4 | 0 | 1 | 2 | 1 | 1 |
| `false_premise_correction` | 4 | 0 | 0 | 4 | 0 | 0 |

### By question class

- `normal`: 25 cases; 6 minimally adequate.
- `collision_comparison`: 3 cases; 0 minimally adequate.
- `false_premise_correction`: 4 cases; 0 minimally adequate.

### Case-by-case annotations

All IDs below are `v14a4_holdout_gradient_clipping_NNN`. The full original prompts and generated strings remain in the S3 evidence above.

| ID suffix | Family | Question class | Automated | Qualitative | Reason |
|---|---|---|---|---|---|
| 000 | `definition_mechanism` | normal | pass | incorrect | Says bounded signal is 'not bounded' before the optimizer; factual contradiction. |
| 001 | `exploding_gradients` | normal | fail | incorrect | Garbled 'Do not help training stability' clause misidentifies the problem. |
| 002 | `norm_threshold` | normal | fail | incomplete | Names clipping but omits maximum-norm threshold behavior. |
| 003 | `optimizer_stability` | collision_comparison | pass | incomplete | Names clipping but does not distinguish it causally from overfitting. |
| 004 | `backprop_stability` | normal | pass | minimally adequate | Identifies clipping and reduction in extreme-update instability. |
| 005 | `learning_rate_comparison` | normal | pass | incorrect | Falsely says clipping limits the learning rate, rather than the gradient. |
| 006 | `false_premise_correction` | false_premise_correction | fail | incomplete | Does not explicitly reject the false claim that exploding gradients help. |
| 007 | `definition_mechanism` | normal | fail | incomplete | States bounded gradient without explaining the actual clipping operation. |
| 008 | `exploding_gradients` | normal | fail | incomplete | Names clipping and stability but not why exploding gradients harm updates. |
| 009 | `norm_threshold` | normal | fail | incomplete | Describes limiting the gradient, but misses conditional norm rescaling. |
| 010 | `optimizer_stability` | normal | pass | minimally adequate | Names clipping and correctly links it to preventing extreme update jumps. |
| 011 | `backprop_stability` | collision_comparison | fail | incomplete | Fails the requested comparison against checkpointing; unclear mechanism. |
| 012 | `learning_rate_comparison` | normal | fail | incomplete | Only says there are two controls, not how clipping differs from LR. |
| 013 | `false_premise_correction` | false_premise_correction | fail | incomplete | Names clipping without correcting the false premise. |
| 014 | `definition_mechanism` | normal | pass | incorrect | Circular claim: bounded signal 'instead of a bounded signal'. |
| 015 | `exploding_gradients` | normal | fail | minimally adequate | Connects gradient clipping with reduction of extreme-update instability. |
| 016 | `norm_threshold` | normal | fail | incorrect | Reverses causal direction: magnitude 'prevented from gradient clipping'. |
| 017 | `optimizer_stability` | normal | pass | incorrect | Garbled 'Do not help training stability' statement contradicts purpose. |
| 018 | `backprop_stability` | normal | fail | incorrect | Explicitly denies controlling the excessive backward signal. |
| 019 | `learning_rate_comparison` | collision_comparison | fail | incomplete | Says two mechanisms are distinct but supplies neither mechanism. |
| 020 | `false_premise_correction` | false_premise_correction | fail | incomplete | Does not reject the false premise; repeats vague mechanism framing. |
| 021 | `definition_mechanism` | normal | pass | incorrect | Says large gradient magnitude is 'prevented from clipping'. |
| 022 | `exploding_gradients` | normal | fail | incorrect | Malformed explanation again says 'Do not help training stability'. |
| 023 | `norm_threshold` | normal | pass | incorrect | Claims norm is reduced before crossing the threshold; mechanism reversed. |
| 024 | `optimizer_stability` | normal | pass | minimally adequate | Correctly identifies clipping as preventing one extreme update. |
| 025 | `backprop_stability` | normal | pass | minimally adequate | States oversized backward gradient is bounded before optimizer step. |
| 026 | `learning_rate_comparison` | normal | fail | residual repetition | Repeats the same meta statement despite explicit EOS training. |
| 027 | `false_premise_correction` | false_premise_correction | fail | incomplete | Names clipping but fails to reject unbounded-gradients safety claim. |
| 028 | `definition_mechanism` | normal | pass | incorrect | Again contrasts bounded gradient with itself (circular contradiction). |
| 029 | `exploding_gradients` | normal | fail | incomplete | Names gradient clipping but offers no mechanism or stability rationale. |
| 030 | `norm_threshold` | normal | fail | incomplete | Repeats the label and omits norm-threshold operation. |
| 031 | `optimizer_stability` | normal | pass | minimally adequate | States gradient signal is bounded before optimizer update; concise. |

## Root-cause findings grounded in current source

1. **EOS changes answer length far more than opening semantics.** 20/32 pairs have exactly the same first punctuation-terminated sentence in EOS and non-EOS arms; 15/32 EOS generations are complete character-for-character prefixes of the corresponding non-EOS generation. Mean generated character length drops from ~410.9 to ~192.2. These are string comparisons, not proof of the model's actual stopping token.
2. **An authored supervision-template issue is directly visible.** In `Erratum/build_v14a4_family_data.py`, `DETAIL_VARIANTS` contains generic process/meta-answer phrases. The `_answer` builder appends these phrases to otherwise substantive chosen-answer templates. **28/32** explicit-EOS generations contain at least one exact full sentence from `DETAIL_VARIANTS`. This strongly suggests the model is reproducing low-information target prose. It does **not** prove all individual generation errors are caused by that one source.
3. **The gradient-clipping target rows did not receive the v14a5 repair.** `ardor_v14a5_signal_first_trainer.py::normalize_behavior_rows` only repairs `dropout/overfitting_relationship`. `ardor_v14a6_learning_dynamics_probe.py::_repaired_behavior_rows` delegates to it; v14a10 uses these rows. Thus gradient-clipping chosen text retains the original v14a4 answer-template structure.
4. **Malformed clause concatenation is a plausible supervision hazard.** For false-premise cases, the `FACTS` tuple contains the predicate fragment 'do not help training stability; they are the problem ...'. The `_answer` branch sometimes inserts a capitalized predicate fragment as a standalone sentence. The generated malformed phrase 'Do not help training stability; they are the problem ...' resembles that source construction. Inspect the exact selected 28 chosen rows before asserting a token-for-token target match.
5. **The evaluation gate is susceptible to factual false positives.** `Erratum/v14a4_diagnostics.py::family_sem_check` uses keyword lists and surface correction markers, not logical entailment. Example `000` passes the full rubric after stating the bounded signal is 'not bounded before' the optimizer; `005` passes despite conflating gradient clipping with limiting the learning rate.
6. **The weakest heldout teacher-forced family is learning-rate comparison.** The EOS-arm by-family reference NLL for `learning_rate_comparison` is **2.33356**, versus **0.87163–1.41342** for all other families. NLL does not establish semantic correctness, but this is consistent with prioritizing the comparison supervision.

## v14a11 decision

**Preserve v14a10 as a positive termination-control result. Do not promote a checkpoint.** The audited failure is primarily content/signal quality, not a reason to undo EOS supervision. No evidence from this audit warrants optimizer, attention, decoding, or architecture changes.

## Next experimental design (proposal only; NOT executed)

**v14a12 candidate: same EOS supervision, improved gradient-clipping target text.** Keep the same SHA-verified v14a2 parent, tokenizer-v9 special IDs, 28 training IDs, 32 untouched holdout IDs, deterministic sample order, optimizer, learning rate, 40 updates, and EOS appended in **both** experimental arms. Compare:

- **Control:** original v14a10 `explicit_eos` chosen answer text + EOS.
- **Intervention:** manually audited corrections to those same 28 chosen answers + the identical EOS label, emphasizing meaningful causal explanations: norm threshold before update, conditional clipping vs learning-rate scaling, correct negation of harmful false premises, and clear contrast against checkpoint/overfitting; remove meta-template filler.

The experiment compares **supervision text quality**, not EOS presence. Answer edits alter target token counts and loss normalization, so declare these as unavoidable measured differences; log per-row target lengths and token-normalized NLL. Keep the semantic holdout sealed from selection and training, and use an explicit factual-check rubric plus blinded independent review rather than keyword matching alone.

**Proposed exploratory checks:** EOS/EOT boundary mean ≥ 0.80; high repetition ≤ 2/32; heldout reference NLL no worse by >0.10 relative to the EOS control; strict semantic adequacy improves at least 6/32 over its reproduced control and improves the `norm_threshold`, `learning_rate_comparison`, and `false_premise_correction` families. These thresholds are recommendations for a **new predeclared experiment**, not results. Reproduce across multiple seeds and routes before considering promotion.
