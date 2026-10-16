#!/usr/bin/env python3
"""v14a9 read-only generalization diagnostic over one exact v14a8 replay.

Replays only the v14a8 lr_2e-5 arm using the original training implementation.
Compares the canonical parent (u0), intermediate (u20), and u40 without ever
saving/promoting model weights. Reports four separate heldout metrics:
reference-answer continuation NLL, first-sentence semantic-rubric success,
reference-answer-boundary EOS/EOT probability, and generated repetition.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from statistics import mean, median
from typing import Any, Sequence

import torch
from tokenizers import Tokenizer

from Erratum import ardor_v14a4_family_balanced_trainer as base
from Erratum import ardor_v14a6_learning_dynamics_probe as pulse
from Erratum import ardor_v14a8_lr_ablation as exp
from Erratum.v14a4_diagnostics import clean_eval_prompt, family_sem_check

OUTPUT_DIR = Path("/workspace/Ardor/training/runs/v14a9_holdout_generalization_lr2e5_u40")
ARM = {"name": "lr_2e-5", "lr": 2e-5}
AUDIT_UPDATES = (0, 20, 40)
REPETITION_FAIL_THRESHOLD = 0.45
V14A8_EXPECTED_U40_NLL = 0.2719491542331756
V14A8_EXPECTED_U40_TOP1 = 0.9571428692530072
V14A8_EXPECTED_U40_SEMANTIC_MACRO = 0.1


def first_sentence(text: str) -> tuple[str, bool]:
    """Return the first punctuation-terminated sentence or all generated text."""
    cleaned = " ".join(str(text).split())
    hit = re.search(r"[.!?](?=\s|$)", cleaned)
    if hit is None:
        return cleaned, False
    return cleaned[:hit.end()], True


def first_sentence_rubric(row: dict[str, Any], generation: str) -> dict[str, Any]:
    """Family/route/false-premise keyword rubric, not human semantic judgment."""
    text, terminated = first_sentence(generation)
    checks = family_sem_check(row, text)
    failures = list(checks["failures"])
    return {
        "first_sentence": text,
        "has_sentence_boundary": terminated,
        "passes_keyword_semantic_rubric": bool(checks["passed"]),
        "failure_reasons": failures,
        "has_route_signal": not any(x.endswith("_missing_route_signal") for x in failures),
        "has_family_signal": not any("_missing_family_signal" in x for x in failures),
        "premise_corrected_when_applicable": not any(
            x in ("false_premise_not_corrected", "false_premise_affirmed")
            for x in failures
        ),
    }


@torch.inference_mode()
def heldout_continuation_nll(
    model, tok: Tokenizer, rows: Sequence[dict[str, Any]], device: torch.device,
    *, max_len: int, batch_rows: int,
) -> dict[str, Any]:
    """Teacher-forced chosen answer NLL; prompt masked exactly as in v14a8."""
    total_tokens = 0
    weighted_nll = 0.0
    weighted_top1 = 0.0
    family_acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    prepared = [
        {
            **dict(row),
            "prompt": clean_eval_prompt(row),
            "chosen": str(row["chosen"]),
            "source": "heldout_reference",
        }
        for row in rows
    ]
    model.eval()
    for start in range(0, len(prepared), batch_rows):
        batch = prepared[start:start + batch_rows]
        loss, stats = base.continuation_ce(
            model, tok, batch, device,
            max_len=max_len, pad_id=base.EXPECTED_SPECIAL_IDS["<pad>"],
        )
        tokens = int(stats["tokens"])
        nll = float(loss.item())
        top1 = float(stats["top1_rate"])
        if not (math.isfinite(nll) and math.isfinite(top1)):
            raise RuntimeError("Nonfinite heldout continuation measurement")
        total_tokens += tokens
        weighted_nll += nll * tokens
        weighted_top1 += top1 * tokens
        # Separate family averages are measured with family-only batches below.
    if total_tokens == 0:
        raise RuntimeError("Empty heldout continuation")
    for family in sorted({str(row["semantic_family"]) for row in prepared}):
        selected = [r for r in prepared if r["semantic_family"] == family]
        for start in range(0, len(selected), batch_rows):
            batch = selected[start:start + batch_rows]
            loss, stats = base.continuation_ce(
                model, tok, batch, device, max_len=max_len,
                pad_id=base.EXPECTED_SPECIAL_IDS["<pad>"],
            )
            n = int(stats["tokens"])
            family_acc[family][0] += n
            family_acc[family][1] += float(loss.item()) * n
            family_acc[family][2] += float(stats["top1_rate"]) * n
    return {
        "rows": len(prepared),
        "tokens": total_tokens,
        "reference_continuation_nll": weighted_nll / total_tokens,
        "reference_continuation_top1": weighted_top1 / total_tokens,
        "by_family": {
            name: {
                "tokens": int(acc[0]),
                "nll": acc[1] / acc[0],
                "top1": acc[2] / acc[0],
            }
            for name, acc in sorted(family_acc.items())
        },
    }


@torch.inference_mode()
def reference_boundary_termination_probs(
    model, tok: Tokenizer, rows: Sequence[dict[str, Any]], device: torch.device,
    *, max_len: int,
) -> dict[str, Any]:
    """P(eos), P(eot) after each complete heldout chosen answer, not at prompt."""
    eos_id = int(base.EXPECTED_SPECIAL_IDS["<eos>"])
    eot_id = int(base.EXPECTED_SPECIAL_IDS["<|eot|>"])
    probabilities = []
    model.eval()
    for row in rows:
        pids = base.encode(tok, clean_eval_prompt(row))
        cids = base.encode(tok, str(row["chosen"]))
        if not pids or not cids or len(cids) >= max_len:
            raise RuntimeError("Invalid heldout EOS probe tokenization")
        ids = pids[-max(1, max_len - len(cids)):] + cids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        eos_p, eot_p = float(probs[eos_id].item()), float(probs[eot_id].item())
        if not (math.isfinite(eos_p) and math.isfinite(eot_p)):
            raise RuntimeError("Nonfinite EOS/EOT probability")
        probabilities.append({
            "id": str(row["id"]),
            "family": str(row["semantic_family"]),
            "p_eos_after_reference": eos_p,
            "p_eot_after_reference": eot_p,
            "p_either_termination_after_reference": eos_p + eot_p,
        })
    combined = [r["p_either_termination_after_reference"] for r in probabilities]
    return {
        "rows": len(probabilities),
        "mean_p_eos_after_reference": mean(r["p_eos_after_reference"] for r in probabilities),
        "mean_p_eot_after_reference": mean(r["p_eot_after_reference"] for r in probabilities),
        "mean_p_either_termination_after_reference": mean(combined),
        "median_p_either_termination_after_reference": median(combined),
        "values": probabilities,
    }


def generation_diagnostics(
    outputs: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], history,
) -> dict[str, Any]:
    lookup = {str(row["id"]): row for row in rows}
    if len(outputs) != len(rows):
        raise RuntimeError("Mismatch in generated and heldout row counts")
    observed = set()
    table = []
    for out in outputs:
        key = str(out["id"])
        if key not in lookup or key in observed:
            raise RuntimeError("Unknown or duplicated heldout output id: " + key)
        observed.add(key)
        text = str(out["generation"])
        first = first_sentence_rubric(lookup[key], text)
        rep = float(history.repetition_rate(text))
        table.append({
            "id": key,
            "family": str(out["semantic_family"]),
            "question_class": str(out["question_class"]),
            "full_generation": text,
            "full_generation_passed_v14a8": bool(out["passed"]),
            "full_generation_failure_reasons": list(out["failures"]),
            "first_sentence": first,
            "trigram_repetition_rate": rep,
            "high_repetition": rep >= REPETITION_FAIL_THRESHOLD,
        })
    mean_rep = mean(x["trigram_repetition_rate"] for x in table)
    return {
        "rows": len(table),
        "first_sentence_keyword_semantic_success_rate": mean(
            float(x["first_sentence"]["passes_keyword_semantic_rubric"]) for x in table
        ),
        "first_sentence_with_punctuation_fraction": mean(
            float(x["first_sentence"]["has_sentence_boundary"]) for x in table
        ),
        "first_sentence_route_signal_rate": mean(
            float(x["first_sentence"]["has_route_signal"]) for x in table
        ),
        "first_sentence_family_signal_rate": mean(
            float(x["first_sentence"]["has_family_signal"]) for x in table
        ),
        "mean_trigram_repetition_rate": mean_rep,
        "high_repetition_fraction": mean(float(x["high_repetition"]) for x in table),
        "high_repetition_count": sum(int(x["high_repetition"]) for x in table),
        "full_answer_original_rubric_success_count": sum(
            int(x["full_generation_passed_v14a8"]) for x in table
        ),
        "by_family": {
            family: {
                "n": len(items),
                "first_sentence_success_rate": mean(
                    float(x["first_sentence"]["passes_keyword_semantic_rubric"]) for x in items
                ),
                "mean_repetition_rate": mean(x["trigram_repetition_rate"] for x in items),
                "high_repetition_count": sum(int(x["high_repetition"]) for x in items),
            }
            for family in sorted({x["family"] for x in table})
            for items in [[x for x in table if x["family"] == family]]
        },
        "rows_detail": table,
    }


def evaluate_separately(
    *, model, tok, target_holdout, device, history, args, arm_dir, update,
) -> dict[str, Any]:
    original = json.loads(
        (arm_dir / f"gradient_clipping_outputs_u{update:04d}.json").read_text(encoding="utf-8")
    )
    holdout_ce = heldout_continuation_nll(
        model, tok, target_holdout, device, max_len=args.max_len,
        batch_rows=args.batch_rows,
    )
    eos = reference_boundary_termination_probs(
        model, tok, target_holdout, device, max_len=args.max_len,
    )
    generated = generation_diagnostics(original, target_holdout, history)
    report = {
        "update": update,
        "holdout_reference_teacher_forcing": holdout_ce,
        "eos_eot_probability_at_reference_answer_boundary": eos,
        "first_sentence_and_generation_repetition": generated,
    }
    exp.write_json(arm_dir / f"generalization_diagnostic_u{update:04d}.json", report)
    compact = {
        "update": update,
        "holdout_reference_nll": holdout_ce["reference_continuation_nll"],
        "holdout_reference_top1": holdout_ce["reference_continuation_top1"],
        "first_sentence_keyword_rubric_success_rate": generated[
            "first_sentence_keyword_semantic_success_rate"
        ],
        "first_sentence_route_signal_rate": generated["first_sentence_route_signal_rate"],
        "first_sentence_family_signal_rate": generated["first_sentence_family_signal_rate"],
        "mean_eos_eot_p_at_reference_boundary": eos[
            "mean_p_either_termination_after_reference"
        ],
        "mean_eos_p_at_reference_boundary": eos["mean_p_eos_after_reference"],
        "mean_eot_p_at_reference_boundary": eos["mean_p_eot_after_reference"],
        "mean_trigram_repetition_rate": generated["mean_trigram_repetition_rate"],
        "high_repetition_count": generated["high_repetition_count"],
        "full_answer_rubric_success_count": generated[
            "full_answer_original_rubric_success_count"
        ],
        "holdout_rows": len(target_holdout),
    }
    print("[generalization] " + json.dumps(compact, sort_keys=True), flush=True)
    return compact


def run(args) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this exact v14a8 replay")
    if args.batch_rows != 4 or args.seed != base.SEED or args.max_len != base.MAX_LEN:
        raise ValueError("Do not alter v14a8 pulse seed, batch rows, or max sequence length")
    if args.weight_decay != 0.0 or args.warmup_updates != 20 or args.min_lr_ratio != 0.2:
        raise ValueError("Do not alter v14a8 optimizer/schedule controls")
    if args.max_gen_tokens != 72 or args.min_eval_words != 3:
        raise ValueError("Do not alter v14a8 generation/evaluation controls")

    base.validate_local_files(verify_checkpoint_sha256=args.verify_parent_sha256)
    contract = base.training_contract()
    behavior_raw, clean_holdout, family_summary = base.ensure_family_datasets(contract)
    behavior = pulse._repaired_behavior_rows(behavior_raw)
    target_rows = pulse.select_target_rows(
        behavior, seed=args.seed + 701,
        rows_per_family=pulse.TARGET_ROWS_PER_FAMILY,
    )
    holdout = [dict(r) for r in clean_holdout if r["route"] == pulse.TARGET_ROUTE]
    if len(target_rows) != 28 or len(holdout) != 32:
        raise RuntimeError("v14a8 gradient_clipping target/holdout size changed")
    if {str(r["id"]) for r in target_rows} & {str(r["id"]) for r in holdout}:
        raise RuntimeError("Heldout row ID leakage into pulse train set")
    tok = Tokenizer.from_file(str(base.TOKENIZER_PATH))
    token_contract = base._tokenizer_contract(tok)
    if not token_contract["passed"] or tok.get_vocab_size() != base.EXPECTED_VOCAB_SIZE:
        raise RuntimeError("Tokenizer contract mismatch")
    history = base._load_historical_eval_module()
    eval_args = base.configure_history_args(history, args)
    special = history.special_ids(tok, eval_args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exp.write_json(output_dir / "diagnostic_contract.json", {
        "name": "v14a9_holdout_generalization_diagnostic",
        "basis": "exact v14a8 lr_2e-5 pulse replay",
        "trainer": "Erratum/ardor_v14a8_lr_ablation.py::run_arm",
        "parent": base.canonical_parent_reference(),
        "tokenizer_v9_contract": token_contract,
        "train_row_ids": [str(x["id"]) for x in target_rows],
        "holdout_row_ids": [str(x["id"]) for x in holdout],
        "dataset_manifest": family_summary,
        "evaluated_updates": list(AUDIT_UPDATES),
        "metrics": [
            "teacher-forced heldout reference-answer continuation NLL/top1",
            "automated first-sentence family/route keyword rubric",
            "mean probability of eos/eot after complete heldout reference answer",
            "free-generation trigram repetition rate",
        ],
        "first_sentence_caveat": "keyword rubric is not human-verified factual correctness",
        "eos_caveat": "EOS/EOT probability is teacher-forced at reference-answer end, not model stopping rate",
        "diagnostic_only": True,
        "promote_weights": False,
        "weight_checkpoint_saved": False,
    })

    orig_snapshot = exp.make_snapshot
    measurements = {}
    def instrumented_snapshot(**kwargs):
        snapshot = orig_snapshot(**kwargs)
        u = int(kwargs["update"])
        if u in AUDIT_UPDATES:
            diag = evaluate_separately(
                model=kwargs["model"], tok=kwargs["tok"],
                target_holdout=kwargs["target_holdout"],
                device=kwargs["device"], history=kwargs["history"],
                args=kwargs["args"], arm_dir=kwargs["arm_dir"], update=u,
            )
            measurements[u] = diag
            snapshot["generalization_diagnostic"] = diag
        return snapshot

    exp.make_snapshot = instrumented_snapshot
    try:
        result = exp.run_arm(
            arm=dict(ARM), target_rows=target_rows, target_holdout=holdout,
            tok=tok, history=history, eval_args=eval_args, special=special,
            args=args, output_dir=output_dir,
        )
    finally:
        exp.make_snapshot = orig_snapshot

    if set(measurements) != set(AUDIT_UPDATES):
        raise RuntimeError("Diagnostic did not measure all promised updates")
    if abs(float(result["final_exact_nll"]) - V14A8_EXPECTED_U40_NLL) > 0.005:
        raise RuntimeError("Replay mismatch: v14a8 u40 exact NLL changed")
    if abs(float(result["final_exact_top1"]) - V14A8_EXPECTED_U40_TOP1) > 0.005:
        raise RuntimeError("Replay mismatch: v14a8 u40 exact top1 changed")
    if abs(float(result["final_semantic_macro_success"]) - V14A8_EXPECTED_U40_SEMANTIC_MACRO) > 0.04:
        raise RuntimeError("Replay mismatch: v14a8 u40 semantic evaluation changed")

    compact = [measurements[u] for u in AUDIT_UPDATES]
    exp.write_json(output_dir / "v14a9_generalization_decision.json", {
        "name": "v14a9_holdout_generalization_diagnostic",
        "status": "completed",
        "replay_verified_against_v14a8": True,
        "replay_train_target_final_nll": result["final_exact_nll"],
        "replay_train_target_final_top1": result["final_exact_top1"],
        "replay_final_semantic_macro": result["final_semantic_macro_success"],
        "measurements": compact,
        "delta_u40_minus_u0": {
            key: compact[-1][key] - compact[0][key]
            for key in (
                "holdout_reference_nll",
                "holdout_reference_top1",
                "first_sentence_keyword_rubric_success_rate",
                "mean_eos_eot_p_at_reference_boundary",
                "mean_trigram_repetition_rate",
                "high_repetition_count",
                "full_answer_rubric_success_count",
            )
        },
        "diagnostic_only": True,
        "promotion_attempted": False,
    })
    print("[done] v14a9_holdout_generalization_diagnostic", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    return exp.build_argparser()


if __name__ == "__main__":
    raise SystemExit(run(build_argparser().parse_args()))
