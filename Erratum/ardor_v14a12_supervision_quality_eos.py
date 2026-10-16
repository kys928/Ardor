#!/usr/bin/env python3
"""v14a12: controlled authored-answer quality ablation with EOS in both arms.

Reuse the verified v14a10 CUDA trainer unchanged. Both arms strictly reload the
v14a2 parent and add the same one-token EOS label. Only the 28 training chosen
texts are replaced in the intervention. Never save or promote model weights.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import torch
from tokenizers import Tokenizer

from Erratum import ardor_v14a10_eos_supervision_ablation as eos
from Erratum import ardor_v14a6_learning_dynamics_probe as pulse
from Erratum import ardor_v14a4_family_balanced_trainer as base
from Erratum import ardor_v14a8_lr_ablation as prev
from Erratum.build_v14a4_family_data import COLLISION_DISTINCTIONS

OUTPUT_DIR = Path("/workspace/Ardor/training/runs/v14a12_supervision_quality_eos_u40")
CONTRACT_PATH = Path(__file__).with_name("v14a12_supervision_quality_contract_20261016.json")
ARMS = ("original_eos_control", "corrected_eos")
EOS_ID = 3

# Four researcher-authored, family-specific teaching targets per family.
# These are authored from standard gradient-clipping mechanics, NOT holdout
# answers. Assigned deterministically by sorted TRAIN ID within each family.
TARGETS: dict[str, tuple[str, str, str, str]] = {
    "definition_mechanism": (
        "Gradient clipping limits an unusually large gradient before the optimizer updates model weights. After backpropagation, it rescales a gradient that exceeds the chosen norm bound instead of letting the full spike reach the optimizer.",
        "Gradient clipping is a gradient-magnitude control applied after gradients are computed and before the optimizer step. When the norm is too high, the gradient is reduced; otherwise, norm clipping leaves it unchanged.",
        "This describes gradient clipping: the algorithm checks the gradient size and reduces exceptional gradients before updating parameters. It mitigates unusually large optimizer inputs rather than changing the learning target.",
        "Gradient clipping constrains an excessive backward-pass gradient prior to the parameter update. With global-norm clipping, the direction is preserved while the magnitude is scaled down when the threshold is exceeded.",
    ),
    "exploding_gradients": (
        "Exploding gradients are very large backward-pass derivatives that can destabilize training. Gradient clipping reduces these oversized gradients before the optimizer uses them, lowering the risk of an extreme update.",
        "The dangerous signal is an exploding gradient: its magnitude can become too large for stable optimization. Gradient clipping bounds the gradient supplied to the optimizer when it exceeds a preset limit.",
        "When gradients explode, the raw update-driving signal can be extreme and make training unstable. Clipping reduces excessive gradient magnitude before the update; it does not make exploding gradients beneficial.",
        "Gradient clipping addresses exploding gradients by scaling down unusually large gradient vectors after backpropagation. This can reduce training instability from gradient spikes, though it does not guarantee perfect stability.",
    ),
    "norm_threshold": (
        "For global-norm clipping, first measure the gradient vector norm. If it exceeds the maximum norm, multiply the gradient by maximum_norm / observed_norm; otherwise, keep the gradient unchanged. Then perform the optimizer step.",
        "A maximum gradient norm is a conditional ceiling, not a target for every gradient. Norm clipping rescales the gradient only when its observed norm exceeds that threshold, before parameters are updated.",
        "Norm-based gradient clipping computes the gradient norm after backpropagation and compares it with the allowed maximum. Oversized vectors are scaled down to the bound while smaller ones pass through unchanged.",
        "The threshold in gradient clipping applies to the norm of the gradient vector. The operation uses a scale factor min(1, threshold / norm) to reduce oversized gradients before the optimizer update.",
    ),
    "optimizer_stability": (
        "Gradient clipping helps optimizer stability by reducing an exceptionally large gradient before the optimizer processes it. This can avoid a destabilizing jump without requiring every normal gradient to be shrunk.",
        "A sudden gradient spike can make optimization unstable. Gradient clipping bounds that incoming gradient prior to the parameter update, reducing the effect of one extreme backward-pass signal.",
        "The mechanism protecting the update is gradient clipping: excessive gradients are rescaled before the optimizer step. This mitigates sharp update pressure; it is not a guarantee that every update will be safe.",
        "To mitigate optimizer instability caused by a gradient spike, gradient clipping limits the gradient magnitude after backpropagation. The optimizer then acts on the adjusted gradient rather than the full extreme signal.",
    ),
    "backprop_stability": (
        "Backpropagation computes parameter gradients; gradient clipping acts afterward, before the optimizer step. It reduces excessively large computed gradients so the subsequent update is less likely to become unstable.",
        "An oversized gradient produced during backpropagation can destabilize an optimizer update. Gradient clipping limits the computed gradient before optimization; it does not prevent every numerical problem within backpropagation itself.",
        "The backward pass creates gradients, and clipping constrains their magnitude after they have been calculated. This helps stabilize the following optimization step when a gradient would otherwise be extreme.",
        "A safe training step can clip a very large backpropagated gradient before handing it to the optimizer. The operation controls the gradient used for updating weights, not the entire backward computation.",
    ),
    "learning_rate_comparison": (
        "Gradient clipping and lowering the learning rate are different controls. Clipping conditionally rescales overly large gradients, while the learning rate changes the step scale used by the optimizer more generally.",
        "Clipping acts on gradient magnitude when a norm threshold is exceeded. A smaller learning rate changes how aggressively the optimizer updates parameters across steps, including steps without gradient spikes.",
        "To control occasional gradient spikes, norm clipping reduces the offending gradient only above a threshold. Lowering the learning rate instead reduces the optimizer's general step scale; it does not selectively clip a gradient.",
        "A gradient-norm cap and the learning rate should not be confused. Clipping bounds an excessive gradient before optimization, whereas the learning rate determines the update scale in the optimizer.",
    ),
    "false_premise_correction": (
        "No. Exploding gradients do not make training safer: they can cause unstable parameter updates. Gradient clipping reduces an oversized gradient before the optimizer step to mitigate that risk.",
        "That premise is backwards. Unbounded or exploding gradients can destabilize optimization. Gradient clipping limits excessive gradient magnitude before the optimizer uses the computed backward signal.",
        "No: very large gradients are a potential source of unstable updates, not protection against them. The protective mechanism is clipping, which scales down gradients that exceed the allowed norm.",
        "Exploding gradients do not safeguard learning. Gradient clipping addresses this problem by conditionally rescaling very large gradient vectors before the optimizer step, reducing the risk of a destabilizing update.",
    ),
}

PROHIBITED_FILLER = (
    "a correct answer should",
    "important signal is the mechanism itself",
    "key distinction is where the mechanism acts",
    "instead of a memorized phrase",
    "identified from its operation, not from generic",
)

# Frozen prior v14a10 explicit-EOS arm, retrieved from RunPod S3 decision JSON.
HISTORICAL_EXPLICIT_EOS = {
    "holdout_reference_nll": 1.2563423216552358,
    "holdout_reference_top1": 0.7814476469915179,
    "mean_eos_eot_p_at_reference_boundary": 0.9220029561221499,
    "high_repetition_count": 1,
}


def corrected_training_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 28 or len({str(r["id"]) for r in rows}) != 28:
        raise ValueError("v14a12 requires 28 unique frozen training rows")
    counts = Counter(str(r["semantic_family"]) for r in rows)
    if set(counts) != set(TARGETS) or any(n != 4 for n in counts.values()):
        raise ValueError("v14a12 requires exactly four rows from every target family")
    assigned: dict[str, str] = {}
    for family, targets in TARGETS.items():
        same_family = sorted((r for r in rows if r["semantic_family"] == family), key=lambda r: str(r["id"]))
        for index, row in enumerate(same_family):
            if row["route"] != "gradient_clipping" or row["source"] != "behavior":
                raise ValueError("Not a gradient-clipping authored training row")
            text = targets[index]
            collision = row.get("collision_route")
            if collision:
                key = ("gradient_clipping", str(collision))
                if key not in COLLISION_DISTINCTIONS:
                    raise ValueError(f"Unexpected collision route: {collision}")
                text += " " + COLLISION_DISTINCTIONS[key]
            if row.get("question_class") == "false_premise_correction" and family != "false_premise_correction":
                raise ValueError("Contradictory question class / family")
            if any(fragment in text.lower() for fragment in PROHIBITED_FILLER):
                raise ValueError("Low-information supervision filler detected")
            assigned[str(row["id"])] = text
    corrected = [{**row, "chosen": assigned[str(row["id"])]} for row in rows]
    if any(new["chosen"] == original["chosen"] for original, new in zip(rows, corrected)):
        raise RuntimeError("Every intervention answer must differ from the control")
    for old, new in zip(rows, corrected):
        if {k: v for k, v in old.items() if k != "chosen"} != {
            k: v for k, v in new.items() if k != "chosen"
        }:
            raise RuntimeError("An intervention modified non-answer metadata")
    return corrected


def chosen_digest(rows: Sequence[dict[str, Any]]) -> str:
    records = [{"id": str(r["id"]), "chosen": str(r["chosen"])} for r in rows]
    payload = json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def target_manifest(control: Sequence[dict[str, Any]], corrected: Sequence[dict[str, Any]], tok: Tokenizer) -> dict[str, Any]:
    details = []
    for old, new in zip(control, corrected):
        old_tokens = len(base.encode(tok, old["chosen"]))
        new_tokens = len(base.encode(tok, new["chosen"]))
        details.append({
            "id": str(old["id"]), "family": str(old["semantic_family"]),
            "question_class": str(old["question_class"]),
            "original_answer_tokens": old_tokens,
            "corrected_answer_tokens": new_tokens,
            "original_answer_sha256": hashlib.sha256(old["chosen"].encode()).hexdigest(),
            "corrected_answer_sha256": hashlib.sha256(new["chosen"].encode()).hexdigest(),
        })
    return {
        "rows": details, "control_chosen_sha256": chosen_digest(control),
        "corrected_chosen_sha256": chosen_digest(corrected),
        "control_answer_tokens": sum(r["original_answer_tokens"] for r in details),
        "corrected_answer_tokens": sum(r["corrected_answer_tokens"] for r in details),
        "special_eos_id_both_arms": EOS_ID,
    }


def control_reproduced(result: dict[str, Any]) -> dict[str, bool]:
    observed = result["final_generalization"]
    return {
        "nll": abs(observed["holdout_reference_nll"] - HISTORICAL_EXPLICIT_EOS["holdout_reference_nll"]) <= 0.025,
        "top1": abs(observed["holdout_reference_top1"] - HISTORICAL_EXPLICIT_EOS["holdout_reference_top1"]) <= 0.02,
        "reference_boundary_eos": abs(
            observed["mean_eos_eot_p_at_reference_boundary"]
            - HISTORICAL_EXPLICIT_EOS["mean_eos_eot_p_at_reference_boundary"]
        ) <= 0.10,
        "high_repetition": observed["high_repetition_count"] <= 3,
    }


def comparison_decision(control: dict[str, Any], corrected: dict[str, Any]) -> dict[str, Any]:
    if control["batch_sequence_ids"] != corrected["batch_sequence_ids"]:
        raise RuntimeError("Batch row ID order differs between training arms")
    original = control["final_generalization"]
    intervention = corrected["final_generalization"]
    names = (
        "holdout_reference_nll", "holdout_reference_top1",
        "first_sentence_keyword_rubric_success_rate",
        "mean_eos_eot_p_at_reference_boundary",
        "mean_trigram_repetition_rate", "high_repetition_count",
        "full_answer_rubric_success_count",
    )
    delta = {name: float(intervention[name]) - float(original[name]) for name in names}
    checks = {
        "mean_reference_boundary_eos_ge_0p80": intervention["mean_eos_eot_p_at_reference_boundary"] >= 0.80,
        "high_repetition_le_2_of_32": intervention["high_repetition_count"] <= 2,
        "heldout_reference_nll_regression_le_0p10": delta["holdout_reference_nll"] <= 0.10,
        "heldout_reference_top1_drop_le_0p02": delta["holdout_reference_top1"] >= -0.02,
    }
    return {
        "experiment": "v14a12_supervision_quality_eos",
        "diagnostic_only": True,
        "control": "original_eos_control",
        "intervention": "corrected_eos",
        "all_automatic_retention_checks_passed": all(checks.values()),
        "automatic_retention_checks": checks,
        "delta_corrected_minus_control": delta,
        "control_reproduction": control_reproduced(control),
        "manual_factual_adequacy_adjudication": "pending",
        "manual_success_criterion": "at least 6/32 more strictly adequate cases than the reproduced EOS control, with gains in weakest families",
        "not_blind_holdout_warning": "v14a11 inspected the 32 holdout prompts; this is a fixed diagnostic panel, not a fresh blind generalization test",
        "limitations": [
            "Changed chosen-answer text changes target token count and CE normalization; measure both.",
            "Both arms use identical EOS label and existing historical greedy EOS/EOT stopping.",
            "Teacher-forced EOS at reference ending is not natural stop frequency.",
            "Keyword rubric passes are not validated factual correctness.",
            "Single route/seed/40-update probe; no promotion even if automated gates pass.",
        ],
        "checkpoint_saved": False,
        "promotion_attempted": False,
    }


def execute_arm(label: str, train_rows: Sequence[dict[str, Any]], holdout, tok, history, eval_args, special, args, output_dir) -> dict[str, Any]:
    if label not in ARMS:
        raise ValueError("Unknown arm")
    # Reuse the v14a10 exact same EOS trainer for both arms, isolating
    # different artifacts under separate roots while preserving all controls.
    result = eos.run_one_arm(
        arm_name="explicit_eos", target_rows=train_rows, target_holdout=holdout,
        tok=tok, history=history, eval_args=eval_args, special=special,
        args=args, output_dir=output_dir / label,
    )
    arm_dir = output_dir / label / "explicit_eos"
    config = json.loads((arm_dir / "run_config.json").read_text(encoding="utf-8"))
    config.update({
        "experiment": "v14a12_supervision_quality_eos",
        "logical_arm": label,
        "actual_single_changing_factor": "chosen_answer_text_quality",
        "eos_supervision_in_both_arms": True,
        "v14a10_core_runner_reused": True,
        "v14a10_legacy_only_variable_field": config.pop("only_variable"),
        "original_chosen_text_unchanged": label == "original_eos_control",
    })
    prev.write_json(arm_dir / "run_config.json", config)
    result.update({
        "arm": label,
        "original_chosen_text_unchanged": label == "original_eos_control",
        "eos_supervision": True,
        "chosen_answer_sha256": chosen_digest(train_rows),
    })
    prev.write_json(arm_dir / "arm_result.json", {k: v for k, v in result.items() if k != "batch_sequence_ids"})
    return result


def run(args) -> None:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v14a12 must run on CUDA")
    if not args.verify_parent_sha256:
        raise ValueError("v14a12 requires --verify-parent-sha256")
    if (
        args.seed != base.SEED or args.batch_rows != 4 or args.max_len != base.MAX_LEN
        or args.weight_decay != 0.0 or args.warmup_updates != 20
        or args.min_lr_ratio != 0.2 or args.max_gen_tokens != 72
        or args.min_eval_words != 3 or args.lr != 2e-8
    ):
        raise ValueError("v14a12 may not vary frozen v14a10 controls")
    if (eos.PEAK_LR, eos.TOTAL_UPDATES, eos.GRAD_CLIP, eos.END_TOKEN) != (2e-5, 40, 0.5, "<eos>"):
        raise RuntimeError("Frozen v14a10 train implementation changed")
    base.validate_local_files(verify_checkpoint_sha256=True)
    contract = base.training_contract()
    raw, all_holdout, summary = base.ensure_family_datasets(contract)
    behavior = pulse._repaired_behavior_rows(raw)
    control = pulse.select_target_rows(
        behavior, seed=args.seed + 701, rows_per_family=pulse.TARGET_ROWS_PER_FAMILY
    )
    holdout = [dict(r) for r in all_holdout if r["route"] == pulse.TARGET_ROUTE]
    corrected = corrected_training_rows(control)
    if len(holdout) != 32 or set(r["id"] for r in control) & set(r["id"] for r in holdout):
        raise RuntimeError("Frozen 32-case diagnostic holdout failed integrity checks")
    if [r["id"] for r in control] != [r["id"] for r in corrected]:
        raise RuntimeError("Target ID order changed")
    tok = Tokenizer.from_file(str(base.TOKENIZER_PATH))
    tcontract = base._tokenizer_contract(tok)
    if not tcontract["passed"] or tok.get_vocab_size() != base.EXPECTED_VOCAB_SIZE:
        raise RuntimeError("tokenizer-v9 contract changed")
    if {k: tok.token_to_id(k) for k in base.EXPECTED_SPECIAL_IDS} != base.EXPECTED_SPECIAL_IDS:
        raise RuntimeError("special token IDs changed")
    if base.EXPECTED_SPECIAL_IDS["<eos>"] != EOS_ID:
        raise RuntimeError("EOS token ID changed")
    history = base._load_historical_eval_module()
    eval_args = base.configure_history_args(history, args)
    special = history.special_ids(tok, eval_args)
    if EOS_ID not in special["eos_ids"]:
        raise RuntimeError("EOS missing from historical greedy stop IDs")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = target_manifest(control, corrected, tok)
    prev.write_json(root / "training_target_manifest.json", manifest)
    prev.write_json(root / "run_config.json", {
        "experiment": "v14a12_supervision_quality_eos",
        "parent": base.canonical_parent_reference(),
        "tokenizer_contract": tcontract,
        "v14a4_dataset_manifest": summary,
        "original_train_ids": [str(x["id"]) for x in control],
        "holdout_ids": [str(x["id"]) for x in holdout],
        "policy_contract": str(CONTRACT_PATH),
        "only_factor": "chosen_answer_text_quality",
        "EOS_in_both_arms": True,
        "EOT_not_added_as_label": True,
        "no_checkpoint_promotion": True,
    })
    original = execute_arm(ARMS[0], control, holdout, tok, history, eval_args, special, args, root)
    replay = control_reproduced(original)
    prev.write_json(root / "control_reproduction.json", replay)
    if not all(replay.values()):
        raise RuntimeError("Historical v14a10 explicit-EOS control failed reproduction gates; abort corrected arm")
    edited = execute_arm(ARMS[1], corrected, holdout, tok, history, eval_args, special, args, root)
    decision = comparison_decision(original, edited)
    prev.write_json(root / "v14a12_supervision_quality_decision.json", decision)
    prev.write_json(root / "run_summary.json", {
        "experiment": decision["experiment"], "diagnostic_only": True,
        "automatic_checks_passed": decision["all_automatic_retention_checks_passed"],
        "manual_review_pending": True,
        "promotion_attempted": False, "timestamp": time.time(),
    })
    print("[done] v14a12 automatic retention checks: " + str(decision["all_automatic_retention_checks_passed"]), flush=True)


def build_argparser():
    parser = eos.build_argparser()
    parser.set_defaults(output_dir=str(OUTPUT_DIR))
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
