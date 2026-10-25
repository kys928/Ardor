#!/usr/bin/env python3
"""v14a14: matched 80-prompt original vs causal supervision, frozen EOS.

Training is hard-gated by 80-row independent-review attestation, pinned data,
pinned holdout and canonical parent/tokenizer contracts. Reuse v14a10 40-update
runner and EOS objective WITHOUT editing it. No weights are saved/promoted.

This script is intentionally *not* called by GitHub Actions. Running it
requires explicit data-review certification and a separate CUDA operator run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from Erratum.v14a14_signal_dataset import build_pairs, digest
from Erratum.v14a14_blind_panel import build_holdout

PIN = {
    "control_sha256": "09341553fb2b689c244e52ec47c6cdc736c964ca68d8ea61843cf5eb7f92745d",
    "revised_sha256": "08f95290f2a49d0f8bd01fdd45389c9a3dd53639a6c6120b7938878b726423d6",
    "proposition_audit_sha256": "854d13cd860961de6a631a12d41a447301c3dbea812d40ed1cf00b47a877015b",
    "train_prompt_sha256": "b94c802089c466d3fc64f1b0ba0274d4fa032c44da3c8fc4a9b5a56a41422627",
    "holdout_prompt_sha256": "cb1b0a03e139039fc1a4e0301f6e62f1a6b298e9920f7e9a944d7745a5c4a589",
    "holdout_rows_and_references_sha256": "9e1b57a09804cea5c3d4ee6fb505c68c220c6925868939e19814b44d3ee81220",
}

PROMOTION_CRITERIA = {
    "primary_human_review": "blinded paired human adjudication of all 28 fresh panel prompts",
    "minimum_factual_adequacy_gain": "at least six more adequate answers / 28 intervention than control",
    "weak_families": "improvement of at least one of four in each backprop_stability, learning_rate_comparison and false_premise_correction",
    "false_premise_accuracy": "at least two of four false-premise corrections adequate",
    "repetition": "intervention high-trigram-repetition count <=2/28 and not worse than control",
    "reference_boundary_eos": "mean EOS/EOT probability after teacher-forced reference >=0.80; not a natural-stop metric",
    "safety": "no checkpoint saved or promoted by diagnostic runner",
    "report_separately": "own-target top1/NLL, holdout reference top1/NLL, free-generation repetition, reference-boundary EOS, human factual adequacy",
}

def preflight() -> tuple[list[dict], list[dict], list[dict], dict]:
    control, causal, signal = build_pairs()
    holdout, panel = build_holdout()
    m = signal["manifest"]
    check = {
        "control_sha256": m["control_sha256"],
        "revised_sha256": m["revised_sha256"],
        "proposition_audit_sha256": m["proposition_audit_sha256"],
        "train_prompt_sha256": m["revised_paired_prompt_sha256"],
        "holdout_prompt_sha256": panel["holdout_prompt_sha256"],
        "holdout_rows_and_references_sha256": panel["holdout_rows_and_references_sha256"],
    }
    if check != PIN:
        raise RuntimeError("v14a14 source or split changed after panel freeze: " + json.dumps(check))
    if len(control) != 80 or len(causal) != 80 or len(holdout) != 28:
        raise RuntimeError("Unexpected train or holdout size")
    if [(r["id"], r["prompt"]) for r in control] != [(r["id"], r["prompt"]) for r in causal]:
        raise RuntimeError("Arms do not share identical prompt exposure")
    return control, causal, holdout, check


def verify_data_review(approval_path: Path, row_ids: list[str]) -> dict:
    """Independent review attestation; no implicit or self-generated approval."""
    document = json.loads(approval_path.read_text(encoding="utf-8"))
    if document.get("proposition_audit_sha256") != PIN["proposition_audit_sha256"]:
        raise ValueError("Reviewer approved a different proposition manifest")
    if document.get("train_prompt_sha256") != PIN["train_prompt_sha256"]:
        raise ValueError("Reviewer approved a different training prompt panel")
    reviewers = document.get("reviewers")
    if not isinstance(reviewers, list) or len(reviewers) < 2:
        raise ValueError("Two independent reviewer attestations required")
    identities = [r.get("reviewer_id") for r in reviewers]
    if not all(identities) or len(set(identities)) != len(identities):
        raise ValueError("Reviewers must have two distinct names")
    for r in reviewers:
        if r.get("status") != "approved_after_review":
            raise ValueError("Review still pending")
        if sorted(r.get("approved_row_ids", [])) != sorted(row_ids):
            raise ValueError("Each reviewer must independently approve all 80 rows")
        if r.get("contradictions_found_unresolved"):
            raise ValueError("Unresolved contradictions block GPU work")
    if document.get("resolved_disagreements") is not True:
        raise ValueError("Disagreements not adjudicated")
    return document


def verify_single_user_review(causal_rows: list[dict], manifest: dict) -> dict:
    """Use the first review truthfully; do NOT synthesize an independent second."""
    info = manifest.get("first_user_review_current", {})
    ledger = info.get("review_ledger", [])
    if info.get("adequate") != 80 or info.get("reviewer_count") != 1:
        raise ValueError("Not all 80 answers have one documented user review")
    if info.get("second_independent_reviewer_status") != "NOT_PERFORMED":
        raise ValueError("Unexpected review state; manual contract reconciliation needed")
    if len(ledger) != 80 or digest(ledger) != info.get("review_ledger_sha256"):
        raise ValueError("First-review ledger missing or changed")
    for row, judged in zip(causal_rows, ledger):
        actual = hashlib.sha256(row["chosen"].encode("utf-8")).hexdigest()
        if row["id"] != judged.get("id") or judged.get("answer_sha256") != actual:
            raise ValueError("Reviewed answer differs from the current training target")
        if judged.get("verdict") != "ADEQUATE":
            raise ValueError("Unapproved current target")
    return {
        "policy": "one_user_reviewer_diagnostic_exception",
        "first_user_approved": len(ledger),
        "independent_second_review": "NOT_PERFORMED",
        "review_ledger_sha256": info["review_ledger_sha256"],
        "no_auto_promotion": True,
    }


def candidate_checkpoint_hook(*, model, optimizer, arm_name: str,
                              arm_dir: Path, result: dict, args,
                              pinned: dict, review: dict) -> dict:
    """Retain a reproducible candidate, never overwrite the canonical parent.

    Uses exact model/optimizer/meta wrapper verified in canonical_eval_v14a2;
    stage and provenance declare this *not* an already-promoted checkpoint.
    """
    import torch
    from Erratum import ardor_v14a4_family_balanced_trainer as base

    if arm_name != "explicit_eos" or arm_dir.name != "explicit_eos":
        raise RuntimeError("Candidate saving is permitted only for explicit EOS")
    if result["steps"] != 40 or result["checkpoints_saved"]:
        raise RuntimeError("Cannot capture incomplete or previously saved training")
    root = arm_dir.parent
    path = root / "candidate" / "v14a14_causal_u0040_candidate.pt"
    if path.exists():
        raise RuntimeError(f"Refusing overwrite of existing candidate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.pending")
    if temporary.exists():
        raise RuntimeError("Incomplete candidate save exists; inspect it first")
    wrapper = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "meta": {
            "stage": "v14a14_supervision_quality_candidate_NOT_PROMOTED",
            "step": 40,
            "trainer": "Erratum/v14a14_controlled_eos_train.py",
            "model_config": dict(base.MODEL_CONFIG),
            "canonical_parent": base.canonical_parent_reference(),
            "tokenizer_contract": {
                "path": str(base.TOKENIZER_PATH),
                "vocab_size": base.EXPECTED_VOCAB_SIZE,
                "special_ids": dict(base.EXPECTED_SPECIAL_IDS),
            },
            "objective": {
                "masked_answer_ce": True,
                "one_supervised_eos_label_per_answer": True,
                "loss_variant": "v14a10_explicit_eos",
            },
            "training_provenance": {
                "dataset_pins": pinned, "review": review,
                "seed": args.seed, "steps": 40,
                "source_arm": "causal_80",
            },
            "promotion": {"status": "CANDIDATE_ONLY",
                          "requires_frozen_28_case_evaluation": True},
        },
    }
    try:
        torch.save(wrapper, str(temporary))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    file_sha256 = base.sha256_file(path)
    record = {
        "path": str(path), "sha256": file_sha256,
        "size_bytes": path.stat().st_size,
        "schema": "model/optimizer/meta",
        "status": "CANDIDATE_NOT_PROMOTED",
    }
    base.write_json(path.with_suffix(".manifest.json"), record)
    return record


def run(args) -> None:
    original, corrected, holdout, pins = preflight()
    if args.preflight_only:
        print(json.dumps({"contract": "v14a14", "status": "PINNED_NOT_TRAINED",
                          "train_rows": 80, "holdout_rows": 28,
                          "pins": pins, "success_gate": PROMOTION_CRITERIA}, sort_keys=True))
        return
    # User-authorized one-review diagnostic exception. Never impersonate the
    # second reviewer; all other canonical/data/evaluation gates stay intact.
    _, _, first_review_manifest = build_pairs()
    first_review = verify_single_user_review(corrected, first_review_manifest["manifest"])
    if args.one_review_diagnostic:
        review = first_review
    else:
        if not args.review_attestation:
            raise ValueError("Two independent reviewers required without --one-review-diagnostic")
        reviewer_data = verify_data_review(Path(args.review_attestation),
                                           [r["id"] for r in original])
        review = {
            "policy": "two_independent_reviewers",
            "reviewer_ids": [r["reviewer_id"] for r in reviewer_data["reviewers"]],
            "first_review_ledger_sha256": first_review["review_ledger_sha256"],
        }

    import torch
    from tokenizers import Tokenizer
    from Erratum import ardor_v14a10_eos_supervision_ablation as eos
    from Erratum import ardor_v14a4_family_balanced_trainer as base
    from Erratum import ardor_v14a8_lr_ablation as prev

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the controlled experiment")
    if not args.verify_parent_sha256:
        raise ValueError("Canonical parent must be SHA-256 checked")
    if (
        args.seed != base.SEED or args.batch_rows != 4 or args.max_len != base.MAX_LEN
        or args.weight_decay != 0.0 or args.warmup_updates != 20
        or args.min_lr_ratio != 0.2 or args.max_gen_tokens != 72
        or args.min_eval_words != 3 or args.lr != 2e-8
        or (eos.PEAK_LR, eos.TOTAL_UPDATES, eos.GRAD_CLIP, eos.END_TOKEN) != (2e-5, 40, 0.5, "<eos>")
    ):
        raise RuntimeError("Frozen v14a10 EOS/optimizer/schedule contract changed")

    base.validate_local_files(verify_checkpoint_sha256=True)
    tok = Tokenizer.from_file(str(base.TOKENIZER_PATH))
    tcontract = base._tokenizer_contract(tok)
    if not tcontract["passed"] or tok.get_vocab_size() != base.EXPECTED_VOCAB_SIZE:
        raise RuntimeError("Tokenizer/vocabulary contract violated")
    if {k: tok.token_to_id(k) for k in base.EXPECTED_SPECIAL_IDS} != base.EXPECTED_SPECIAL_IDS:
        raise RuntimeError("Special-token contract violated")
    if base.EXPECTED_SPECIAL_IDS.get("<eos>") != 3:
        raise RuntimeError("EOS ID is not canonical 3")
    history = base._load_historical_eval_module()
    eval_args = base.configure_history_args(history, args)
    special = history.special_ids(tok, eval_args)
    if base.EXPECTED_SPECIAL_IDS["<eos>"] not in special["eos_ids"]:
        raise RuntimeError("EOS is not an allowed generation stop token")

    # Quantify the unavoidable different CE-token exposure caused by answer
    # length. Matched prompt rows do NOT guarantee matched target token counts.
    token_exposure = {}
    for label, rows in (("original_80", original), ("causal_80", corrected)):
        lengths = [len(base.encode(tok, r["chosen"])) for r in rows]
        if not all(0 < n + 1 < args.max_len for n in lengths):
            raise RuntimeError("EOS answer exceeds or empties max token context")
        by_family = {}
        for family in sorted({r["semantic_family"] for r in rows}):
            ix = [i for i, r in enumerate(rows) if r["semantic_family"] == family]
            by_family[family] = {
                "rows": len(ix), "chosen_tokens": sum(lengths[i] for i in ix),
                "mean_chosen_tokens": sum(lengths[i] for i in ix) / len(ix),
            }
        token_exposure[label] = {
            "chosen_tokens_per_one_pass": sum(lengths),
            "mean_chosen_tokens": sum(lengths) / len(lengths),
            "min_chosen_tokens": min(lengths),
            "max_chosen_tokens": max(lengths),
            "one_eos_label_per_row_per_exposure": True,
            "by_family": by_family,
        }
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prev.write_json(root / "v14a14_frozen_experiment.json", {
        "pins": pins, "review": review,
        "retention": PROMOTION_CRITERIA, "arms": ["original_80", "causal_80"],
        "same_train_prompts": True, "same_seed": args.seed,
        "target_token_exposure_per_single_dataset_pass": token_exposure,
        "eos_one_token_per_answer": True, "steps_each": eos.TOTAL_UPDATES,
        "control_original_v14a4_answers_60_and_cue_matched_legacy_answers_20": True,
        "heldout_seen_during_training": False, "checkpoint_saved": False,
        "promotion_policy": "eligible_if_frozen_evaluation_gates_pass",
        "automatic_promotion": False,
    })
    results = {}
    for label, rows in (("original_80", original), ("causal_80", corrected)):
        results[label] = eos.run_one_arm(
            arm_name="explicit_eos", target_rows=rows, target_holdout=holdout,
            tok=tok, history=history, eval_args=eval_args, special=special,
            args=args, output_dir=root / label,
            candidate_checkpoint_hook=(
                (lambda **kw: candidate_checkpoint_hook(**kw, pinned=pins, review=review))
                if label == "causal_80" else None
            ),
        )
    if results["original_80"]["batch_sequence_ids"] != results["causal_80"]["batch_sequence_ids"]:
        raise RuntimeError("Batch IDs differ across arms")
    prev.write_json(root / "v14a14_posttraining_diagnostic.json", {
        "arms": {
            k: {key: val for key, val in v.items() if key != "batch_sequence_ids"}
            for k, v in results.items()
        },
        "matched_batch_order": True,
        "factual_adequacy": "PENDING_BLIND_HUMAN_ADJUDICATION",
        "promotion_attempted": False,
        "candidate_retained": bool(results["causal_80"].get("candidate_checkpoint")),
        "promotion_policy": "frozen_evaluation_gate_required",
    })
    print("[done] matched EOS diagnostic; candidate retained; promotion requires evidence gate", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    # Importing EOS parser only in actual CUDA execution avoids torch dependency
    # for pure CPU preflight; parse a minimal preflight flag first.
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--preflight-only", action="store_true")
    opts, _ = first.parse_known_args()
    if opts.preflight_only:
        parser = argparse.ArgumentParser()
        parser.add_argument("--preflight-only", action="store_true")
        parser.set_defaults(device="cpu", review_attestation=None, output_dir="",
                            verify_parent_sha256=False)
        return parser
    from Erratum import ardor_v14a10_eos_supervision_ablation as eos
    parser = eos.build_argparser()
    parser.add_argument("--review-attestation", type=str, default=None)
    parser.add_argument("--one-review-diagnostic", action="store_true",
                        help="User-authorized diagnostic exception; single reviewed ledger; never fake a second reviewer")
    parser.set_defaults(output_dir="/workspace/Ardor/training/runs/v14a14_strong_supervision_eos_u40",
                        preflight_only=False)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
