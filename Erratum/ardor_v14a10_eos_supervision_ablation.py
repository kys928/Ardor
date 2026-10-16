#!/usr/bin/env python3
"""v14a10 single-factor EOS-supervision ablation on the exact v14a8 high-LR pulse.

The baseline trains with the original v14a4 continuation_ce. The only
intervention adds one canonical <eos> token as a supervised next-token label
after each identical chosen answer. The 28 rows, 40 batches, seed, parent,
AdamW, learning rate, cosine schedule, 0.5 clip and teacher-forced answer text
are unchanged. No checkpoints are written or promoted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from Erratum import ardor_v14a4_family_balanced_trainer as base
from Erratum import ardor_v14a6_learning_dynamics_probe as pulse
from Erratum import ardor_v14a8_lr_ablation as prev
from Erratum import ardor_v14a9_holdout_generalization_diagnostic as audit

OUTPUT_DIR = Path("/workspace/Ardor/training/runs/v14a10_eos_supervision_ablation_u40")
CONTRACT_PATH = Path(__file__).with_name("v14a10_eos_supervision_contract_20261016.json")
PEAK_LR = 2e-5
ARMS = ("no_eos", "explicit_eos")
PROBE_UPDATES = (0, 20, 40)
TOTAL_UPDATES = 40
GRAD_CLIP = 0.5
END_TOKEN = "<eos>"


def eos_supervised_batch(
    rows: Sequence[dict[str, Any]], tok: Tokenizer,
    *, max_len: int, pad_id: int, eos_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Same masked continuation labels, with ONE EOS label at answer end.

    Explicit integer token ID avoids textual EOS re-tokenization. Compared to
    v14a4 continuation_ce, this adds precisely one supervised token per row.
    """
    examples = []
    routes: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    families: Counter[str] = Counter()
    for row in rows:
        pids = base.encode(tok, row["prompt"])
        cids = base.encode(tok, row["chosen"])
        if not pids or not cids or len(cids) + 1 >= max_len:
            raise RuntimeError(f"Invalid v14a10 tokenization {row['id']}")
        pids = pids[-max(1, max_len - len(cids) - 1):]
        # Last input is last answer token; label at that position is EOS.
        xids = (pids + cids + [eos_id])[:-1]
        labels = [-100] * (len(pids) - 1) + cids + [eos_id]
        if len(xids) != len(labels) or labels[-1] != eos_id:
            raise RuntimeError(f"Invalid EOS supervision alignment for {row['id']}")
        assert sum(label != -100 for label in labels) == len(cids) + 1
        examples.append((xids, labels))
        routes[str(row["route"])] += 1
        sources[str(row["source"])] += 1
        families[f"{row['route']}/{row['semantic_family']}"] += 1

    width = max(len(xids) for xids, _ in examples)
    x = torch.full((len(examples), width), int(pad_id), dtype=torch.long, device=device)
    y = torch.full((len(examples), width), -100, dtype=torch.long, device=device)
    for i, (xids, labels) in enumerate(examples):
        x[i, :len(xids)] = torch.tensor(xids, dtype=torch.long, device=device)
        y[i, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
    return x, y, {
        "routes": dict(routes), "sources": dict(sources), "families": dict(families),
        "eos_labels": len(examples),
    }


def eos_continuation_ce(
    model, tok: Tokenizer, rows: Sequence[dict[str, Any]],
    device: torch.device, *, max_len: int, pad_id: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    x, y, batch_meta = eos_supervised_batch(
        rows, tok, max_len=max_len, pad_id=pad_id,
        eos_id=base.EXPECTED_SPECIAL_IDS[END_TOKEN], device=device,
    )
    logits = model(x).float()
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100,
    )
    mask = y != -100
    top1 = float((torch.argmax(logits, dim=-1)[mask] == y[mask]).float().mean().detach().item())
    return loss, {
        "tokens": int(mask.sum().item()),
        "top1_rate": top1,
        **batch_meta,
    }


def snapshot(
    *, arm_name: str, update: int, model, tok, train_rows, holdout,
    device, history, eval_args, special, args, arm_dir,
) -> dict[str, Any]:
    standard = prev.make_snapshot(
        arm_name=arm_name,
        update=update,
        model=model,
        tok=tok,
        target_rows=train_rows,
        target_holdout=holdout,
        device=device,
        history=history,
        eval_args=eval_args,
        special=special,
        args=args,
        arm_dir=arm_dir,
    )
    distinct = audit.evaluate_separately(
        model=model, tok=tok, target_holdout=holdout, device=device,
        history=history, args=args, arm_dir=arm_dir, update=update,
    )
    return {**standard, "generalization": distinct}


def run_one_arm(
    *, arm_name: str, target_rows, target_holdout, tok,
    history, eval_args, special, args, output_dir,
) -> dict[str, Any]:
    if arm_name not in ARMS:
        raise ValueError("Unknown EOS ablation arm")
    arm_dir = output_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "train_steps.jsonl").write_text("", encoding="utf-8")
    (arm_dir / "trajectory.jsonl").write_text("", encoding="utf-8")
    prev.set_reproducible_seed(args.seed)
    model, strict_load, checkpoint_meta = base._load_canonical_model(args.device)
    config = model.model_config()
    for name, expected in base.MODEL_CONFIG.items():
        if config.get(name) != expected:
            raise RuntimeError(f"Canonical model config mismatch: {name}")
    if any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError("All parameters must remain trainable")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=PEAK_LR, weight_decay=args.weight_decay,
    )
    sampler = pulse.TargetPulseSampler(target_rows, seed=args.seed + 801)
    lr_args = argparse.Namespace(**vars(args))
    lr_args.lr = PEAK_LR
    lr_args.max_updates = prev.V14A6_LR_HORIZON
    device = torch.device(args.device)
    prev.write_json(arm_dir / "run_config.json", {
        "arm": arm_name, "only_variable": "append_one_supervised_eos_label",
        "explicit_eos": arm_name == "explicit_eos",
        "parent": base.canonical_parent_reference(),
        "strict_load": strict_load, "checkpoint_meta": checkpoint_meta,
        "train_ids": [str(row["id"]) for row in target_rows],
        "batch_seed": args.seed + 801,
        "pulse_updates": TOTAL_UPDATES, "probe_updates": list(PROBE_UPDATES),
        "optimizer": "fresh AdamW, all parameters trainable",
        "lr": PEAK_LR, "lr_horizon": prev.V14A6_LR_HORIZON,
        "warmup_updates": args.warmup_updates, "min_lr_ratio": args.min_lr_ratio,
        "weight_decay": args.weight_decay, "grad_clip": GRAD_CLIP,
        "objective": "original masked chosen CE" if arm_name == "no_eos" else (
            "original masked chosen CE plus exactly one supervised eos label per row"
        ),
        "no_checkpoint_promotion": True,
    })

    snapshots: dict[int, dict[str, Any]] = {}
    for update in range(TOTAL_UPDATES + 1):
        if update > 0:
            batch = sampler.next_batch(args.batch_rows)
            model.train()
            lr = base.lr_at(update, lr_args)
            base.set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            ce_func = base.continuation_ce if arm_name == "no_eos" else eos_continuation_ce
            loss, stats = ce_func(
                model, tok, batch, device,
                max_len=args.max_len, pad_id=base.EXPECTED_SPECIAL_IDS["<pad>"],
            )
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Nonfinite CE for {arm_name} u{update}")
            loss.backward()
            raw_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not math.isfinite(raw_grad_norm):
                raise RuntimeError(f"Nonfinite gradient norm for {arm_name} u{update}")
            optimizer.step()
            rec = {
                "arm": arm_name, "update": update, "lr": lr,
                "ce": float(loss.detach().item()), "raw_grad_norm": raw_grad_norm,
                "max_norm": GRAD_CLIP,
                "clip_scale": min(1.0, GRAD_CLIP / (raw_grad_norm + 1e-6)),
                "batch_ids": [str(row["id"]) for row in batch],
                **stats,
            }
            prev.append_jsonl(arm_dir / "train_steps.jsonl", rec)
            if update == 1 or update % 5 == 0:
                print(
                    f"[train] {arm_name} u={update} lr={lr:.2e} "
                    f"ce={rec['ce']:.6f} norm={raw_grad_norm:.3f}",
                    flush=True,
                )
        if update in PROBE_UPDATES:
            s = snapshot(
                arm_name=arm_name, update=update,
                model=model, tok=tok, train_rows=target_rows,
                holdout=target_holdout, device=device, history=history,
                eval_args=eval_args, special=special, args=args, arm_dir=arm_dir,
            )
            snapshots[update] = s
            prev.append_jsonl(arm_dir / "trajectory.jsonl", s)

    records = [
        json.loads(line) for line in (arm_dir / "train_steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != TOTAL_UPDATES:
        raise RuntimeError("Incomplete EOS pulse")
    result = {
        "arm": arm_name,
        "final_train_target_nll": float(snapshots[TOTAL_UPDATES]["exact"]["nll"]),
        "final_train_target_top1": float(snapshots[TOTAL_UPDATES]["exact"]["top1_rate"]),
        "final_semantic_macro": float(
            snapshots[TOTAL_UPDATES]["semantic"]["family_macro_success_rate"]
        ),
        "final_generalization": snapshots[TOTAL_UPDATES]["generalization"],
        "baseline_generalization": snapshots[0]["generalization"],
        "steps": len(records),
        "batch_sequence_ids": [
            bid for record in records for bid in record["batch_ids"]
        ],
        "identical_training_example_count": len(target_rows),
        "original_chosen_text_unchanged": True,
        "checkpoints_saved": False,
    }
    prev.write_json(arm_dir / "arm_result.json", result)
    del optimizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def verdict(no_eos: dict[str, Any], yes_eos: dict[str, Any]) -> dict[str, Any]:
    baseline = no_eos["final_generalization"]
    amended = yes_eos["final_generalization"]
    if abs(no_eos["final_train_target_nll"] - audit.V14A8_EXPECTED_U40_NLL) > 0.005:
        raise RuntimeError("v14a10 baseline did not replicate v14a8/v14a9 final NLL")
    if abs(no_eos["final_train_target_top1"] - audit.V14A8_EXPECTED_U40_TOP1) > 0.005:
        raise RuntimeError("v14a10 baseline did not replicate v14a8/v14a9 final top1")
    if abs(no_eos["final_semantic_macro"] - audit.V14A8_EXPECTED_U40_SEMANTIC_MACRO) > 0.04:
        raise RuntimeError("v14a10 baseline did not replicate semantic macro")
    if no_eos["batch_sequence_ids"] != yes_eos["batch_sequence_ids"]:
        raise RuntimeError("Batch order mismatch between EOS ablation arms")
    keys = (
        "holdout_reference_nll", "holdout_reference_top1",
        "first_sentence_keyword_rubric_success_rate",
        "mean_eos_eot_p_at_reference_boundary",
        "mean_trigram_repetition_rate", "high_repetition_count",
        "full_answer_rubric_success_count",
    )
    delta = {k: amended[k] - baseline[k] for k in keys}
    eos_restored = amended["mean_eos_eot_p_at_reference_boundary"] >= 0.1
    repetition_reduced = (
        amended["high_repetition_count"] <= baseline["high_repetition_count"] - 8
        and amended["mean_trigram_repetition_rate"] < baseline["mean_trigram_repetition_rate"]
    )
    heldout_preserved = (
        delta["holdout_reference_nll"] <= 0.1
        and delta["holdout_reference_top1"] >= -0.02
    )
    if eos_restored and repetition_reduced and heldout_preserved:
        interpretation = "eos_supervision_restored_termination_and_reduced_repetition_without_heldout_loss"
    elif eos_restored and repetition_reduced:
        interpretation = "eos_helped_generation_but_heldout_learning_regressed"
    elif eos_restored:
        interpretation = "eos_probability_restored_without_sufficient_repetition_recovery"
    else:
        interpretation = "explicit_eos_supervision_did_not_restore_termination"
    return {
        "experiment": "v14a10_eos_supervision_ablation",
        "interpretation": interpretation,
        "baseline_arm": no_eos["arm"],
        "intervention_arm": yes_eos["arm"],
        "controls": {
            "parent": "same exact sha256-verified canonical v14a2 parent",
            "tokenizer": "same tokenizer-v9 and canonical eos id",
            "train_rows": "same 28 target chosen rows, all answer text unchanged",
            "batch_order_verified_equal": True,
            "optimizer": "fresh AdamW each arm",
            "learning_rate": PEAK_LR,
            "training_updates": TOTAL_UPDATES,
            "grad_clip": GRAD_CLIP,
            "only_training_difference": "one extra eos token label after each chosen answer",
        },
        "decision_thresholds": {
            "min_mean_heldout_end_token_probability": 0.1,
            "min_high_repetition_count_reduction": 8,
            "max_heldout_nll_regression": 0.1,
            "max_heldout_top1_regression": 0.02,
        },
        "checks": {
            "eos_probability_restored": eos_restored,
            "repetition_reduced": repetition_reduced,
            "heldout_reference_learning_preserved": heldout_preserved,
        },
        "u40_absolute_deltas_eos_minus_no_eos": delta,
        "arms": {
            no_eos["arm"]: {k: v for k, v in no_eos.items() if k != "batch_sequence_ids"},
            yes_eos["arm"]: {k: v for k, v in yes_eos.items() if k != "batch_sequence_ids"},
        },
        "limitations": [
            "First sentence is scored by automatic keyword rubric, not human judgment.",
            "EOS probability is after full teacher-forced reference answer.",
            "EOS CE changes the mean token-loss normalization by adding one target per row.",
            "Small one-route, one-seed ablation is not promotion evidence.",
        ],
        "promotion_attempted": False,
        "checkpoint_saved": False,
    }


def run(args) -> None:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v14a10 fixed CUDA worker required")
    if (
        args.seed != base.SEED or args.batch_rows != 4 or args.max_len != base.MAX_LEN
        or args.weight_decay != 0.0 or args.warmup_updates != 20
        or args.min_lr_ratio != 0.2 or args.max_gen_tokens != 72
        or args.min_eval_words != 3 or args.lr != 2e-8
    ):
        raise ValueError("v14a10 may not vary frozen v14a8 controls")
    base.validate_local_files(verify_checkpoint_sha256=args.verify_parent_sha256)
    contract = base.training_contract()
    raw, holdout_all, family_summary = base.ensure_family_datasets(contract)
    behavior = pulse._repaired_behavior_rows(raw)
    train_rows = pulse.select_target_rows(
        behavior, seed=args.seed + 701, rows_per_family=pulse.TARGET_ROWS_PER_FAMILY,
    )
    holdout = [dict(row) for row in holdout_all if row["route"] == pulse.TARGET_ROUTE]
    if len(train_rows) != 28 or len(holdout) != 32:
        raise RuntimeError("Invalid frozen train/holdout cardinality")
    if {str(row["id"]) for row in train_rows} & {str(row["id"]) for row in holdout}:
        raise RuntimeError("Heldout contamination by train rows")
    tok = Tokenizer.from_file(str(base.TOKENIZER_PATH))
    contract_t = base._tokenizer_contract(tok)
    if not contract_t["passed"] or tok.get_vocab_size() != base.EXPECTED_VOCAB_SIZE:
        raise RuntimeError("tokenizer-v9 verification failed")
    if {
        k: tok.token_to_id(k) for k in base.EXPECTED_SPECIAL_IDS
    } != base.EXPECTED_SPECIAL_IDS:
        raise RuntimeError("Special token IDs changed")
    history = base._load_historical_eval_module()
    eval_args = base.configure_history_args(history, args)
    special = history.special_ids(tok, eval_args)
    if base.EXPECTED_SPECIAL_IDS[END_TOKEN] not in special["eos_ids"]:
        raise RuntimeError("Chosen EOS not supported by current greedy stop criterion")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prev.write_json(output_dir / "run_config.json", {
        "name": "v14a10_eos_supervision_ablation",
        "parent": base.canonical_parent_reference(),
        "policy_contract": str(CONTRACT_PATH),
        "train_ids": [str(x["id"]) for x in train_rows],
        "holdout_ids": [str(x["id"]) for x in holdout],
        "family_manifest": family_summary,
        "tokenizer_contract": contract_t,
        "arms": list(ARMS),
        "single_changing_training_variable": "explicit_eos_label",
        "no_checkpoint_promotion": True,
        "probe_updates": list(PROBE_UPDATES),
    })
    arms = [
        run_one_arm(
            arm_name=arm, target_rows=train_rows, target_holdout=holdout, tok=tok,
            history=history, eval_args=eval_args, special=special, args=args,
            output_dir=output_dir,
        )
        for arm in ARMS
    ]
    decision = verdict(arms[0], arms[1])
    prev.write_json(output_dir / "v14a10_eos_ablation_decision.json", decision)
    prev.write_json(output_dir / "run_summary.json", {
        "experiment": decision["experiment"],
        "interpretation": decision["interpretation"],
        "diagnostic_only": True,
        "promotion_attempted": False,
        "timestamp": time.time(),
    })
    print("[done] interpretation=" + decision["interpretation"], flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = prev.build_argparser()
    parser.set_defaults(output_dir=str(OUTPUT_DIR))
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
