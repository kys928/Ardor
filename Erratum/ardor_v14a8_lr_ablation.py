#!/usr/bin/env python3
"""v14a8 single-factor learning-rate ablation: exact v14a7 clip_0p5 target pulse.

Every arm starts from the canonical v14a2 parent with a fresh AdamW state,
uses the identical v14a6/v14a7 28 target rows and 40 batch updates, same
continuation-CE objective, architecture, tokenizer and fixed gradient clip 0.5.
Only the peak LR scale changes; warmup/cosine schedule SHAPE stays fixed.
Diagnostic-only: no checkpoint promotion and no broad training changes.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch
from tokenizers import Tokenizer

from Erratum import ardor_v14a4_family_balanced_trainer as base
from Erratum import ardor_v14a6_learning_dynamics_probe as v14a6
from Erratum.v14a4_diagnostics import compact_clean, evaluate_clean_family

OUTPUT_DIR = Path(
    "/workspace/Ardor/training/runs/"
    "sft_v14a8_lr_ablation_v14a7_pulse_u40"
)
POLICY_CONTRACT_PATH = Path(
    "/opt/Ardor/Erratum/v14a8_lr_ablation_contract_20261016.json"
)

PULSE_UPDATES = 40
PROBE_UPDATES = (0, 1, 5, 10, 20, 30, 40)
V14A6_LR_HORIZON = 100

ARMS: tuple[dict[str, Any], ...] = (
    {"name": "lr_2e-8", "lr": 2e-8},
    {"name": "lr_2e-7", "lr": 2e-7},
    {"name": "lr_2e-6", "lr": 2e-6},
    {"name": "lr_2e-5", "lr": 2e-5},
)
FIXED_GRAD_CLIP = 0.5


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_snapshot(
    *,
    arm_name: str,
    update: int,
    model,
    tok: Tokenizer,
    target_rows: Sequence[dict[str, Any]],
    target_holdout: Sequence[dict[str, Any]],
    device: torch.device,
    history,
    eval_args,
    special,
    args,
    arm_dir: Path,
) -> dict[str, Any]:
    exact = v14a6.score_exact_targets(
        model,
        tok,
        target_rows,
        device,
        max_len=args.max_len,
    )
    semantic_full = evaluate_clean_family(
        model,
        tok,
        target_holdout,
        device,
        history,
        eval_args,
        special,
        name=f"v14a8_{arm_name}_gradient_clipping_u{update}",
    )
    write_json(
        arm_dir / f"gradient_clipping_outputs_u{update:04d}.json",
        semantic_full["outputs"],
    )
    semantic = compact_clean(semantic_full)
    snapshot = {
        "arm": arm_name,
        "update": update,
        "exact": exact,
        "semantic": semantic,
    }
    print(
        f"[probe] arm={arm_name} u={update:04d} "
        f"exact_nll={exact['nll']:.6f} "
        f"exact_top1={exact['top1_rate']:.6f} "
        f"semantic_macro={semantic['family_macro_success_rate']:.6f}"
    )
    return snapshot


def arm_result(
    *,
    arm: dict[str, Any],
    baseline: dict[str, Any],
    final: dict[str, Any],
    train_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_nll = float(baseline["exact"]["nll"])
    final_nll = float(final["exact"]["nll"])
    baseline_top1 = float(baseline["exact"]["top1_rate"])
    final_top1 = float(final["exact"]["top1_rate"])
    nll_gain = baseline_nll - final_nll
    top1_gain = final_top1 - baseline_top1
    nll_threshold = max(0.05, 0.01 * baseline_nll)
    learned = nll_gain >= nll_threshold or top1_gain >= 0.02

    clip_scales = [
        float(row["clip_scale"])
        for row in train_records
        if row["clip_scale"] is not None
    ]
    clipped_updates = sum(
        1
        for row in train_records
        if row["clip_scale"] is not None and float(row["clip_scale"]) < 0.999999
    )
    raw_norms = [float(row["raw_grad_norm"]) for row in train_records]

    return {
        "arm": str(arm["name"]),
        "lr": float(arm["lr"]),
        "max_norm": FIXED_GRAD_CLIP,
        "baseline_exact_nll": baseline_nll,
        "final_exact_nll": final_nll,
        "exact_nll_gain": nll_gain,
        "baseline_exact_top1": baseline_top1,
        "final_exact_top1": final_top1,
        "exact_top1_gain": top1_gain,
        "baseline_semantic_macro_success": float(
            baseline["semantic"]["family_macro_success_rate"]
        ),
        "final_semantic_macro_success": float(
            final["semantic"]["family_macro_success_rate"]
        ),
        "token_learning_supported": learned,
        "decision_thresholds": {
            "minimum_nll_gain": nll_threshold,
            "minimum_top1_gain": 0.02,
        },
        "gradient_diagnostics": {
            "updates": len(train_records),
            "clipped_updates": clipped_updates,
            "clipped_fraction": clipped_updates / max(1, len(train_records)),
            "mean_raw_grad_norm": sum(raw_norms) / max(1, len(raw_norms)),
            "min_raw_grad_norm": min(raw_norms) if raw_norms else None,
            "max_raw_grad_norm": max(raw_norms) if raw_norms else None,
            "mean_clip_scale_when_defined": (
                sum(clip_scales) / len(clip_scales) if clip_scales else None
            ),
            "min_clip_scale_when_defined": min(clip_scales) if clip_scales else None,
        },
    }


def build_decision(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(row["arm"]): row for row in results}
    baseline = by_name["lr_2e-8"]
    # Ensure the positive control reproduces the original v14a7 pulse under clip=0.5.
    if abs(float(baseline["final_exact_nll"]) - 5.196875697090512) > 0.005:
        raise RuntimeError("v14a8 baseline did not reproduce v14a7 u40 exact NLL")
    if abs(float(baseline["final_exact_top1"]) - 0.27301587507365244) > 0.005:
        raise RuntimeError("v14a8 baseline did not reproduce v14a7 u40 exact top-1")
    higher = [by_name[name] for name in ("lr_2e-7", "lr_2e-6", "lr_2e-5")]
    recovered = [row for row in higher if row["token_learning_supported"]]

    if not baseline["token_learning_supported"] and recovered:
        primary = "lr_scale_bottleneck_supported"
    elif not baseline["token_learning_supported"]:
        primary = "lr_scale_alone_did_not_restore_learning"
    else:
        primary = "baseline_learning_failure_not_replicated"

    best_nll = max(results, key=lambda row: float(row["exact_nll_gain"]))
    best_top1 = max(results, key=lambda row: float(row["exact_top1_gain"]))
    return {
        "experiment": "v14a8_lr_ablation_v14a7_pulse_u40",
        "primary_interpretation": primary,
        "interpretation_strength": "controlled single-factor diagnostic evidence, not causal proof",
        "baseline_arm": "lr_2e-8",
        "recovered_higher_lr_arms": [str(row["arm"]) for row in recovered],
        "best_nll_gain_arm": str(best_nll["arm"]),
        "best_nll_gain": float(best_nll["exact_nll_gain"]),
        "best_top1_gain_arm": str(best_top1["arm"]),
        "best_top1_gain": float(best_top1["exact_top1_gain"]),
        "arms": [
            {**row,
             "nll_gain_relative_to_baseline": float(row["exact_nll_gain"]) - float(baseline["exact_nll_gain"]),
             "top1_gain_relative_to_baseline": float(row["exact_top1_gain"]) - float(baseline["exact_top1_gain"])}
            for row in results
        ],
    }


def run_arm(
    *,
    arm: dict[str, Any],
    target_rows: Sequence[dict[str, Any]],
    target_holdout: Sequence[dict[str, Any]],
    tok: Tokenizer,
    history,
    eval_args,
    special,
    args,
    output_dir: Path,
) -> dict[str, Any]:
    arm_name = str(arm["name"])
    max_norm = FIXED_GRAD_CLIP
    arm_lr = float(arm["lr"])
    arm_dir = output_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = arm_dir / "trajectory.jsonl"
    train_path = arm_dir / "train_steps.jsonl"
    trajectory_path.write_text("", encoding="utf-8")
    train_path.write_text("", encoding="utf-8")

    set_reproducible_seed(args.seed)
    model, strict_load, checkpoint_meta = base._load_canonical_model(args.device)
    cfg = model.model_config()
    for key, expected in base.MODEL_CONFIG.items():
        if cfg.get(key) != expected:
            raise RuntimeError(
                f"Canonical model config mismatch at {key}: "
                f"expected={expected!r} actual={cfg.get(key)!r}"
            )
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("v14a8 requires all model parameters trainable")

    device = torch.device(args.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=arm_lr,
        weight_decay=args.weight_decay,
    )
    sampler = v14a6.TargetPulseSampler(target_rows, seed=args.seed + 801)
    schedule_args = argparse.Namespace(**vars(args))
    schedule_args.lr = arm_lr
    schedule_args.max_updates = V14A6_LR_HORIZON

    write_json(
        arm_dir / "run_config.json",
        {
            "experiment": "v14a8_lr_ablation_v14a7_pulse_u40",
            "arm": arm_name,
            "max_norm": max_norm,
            "canonical_parent": base.canonical_parent_reference(),
            "strict_load": strict_load,
            "canonical_checkpoint_meta": checkpoint_meta,
            "target_row_ids": [str(row["id"]) for row in target_rows],
            "pulse_updates": PULSE_UPDATES,
            "probe_updates": list(PROBE_UPDATES),
            "optimizer": "AdamW",
            "lr": arm_lr,
            "lr_horizon_updates": V14A6_LR_HORIZON,
            "warmup_updates": args.warmup_updates,
            "min_lr_ratio": args.min_lr_ratio,
            "weight_decay": args.weight_decay,
            "objective": "full chosen-answer continuation CE with prompt tokens masked",
            "trainability": "all model parameters",
            "diagnostic_only": True,
        },
    )

    snapshots: dict[int, dict[str, Any]] = {}
    baseline = make_snapshot(
        arm_name=arm_name,
        update=0,
        model=model,
        tok=tok,
        target_rows=target_rows,
        target_holdout=target_holdout,
        device=device,
        history=history,
        eval_args=eval_args,
        special=special,
        args=args,
        arm_dir=arm_dir,
    )
    snapshots[0] = baseline
    append_jsonl(trajectory_path, baseline)

    train_records: list[dict[str, Any]] = []
    for update in range(1, PULSE_UPDATES + 1):
        batch = sampler.next_batch(args.batch_rows)

        model.train()
        lr = base.lr_at(update, schedule_args)
        base.set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        loss, train_stats = base.continuation_ce(
            model,
            tok,
            batch,
            device,
            max_len=args.max_len,
            pad_id=base.EXPECTED_SPECIAL_IDS["<pad>"],
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Nonfinite CE in arm={arm_name} update={update}")
        loss.backward()

        effective_max_norm = math.inf if max_norm is None else float(max_norm)
        raw_grad_norm_t = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            effective_max_norm,
        )
        raw_grad_norm = float(raw_grad_norm_t)
        if not math.isfinite(raw_grad_norm):
            raise RuntimeError(f"Nonfinite gradient norm in arm={arm_name} update={update}")
        if max_norm is None:
            clip_scale = None
        else:
            clip_scale = min(
                1.0,
                float(max_norm) / (raw_grad_norm + 1.0e-6),
            )

        optimizer.step()

        record = {
            "arm": arm_name,
            "update": update,
            "lr": lr,
            "continuation_ce": float(loss.detach().item()),
            "raw_grad_norm": raw_grad_norm,
            "max_norm": max_norm,
            "clip_scale": clip_scale,
            **train_stats,
            "batch_ids": [str(row["id"]) for row in batch],
        }
        train_records.append(record)
        append_jsonl(train_path, record)

        if update == 1 or update % 5 == 0:
            clip_text = "none" if max_norm is None else f"{float(max_norm):g}"
            print(
                f"[train] arm={arm_name} u={update:04d}/{PULSE_UPDATES} "
                f"clip={clip_text} lr={lr:.2e} "
                f"ce={record['continuation_ce']:.5f} "
                f"top1={train_stats['top1_rate']:.3f} "
                f"raw_grad={raw_grad_norm:.3f} "
                f"clip_scale={clip_scale}"
            )

        if update in PROBE_UPDATES:
            snapshot = make_snapshot(
                arm_name=arm_name,
                update=update,
                model=model,
                tok=tok,
                target_rows=target_rows,
                target_holdout=target_holdout,
                device=device,
                history=history,
                eval_args=eval_args,
                special=special,
                args=args,
                arm_dir=arm_dir,
            )
            snapshots[update] = snapshot
            append_jsonl(trajectory_path, snapshot)

    final = snapshots[PULSE_UPDATES]
    result = arm_result(
        arm=arm,
        baseline=baseline,
        final=final,
        train_records=train_records,
    )
    result["batch_sequence_ids"] = [
        str(batch_id)
        for row in train_records
        for batch_id in row["batch_ids"]
    ]
    write_json(arm_dir / "arm_result.json", result)

    del optimizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run(args) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v14a8 training requires CUDA")

    if args.lr != ARMS[0]["lr"]:
        raise ValueError("v14a8 keeps CLI reference LR fixed at 2e-8; arm LR is internal")
    base.validate_local_files(verify_checkpoint_sha256=args.verify_parent_sha256)
    contract = base.training_contract()
    behavior_raw, clean_holdout, family_summary = base.ensure_family_datasets(contract)
    behavior_rows = v14a6._repaired_behavior_rows(behavior_raw)

    target_rows = v14a6.select_target_rows(
        behavior_rows,
        seed=args.seed + 701,
        rows_per_family=v14a6.TARGET_ROWS_PER_FAMILY,
    )
    target_holdout = [
        dict(row)
        for row in clean_holdout
        if row["route"] == v14a6.TARGET_ROUTE
    ]
    target_ids = [str(row["id"]) for row in target_rows]
    holdout_ids = {str(row["id"]) for row in target_holdout}
    if set(target_ids) & holdout_ids:
        raise RuntimeError("Target pulse rows overlap the clean semantic holdout")

    tok = Tokenizer.from_file(str(base.TOKENIZER_PATH))
    tok_contract = base._tokenizer_contract(tok)
    if not tok_contract["passed"] or tok.get_vocab_size() != base.EXPECTED_VOCAB_SIZE:
        raise RuntimeError(f"Tokenizer-v9 contract failed: {tok_contract}")
    if {
        token: tok.token_to_id(token)
        for token in base.EXPECTED_SPECIAL_IDS
    } != base.EXPECTED_SPECIAL_IDS:
        raise RuntimeError("Tokenizer-v9 special-token IDs changed")

    history = base._load_historical_eval_module()
    eval_args = base.configure_history_args(history, args)
    special = history.special_ids(tok, eval_args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_config.json",
        {
            "experiment": "v14a8_lr_ablation_v14a7_pulse_u40",
            "trainer": "Erratum/ardor_v14a8_lr_ablation.py",
            "policy_contract": str(POLICY_CONTRACT_PATH),
            "parent_data_contract": str(base.CONTRACT_PATH),
            "canonical_parent": base.canonical_parent_reference(),
            "tokenizer_contract": {
                "path": str(base.TOKENIZER_PATH),
                "vocab_size": base.EXPECTED_VOCAB_SIZE,
                "special_ids": dict(base.EXPECTED_SPECIAL_IDS),
            },
            "family_dataset_manifest": family_summary,
            "target_route": v14a6.TARGET_ROUTE,
            "target_rows_per_family": v14a6.TARGET_ROWS_PER_FAMILY,
            "target_row_ids": target_ids,
            "target_holdout_rows": len(target_holdout),
            "fixed_grad_clip": FIXED_GRAD_CLIP,
            "pulse_updates": PULSE_UPDATES,
            "probe_updates": list(PROBE_UPDATES),
            "arms": list(ARMS),
            "optimizer": "AdamW",
            "lr_reference": args.lr,
            "lr_horizon_updates": V14A6_LR_HORIZON,
            "warmup_updates": args.warmup_updates,
            "min_lr_ratio": args.min_lr_ratio,
            "weight_decay": args.weight_decay,
            "objective": "full chosen-answer continuation CE with prompt tokens masked",
            "batch_rows": args.batch_rows,
            "grad_clip": FIXED_GRAD_CLIP,
            "diagnostic_only": True,
            "promotion": False,
        },
    )

    results: list[dict[str, Any]] = []
    reference_batch_sequence: list[str] | None = None
    for arm in ARMS:
        result = run_arm(
            arm=arm,
            target_rows=target_rows,
            target_holdout=target_holdout,
            tok=tok,
            history=history,
            eval_args=eval_args,
            special=special,
            args=args,
            output_dir=output_dir,
        )
        sequence = list(result.pop("batch_sequence_ids"))
        if reference_batch_sequence is None:
            reference_batch_sequence = sequence
        elif sequence != reference_batch_sequence:
            raise RuntimeError(
                f"Batch-order mismatch in arm {result['arm']}; "
                "v14a8 requires identical v14a6 pulse ordering"
            )
        results.append(result)

    decision = build_decision(results)
    decision["controls"] = {
        "same_canonical_parent": True,
        "same_tokenizer": True,
        "same_target_rows": True,
        "same_batch_order": True,
        "same_optimizer": "AdamW",
        "same_lr_schedule_shape": True,
        "same_fixed_grad_clip": FIXED_GRAD_CLIP,
        "same_objective": True,
        "same_architecture": True,
        "only_ablation_variable": "peak_learning_rate",
    }
    write_json(output_dir / "v14a8_lr_ablation_decision.json", decision)
    write_json(
        output_dir / "run_summary.json",
        {
            "experiment": decision["experiment"],
            "diagnostic_only": True,
            "promotion_attempted": False,
            "canonical_parent": base.canonical_parent_reference(),
            "policy_contract": str(POLICY_CONTRACT_PATH),
            "decision": decision,
            "created_at_unix": time.time(),
        },
    )
    print(
        "[done] primary_interpretation="
        + str(decision["primary_interpretation"])
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=base.SEED)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max-len", type=int, default=base.MAX_LEN)
    parser.add_argument("--max-gen-tokens", type=int, default=72)
    parser.add_argument("--min-eval-words", type=int, default=3)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--eval-loss-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-8)
    parser.add_argument("--min-lr-ratio", type=float, default=0.2)
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-rows", type=int, default=4)
    parser.add_argument(
        "--verify-parent-sha256",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.batch_rows != 4:
        raise ValueError("v14a8 is fixed to the v14a6 batch_rows=4 contract")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
