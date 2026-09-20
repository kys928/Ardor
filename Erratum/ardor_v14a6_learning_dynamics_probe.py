#!/usr/bin/env python3
"""v14a6 target-pulse / interference-chase learning-dynamics probe.

Scientific question:
1. Can clean gradient-clipping supervision move the exact chosen continuations at all?
2. If it can, is that learning overwritten when gradient-clipping examples disappear?
3. If exact continuations improve and survive, does that learning transfer to clean semantic paraphrases?

This is diagnostic-only. It keeps the canonical v14a2 parent, tokenizer-v9 contract,
all-parameter trainability, AdamW, continuation CE, LR scale, weight decay, and gradient
clipping used by v14a5. The intervention is only the temporal ordering of already-clean
authored behavior supervision.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch
from tokenizers import Tokenizer

from Erratum import ardor_v14a4_family_balanced_trainer as base
from Erratum import ardor_v14a5_signal_first_trainer as v14a5
from Erratum.build_v14a4_family_data import FAMILIES
from Erratum.v14a4_diagnostics import compact_clean, evaluate_clean_family

OUTPUT_DIR = Path(
    "/workspace/Ardor/training/runs/"
    "sft_v14a6_learning_dynamics_target_pulse_chase_u80"
)
POLICY_CONTRACT_PATH = Path(
    "/opt/Ardor/Erratum/v14a6_learning_dynamics_contract_20260920.json"
)

TARGET_ROUTE = "gradient_clipping"
TARGET_ROWS_PER_FAMILY = 4
PULSE_UPDATES = 40
CHASE_UPDATES = 40
TOTAL_UPDATES = PULSE_UPDATES + CHASE_UPDATES
V14A5_LR_HORIZON = 100
PROBE_UPDATES = (0, 1, 5, 10, 20, 30, 40, 45, 50, 60, 70, 80)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def _repaired_behavior_rows(
    raw_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reuse the exact v14a5 authored-behavior normalization/repair."""
    base._V14A5_ORIGINAL_NORMALIZE_BEHAVIOR = base.normalize_behavior_rows
    try:
        return v14a5.normalize_behavior_rows(raw_rows)
    finally:
        delattr(base, "_V14A5_ORIGINAL_NORMALIZE_BEHAVIOR")


def select_target_rows(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    rows_per_family: int = TARGET_ROWS_PER_FAMILY,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for family in FAMILIES[TARGET_ROUTE]:
        pool = [
            dict(row)
            for row in rows
            if row["source"] == "behavior"
            and row["route"] == TARGET_ROUTE
            and row["semantic_family"] == family
        ]
        stable = sum((i + 1) * ord(ch) for i, ch in enumerate(family))
        rng = random.Random(seed + stable)
        rng.shuffle(pool)
        if len(pool) < rows_per_family:
            raise RuntimeError(
                f"Need {rows_per_family} authored rows for {TARGET_ROUTE}/{family}; "
                f"found {len(pool)}"
            )
        selected.extend(pool[:rows_per_family])
    return selected


class TargetPulseSampler:
    """Round-robin gradient-clipping families; intentional recycling for overfit diagnosis."""

    def __init__(self, rows: Sequence[dict[str, Any]], seed: int):
        self.pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["route"] != TARGET_ROUTE:
                raise ValueError("TargetPulseSampler accepts gradient_clipping rows only")
            self.pools[str(row["semantic_family"])].append(dict(row))
        self.families = list(FAMILIES[TARGET_ROUTE])
        self.rngs: dict[str, random.Random] = {}
        self.orders: dict[str, list[int]] = {}
        self.family_cursor = 0
        for family in self.families:
            pool = self.pools.get(family, [])
            if not pool:
                raise RuntimeError(f"Missing target-pulse family {family}")
            stable = sum((i + 1) * ord(ch) for i, ch in enumerate(family))
            rng = random.Random(seed + stable)
            self.rngs[family] = rng
            order = list(range(len(pool)))
            rng.shuffle(order)
            self.orders[family] = order

    def _take(self, family: str) -> dict[str, Any]:
        order = self.orders[family]
        if not order:
            order.extend(range(len(self.pools[family])))
            self.rngs[family].shuffle(order)
        return dict(self.pools[family][order.pop()])

    def next_batch(self, n: int) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        for _ in range(n):
            family = self.families[self.family_cursor % len(self.families)]
            self.family_cursor += 1
            batch.append(self._take(family))
        return batch


class InterferenceChaseSampler:
    """Broad non-target authored behavior, family-balanced and without recycling."""

    def __init__(self, rows: Sequence[dict[str, Any]], seed: int):
        self.pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["source"] != "behavior" or row["route"] == TARGET_ROUTE:
                continue
            key = (str(row["route"]), str(row["semantic_family"]))
            self.pools[key].append(dict(row))
        self.keys = sorted(self.pools)
        self.cursor = 0
        for key in self.keys:
            stable = sum((i + 1) * ord(ch) for i, ch in enumerate("|".join(key)))
            rng = random.Random(seed + stable)
            rng.shuffle(self.pools[key])
            if not self.pools[key]:
                raise RuntimeError(f"Empty interference pool for {key[0]}/{key[1]}")

    def next_batch(self, n: int) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        for _ in range(n):
            found = None
            for _ in range(len(self.keys)):
                key = self.keys[self.cursor % len(self.keys)]
                self.cursor += 1
                pool = self.pools[key]
                if pool:
                    found = dict(pool.pop())
                    break
            if found is None:
                raise RuntimeError(
                    "v14a6 exhausted non-gradient-clipping authored behavior during chase"
                )
            batch.append(found)
        return batch


@torch.inference_mode()
def score_exact_targets(
    model,
    tok: Tokenizer,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    max_len: int,
) -> dict[str, Any]:
    model.eval()
    by_family: dict[str, Any] = {}
    total_tokens = 0
    nll_weighted = 0.0
    top1_weighted = 0.0
    for family in FAMILIES[TARGET_ROUTE]:
        family_rows = [row for row in rows if row["semantic_family"] == family]
        if not family_rows:
            raise RuntimeError(f"No exact target rows for {family}")
        loss, stats = base.continuation_ce(
            model,
            tok,
            family_rows,
            device,
            max_len=max_len,
            pad_id=base.EXPECTED_SPECIAL_IDS["<pad>"],
        )
        tokens = int(stats["tokens"])
        nll = float(loss.detach().item())
        top1 = float(stats["top1_rate"])
        by_family[family] = {
            "rows": len(family_rows),
            "tokens": tokens,
            "nll": nll,
            "top1_rate": top1,
        }
        total_tokens += tokens
        nll_weighted += nll * tokens
        top1_weighted += top1 * tokens
    return {
        "rows": len(rows),
        "tokens": total_tokens,
        "nll": nll_weighted / max(1, total_tokens),
        "top1_rate": top1_weighted / max(1, total_tokens),
        "by_family": by_family,
    }


def _semantic_family_rate(snapshot: dict[str, Any], family: str) -> float:
    key = f"{TARGET_ROUTE}/{family}"
    return float(snapshot["semantic"]["family_success_rate"].get(key, 0.0))


def classify_family(
    baseline: dict[str, Any],
    pulse: dict[str, Any],
    chase: dict[str, Any],
    family: str,
) -> dict[str, Any]:
    b = baseline["exact"]["by_family"][family]
    p = pulse["exact"]["by_family"][family]
    c = chase["exact"]["by_family"][family]
    nll_gain = float(b["nll"]) - float(p["nll"])
    top1_gain = float(p["top1_rate"]) - float(b["top1_rate"])
    nll_threshold = max(0.05, 0.01 * float(b["nll"]))
    token_learning = nll_gain >= nll_threshold or top1_gain >= 0.02

    semantic_base = _semantic_family_rate(baseline, family)
    semantic_pulse = _semantic_family_rate(pulse, family)
    semantic_chase = _semantic_family_rate(chase, family)
    semantic_transfer = semantic_pulse > semantic_base

    nll_forgetting = float(c["nll"]) - float(p["nll"])
    top1_forgetting = float(p["top1_rate"]) - float(c["top1_rate"])
    forget_nll_threshold = max(0.05, 0.25 * max(0.0, nll_gain))
    forget_top1_threshold = max(0.02, 0.25 * max(0.0, top1_gain))
    semantic_forgetting = semantic_transfer and semantic_chase < semantic_pulse
    interference = token_learning and (
        nll_forgetting >= forget_nll_threshold
        or top1_forgetting >= forget_top1_threshold
        or semantic_forgetting
    )

    if not token_learning:
        primary = "target_token_learning_failure_supported"
    elif interference:
        primary = "interference_or_overwrite_supported"
    elif not semantic_transfer:
        primary = "token_learning_without_semantic_generalization_supported"
    else:
        primary = "target_learning_and_semantic_transfer_supported"

    return {
        "primary_interpretation": primary,
        "token_learning_supported": token_learning,
        "semantic_transfer_supported": semantic_transfer,
        "interference_supported": interference,
        "baseline_exact_nll": float(b["nll"]),
        "pulse_exact_nll": float(p["nll"]),
        "chase_exact_nll": float(c["nll"]),
        "exact_nll_gain_during_pulse": nll_gain,
        "exact_nll_forgetting_during_chase": nll_forgetting,
        "baseline_exact_top1": float(b["top1_rate"]),
        "pulse_exact_top1": float(p["top1_rate"]),
        "chase_exact_top1": float(c["top1_rate"]),
        "exact_top1_gain_during_pulse": top1_gain,
        "exact_top1_forgetting_during_chase": top1_forgetting,
        "baseline_semantic_success": semantic_base,
        "pulse_semantic_success": semantic_pulse,
        "chase_semantic_success": semantic_chase,
        "decision_thresholds": {
            "minimum_nll_gain": nll_threshold,
            "minimum_top1_gain": 0.02,
            "forgetting_nll": forget_nll_threshold,
            "forgetting_top1": forget_top1_threshold,
        },
    }


def build_decision(
    baseline: dict[str, Any],
    pulse: dict[str, Any],
    chase: dict[str, Any],
) -> dict[str, Any]:
    family_results = {
        family: classify_family(baseline, pulse, chase, family)
        for family in FAMILIES[TARGET_ROUTE]
    }

    b_nll = float(baseline["exact"]["nll"])
    p_nll = float(pulse["exact"]["nll"])
    c_nll = float(chase["exact"]["nll"])
    b_top1 = float(baseline["exact"]["top1_rate"])
    p_top1 = float(pulse["exact"]["top1_rate"])
    c_top1 = float(chase["exact"]["top1_rate"])
    nll_gain = b_nll - p_nll
    top1_gain = p_top1 - b_top1
    nll_threshold = max(0.05, 0.01 * b_nll)
    token_learning = nll_gain >= nll_threshold or top1_gain >= 0.02

    sem_base = float(baseline["semantic"]["family_macro_success_rate"])
    sem_pulse = float(pulse["semantic"]["family_macro_success_rate"])
    sem_chase = float(chase["semantic"]["family_macro_success_rate"])
    semantic_transfer = sem_pulse > sem_base

    nll_forgetting = c_nll - p_nll
    top1_forgetting = p_top1 - c_top1
    interference = token_learning and (
        nll_forgetting >= max(0.05, 0.25 * max(0.0, nll_gain))
        or top1_forgetting >= max(0.02, 0.25 * max(0.0, top1_gain))
        or (semantic_transfer and sem_chase < sem_pulse)
    )

    if not token_learning:
        primary = "target_token_learning_failure_supported"
    elif interference:
        primary = "interference_or_overwrite_supported"
    elif not semantic_transfer:
        primary = "token_learning_without_semantic_generalization_supported"
    else:
        primary = "target_learning_and_semantic_transfer_supported"

    return {
        "experiment": "v14a6_learning_dynamics_target_pulse_chase_u80",
        "target_route": TARGET_ROUTE,
        "primary_interpretation": primary,
        "interpretation_strength": "controlled diagnostic evidence, not causal proof",
        "hypotheses": {
            "target_token_learning_failure_supported": not token_learning,
            "interference_or_overwrite_supported": interference,
            "token_learning_without_semantic_generalization_supported": (
                token_learning and not semantic_transfer
            ),
        },
        "aggregate": {
            "baseline_exact_nll": b_nll,
            "pulse_exact_nll": p_nll,
            "chase_exact_nll": c_nll,
            "exact_nll_gain_during_pulse": nll_gain,
            "exact_nll_forgetting_during_chase": nll_forgetting,
            "baseline_exact_top1": b_top1,
            "pulse_exact_top1": p_top1,
            "chase_exact_top1": c_top1,
            "exact_top1_gain_during_pulse": top1_gain,
            "exact_top1_forgetting_during_chase": top1_forgetting,
            "baseline_semantic_macro_success": sem_base,
            "pulse_semantic_macro_success": sem_pulse,
            "chase_semantic_macro_success": sem_chase,
        },
        "families": family_results,
    }


def make_snapshot(
    *,
    update: int,
    phase: str,
    model,
    tok: Tokenizer,
    target_rows: Sequence[dict[str, Any]],
    target_holdout: Sequence[dict[str, Any]],
    device: torch.device,
    history,
    eval_args,
    special,
    args,
    output_dir: Path,
) -> dict[str, Any]:
    exact = score_exact_targets(
        model, tok, target_rows, device, max_len=args.max_len
    )
    semantic_full = evaluate_clean_family(
        model,
        tok,
        target_holdout,
        device,
        history,
        eval_args,
        special,
        name=f"v14a6_gradient_clipping_u{update}",
    )
    _write_json(
        output_dir / f"gradient_clipping_outputs_u{update:04d}.json",
        semantic_full["outputs"],
    )
    semantic = compact_clean(semantic_full)
    snapshot = {
        "update": update,
        "phase": phase,
        "exact": exact,
        "semantic": semantic,
    }
    print(
        f"[probe] u={update:04d} phase={phase} "
        f"exact_nll={exact['nll']:.5f} exact_top1={exact['top1_rate']:.3f} "
        f"semantic_macro={semantic['family_macro_success_rate']:.4f}"
    )
    return snapshot


def run(args) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v14a6 training requires CUDA")

    base.validate_local_files(verify_checkpoint_sha256=args.verify_parent_sha256)
    contract = base.training_contract()
    behavior_raw, clean_holdout, family_summary = base.ensure_family_datasets(contract)
    behavior_rows = _repaired_behavior_rows(behavior_raw)

    target_rows = select_target_rows(
        behavior_rows,
        seed=args.seed + 701,
        rows_per_family=TARGET_ROWS_PER_FAMILY,
    )
    target_holdout = [
        dict(row) for row in clean_holdout if row["route"] == TARGET_ROUTE
    ]
    if not target_holdout:
        raise RuntimeError("No gradient_clipping rows in clean family holdout")

    model, strict_load, checkpoint_meta = base._load_canonical_model(args.device)
    cfg = model.model_config()
    for key, expected in base.MODEL_CONFIG.items():
        if cfg.get(key) != expected:
            raise RuntimeError(
                f"Canonical model config mismatch at {key}: "
                f"expected={expected!r} actual={cfg.get(key)!r}"
            )
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("v14a6 requires the v14a5 all-parameter trainability policy")

    tok = Tokenizer.from_file(str(base.TOKENIZER_PATH))
    tok_contract = base._tokenizer_contract(tok)
    if not tok_contract["passed"] or tok.get_vocab_size() != base.EXPECTED_VOCAB_SIZE:
        raise RuntimeError(f"Tokenizer-v9 contract failed: {tok_contract}")
    if {
        token: tok.token_to_id(token) for token in base.EXPECTED_SPECIAL_IDS
    } != base.EXPECTED_SPECIAL_IDS:
        raise RuntimeError("Tokenizer-v9 special-token IDs changed")

    history = base._load_historical_eval_module()
    eval_args = base.configure_history_args(history, args)
    special = history.special_ids(tok, eval_args)
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.jsonl"
    train_path = output_dir / "train_steps.jsonl"
    trajectory_path.write_text("", encoding="utf-8")
    train_path.write_text("", encoding="utf-8")

    target_ids = [str(row["id"]) for row in target_rows]
    target_id_set = set(target_ids)
    holdout_ids = {str(row["id"]) for row in target_holdout}
    if target_id_set & holdout_ids:
        raise RuntimeError("Target pulse rows overlap the clean semantic holdout")

    _write_json(
        output_dir / "run_config.json",
        {
            "experiment": "v14a6_learning_dynamics_target_pulse_chase_u80",
            "trainer": "Erratum/ardor_v14a6_learning_dynamics_probe.py",
            "policy_contract": str(POLICY_CONTRACT_PATH),
            "parent_data_contract": str(base.CONTRACT_PATH),
            "canonical_parent": base.canonical_parent_reference(),
            "strict_load": strict_load,
            "canonical_checkpoint_meta": checkpoint_meta,
            "tokenizer_contract": {
                "path": str(base.TOKENIZER_PATH),
                "vocab_size": base.EXPECTED_VOCAB_SIZE,
                "special_ids": dict(base.EXPECTED_SPECIAL_IDS),
            },
            "family_dataset_manifest": family_summary,
            "target_route": TARGET_ROUTE,
            "target_rows_per_family": TARGET_ROWS_PER_FAMILY,
            "target_row_ids": target_ids,
            "target_holdout_rows": len(target_holdout),
            "pulse_updates": PULSE_UPDATES,
            "chase_updates": CHASE_UPDATES,
            "total_updates": TOTAL_UPDATES,
            "probe_updates": list(PROBE_UPDATES),
            "optimizer": "AdamW",
            "lr": args.lr,
            "lr_horizon_updates": V14A5_LR_HORIZON,
            "warmup_updates": args.warmup_updates,
            "min_lr_ratio": args.min_lr_ratio,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "objective": "full chosen-answer continuation CE with prompt tokens masked",
            "trainability": "all model parameters",
            "diagnostic_only": True,
        },
    )

    pulse_sampler = TargetPulseSampler(target_rows, seed=args.seed + 801)
    chase_sampler = InterferenceChaseSampler(behavior_rows, seed=args.seed + 901)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    schedule_args = argparse.Namespace(**vars(args))
    schedule_args.max_updates = V14A5_LR_HORIZON

    snapshots: dict[int, dict[str, Any]] = {}
    baseline = make_snapshot(
        update=0,
        phase="baseline",
        model=model,
        tok=tok,
        target_rows=target_rows,
        target_holdout=target_holdout,
        device=device,
        history=history,
        eval_args=eval_args,
        special=special,
        args=args,
        output_dir=output_dir,
    )
    snapshots[0] = baseline
    _append_jsonl(trajectory_path, baseline)

    cumulative_phase_rows: Counter[str] = Counter()
    for update in range(1, TOTAL_UPDATES + 1):
        phase = "target_pulse" if update <= PULSE_UPDATES else "interference_chase"
        batch = (
            pulse_sampler.next_batch(args.batch_rows)
            if phase == "target_pulse"
            else chase_sampler.next_batch(args.batch_rows)
        )
        if phase == "interference_chase" and any(
            row["route"] == TARGET_ROUTE for row in batch
        ):
            raise RuntimeError("Gradient-clipping row leaked into interference chase")

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
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        optimizer.step()

        for row in batch:
            cumulative_phase_rows[phase] += 1

        train_record = {
            "update": update,
            "phase": phase,
            "lr": lr,
            "continuation_ce": float(loss.detach().item()),
            "grad_norm": float(grad_norm),
            **train_stats,
            "batch_ids": [str(row["id"]) for row in batch],
        }
        _append_jsonl(train_path, train_record)
        if update == 1 or update % 5 == 0:
            print(
                f"[train] u={update:04d}/{TOTAL_UPDATES} phase={phase} "
                f"lr={lr:.2e} ce={train_record['continuation_ce']:.5f} "
                f"top1={train_stats['top1_rate']:.3f} grad={float(grad_norm):.3f}"
            )

        if update in PROBE_UPDATES:
            snapshot = make_snapshot(
                update=update,
                phase=phase,
                model=model,
                tok=tok,
                target_rows=target_rows,
                target_holdout=target_holdout,
                device=device,
                history=history,
                eval_args=eval_args,
                special=special,
                args=args,
                output_dir=output_dir,
            )
            snapshots[update] = snapshot
            _append_jsonl(trajectory_path, snapshot)

    pulse = snapshots[PULSE_UPDATES]
    chase = snapshots[TOTAL_UPDATES]
    decision = build_decision(baseline, pulse, chase)
    decision["sampling"] = {
        "target_pulse_rows": cumulative_phase_rows["target_pulse"],
        "interference_chase_rows": cumulative_phase_rows["interference_chase"],
        "target_unique_rows": len(target_id_set),
        "target_families": len(FAMILIES[TARGET_ROUTE]),
        "chase_contains_gradient_clipping": False,
    }
    _write_json(output_dir / "v14a6_learning_dynamics_decision.json", decision)
    _write_json(
        output_dir / "run_summary.json",
        {
            "experiment": decision["experiment"],
            "diagnostic_only": True,
            "promotion_attempted": False,
            "canonical_parent": base.canonical_parent_reference(),
            "policy_contract": str(POLICY_CONTRACT_PATH),
            "baseline": baseline,
            "pulse_end": pulse,
            "chase_end": chase,
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
    parser.add_argument("--grad-clip", type=float, default=0.5)
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
        raise ValueError("v14a6 is fixed to the v14a5 batch_rows=4 contract")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
