#!/usr/bin/env python3
"""Paired v14a4 format-only A/B from the canonical v14a2 parent.

This is an exploratory serialization-isolation experiment. The recorded conversation-audit
materiality gate did not fire, so this runner must not be interpreted as a gate-mandated repair.

Both arms use the same canonical parent, data rows, deterministic row order, chosen answers,
optimizer, LR schedule, batch size, update budget, masking rule, and evaluation rows. The sole
training variable is prompt serialization:

A: current v14a4 training serialization (prompt + "\\n-", no tokenizer postprocessor)
B: exact current single-turn runtime role/system serialization with tokenizer postprocessor

Each arm is trained for exactly 100 updates from a fresh canonical v14a2 load and evaluated in a
2x2 design under both training-matched and exact single-turn runtime serialization.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import Erratum.ardor_v14a4_conversation_audit as audit
from Erratum.ardor_v14a4_family_balanced_trainer import (
    BALANCED_PATH,
    TARGETED_PATH,
    FamilyBalancedSampler,
    ensure_family_datasets,
    lr_at,
    normalize_behavior_rows,
    normalize_replay_rows,
    read_jsonl,
    required_directed_edges,
    training_contract,
    verify_fixed_dataset,
)
from Erratum.canonical_contract_v14a2 import (
    EXPECTED_SPECIAL_IDS,
    MODEL_CONFIG,
    TOKENIZER_PATH,
    canonical_parent_reference,
    validate_local_files,
)
from Erratum.canonical_eval_v14a2 import (
    _load_canonical_model,
    _load_historical_eval_module,
    _tokenizer_contract,
)
from Erratum.v14a4_diagnostics import clean_eval_prompt

OUTPUT_DIR = Path("/workspace/Ardor/training/runs/v14a4_format_only_ab_u100")
MAX_UPDATES = 100
BATCH_ROWS = 4
MAX_LEN = 768
SEED = 928
LR = 2e-8
MIN_LR_RATIO = 0.2
WARMUP_UPDATES = 20
WEIGHT_DECAY = 0.0
GRAD_CLIP = 0.5


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def stable_json_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def raw_prompt_from_training(prompt: str) -> str:
    text = str(prompt)
    if not text.endswith("\n-"):
        raise RuntimeError(f"Expected normalized training prompt to end with \\n-: {text!r}")
    return text[:-2]


def training_prompt_ids(tok: Tokenizer, row: dict[str, Any]) -> list[int]:
    return list(tok.encode(str(row["prompt"]), add_special_tokens=False).ids)


def runtime_prompt_ids(tok: Tokenizer, row: dict[str, Any]) -> list[int]:
    raw_prompt = raw_prompt_from_training(str(row["prompt"]))
    serialized = audit.runtime_chat_prompt(raw_prompt)
    return list(tok.encode(serialized, add_special_tokens=True).ids)


def chosen_ids(tok: Tokenizer, row: dict[str, Any]) -> list[int]:
    return list(tok.encode(str(row["chosen"]), add_special_tokens=False).ids)


def continuation_ce_with_prompt_encoder(
    model,
    tok: Tokenizer,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    prompt_encoder: Callable[[Tokenizer, dict[str, Any]], list[int]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    examples: list[tuple[list[int], list[int]]] = []
    source_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()

    for row in rows:
        pids = prompt_encoder(tok, row)
        cids = chosen_ids(tok, row)
        if not pids:
            raise RuntimeError(f"Empty prompt tokens for {row['id']}")
        if not cids:
            raise RuntimeError(f"Empty chosen continuation for {row['id']}")
        if len(cids) >= MAX_LEN:
            raise RuntimeError(f"Chosen continuation exceeds max_len for {row['id']}")
        pids = pids[-max(1, MAX_LEN - len(cids)):]
        ids = pids + cids
        xids = ids[:-1]
        labels = [-100] * len(xids)
        start = len(pids) - 1
        labels[start:start + len(cids)] = cids
        if len(labels) != len(xids) or not any(label != -100 for label in labels):
            raise RuntimeError(f"Invalid continuation mask for {row['id']}")
        examples.append((xids, labels))
        source_counts[str(row["source"])] += 1
        route_counts[str(row["route"])] += 1
        family_counts[str(row["semantic_family"])] += 1

    width = max(len(xids) for xids, _ in examples)
    x = torch.full(
        (len(examples), width), EXPECTED_SPECIAL_IDS["<pad>"], dtype=torch.long, device=device
    )
    y = torch.full((len(examples), width), -100, dtype=torch.long, device=device)
    for i, (xids, labels) in enumerate(examples):
        x[i, :len(xids)] = torch.tensor(xids, dtype=torch.long, device=device)
        y[i, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)

    logits = model(x).float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
    mask = y != -100
    predictions = torch.argmax(logits, dim=-1)
    top1 = float((predictions[mask] == y[mask]).float().mean().detach().item())
    return loss, {
        "tokens": int(mask.sum().item()),
        "top1": top1,
        "sources": dict(source_counts),
        "routes": dict(route_counts),
        "families": dict(family_counts),
        "max_sequence_tokens": int(width + 1),
    }


def build_shared_schedule(training_rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    sampler = FamilyBalancedSampler(training_rows, SEED + 101)
    schedule: list[list[dict[str, Any]]] = []
    for _update in range(1, MAX_UPDATES + 1):
        schedule.append([dict(row) for row in sampler.next_batch(BATCH_ROWS)])
    return schedule


def compact_schedule(schedule: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for update, batch in enumerate(schedule, 1):
        output.append({
            "update": update,
            "rows": [
                {
                    "id": str(row["id"]),
                    "route": str(row["route"]),
                    "semantic_family": str(row["semantic_family"]),
                    "source": str(row["source"]),
                    "chosen_sha256": hashlib.sha256(str(row["chosen"]).encode("utf-8")).hexdigest(),
                }
                for row in batch
            ],
        })
    return output


def prompt_trace(tok: Tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    raw_prompt = raw_prompt_from_training(str(row["prompt"]))
    runtime_serialized = audit.runtime_chat_prompt(raw_prompt)
    a_ids = training_prompt_ids(tok, row)
    b_ids = runtime_prompt_ids(tok, row)
    cids = chosen_ids(tok, row)
    return {
        "id": str(row["id"]),
        "route": str(row["route"]),
        "source": str(row["source"]),
        "chosen_sha256": hashlib.sha256(str(row["chosen"]).encode("utf-8")).hexdigest(),
        "chosen_ids": cids,
        "arm_a": {
            "serialization": str(row["prompt"]),
            "add_special_tokens": False,
            "prompt_ids": a_ids,
            "prompt_tokens": [tok.id_to_token(i) for i in a_ids],
        },
        "arm_b": {
            "serialization": runtime_serialized,
            "add_special_tokens": True,
            "prompt_ids": b_ids,
            "prompt_tokens": [tok.id_to_token(i) for i in b_ids],
        },
    }


def eval_both_formats(model, tok: Tokenizer, holdout: Sequence[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    history = _load_historical_eval_module()
    audit._HISTORY = history
    eval_args = audit.configure_eval_args(history)
    special = history.special_ids(tok, eval_args)

    training = audit.evaluate_mode(
        model,
        tok,
        holdout,
        device,
        history,
        eval_args,
        special,
        name="training_matched",
        formatter=clean_eval_prompt,
        add_special_tokens=False,
        suppress_stop_tokens_until=0,
    )
    runtime = audit.evaluate_mode(
        model,
        tok,
        holdout,
        device,
        history,
        eval_args,
        special,
        name="runtime_exact_single_turn",
        formatter=lambda row: audit.runtime_chat_prompt(str(row["prompt"])),
        add_special_tokens=True,
        suppress_stop_tokens_until=audit.RUNTIME_MIN_NEW_TOKENS,
    )
    return {"training_matched": training, "runtime_exact_single_turn": runtime}


def save_arm_checkpoint(path: Path, model, arm: str, schedule_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "designation": "v14a4_format_only_ab_u100",
            "arm": arm,
            "update": MAX_UPDATES,
            "canonical_parent": canonical_parent_reference(),
            "shared_schedule_sha256": schedule_sha256,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def train_arm(
    arm: str,
    tok: Tokenizer,
    schedule: Sequence[Sequence[dict[str, Any]]],
    holdout: Sequence[dict[str, Any]],
    schedule_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    if arm not in {"A_training_format", "B_runtime_format"}:
        raise ValueError(arm)
    prompt_encoder = training_prompt_ids if arm == "A_training_format" else runtime_prompt_ids

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, strict_load, checkpoint_meta = _load_canonical_model("cuda")
    cfg = model.model_config()
    for key, expected in MODEL_CONFIG.items():
        if cfg.get(key) != expected:
            raise RuntimeError(
                f"Canonical model config mismatch at {key}: expected={expected!r} actual={cfg.get(key)!r}"
            )
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("A/B requires the same all-parameter trainability as v14a4")

    device = torch.device("cuda")
    args = SimpleNamespace(
        lr=LR,
        min_lr_ratio=MIN_LR_RATIO,
        warmup_updates=WARMUP_UPDATES,
        max_updates=MAX_UPDATES,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = arm_dir / "training_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    for update, batch in enumerate(schedule, 1):
        current_lr = lr_at(update, args)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        loss, stats = continuation_ce_with_prompt_encoder(
            model, tok, batch, device, prompt_encoder=prompt_encoder
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        record = {
            "update": update,
            "arm": arm,
            "lr": current_lr,
            "loss": float(loss.detach().item()),
            "grad_norm": float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
            **stats,
            "row_ids": [str(row["id"]) for row in batch],
        }
        append_jsonl(metrics_path, record)
        if update == 1 or update % 5 == 0:
            print(
                f"[{arm}] u={update:04d}/{MAX_UPDATES} lr={current_lr:.3e} "
                f"loss={record['loss']:.5f} top1={record['top1']:.3f} grad={record['grad_norm']:.3f}",
                flush=True,
            )

    checkpoint_path = arm_dir / "u100_model_state.pt"
    save_arm_checkpoint(checkpoint_path, model, arm, schedule_sha256)
    print(f"[{arm}] evaluating 2x2 formats", flush=True)
    evaluation = eval_both_formats(model, tok, holdout, device)
    write_json(arm_dir / "training_matched_outputs.json", evaluation["training_matched"]["outputs"])
    write_json(arm_dir / "runtime_exact_single_turn_outputs.json", evaluation["runtime_exact_single_turn"]["outputs"])
    compact_eval = {
        key: audit.compact(value)
        for key, value in evaluation.items()
    }
    result = {
        "arm": arm,
        "serialization": (
            "v14a4 prompt + newline-dash; add_special_tokens=false"
            if arm == "A_training_format"
            else "exact single-turn runtime role/system frame; add_special_tokens=true"
        ),
        "strict_load": strict_load,
        "checkpoint_meta": checkpoint_meta,
        "checkpoint_path": str(checkpoint_path),
        "shared_schedule_sha256": schedule_sha256,
        "evaluation": compact_eval,
    }
    write_json(arm_dir / "arm_summary.json", result)

    del optimizer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-parent-sha256",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Full-hash the frozen canonical v14a2 parent before either arm.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v14a4 format A/B")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local_contract = validate_local_files(verify_checkpoint_sha256=bool(args.verify_parent_sha256))
    contract = training_contract()
    behavior_raw, holdout, dataset_summary = ensure_family_datasets(contract)

    replay_manifest = {
        "balanced": verify_fixed_dataset(
            BALANCED_PATH,
            contract["data"]["balanced_replay"]["sha256"],
            int(contract["data"]["balanced_replay"]["rows"]),
        ),
        "targeted": verify_fixed_dataset(
            TARGETED_PATH,
            contract["data"]["targeted_collision_replay"]["sha256"],
            int(contract["data"]["targeted_collision_replay"]["rows"]),
        ),
    }

    balanced_raw = read_jsonl(BALANCED_PATH)
    targeted_raw = read_jsonl(TARGETED_PATH)
    training_rows: list[dict[str, Any]] = []
    training_rows.extend(normalize_behavior_rows(behavior_raw))
    training_rows.extend(normalize_replay_rows(balanced_raw, "balanced"))
    training_rows.extend(
        normalize_replay_rows(
            targeted_raw,
            "targeted",
            required_edges=required_directed_edges(),
        )
    )

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    tok_contract = _tokenizer_contract(tok)
    if not tok_contract["passed"]:
        raise RuntimeError(f"Tokenizer contract failed: {tok_contract['mismatches']}")
    observed_special = {token: tok.token_to_id(token) for token in EXPECTED_SPECIAL_IDS}
    if observed_special != EXPECTED_SPECIAL_IDS:
        raise RuntimeError(f"Tokenizer special IDs changed: {observed_special}")

    schedule = build_shared_schedule(training_rows)
    compact = compact_schedule(schedule)
    schedule_sha256 = stable_json_sha256(compact)
    write_json(OUTPUT_DIR / "shared_schedule.json", compact)

    trace_rows: list[dict[str, Any]] = []
    seen_routes: set[str] = set()
    for batch in schedule:
        for row in batch:
            route = str(row["route"])
            if route not in seen_routes:
                trace_rows.append(prompt_trace(tok, row))
                seen_routes.add(route)
    write_json(OUTPUT_DIR / "prompt_serialization_trace.json", trace_rows)

    manifest = {
        "designation": "v14a4_format_only_ab_u100",
        "interpretation": "exploratory format isolation; conversation-audit materiality gate was false",
        "canonical_parent": canonical_parent_reference(),
        "local_contract_validation": local_contract,
        "model_config": MODEL_CONFIG,
        "tokenizer_special_ids": observed_special,
        "data": dataset_summary,
        "replay": replay_manifest,
        "training_rows": len(training_rows),
        "shared_schedule_sha256": schedule_sha256,
        "updates": MAX_UPDATES,
        "batch_rows": BATCH_ROWS,
        "optimizer": {
            "family": "AdamW",
            "lr": LR,
            "schedule": "cosine",
            "min_lr_ratio": MIN_LR_RATIO,
            "warmup_updates": WARMUP_UPDATES,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
        },
        "fixed_variables": [
            "canonical parent",
            "training rows",
            "row order",
            "chosen continuation text and tokenization",
            "optimizer",
            "learning-rate schedule",
            "batch size",
            "update budget",
            "full-parameter trainability",
            "continuation CE masking",
            "evaluation rows",
        ],
        "sole_training_variable": "prompt serialization/tokenization format",
        "arm_a": "v14a4 prompt + newline-dash, add_special_tokens=false",
        "arm_b": "exact runtime single-turn role/system prompt, add_special_tokens=true",
    }
    write_json(OUTPUT_DIR / "experiment_manifest.json", manifest)

    arm_a = train_arm(
        "A_training_format", tok, schedule, holdout, schedule_sha256, OUTPUT_DIR
    )
    arm_b = train_arm(
        "B_runtime_format", tok, schedule, holdout, schedule_sha256, OUTPUT_DIR
    )

    comparison = {
        "designation": "v14a4_format_only_ab_u100",
        "shared_schedule_sha256": schedule_sha256,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "matrix": {
            "A_train__training_eval": arm_a["evaluation"]["training_matched"],
            "A_train__runtime_eval": arm_a["evaluation"]["runtime_exact_single_turn"],
            "B_train__training_eval": arm_b["evaluation"]["training_matched"],
            "B_train__runtime_eval": arm_b["evaluation"]["runtime_exact_single_turn"],
        },
    }
    write_json(OUTPUT_DIR / "ab_summary.json", comparison)
    print(f"[done] summary={OUTPUT_DIR / 'ab_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
