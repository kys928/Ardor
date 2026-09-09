#!/usr/bin/env python3
"""v14a3 behavior-first semantic landing trainer.

Changes exactly one scientific priority relative to canonical v14a2: the optimizer is driven by
full chosen-answer continuation CE instead of local single-token anti-loop margin loss. Prompt
tokens are excluded from CE. The exact promoted v14a2 u600 optimizer schedule and all-parameter
trainability are preserved. Geometry, fixed-target ranking, loops, and repetition are evaluation
gates only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from Erratum.build_v14a3_behavior_data import build_rows as build_behavior_rows
from Erratum.build_v14a3_behavior_data import validate as validate_behavior_rows
from Erratum.canonical_analysis_v14a2 import ROUTES
from Erratum.canonical_contract_v14a2 import (
    CANONICAL_CHECKPOINT,
    EXPECTED_SPECIAL_IDS,
    EXPECTED_VOCAB_SIZE,
    MODEL_CONFIG,
    TOKENIZER_PATH,
    canonical_parent_reference,
    validate_local_files,
)
from Erratum.canonical_eval_v14a2 import (
    HOLDOUT_PATH,
    _load_canonical_model,
    _load_historical_eval_module,
    _tokenizer_contract,
)
from Erratum.v14a3_diagnostics import (
    PRIORITY_ROUTES,
    calibrated_target_ranking,
    compact_behavior,
    evaluate_balanced_chosen,
    evaluate_behavior,
    evaluate_final_geometry,
    primary_promotion_gate,
    promotion_key,
    secondary_promotion_gate,
)

REPO_ROOT = Path("/workspace/Ardor")
CONTRACT_PATH = Path(__file__).with_name("v14a3_behavior_first_training_contract_20260909.json")
BEHAVIOR_PATH = REPO_ROOT / "training/data/v14_v3/dataset_v14a3_behavior_first.jsonl"
BALANCED_PATH = REPO_ROOT / "training/data/v14_v3/dataset_v3b_route_contrastive_balanced.jsonl"
TARGETED_PATH = REPO_ROOT / "training/data/v14_v3/dataset_v14b3_targeted_route_collisions.jsonl"
OUTPUT_DIR = REPO_ROOT / "training/runs/sft_v14a3_behavior_first_semantic_landing_u600"
MAX_LEN = 768
SEED = 928

# Interleaved counts exactly match the declared 16-row route schedule:
# gradient_clipping=5, dropout=3, rag=2, tokenizer=2, all four guards=1.
ROUTE_SEQUENCE = [
    "gradient_clipping", "dropout", "rag", "tokenizer",
    "gradient_clipping", "checkpoint", "dropout", "correlation",
    "gradient_clipping", "rag", "tokenizer", "direct_answer",
    "gradient_clipping", "dropout", "overfitting", "gradient_clipping",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_contract() -> dict[str, Any]:
    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if value.get("designation") != "v14a3_behavior_first_semantic_landing":
        raise RuntimeError(f"Unexpected v14a3 training contract: {value.get('designation')!r}")
    return value


def verify_fixed_dataset(path: Path, expected_sha256: str, expected_rows: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha256:
        raise RuntimeError(f"Dataset SHA mismatch for {path}: expected={expected_sha256} actual={observed_sha}")
    rows = read_jsonl(path)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Dataset row-count mismatch for {path}: expected={expected_rows} actual={len(rows)}")
    return {"path": str(path), "rows": len(rows), "sha256": observed_sha}


def ensure_behavior_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_holdout = read_jsonl(HOLDOUT_PATH)
    expected = build_behavior_rows()
    expected_summary = validate_behavior_rows(expected, raw_holdout)
    if BEHAVIOR_PATH.exists():
        rows = read_jsonl(BEHAVIOR_PATH)
        observed_summary = validate_behavior_rows(rows, raw_holdout)
        if rows != expected:
            raise RuntimeError(
                "Existing v14a3 behavior dataset does not exactly match the deterministic repo builder; refusing training"
            )
        summary = observed_summary
    else:
        BEHAVIOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        with BEHAVIOR_PATH.open("w", encoding="utf-8") as handle:
            for row in expected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        rows = expected
        summary = expected_summary
    summary = dict(summary)
    summary["path"] = str(BEHAVIOR_PATH)
    summary["sha256"] = sha256_file(BEHAVIOR_PATH)
    write_json(BEHAVIOR_PATH.with_suffix(".summary.json"), summary)
    return rows, summary


def route_of(row: dict[str, Any]) -> str:
    return str(row.get("route") or row.get("positive_group") or "unknown")


def chosen_of(row: dict[str, Any]) -> str:
    return str(row.get("chosen") or row.get("answer") or row.get("response") or "").strip()


def prompt_of(row: dict[str, Any]) -> str:
    text = str(row.get("anchor_context") or row.get("prompt") or row.get("text") or "").strip()
    if text and not text.endswith("\n-"):
        text = text.rstrip() + "\n-"
    return text


def parse_negatives(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [x.strip() for x in text.split(",") if x.strip()]
    return [str(value)]


def required_directed_edges(contract: dict[str, Any]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for pair in contract["required_collision_pairs"]:
        a, b = str(pair[0]), str(pair[1])
        edges.add((a, b))
        edges.add((b, a))
    return edges


def normalize_training_rows(
    rows: Sequence[dict[str, Any]],
    source: str,
    *,
    required_edges: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        route = route_of(row)
        prompt = prompt_of(row)
        chosen = chosen_of(row)
        if route not in ROUTES or not prompt or not chosen:
            continue
        if source == "targeted":
            negatives = parse_negatives(row.get("hard_negatives")) + parse_negatives(row.get("hard_negative"))
            if required_edges is not None and not any((route, neg) in required_edges for neg in negatives):
                continue
        output.append({
            "id": str(row.get("id", f"{source}_{index:06d}")),
            "route": route,
            "prompt": prompt,
            "chosen": chosen,
            "source": source,
        })
    return output


class DeterministicRouteSampler:
    """Exact route-pressure cycle plus deterministic source replay."""

    def __init__(self, rows: Sequence[dict[str, Any]], seed: int):
        self.pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            self.pools[(str(row["source"]), str(row["route"]))].append(dict(row))
        self.orders: dict[tuple[str, str], list[int]] = {}
        self.rngs: dict[tuple[str, str], random.Random] = {}
        self.source_cursor: Counter[str] = Counter()
        self.route_cursor = 0
        for key, pool in self.pools.items():
            stable_offset = sum((i + 1) * ord(ch) for i, ch in enumerate("|".join(key)))
            rng = random.Random(seed + stable_offset)
            order = list(range(len(pool)))
            rng.shuffle(order)
            self.orders[key] = order
            self.rngs[key] = rng
        for route in ROUTES:
            if not self.pools.get(("balanced", route)):
                raise RuntimeError(f"Balanced replay is mandatory for every route; missing {route}")
            if not any(self.pools.get((source, route)) for source in ("behavior", "balanced", "targeted")):
                raise RuntimeError(f"No training rows for route {route}")

    def source_cycle(self, route: str) -> list[str]:
        if route == "gradient_clipping":
            return ["behavior", "behavior", "behavior", "balanced", "targeted"]
        if route in {"dropout", "rag", "tokenizer"}:
            return ["behavior", "behavior", "balanced", "targeted"]
        return ["balanced", "behavior", "targeted"]

    def _take(self, source: str, route: str) -> dict[str, Any] | None:
        key = (source, route)
        pool = self.pools.get(key, [])
        if not pool:
            return None
        order = self.orders[key]
        if not order:
            order.extend(range(len(pool)))
            self.rngs[key].shuffle(order)
        return pool[order.pop()]

    def next_row(self, route: str) -> dict[str, Any]:
        cycle = self.source_cycle(route)
        pos = self.source_cursor[route]
        self.source_cursor[route] += 1
        preferred = cycle[pos % len(cycle)]
        for source in [preferred, "behavior", "balanced", "targeted"]:
            row = self._take(source, route)
            if row is not None:
                return row
        raise RuntimeError(f"Sampler exhausted all sources for route {route}")

    def next_batch(self, n: int) -> list[dict[str, Any]]:
        batch = []
        for _ in range(n):
            route = ROUTE_SEQUENCE[self.route_cursor % len(ROUTE_SEQUENCE)]
            self.route_cursor += 1
            batch.append(self.next_row(route))
        return batch


def encode(tok: Tokenizer, text: str) -> list[int]:
    return list(tok.encode(str(text), add_special_tokens=False).ids)


def continuation_ce(
    model,
    tok: Tokenizer,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    max_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    examples: list[tuple[list[int], list[int]]] = []
    source_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    for row in rows:
        pids = encode(tok, row["prompt"])
        cids = encode(tok, row["chosen"])
        if not pids or not cids:
            raise RuntimeError(f"Empty prompt/chosen tokenization for {row['id']}")
        if len(cids) >= max_len:
            raise RuntimeError(f"Chosen continuation exceeds max_len for {row['id']}")
        pids = pids[-max(1, max_len - len(cids)):]
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

    width = max(len(x) for x, _ in examples)
    x = torch.full((len(examples), width), int(pad_id), dtype=torch.long, device=device)
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
        "top1_rate": top1,
        "routes": dict(route_counts),
        "sources": dict(source_counts),
    }


def lr_at(update: int, args) -> float:
    if update <= args.warmup_updates:
        return args.lr * update / max(1, args.warmup_updates)
    t = (update - args.warmup_updates) / max(1, args.max_updates - args.warmup_updates)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))
    return args.lr * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def configure_history_args(history, args):
    eval_args = history.build_argparser().parse_args([])
    eval_args.device = args.device
    eval_args.max_len = args.max_len
    eval_args.max_gen_tokens = args.max_gen_tokens
    eval_args.eval_samples = args.eval_samples
    eval_args.eval_loss_samples = args.eval_loss_samples
    eval_args.min_eval_words = args.min_eval_words
    eval_args.save_eval_preview = args.eval_samples
    eval_args.eos_ids = [3, 7]
    eval_args.bos_id = 2
    return eval_args


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    *,
    update: int,
    args,
    dataset_manifest: dict[str, Any],
    gate: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "meta": {
            "stage": "v14a3_behavior_first_semantic_landing",
            "step": int(update),
            "trainer": "Erratum/ardor_v14a3_behavior_first_trainer.py",
            "model_config": dict(MODEL_CONFIG),
            "canonical_parent": canonical_parent_reference(),
            "tokenizer_contract": {
                "path": str(TOKENIZER_PATH),
                "vocab_size": EXPECTED_VOCAB_SIZE,
                "special_ids": dict(EXPECTED_SPECIAL_IDS),
            },
            "objective": {
                "continuation_ce_weight": 1.0,
                "local_margin_weight": 0.0,
                "geometry_loss_weight": 0.0,
                "prompt_tokens_masked": True,
            },
            "freeze_policy": "none",
            "dataset_manifest": dataset_manifest,
            "promotion_gate": gate,
            "diagnostics": diagnostics,
            "args": vars(args),
            "created_at_unix": time.time(),
        },
    }, str(path))


def run(args) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v14a3 training requires CUDA")
    contract = training_contract()
    validate_local_files(verify_checkpoint_sha256=args.verify_parent_sha256)

    dataset_manifest = {
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
        "holdout": verify_fixed_dataset(
            HOLDOUT_PATH,
            contract["data"]["holdout"]["sha256"],
            int(contract["data"]["holdout"]["rows"]),
        ),
    }
    behavior_raw, behavior_summary = ensure_behavior_dataset()
    dataset_manifest["behavior"] = behavior_summary

    model, strict_load, checkpoint_meta = _load_canonical_model(args.device)
    cfg = model.model_config()
    for key, expected in MODEL_CONFIG.items():
        if cfg.get(key) != expected:
            raise RuntimeError(f"Canonical model config mismatch at {key}: expected={expected!r} actual={cfg.get(key)!r}")
    if any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError("v14a3 default freeze policy is none, but the loaded canonical model has frozen parameters")

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    tok_contract = _tokenizer_contract(tok)
    if not tok_contract["passed"] or tok.get_vocab_size() != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(f"Tokenizer-v9 contract failed: {tok_contract}")
    if {token: tok.token_to_id(token) for token in EXPECTED_SPECIAL_IDS} != EXPECTED_SPECIAL_IDS:
        raise RuntimeError("Tokenizer-v9 special-token IDs changed")

    history = _load_historical_eval_module()
    eval_args = configure_history_args(history, args)
    special = history.special_ids(tok, eval_args)

    balanced_raw = read_jsonl(BALANCED_PATH)
    targeted_raw = read_jsonl(TARGETED_PATH)
    edges = required_directed_edges(contract)
    training_rows = []
    training_rows.extend(normalize_training_rows(behavior_raw, "behavior"))
    training_rows.extend(normalize_training_rows(balanced_raw, "balanced"))
    training_rows.extend(normalize_training_rows(targeted_raw, "targeted", required_edges=edges))
    source_counts = Counter(row["source"] for row in training_rows)
    route_counts = Counter(row["route"] for row in training_rows)
    dataset_manifest["normalized_training_rows"] = {
        "rows": len(training_rows),
        "by_source": dict(source_counts),
        "by_route": dict(route_counts),
    }

    holdout = history.load_holdout(HOLDOUT_PATH)
    rng = random.Random(args.seed + 17)
    rng.shuffle(holdout)
    holdout = holdout[: args.eval_samples]

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    write_json(output_dir / "run_config.json", {
        **vars(args),
        "canonical_parent": canonical_parent_reference(),
        "training_contract": str(CONTRACT_PATH),
        "strict_load": strict_load,
        "canonical_checkpoint_meta": checkpoint_meta,
        "objective": {"continuation_ce_weight": 1.0, "local_margin_weight": 0.0, "geometry_loss_weight": 0.0},
        "freeze_policy": "none",
    })

    device = torch.device(args.device)
    print("[baseline] behavior")
    base_behavior = evaluate_behavior(model, tok, holdout, device, history, eval_args, special, name="canonical_v14a2")
    print("[baseline] balanced chosen continuation")
    base_chosen = evaluate_balanced_chosen(model, tok, balanced_raw, device)
    print("[baseline] calibrated fixed-target ranking")
    base_calibrated = calibrated_target_ranking(model, tok, balanced_raw, holdout, device)
    print("[baseline] final hidden geometry")
    base_geometry = evaluate_final_geometry(model, tok, balanced_raw, holdout, device)
    baseline = {
        "behavior": compact_behavior(base_behavior),
        "balanced_chosen": base_chosen,
        "calibrated_target_ranking": {k: v for k, v in base_calibrated.items() if k != "examples"},
        "final_geometry": base_geometry,
    }
    write_json(output_dir / "baseline_diagnostics.json", baseline)
    write_json(output_dir / "baseline_behavior_outputs.json", base_behavior["outputs"])

    sampler = DeterministicRouteSampler(training_rows, args.seed + 101)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_key: tuple[float, ...] | None = None
    best_update: int | None = None
    best_checkpoint: str | None = None
    promoted_once = False

    for update in range(1, args.max_updates + 1):
        model.train()
        lr = lr_at(update, args)
        set_lr(optimizer, lr)
        batch = sampler.next_batch(args.batch_rows)
        optimizer.zero_grad(set_to_none=True)
        loss, train_stats = continuation_ce(
            model, tok, batch, device, max_len=args.max_len, pad_id=EXPECTED_SPECIAL_IDS["<pad>"]
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        train_record = {
            "update": update,
            "lr": lr,
            "continuation_ce": float(loss.detach().item()),
            "continuation_ce_weight": 1.0,
            "local_margin_weight": 0.0,
            "geometry_loss_weight": 0.0,
            "grad_norm": float(grad_norm),
            **train_stats,
        }
        if update == 1 or update % args.log_every == 0:
            print(
                f"[train] u={update:04d}/{args.max_updates} lr={lr:.2e} "
                f"ce={train_record['continuation_ce']:.5f} top1={train_stats['top1_rate']:.3f} "
                f"routes={train_stats['routes']} sources={train_stats['sources']} grad={float(grad_norm):.3f}"
            )

        do_eval = update == 1 or update in args.eval_updates or update % args.eval_every == 0
        if not do_eval:
            continue

        print(f"[eval] primary update={update}")
        current_behavior = evaluate_behavior(
            model, tok, holdout, device, history, eval_args, special, name=f"v14a3_u{update}"
        )
        current_chosen = evaluate_balanced_chosen(model, tok, balanced_raw, device)
        primary_gate = primary_promotion_gate(base_behavior, current_behavior, base_chosen, current_chosen)
        record: dict[str, Any] = {
            "update": update,
            "train": train_record,
            "behavior": compact_behavior(current_behavior),
            "balanced_chosen": current_chosen,
            "primary_gate": primary_gate,
            "secondary_gate": None,
            "promoted": False,
        }

        if primary_gate["passed"]:
            print(f"[eval] primary gate passed at u={update}; running calibrated ranking + geometry")
            current_calibrated = calibrated_target_ranking(model, tok, balanced_raw, holdout, device)
            current_geometry = evaluate_final_geometry(model, tok, balanced_raw, holdout, device)
            secondary_gate = secondary_promotion_gate(
                base_calibrated, current_calibrated, base_geometry, current_geometry
            )
            record["calibrated_target_ranking"] = {
                k: v for k, v in current_calibrated.items() if k != "examples"
            }
            record["final_geometry"] = current_geometry
            record["secondary_gate"] = secondary_gate
            if secondary_gate["passed"]:
                key = promotion_key(current_behavior, current_chosen, current_calibrated, current_geometry)
                if best_key is None or key > best_key:
                    best_key = key
                    best_update = update
                    best_path = checkpoint_dir / "best_model.pt"
                    combined_gate = {"passed": True, "primary": primary_gate, "secondary": secondary_gate}
                    diagnostics = {
                        "behavior": compact_behavior(current_behavior),
                        "balanced_chosen": current_chosen,
                        "calibrated_target_ranking": record["calibrated_target_ranking"],
                        "final_geometry": current_geometry,
                    }
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        update=update,
                        args=args,
                        dataset_manifest=dataset_manifest,
                        gate=combined_gate,
                        diagnostics=diagnostics,
                    )
                    best_checkpoint = str(best_path)
                    promoted_once = True
                    record["promoted"] = True
                    print(f"[promote] behavior-first gate passed; saved {best_path}")

        append_jsonl(metrics_path, record)
        write_json(output_dir / f"eval_u{update:04d}.json", record)
        print(
            f"[eval] bad={current_behavior['bad_rate']:.4f}/{base_behavior['bad_rate']:.4f} "
            f"chosen_nll={current_chosen['overall_nll']:.4f}/{base_chosen['overall_nll']:.4f} "
            f"primary={primary_gate['passed']} promoted={record['promoted']}"
        )

        if args.stop_on_regression:
            if current_behavior["model_loop_rate"] > base_behavior["model_loop_rate"] + 0.008:
                print("[stop] model-loop regression guard")
                break
            if current_behavior["mean_repetition_rate"] > base_behavior["mean_repetition_rate"] + 0.01:
                print("[stop] repetition regression guard")
                break
            guard_regression = False
            for route in ROUTES:
                if route in PRIORITY_ROUTES:
                    limit = base_behavior["route_bad_rate"][route]
                else:
                    limit = base_behavior["route_bad_rate"][route] + 0.04
                if current_behavior["route_bad_rate"][route] > limit:
                    print(
                        f"[stop] route behavior regression route={route} "
                        f"base={base_behavior['route_bad_rate'][route]:.4f} current={current_behavior['route_bad_rate'][route]:.4f}"
                    )
                    guard_regression = True
                    break
            if guard_regression:
                break

    print("[final] behavior + chosen + calibrated + geometry")
    final_behavior = evaluate_behavior(model, tok, holdout, device, history, eval_args, special, name="v14a3_final")
    final_chosen = evaluate_balanced_chosen(model, tok, balanced_raw, device)
    final_calibrated = calibrated_target_ranking(model, tok, balanced_raw, holdout, device)
    final_geometry = evaluate_final_geometry(model, tok, balanced_raw, holdout, device)
    final_primary = primary_promotion_gate(base_behavior, final_behavior, base_chosen, final_chosen)
    final_secondary = secondary_promotion_gate(base_calibrated, final_calibrated, base_geometry, final_geometry)
    summary = {
        "trainer": "ardor_v14a3_behavior_first_trainer",
        "canonical_parent": canonical_parent_reference(),
        "objective": {"continuation_ce_weight": 1.0, "local_margin_weight": 0.0, "geometry_loss_weight": 0.0},
        "freeze_policy": "none",
        "dataset_manifest": dataset_manifest,
        "baseline": baseline,
        "final": {
            "behavior": compact_behavior(final_behavior),
            "balanced_chosen": final_chosen,
            "calibrated_target_ranking": {k: v for k, v in final_calibrated.items() if k != "examples"},
            "final_geometry": final_geometry,
            "primary_gate": final_primary,
            "secondary_gate": final_secondary,
        },
        "promoted_once": promoted_once,
        "best_update": best_update,
        "best_checkpoint": best_checkpoint,
        "created_at_unix": time.time(),
    }
    write_json(output_dir / "run_summary.json", summary)
    write_json(output_dir / "final_behavior_outputs.json", final_behavior["outputs"])
    print(f"[done] promoted_once={promoted_once} best_update={best_update} best_checkpoint={best_checkpoint}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--max-gen-tokens", type=int, default=72)
    parser.add_argument("--min-eval-words", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-8)
    parser.add_argument("--min-lr-ratio", type=float, default=0.2)
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--max-updates", type=int, default=600)
    parser.add_argument("--batch-rows", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-updates", type=int, nargs="*", default=[1, 10])
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--eval-loss-samples", type=int, default=128)
    parser.add_argument("--stop-on-regression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-parent-sha256", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
