#!/usr/bin/env python3
"""v14a4 family-balanced semantic-landing trainer.

Relative to canonical v14a2, training keeps the verified v14a2 u600 optimizer schedule, all-parameter
trainability, and full chosen-answer continuation CE introduced by v14a3. The scientific change is
supervision distribution: behavior rows are balanced by route x semantic_family and consumed without
recycling over the 600-update run. The clean family holdout is the primary fast gate; historical
behavior remains a secondary comparability gate.
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

from Erratum.build_v14a4_family_data import (
    COLLISIONS,
    FAMILIES,
    PRIORITY_ROUTES,
    ROUTES,
    TARGET_TRAIN_COUNTS,
    build_holdout_rows,
    build_train_rows,
    validate as validate_family_data,
    write_jsonl,
)
from Erratum.canonical_contract_v14a2 import (
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
from Erratum.v14a4_diagnostics import (
    calibrated_target_ranking,
    clean_primary_gate,
    compact_behavior,
    compact_clean,
    evaluate_balanced_chosen,
    evaluate_behavior,
    evaluate_clean_family,
    evaluate_final_geometry,
    promotion_key,
    secondary_gate,
    tertiary_gate,
)

REPO_ROOT = Path("/workspace/Ardor")
CONTRACT_PATH = Path(__file__).with_name("v14a4_family_balanced_training_contract_20260913.json")
BEHAVIOR_PATH = REPO_ROOT / "training/data/v14_v3/dataset_v14a4_family_balanced_behavior.jsonl"
CLEAN_HOLDOUT_PATH = REPO_ROOT / "training/data/v14_v3/dataset_v14a4_family_balanced_holdout.jsonl"
BALANCED_PATH = REPO_ROOT / "training/data/v14_v3/dataset_v3b_route_contrastive_balanced.jsonl"
TARGETED_PATH = REPO_ROOT / "training/data/v14_v3/dataset_v14b3_targeted_route_collisions.jsonl"
OUTPUT_DIR = REPO_ROOT / "training/runs/sft_v14a4_family_balanced_semantic_landing_u600"
MAX_LEN = 768
SEED = 928

ROUTE_SEQUENCE = [
    "gradient_clipping", "dropout", "rag", "tokenizer",
    "gradient_clipping", "checkpoint", "dropout", "correlation",
    "gradient_clipping", "rag", "tokenizer", "direct_answer",
    "gradient_clipping", "dropout", "overfitting", "gradient_clipping",
]

SOURCE_CYCLES = {
    "gradient_clipping": ["behavior", "behavior", "behavior", "balanced", "targeted"],
    "dropout": ["behavior", "balanced", "targeted", "behavior"],
    "rag": ["behavior", "balanced", "targeted", "behavior"],
    "tokenizer": ["behavior", "balanced", "targeted", "behavior"],
}
GUARD_SOURCE_CYCLE = ["balanced", "behavior", "targeted"]


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
    if value.get("designation") != "v14a4_family_balanced_semantic_landing":
        raise RuntimeError(f"Unexpected v14a4 contract: {value.get('designation')!r}")
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


def ensure_family_datasets(contract: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expected_train = build_train_rows()
    expected_holdout = build_holdout_rows()
    historical_raw = read_jsonl(HOLDOUT_PATH)
    summary = validate_family_data(expected_train, expected_holdout, historical_raw)

    train_contract = contract["data"]["behavior_train"]
    holdout_contract = contract["data"]["clean_family_holdout"]
    if len(expected_train) != int(train_contract["rows"]) or summary["train_sha256"] != train_contract["sha256"]:
        raise RuntimeError("Deterministic v14a4 behavior dataset does not match pinned contract")
    if len(expected_holdout) != int(holdout_contract["rows"]) or summary["holdout_sha256"] != holdout_contract["sha256"]:
        raise RuntimeError("Deterministic v14a4 clean holdout does not match pinned contract")

    for path, expected in ((BEHAVIOR_PATH, expected_train), (CLEAN_HOLDOUT_PATH, expected_holdout)):
        if path.exists():
            observed = read_jsonl(path)
            if observed != expected:
                raise RuntimeError(f"Existing deterministic v14a4 dataset differs from repo builder: {path}")
        else:
            write_jsonl(path, expected)

    summary = dict(summary)
    summary["behavior_path"] = str(BEHAVIOR_PATH)
    summary["holdout_path"] = str(CLEAN_HOLDOUT_PATH)
    write_json(BEHAVIOR_PATH.with_suffix(".summary.json"), summary)
    return expected_train, expected_holdout, summary


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


def required_directed_edges() -> set[tuple[str, str]]:
    return {(route, neg) for route, negatives in COLLISIONS.items() for neg in negatives}


def normalize_replay_rows(
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
            "semantic_family": "legacy_generic",
            "prompt": prompt,
            "chosen": chosen,
            "source": source,
        })
    return output


def normalize_behavior_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "id": str(row["id"]),
        "route": str(row["route"]),
        "semantic_family": str(row["semantic_family"]),
        "prompt": str(row["prompt"]).rstrip() + "\n-",
        "chosen": str(row["chosen"]).strip(),
        "source": "behavior",
        "question_class": str(row["question_class"]),
        "prompt_style": str(row["prompt_style"]),
        "answer_style": str(row["answer_style"]),
        "collision_route": row.get("collision_route"),
    } for row in rows]


class FamilyBalancedSampler:
    """Deterministic route/source schedule with route x family balancing for behavior rows.

    Behavior rows never recycle. Targeted replay falls back to balanced replay before behavior,
    so routes absent from the targeted corpus cannot steal extra behavior exposures.
    """

    def __init__(self, rows: Sequence[dict[str, Any]], seed: int):
        self.behavior_pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.replay_pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            route = str(row["route"])
            source = str(row["source"])
            if source == "behavior":
                self.behavior_pools[(route, str(row["semantic_family"]))].append(dict(row))
            else:
                self.replay_pools[(source, route)].append(dict(row))

        self.behavior_orders: dict[tuple[str, str], list[int]] = {}
        self.behavior_rngs: dict[tuple[str, str], random.Random] = {}
        self.replay_orders: dict[tuple[str, str], list[int]] = {}
        self.replay_rngs: dict[tuple[str, str], random.Random] = {}
        for key, pool in self.behavior_pools.items():
            stable = sum((i + 1) * ord(ch) for i, ch in enumerate("behavior|" + "|".join(key)))
            rng = random.Random(seed + stable)
            order = list(range(len(pool)))
            rng.shuffle(order)
            self.behavior_orders[key] = order
            self.behavior_rngs[key] = rng
        for key, pool in self.replay_pools.items():
            stable = sum((i + 1) * ord(ch) for i, ch in enumerate("replay|" + "|".join(key)))
            rng = random.Random(seed + stable)
            order = list(range(len(pool)))
            rng.shuffle(order)
            self.replay_orders[key] = order
            self.replay_rngs[key] = rng

        self.family_cursor: Counter[str] = Counter()
        self.source_cursor: Counter[str] = Counter()
        self.route_cursor = 0
        for route in ROUTES:
            if not self.replay_pools.get(("balanced", route)):
                raise RuntimeError(f"Balanced replay is mandatory for every route; missing {route}")
            for family in FAMILIES[route]:
                if not self.behavior_pools.get((route, family)):
                    raise RuntimeError(f"Missing behavior family {route}/{family}")

    def source_cycle(self, route: str) -> list[str]:
        return list(SOURCE_CYCLES.get(route, GUARD_SOURCE_CYCLE))

    def _take_behavior(self, route: str) -> dict[str, Any]:
        families = FAMILIES[route]
        for _ in range(len(families)):
            pos = self.family_cursor[route]
            family = families[pos % len(families)]
            self.family_cursor[route] += 1
            key = (route, family)
            order = self.behavior_orders[key]
            if order:
                return self.behavior_pools[key][order.pop()]
        raise RuntimeError(
            f"Behavior rows exhausted for {route}; v14a4 forbids behavior recycling because "
            "the u600 schedule is pinned to exactly one exposure per authored behavior row"
        )

    def _take_replay(self, source: str, route: str) -> dict[str, Any] | None:
        key = (source, route)
        pool = self.replay_pools.get(key, [])
        if not pool:
            return None
        order = self.replay_orders[key]
        if not order:
            order.extend(range(len(pool)))
            self.replay_rngs[key].shuffle(order)
        return pool[order.pop()]

    def next_row(self, route: str) -> dict[str, Any]:
        cycle = self.source_cycle(route)
        pos = self.source_cursor[route]
        self.source_cursor[route] += 1
        preferred = cycle[pos % len(cycle)]

        if preferred == "behavior":
            return self._take_behavior(route)
        if preferred == "targeted":
            row = self._take_replay("targeted", route)
            if row is not None:
                return row
            row = self._take_replay("balanced", route)
            if row is not None:
                return row
            return self._take_behavior(route)

        row = self._take_replay("balanced", route)
        if row is not None:
            return row
        row = self._take_replay("targeted", route)
        if row is not None:
            return row
        return self._take_behavior(route)

    def next_batch(self, n: int) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
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
    family_counts: Counter[str] = Counter()
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
        family_counts[f"{row['route']}/{row['semantic_family']}"] += 1

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
        "families": dict(family_counts),
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
            "stage": "v14a4_family_balanced_semantic_landing",
            "step": int(update),
            "trainer": "Erratum/ardor_v14a4_family_balanced_trainer.py",
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
            "sampling_policy": "route x semantic_family behavior balance; no behavior recycling",
            "dataset_manifest": dataset_manifest,
            "promotion_gate": gate,
            "diagnostics": diagnostics,
            "args": vars(args),
            "created_at_unix": time.time(),
        },
    }, str(path))


def run(args) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v14a4 training requires CUDA")
    contract = training_contract()
    validate_local_files(verify_checkpoint_sha256=args.verify_parent_sha256)

    behavior_raw, clean_holdout, family_summary = ensure_family_datasets(contract)
    dataset_manifest = {
        "behavior": family_summary,
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
        "historical_holdout": verify_fixed_dataset(
            HOLDOUT_PATH,
            contract["data"]["historical_holdout"]["sha256"],
            int(contract["data"]["historical_holdout"]["rows"]),
        ),
    }

    model, strict_load, checkpoint_meta = _load_canonical_model(args.device)
    cfg = model.model_config()
    for key, expected in MODEL_CONFIG.items():
        if cfg.get(key) != expected:
            raise RuntimeError(f"Canonical model config mismatch at {key}: expected={expected!r} actual={cfg.get(key)!r}")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("v14a4 freeze policy is none, but the canonical model has frozen parameters")

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
    training_rows: list[dict[str, Any]] = []
    training_rows.extend(normalize_behavior_rows(behavior_raw))
    training_rows.extend(normalize_replay_rows(balanced_raw, "balanced"))
    training_rows.extend(normalize_replay_rows(
        targeted_raw, "targeted", required_edges=required_directed_edges()
    ))
    dataset_manifest["normalized_training_rows"] = {
        "rows": len(training_rows),
        "by_source": dict(Counter(row["source"] for row in training_rows)),
        "by_route": dict(Counter(row["route"] for row in training_rows)),
    }

    historical_holdout = history.load_holdout(HOLDOUT_PATH)
    rng = random.Random(args.seed + 17)
    rng.shuffle(historical_holdout)
    historical_holdout = historical_holdout[: args.eval_samples]

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    samples_path = output_dir / "sampled_rows.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    samples_path.write_text("", encoding="utf-8")
    write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    write_json(output_dir / "run_config.json", {
        **vars(args),
        "canonical_parent": canonical_parent_reference(),
        "training_contract": str(CONTRACT_PATH),
        "strict_load": strict_load,
        "canonical_checkpoint_meta": checkpoint_meta,
        "objective": {"continuation_ce_weight": 1.0, "local_margin_weight": 0.0, "geometry_loss_weight": 0.0},
        "freeze_policy": "none",
        "sampling_policy": "route x semantic_family behavior balance; no behavior recycling",
    })

    device = torch.device(args.device)
    print("[baseline] clean family holdout")
    base_clean = evaluate_clean_family(
        model, tok, clean_holdout, device, history, eval_args, special, name="canonical_v14a2_clean_family"
    )
    print("[baseline] historical holdout")
    base_historical = evaluate_behavior(
        model, tok, historical_holdout, device, history, eval_args, special, name="canonical_v14a2_historical"
    )
    print("[baseline] balanced chosen continuation")
    base_chosen = evaluate_balanced_chosen(model, tok, balanced_raw, device)
    print("[baseline] calibrated fixed-target ranking")
    base_calibrated = calibrated_target_ranking(model, tok, balanced_raw, historical_holdout, device)
    print("[baseline] final hidden geometry")
    base_geometry = evaluate_final_geometry(model, tok, balanced_raw, historical_holdout, device)
    baseline = {
        "clean_family": compact_clean(base_clean),
        "historical": compact_behavior(base_historical),
        "balanced_chosen": base_chosen,
        "calibrated_target_ranking": {k: v for k, v in base_calibrated.items() if k != "examples"},
        "final_geometry": base_geometry,
    }
    write_json(output_dir / "baseline_diagnostics.json", baseline)
    write_json(output_dir / "baseline_clean_outputs.json", base_clean["outputs"])
    write_json(output_dir / "baseline_historical_outputs.json", base_historical["outputs"])

    sampler = FamilyBalancedSampler(training_rows, args.seed + 101)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_key: tuple[float, ...] | None = None
    best_update: int | None = None
    best_checkpoint: str | None = None
    promoted_once = False

    cumulative_sources: Counter[str] = Counter()
    cumulative_routes: Counter[str] = Counter()
    cumulative_families: Counter[str] = Counter()
    behavior_ids_seen: set[str] = set()

    for update in range(1, args.max_updates + 1):
        model.train()
        lr = lr_at(update, args)
        set_lr(optimizer, lr)
        batch = sampler.next_batch(args.batch_rows)
        for row in batch:
            cumulative_sources[row["source"]] += 1
            cumulative_routes[row["route"]] += 1
            cumulative_families[f"{row['route']}/{row['semantic_family']}"] += 1
            if row["source"] == "behavior":
                if row["id"] in behavior_ids_seen:
                    raise RuntimeError(f"Behavior row recycled despite v14a4 no-recycle contract: {row['id']}")
                behavior_ids_seen.add(row["id"])
            append_jsonl(samples_path, {
                "update": update,
                "id": row["id"],
                "source": row["source"],
                "route": row["route"],
                "semantic_family": row["semantic_family"],
            })

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
            "grad_norm": float(grad_norm),
            **train_stats,
            "cumulative_sources": dict(cumulative_sources),
            "cumulative_routes": dict(cumulative_routes),
            "cumulative_families": dict(cumulative_families),
            "unique_behavior_rows_seen": len(behavior_ids_seen),
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

        print(f"[eval] clean-family primary update={update}")
        current_clean = evaluate_clean_family(
            model, tok, clean_holdout, device, history, eval_args, special, name=f"v14a4_clean_u{update}"
        )
        primary = clean_primary_gate(
            base_clean, current_clean, max_family_regression=args.max_family_regression
        )
        record: dict[str, Any] = {
            "update": update,
            "train": train_record,
            "clean_family": compact_clean(current_clean),
            "primary_gate": primary,
            "secondary_gate": None,
            "tertiary_gate": None,
            "promoted": False,
        }
        write_json(output_dir / f"clean_outputs_u{update:04d}.json", current_clean["outputs"])

        if primary["passed"]:
            print(f"[eval] clean gate passed at u={update}; running historical + chosen")
            current_historical = evaluate_behavior(
                model, tok, historical_holdout, device, history, eval_args, special,
                name=f"v14a4_historical_u{update}",
            )
            current_chosen = evaluate_balanced_chosen(model, tok, balanced_raw, device)
            secondary = secondary_gate(base_historical, current_historical, base_chosen, current_chosen)
            record["historical"] = compact_behavior(current_historical)
            record["balanced_chosen"] = current_chosen
            record["secondary_gate"] = secondary
            write_json(output_dir / f"historical_outputs_u{update:04d}.json", current_historical["outputs"])

            if secondary["passed"]:
                print(f"[eval] secondary gate passed at u={update}; running calibrated ranking + geometry")
                current_calibrated = calibrated_target_ranking(
                    model, tok, balanced_raw, historical_holdout, device
                )
                current_geometry = evaluate_final_geometry(
                    model, tok, balanced_raw, historical_holdout, device
                )
                tertiary = tertiary_gate(base_calibrated, current_calibrated, base_geometry, current_geometry)
                record["calibrated_target_ranking"] = {
                    k: v for k, v in current_calibrated.items() if k != "examples"
                }
                record["final_geometry"] = current_geometry
                record["tertiary_gate"] = tertiary

                if tertiary["passed"]:
                    key = promotion_key(
                        current_clean, current_historical, current_chosen,
                        current_calibrated, current_geometry,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_update = update
                        best_path = checkpoint_dir / "best_model.pt"
                        gate = {"passed": True, "primary": primary, "secondary": secondary, "tertiary": tertiary}
                        diagnostics = {
                            "clean_family": compact_clean(current_clean),
                            "historical": compact_behavior(current_historical),
                            "balanced_chosen": current_chosen,
                            "calibrated_target_ranking": record["calibrated_target_ranking"],
                            "final_geometry": current_geometry,
                        }
                        save_checkpoint(
                            best_path, model, optimizer, update=update, args=args,
                            dataset_manifest=dataset_manifest, gate=gate, diagnostics=diagnostics,
                        )
                        best_checkpoint = str(best_path)
                        promoted_once = True
                        record["promoted"] = True
                        print(f"[promote] family-balanced gate passed; saved {best_path}")

        append_jsonl(metrics_path, record)
        write_json(output_dir / f"eval_u{update:04d}.json", record)
        print(
            f"[eval] clean_success={current_clean['success_rate']:.4f}/{base_clean['success_rate']:.4f} "
            f"priority_family={current_clean['priority_family_macro_success_rate']:.4f}/"
            f"{base_clean['priority_family_macro_success_rate']:.4f} "
            f"primary={primary['passed']} promoted={record['promoted']}"
        )

        if args.stop_on_regression:
            if current_clean["model_loop_rate"] > base_clean["model_loop_rate"] + 0.008:
                print("[stop] clean-family model-loop regression guard")
                break
            if current_clean["mean_repetition_rate"] > base_clean["mean_repetition_rate"] + 0.01:
                print("[stop] clean-family repetition regression guard")
                break

    print("[final] clean + historical + chosen + calibrated + geometry")
    final_clean = evaluate_clean_family(
        model, tok, clean_holdout, device, history, eval_args, special, name="v14a4_final_clean"
    )
    final_historical = evaluate_behavior(
        model, tok, historical_holdout, device, history, eval_args, special, name="v14a4_final_historical"
    )
    final_chosen = evaluate_balanced_chosen(model, tok, balanced_raw, device)
    final_calibrated = calibrated_target_ranking(model, tok, balanced_raw, historical_holdout, device)
    final_geometry = evaluate_final_geometry(model, tok, balanced_raw, historical_holdout, device)
    final_primary = clean_primary_gate(
        base_clean, final_clean, max_family_regression=args.max_family_regression
    )
    final_secondary = secondary_gate(base_historical, final_historical, base_chosen, final_chosen)
    final_tertiary = tertiary_gate(base_calibrated, final_calibrated, base_geometry, final_geometry)

    summary = {
        "trainer": "ardor_v14a4_family_balanced_trainer",
        "canonical_parent": canonical_parent_reference(),
        "objective": {"continuation_ce_weight": 1.0, "local_margin_weight": 0.0, "geometry_loss_weight": 0.0},
        "freeze_policy": "none",
        "dataset_manifest": dataset_manifest,
        "baseline": baseline,
        "final": {
            "clean_family": compact_clean(final_clean),
            "historical": compact_behavior(final_historical),
            "balanced_chosen": final_chosen,
            "calibrated_target_ranking": {k: v for k, v in final_calibrated.items() if k != "examples"},
            "final_geometry": final_geometry,
            "primary_gate": final_primary,
            "secondary_gate": final_secondary,
            "tertiary_gate": final_tertiary,
        },
        "sampling": {
            "cumulative_sources": dict(cumulative_sources),
            "cumulative_routes": dict(cumulative_routes),
            "cumulative_families": dict(cumulative_families),
            "unique_behavior_rows_seen": len(behavior_ids_seen),
        },
        "promoted_once": promoted_once,
        "best_update": best_update,
        "best_checkpoint": best_checkpoint,
        "created_at_unix": time.time(),
    }
    write_json(output_dir / "run_summary.json", summary)
    write_json(output_dir / "final_clean_outputs.json", final_clean["outputs"])
    write_json(output_dir / "final_historical_outputs.json", final_historical["outputs"])
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
    parser.add_argument("--max-family-regression", type=float, default=0.05)
    parser.add_argument("--stop-on-regression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-parent-sha256", action=argparse.BooleanOptionalAction, default=True,
        help="Full-hash the canonical ~12GB parent before training; enabled by default.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
