#!/usr/bin/env python3
"""Extensive read-only scientific analysis of the frozen 24-head canonical v14a2 parent.

The model is loaded only through the explicit canonical wrapper contract used by
canonical_eval_v14a2.py. No training, decoding trick, architecture inference, or
checkpoint mutation is performed here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gc
import json
import math
from pathlib import Path
import random
from statistics import mean, median
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from Erratum.canonical_eval_v14a2 import (
    CANONICAL_CHECKPOINT,
    EXPECTED_ROPE_THETA,
    EXPECTED_SPECIAL_IDS,
    EXPECTED_VOCAB_SIZE,
    HOLDOUT_PATH,
    TOKENIZER_PATH,
    _load_canonical_model,
    _load_historical_eval_module,
    _tokenizer_contract,
)

PERSISTENT_ROOT = Path("/workspace/Ardor")
BALANCED_PATH = PERSISTENT_ROOT / "training/data/v14_v3/dataset_v3b_route_contrastive_balanced.jsonl"
ROUTES = [
    "tokenizer", "rag", "overfitting", "direct_answer",
    "checkpoint", "dropout", "gradient_clipping", "correlation",
]
ROUTE_TARGET = {
    "tokenizer": "A tokenizer converts text into token IDs using a vocabulary so the model can process text as numbers.",
    "rag": "RAG retrieves relevant external context before generation so the answer can use grounded evidence.",
    "overfitting": "Overfitting means a model fits training data too closely and performs worse on new or unseen examples.",
    "direct_answer": "A direct answer gives the main point first and avoids unnecessary setup.",
    "checkpoint": "A checkpoint is a saved model state that can be evaluated or used to resume training.",
    "dropout": "Dropout randomly masks activations during training to improve generalization.",
    "gradient_clipping": "Gradient clipping limits large gradients so optimizer updates stay stable.",
    "correlation": "Correlation means variables are statistically related, but it does not prove causation.",
}
SEED = 928
MAX_LEN = 768
REP_BATCH = 8
PROTO_TRAIN_PER_ROUTE = 64
TRAIN_CE_PER_ROUTE = 32


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


def enc(tok: Tokenizer, text: str) -> list[int]:
    return list(tok.encode(str(text), add_special_tokens=False).ids)


def dec(tok: Tokenizer, ids: Sequence[int]) -> str:
    return tok.decode([int(x) for x in ids]) if ids else ""


def anchor_of(row: dict[str, Any]) -> str:
    text = str(row.get("anchor_context") or row.get("prompt") or row.get("text") or "").strip()
    if text and not text.endswith("\n-"):
        text = text.rstrip() + "\n-"
    return text


def chosen_of(row: dict[str, Any]) -> str:
    return str(row.get("chosen") or row.get("answer") or row.get("response") or "").strip()


def route_of(row: dict[str, Any]) -> str:
    return str(row.get("route") or row.get("positive_group") or "unknown")


def load_balanced() -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(read_jsonl(BALANCED_PATH)):
        route = route_of(row)
        anchor = anchor_of(row)
        chosen = chosen_of(row)
        if route not in ROUTES or not anchor or not chosen:
            continue
        out.append({
            "id": str(row.get("id", f"balanced_{i:06d}")),
            "route": route,
            "anchor_context": anchor,
            "chosen": chosen,
        })
    return out


def load_holdout(history) -> list[dict[str, Any]]:
    rows = history.load_holdout(HOLDOUT_PATH)
    rng = random.Random(SEED + 17)
    rng.shuffle(rows)
    return rows[:256]


def stratified(rows: Sequence[dict[str, Any]], per_route: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if route_of(row) in ROUTES:
            by[route_of(row)].append(row)
    out = []
    for route in ROUTES:
        pool = list(by.get(route, []))
        rng.shuffle(pool)
        out.extend(pool[:per_route])
    return out


def logits_from(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    if isinstance(out, dict):
        for key in ("logits", "lm_logits", "output"):
            if torch.is_tensor(out.get(key)):
                return out[key]
    if hasattr(out, "logits") and torch.is_tensor(out.logits):
        return out.logits
    raise RuntimeError(f"Cannot extract logits from {type(out)}")


class LayerCapture:
    def __init__(self, model):
        selected = [0, 8, 16, 24, 32, 35]
        self.layers: dict[str, Any] = {}
        for idx in selected:
            if idx < len(model.blocks):
                self.layers[f"block_{idx:02d}"] = model.blocks[idx]
        self.layers["final_norm"] = model.norm
        self.values: dict[str, torch.Tensor] = {}
        self.handles = []
        for name, module in self.layers.items():
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not torch.is_tensor(out):
                raise RuntimeError(f"Layer {name} did not return a tensor")
            self.values[name] = out
        return hook

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def collect_reps(model, tok, capture: LayerCapture, rows: Sequence[dict[str, Any]], device: torch.device) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    all_reps: dict[str, list[torch.Tensor]] = defaultdict(list)
    labels: list[str] = []
    ids_out: list[str] = []
    for start in range(0, len(rows), REP_BATCH):
        batch = list(rows[start:start + REP_BATCH])
        encoded = [enc(tok, anchor_of(row))[-MAX_LEN:] for row in batch]
        if any(not ids for ids in encoded):
            raise RuntimeError("Empty prompt in representation batch")
        max_t = max(len(ids) for ids in encoded)
        x = torch.zeros((len(batch), max_t), dtype=torch.long, device=device)
        lengths = []
        for i, ids in enumerate(encoded):
            x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            lengths.append(len(ids))
        capture.clear()
        _ = logits_from(model(x))
        for layer in capture.layers:
            value = capture.values[layer]
            picked = torch.stack([value[i, lengths[i] - 1].float() for i in range(len(batch))])
            picked = F.normalize(picked, dim=1).cpu()
            all_reps[layer].append(picked)
        labels.extend(route_of(row) for row in batch)
        ids_out.extend(str(row.get("id", "")) for row in batch)
    return {k: torch.cat(v, dim=0) for k, v in all_reps.items()}, labels, ids_out


def prototypes(reps: torch.Tensor, labels: Sequence[str]) -> torch.Tensor:
    out = []
    for route in ROUTES:
        idx = [i for i, lab in enumerate(labels) if lab == route]
        if not idx:
            raise RuntimeError(f"Missing prototype route {route}")
        out.append(F.normalize(reps[idx].mean(dim=0), dim=0))
    return torch.stack(out, dim=0)


def geometry_stats(train_reps: torch.Tensor, train_labels: Sequence[str], test_reps: torch.Tensor, test_labels: Sequence[str]) -> dict[str, Any]:
    proto = prototypes(train_reps, train_labels)
    sims = test_reps @ proto.t()
    pred = torch.argmax(sims, dim=1)
    true = torch.tensor([ROUTES.index(x) for x in test_labels], dtype=torch.long)
    correct = pred == true
    own = sims[torch.arange(len(test_labels)), true]
    masked = sims.clone()
    masked[torch.arange(len(test_labels)), true] = -1e9
    wrong = masked.max(dim=1).values
    gap = own - wrong
    confusion = Counter(f"{test_labels[i]}->{ROUTES[int(pred[i])]}" for i in range(len(test_labels)) if not bool(correct[i]))

    pair_cos = {}
    for i, a in enumerate(ROUTES):
        for j in range(i + 1, len(ROUTES)):
            b = ROUTES[j]
            pair_cos[f"{a}<->{b}"] = float((proto[i] @ proto[j]).item())

    by_route = {}
    for route in ROUTES:
        idx = [i for i, lab in enumerate(test_labels) if lab == route]
        if not idx:
            continue
        by_route[route] = {
            "n": len(idx),
            "prototype_accuracy": float(correct[idx].float().mean().item()),
            "mean_own_vs_best_wrong_gap": float(gap[idx].mean().item()),
        }

    # Holdout-only nearest-neighbor geometry, excluding self.
    sim_nn = test_reps @ test_reps.t()
    sim_nn.fill_diagonal_(-1e9)
    nn = torch.argmax(sim_nn, dim=1)
    nn_cross = torch.tensor([test_labels[i] != test_labels[int(nn[i])] for i in range(len(test_labels))], dtype=torch.float32)
    k = min(8, max(1, len(test_labels) - 1))
    order = torch.topk(sim_nn, k=k, dim=1).indices
    purity = []
    for i in range(len(test_labels)):
        purity.append(sum(test_labels[int(j)] == test_labels[i] for j in order[i]) / k)

    return {
        "prototype_accuracy": float(correct.float().mean().item()),
        "mean_own_vs_best_wrong_gap": float(gap.mean().item()),
        "median_own_vs_best_wrong_gap": float(median(gap.tolist())),
        "nn_cross_rate": float(nn_cross.mean().item()),
        "knn_k": k,
        "mean_knn_purity": float(mean(purity)),
        "confusion": dict(confusion.most_common()),
        "by_route": by_route,
        "train_prototype_pair_cosine": dict(sorted(pair_cos.items(), key=lambda kv: kv[1], reverse=True)),
        "predictions": [ROUTES[int(x)] for x in pred],
        "correct": [bool(x) for x in correct],
        "gaps": [float(x) for x in gap],
    }


@torch.inference_mode()
def score_target_batch(model, tok, prompt: str, device: torch.device) -> dict[str, Any]:
    pids = enc(tok, prompt)
    if not pids:
        raise RuntimeError("Empty prompt for target scoring")
    rows = []
    lengths = []
    prompt_lengths = []
    target_ids_by_route = {}
    for route in ROUTES:
        tids = enc(tok, ROUTE_TARGET[route])
        target_ids_by_route[route] = tids
        ids = (pids + tids)[-MAX_LEN:]
        removed = max(0, len(pids) + len(tids) - len(ids))
        effective_prompt_len = max(0, len(pids) - removed)
        if effective_prompt_len < 1:
            raise RuntimeError("Prompt fully truncated during target scoring")
        rows.append(ids)
        lengths.append(len(ids))
        prompt_lengths.append(effective_prompt_len)
    max_t = max(lengths)
    x = torch.zeros((len(ROUTES), max_t - 1), dtype=torch.long, device=device)
    for i, ids in enumerate(rows):
        x[i, :len(ids) - 1] = torch.tensor(ids[:-1], dtype=torch.long, device=device)
    logits = logits_from(model(x)).float()

    nlls = []
    route_details = {}
    for i, route in enumerate(ROUTES):
        ids = rows[i]
        plen = prompt_lengths[i]
        start = plen - 1
        target = torch.tensor(ids[plen:], dtype=torch.long, device=device)
        relevant = logits[i, start:start + len(target)]
        logp = F.log_softmax(relevant, dim=-1)
        token_nll = -logp[torch.arange(len(target), device=device), target]
        ranks = 1 + (relevant > relevant[torch.arange(len(target), device=device), target].unsqueeze(1)).sum(dim=1)
        top1 = torch.argmax(relevant, dim=-1) == target
        avg_nll = float(token_nll.mean().item())
        nlls.append(avg_nll)
        route_details[route] = {
            "mean_target_nll": avg_nll,
            "target_ppl": float(math.exp(min(avg_nll, 20.0))),
            "target_token_top1_rate": float(top1.float().mean().item()),
            "mean_target_token_rank": float(ranks.float().mean().item()),
            "first_8_token_nll": [float(x) for x in token_nll[:8].tolist()],
            "first_8_token_rank": [int(x) for x in ranks[:8].tolist()],
            "first_8_tokens": [dec(tok, [int(x)]) for x in target[:8].tolist()],
        }
    pred_idx = min(range(len(ROUTES)), key=lambda i: nlls[i])
    return {"predicted_route": ROUTES[pred_idx], "route_scores": route_details}


@torch.inference_mode()
def score_actual_chosen(model, tok, prompt: str, chosen: str, device: torch.device) -> dict[str, Any]:
    pids, tids = enc(tok, prompt), enc(tok, chosen)
    if not pids or not tids:
        return {"tokens": 0}
    ids = (pids + tids)[-MAX_LEN:]
    removed = max(0, len(pids) + len(tids) - len(ids))
    plen = max(0, len(pids) - removed)
    if plen < 1 or plen >= len(ids):
        return {"tokens": 0}
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    logits = logits_from(model(x))[0].float()
    target = torch.tensor(ids[plen:], dtype=torch.long, device=device)
    relevant = logits[plen - 1:plen - 1 + len(target)]
    logp = F.log_softmax(relevant, dim=-1)
    nll = -logp[torch.arange(len(target), device=device), target]
    ranks = 1 + (relevant > relevant[torch.arange(len(target), device=device), target].unsqueeze(1)).sum(dim=1)
    return {
        "tokens": len(target),
        "mean_nll": float(nll.mean().item()),
        "ppl": float(math.exp(min(float(nll.mean().item()), 20.0))),
        "top1_rate": float((torch.argmax(relevant, dim=-1) == target).float().mean().item()),
        "mean_rank": float(ranks.float().mean().item()),
        "first_8_nll": [float(x) for x in nll[:8].tolist()],
        "first_8_rank": [int(x) for x in ranks[:8].tolist()],
    }


@torch.inference_mode()
def greedy_generate(model, tok, prompt: str, device: torch.device, eos_ids: set[int], max_tokens: int = 40) -> tuple[str, list[int]]:
    ids = enc(tok, prompt)[-MAX_LEN:]
    generated: list[int] = []
    for _ in range(max_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        nxt = int(torch.argmax(logits_from(model(x))[0, -1].float()).item())
        if nxt in eos_ids:
            break
        generated.append(nxt)
        ids.append(nxt)
        ids = ids[-MAX_LEN:]
    return dec(tok, generated).strip(), generated


def compact_layer_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stats.items() if k not in {"predictions", "correct", "gaps"}}


def summarize_numeric(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": mean(vals), "median": median(vals), "min": min(vals), "max": max(vals)}


def evaluate(run_dir: Path) -> dict[str, Any]:
    for required in (CANONICAL_CHECKPOINT, TOKENIZER_PATH, HOLDOUT_PATH, BALANCED_PATH):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical analysis requires CUDA")
    device = torch.device("cuda")

    model, strict_load, checkpoint_meta = _load_canonical_model("cuda")
    cfg = model.model_config()
    if int(cfg.get("n_heads", -1)) != 24 or not bool(cfg.get("use_rope")) or float(cfg.get("rope_theta", 0)) != EXPECTED_ROPE_THETA:
        raise RuntimeError(f"Canonical architecture contract failed: {cfg}")

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    tok_contract = _tokenizer_contract(tok)
    if not tok_contract["passed"] or tok.get_vocab_size() != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(f"Tokenizer contract failed: {tok_contract}")
    history = _load_historical_eval_module()
    args = history.build_argparser().parse_args([])
    args.device = "cuda"
    special = history.special_ids(tok, args)
    eos_ids = set(int(x) for x in special.get("eos_ids", []))

    balanced = load_balanced()
    holdout = load_holdout(history)
    proto_train = stratified(balanced, PROTO_TRAIN_PER_ROUTE, SEED + 101)
    train_ce_rows = stratified(balanced, TRAIN_CE_PER_ROUTE, SEED + 103)

    capture = LayerCapture(model)
    print(f"[analysis] collecting representations: train={len(proto_train)} holdout={len(holdout)} layers={list(capture.layers)}")
    train_reps, train_labels, _ = collect_reps(model, tok, capture, proto_train, device)
    holdout_reps, holdout_labels, holdout_ids = collect_reps(model, tok, capture, holdout, device)
    layer_stats = {
        layer: geometry_stats(train_reps[layer], train_labels, holdout_reps[layer], holdout_labels)
        for layer in capture.layers
    }
    capture.close()
    gc.collect()
    torch.cuda.empty_cache()

    print("[analysis] scoring canonical route targets on all holdout prompts")
    target_rows = []
    for idx, row in enumerate(holdout):
        score = score_target_batch(model, tok, row["prompt"], device)
        own = score["route_scores"][row["route"]]
        wrong = [v["mean_target_nll"] for r, v in score["route_scores"].items() if r != row["route"]]
        own_gap = min(wrong) - own["mean_target_nll"]
        target_rows.append({
            "id": str(row["id"]),
            "route": row["route"],
            "predicted_route": score["predicted_route"],
            "route_class_correct": score["predicted_route"] == row["route"],
            "own_target_nll": own["mean_target_nll"],
            "own_target_ppl": own["target_ppl"],
            "own_vs_best_wrong_nll_gap": own_gap,
            "target_token_top1_rate": own["target_token_top1_rate"],
            "mean_target_token_rank": own["mean_target_token_rank"],
            "first_8_token_nll": own["first_8_token_nll"],
            "first_8_token_rank": own["first_8_token_rank"],
            "first_8_tokens": own["first_8_tokens"],
        })
        if (idx + 1) % 32 == 0:
            print(f"[analysis] target-score {idx+1}/{len(holdout)}")

    print("[analysis] scoring actual balanced chosen responses")
    train_ce = []
    for idx, row in enumerate(train_ce_rows):
        s = score_actual_chosen(model, tok, row["anchor_context"], row["chosen"], device)
        train_ce.append({"id": row["id"], "route": row["route"], **s})
        if (idx + 1) % 64 == 0:
            print(f"[analysis] train-chosen-score {idx+1}/{len(train_ce_rows)}")

    print("[analysis] deterministic greedy generation on full canonical holdout")
    final_geom = layer_stats["final_norm"]
    final_hidden_pred = final_geom["predictions"]
    final_hidden_correct = final_geom["correct"]
    generations = []
    cross_route_signal = Counter()
    behavior_by_route = Counter()
    for idx, row in enumerate(holdout):
        text, tids = greedy_generate(model, tok, row["prompt"], device, eos_ids, max_tokens=int(args.max_gen_tokens))
        chk = history.sem_check(row["route"], text)
        failures = list(chk["failures"])
        if not tids:
            failures.append("empty_generation")
        if history.wc(text) < int(args.min_eval_words):
            failures.append("too_short_generation")
        bad = bool(failures)
        passed_routes = []
        for route in ROUTES:
            if history.sem_check(route, text)["passed"]:
                passed_routes.append(route)
                if route != row["route"]:
                    cross_route_signal[f"{row['route']}->{route}"] += 1
        target = target_rows[idx]
        generations.append({
            "id": str(row["id"]),
            "route": row["route"],
            "generation": text,
            "bad": bad,
            "failures": failures,
            "semantic_routes_passed": passed_routes,
            "hidden_predicted_route": final_hidden_pred[idx],
            "hidden_route_correct": bool(final_hidden_correct[idx]),
            "lm_target_predicted_route": target["predicted_route"],
            "lm_target_route_correct": bool(target["route_class_correct"]),
            "lm_own_vs_wrong_gap": target["own_vs_best_wrong_nll_gap"],
            "lm_target_token_top1_rate": target["target_token_top1_rate"],
        })
        behavior_by_route[f"{row['route']}|bad={bad}"] += 1
        if (idx + 1) % 32 == 0:
            print(f"[analysis] generation {idx+1}/{len(holdout)}")

    hidden_ok = [r for r in generations if r["hidden_route_correct"]]
    hidden_bad = [r for r in generations if not r["hidden_route_correct"]]
    lm_ok = [r for r in generations if r["lm_target_route_correct"]]
    lm_bad = [r for r in generations if not r["lm_target_route_correct"]]
    both_ok = [r for r in generations if r["hidden_route_correct"] and r["lm_target_route_correct"]]
    route_chain = {
        "hidden_route_accuracy": sum(r["hidden_route_correct"] for r in generations) / max(1, len(generations)),
        "lm_target_route_accuracy": sum(r["lm_target_route_correct"] for r in generations) / max(1, len(generations)),
        "behavior_success_rate": sum(not r["bad"] for r in generations) / max(1, len(generations)),
        "bad_rate_when_hidden_correct": sum(r["bad"] for r in hidden_ok) / max(1, len(hidden_ok)),
        "bad_rate_when_hidden_wrong": sum(r["bad"] for r in hidden_bad) / max(1, len(hidden_bad)),
        "bad_rate_when_lm_target_classifier_correct": sum(r["bad"] for r in lm_ok) / max(1, len(lm_ok)),
        "bad_rate_when_lm_target_classifier_wrong": sum(r["bad"] for r in lm_bad) / max(1, len(lm_bad)),
        "bad_rate_when_hidden_and_lm_classifier_correct": sum(r["bad"] for r in both_ok) / max(1, len(both_ok)),
        "hidden_correct_lm_wrong_count": sum(r["hidden_route_correct"] and not r["lm_target_route_correct"] for r in generations),
        "hidden_correct_lm_correct_behavior_bad_count": sum(r["hidden_route_correct"] and r["lm_target_route_correct"] and r["bad"] for r in generations),
        "hidden_wrong_count": len(hidden_bad),
    }

    by_route = {}
    for route in ROUTES:
        rs = [r for r in generations if r["route"] == route]
        tr = [r for r in target_rows if r["route"] == route]
        by_route[route] = {
            "n": len(rs),
            "behavior_bad_rate": sum(r["bad"] for r in rs) / max(1, len(rs)),
            "hidden_route_accuracy_final_norm": sum(r["hidden_route_correct"] for r in rs) / max(1, len(rs)),
            "lm_target_route_accuracy": sum(r["lm_target_route_correct"] for r in rs) / max(1, len(rs)),
            "mean_own_target_nll": mean(r["own_target_nll"] for r in tr),
            "mean_own_target_ppl": mean(r["own_target_ppl"] for r in tr),
            "mean_own_vs_best_wrong_nll_gap": mean(r["own_vs_best_wrong_nll_gap"] for r in tr),
            "mean_target_token_top1_rate": mean(r["target_token_top1_rate"] for r in tr),
            "mean_target_token_rank": mean(r["mean_target_token_rank"] for r in tr),
        }

    train_by_route = {}
    for route in ROUTES:
        rs = [r for r in train_ce if r["route"] == route and r.get("tokens", 0)]
        train_by_route[route] = {
            "n": len(rs),
            "mean_chosen_nll": mean(r["mean_nll"] for r in rs) if rs else None,
            "mean_chosen_ppl": mean(r["ppl"] for r in rs) if rs else None,
            "mean_chosen_top1_rate": mean(r["top1_rate"] for r in rs) if rs else None,
            "mean_chosen_token_rank": mean(r["mean_rank"] for r in rs) if rs else None,
        }

    result = {
        "schema_version": 1,
        "passed": True,
        "purpose": "Locate the exact route-specific behavioral bottleneck in frozen canonical v14a2",
        "checkpoint": str(CANONICAL_CHECKPOINT),
        "checkpoint_meta": checkpoint_meta,
        "strict_load": strict_load,
        "model_config": cfg,
        "tokenizer": tok_contract,
        "analysis_contract": {
            "seed": SEED,
            "holdout_rows": len(holdout),
            "prototype_train_rows": len(proto_train),
            "balanced_chosen_ce_rows": len(train_ce_rows),
            "prototype_train_per_route": PROTO_TRAIN_PER_ROUTE,
            "layers": list(layer_stats),
            "generation": "greedy_argmax",
            "max_gen_tokens": int(args.max_gen_tokens),
            "route_target_scoring": "target-only mean token NLL; all 8 canonical route targets scored per prompt",
            "architecture_source": "canonical_checkpoint.meta.model_config",
            "architecture_inference_allowed": False,
        },
        "representation_geometry": {layer: compact_layer_stats(stats) for layer, stats in layer_stats.items()},
        "route_chain": route_chain,
        "by_route": by_route,
        "balanced_chosen_teacher_forcing": {
            "overall_nll": summarize_numeric(train_ce, "mean_nll"),
            "overall_top1_rate": summarize_numeric(train_ce, "top1_rate"),
            "overall_mean_rank": summarize_numeric(train_ce, "mean_rank"),
            "by_route": train_by_route,
        },
        "holdout_target_teacher_forcing": {
            "own_target_nll": summarize_numeric(target_rows, "own_target_nll"),
            "own_vs_best_wrong_nll_gap": summarize_numeric(target_rows, "own_vs_best_wrong_nll_gap"),
            "target_token_top1_rate": summarize_numeric(target_rows, "target_token_top1_rate"),
            "mean_target_token_rank": summarize_numeric(target_rows, "mean_target_token_rank"),
        },
        "cross_route_semantic_signal": dict(cross_route_signal.most_common()),
        "behavior_counts": dict(behavior_by_route),
        "holdout_samples": generations,
        "target_samples": target_rows,
        "balanced_chosen_samples": train_ce,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "canonical_analysis.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": True,
        "route_chain": route_chain,
        "by_route": by_route,
        "final_geometry": compact_layer_stats(layer_stats["final_norm"]),
        "balanced_chosen_teacher_forcing": result["balanced_chosen_teacher_forcing"],
        "holdout_target_teacher_forcing": result["holdout_target_teacher_forcing"],
        "result_path": str(out),
    }, indent=2, sort_keys=True))
    return result


def main() -> int:
    evaluate(Path("/tmp/ardor-canonical-v14a2-analysis"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
