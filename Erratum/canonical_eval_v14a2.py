#!/usr/bin/env python3
"""Fixed-purpose canonical v14a2 behavioral evaluation under the current Ardor architecture.

This evaluator deliberately reuses the historical v14a2 trainer's evaluation functions and
sampling order, while loading the canonical checkpoint through the current strict native
checkpoint loader. It performs no training and writes no model checkpoint.
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from Cerebrum.Cortex.loaders.native_checkpoint import load_native_decoder

PERSISTENT_ROOT = Path("/workspace/Ardor")
CANONICAL_CHECKPOINT = (
    PERSISTENT_ROOT
    / "training/runs/v14_promoted_candidates/"
    "v14a2_local_anti_loop_margin_best_u600_CANONICAL_FULLSTATE.pt"
)
TOKENIZER_PATH = PERSISTENT_ROOT / "tokenizer_v9.json"
HISTORICAL_TRAINER = (
    PERSISTENT_ROOT / "training/scripts/ardor_v14a2_local_anti_loop_margin_trainer.py"
)
HOLDOUT_PATH = PERSISTENT_ROOT / "training/data/v14_v3/dataset_v3_holdout_audit.jsonl"
LOCAL_DATA_PATH = (
    PERSISTENT_ROOT / "training/data/v14_v3/dataset_v14a2_local_anti_loop_margin.jsonl"
)
EXPECTED_SPECIAL_IDS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<user>": 4,
    "<assistant>": 5,
    "<system>": 6,
    "<eot>": 7,
}
EXPECTED_VOCAB_SIZE = 52224
EXPECTED_ROPE_THETA = 10000.0


def _load_historical_eval_module():
    if not HISTORICAL_TRAINER.is_file():
        raise FileNotFoundError(f"Historical v14a2 trainer is missing: {HISTORICAL_TRAINER}")
    spec = importlib.util.spec_from_file_location("ardor_v14a2_historical_eval", HISTORICAL_TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import historical evaluator: {HISTORICAL_TRAINER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compact_eval(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in {"outputs", "worst"}}


def _tokenizer_contract(tok: Tokenizer) -> dict[str, Any]:
    specials = {token: tok.token_to_id(token) for token in EXPECTED_SPECIAL_IDS}
    mismatches = {
        token: {"expected": expected, "actual": specials[token]}
        for token, expected in EXPECTED_SPECIAL_IDS.items()
        if specials[token] != expected
    }
    vocab_size = int(tok.get_vocab_size())
    if vocab_size != EXPECTED_VOCAB_SIZE:
        mismatches["vocab_size"] = {
            "expected": EXPECTED_VOCAB_SIZE,
            "actual": vocab_size,
        }
    return {
        "path": str(TOKENIZER_PATH),
        "vocab_size": vocab_size,
        "special_ids": specials,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def evaluate(run_dir: Path) -> dict[str, Any]:
    for required in (CANONICAL_CHECKPOINT, TOKENIZER_PATH, HOLDOUT_PATH, LOCAL_DATA_PATH):
        if not required.is_file():
            raise FileNotFoundError(f"Required canonical evaluation artifact is missing: {required}")

    if not torch.cuda.is_available():
        raise RuntimeError("Canonical v14a2 evaluation requires a CUDA worker")
    device = "cuda"

    history = _load_historical_eval_module()
    args = history.build_argparser().parse_args([])
    args.device = device
    args.resume = str(CANONICAL_CHECKPOINT)
    args.holdout = str(HOLDOUT_PATH)
    args.local_data = str(LOCAL_DATA_PATH)
    args.rebuild_local_data = False

    model, description, want_vocab, checkpoint_meta = load_native_decoder(
        str(CANONICAL_CHECKPOINT),
        device,
        allow_partial_load=False,
    )
    if not bool(description.get("strict_loaded")) or bool(description.get("partial_loaded")):
        raise RuntimeError(f"Canonical checkpoint did not strict-load cleanly: {description}")

    cfg = model.model_config() if hasattr(model, "model_config") else {}
    if not bool(cfg.get("use_rope")):
        raise RuntimeError(f"Canonical model is not using RoPE under the current architecture: {cfg}")
    if float(cfg.get("rope_theta", 0.0)) != EXPECTED_ROPE_THETA:
        raise RuntimeError(f"Unexpected canonical rope_theta: {cfg.get('rope_theta')}")

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    tok_contract = _tokenizer_contract(tok)
    if not tok_contract["passed"]:
        raise RuntimeError(f"Tokenizer v9 contract mismatch: {tok_contract['mismatches']}")
    if int(want_vocab) != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(
            f"Checkpoint/model vocab mismatch: expected={EXPECTED_VOCAB_SIZE} actual={want_vocab}"
        )

    special = history.special_ids(tok, args)
    local_rows = history.read_jsonl(LOCAL_DATA_PATH)
    holdout = history.load_holdout(HOLDOUT_PATH)

    rng = random.Random(int(args.seed) + 17)
    rng.shuffle(holdout)
    rng.shuffle(local_rows)

    local_eval = history.eval_local_margins(
        model, tok, local_rows, torch.device(device), args, "canonical_v14a2_local"
    )
    holdout_eval = history.eval_holdout(
        model,
        tok,
        holdout,
        torch.device(device),
        args,
        special,
        "canonical_v14a2_holdout",
    )

    result = {
        "schema_version": 1,
        "passed": True,
        "checkpoint": str(CANONICAL_CHECKPOINT),
        "strict_load": {
            "strict_loaded": bool(description.get("strict_loaded")),
            "partial_loaded": bool(description.get("partial_loaded")),
            "missing_keys": list(description.get("missing_keys") or []),
            "unexpected_keys": list(description.get("unexpected_keys") or []),
        },
        "model_config": cfg,
        "checkpoint_meta": checkpoint_meta,
        "tokenizer": tok_contract,
        "evaluation_contract": {
            "source": str(HISTORICAL_TRAINER),
            "seed": int(args.seed),
            "shuffle_seed": int(args.seed) + 17,
            "max_len": int(args.max_len),
            "max_gen_tokens": int(args.max_gen_tokens),
            "eval_samples": int(args.eval_samples),
            "eval_local_samples": int(args.eval_local_samples),
            "eval_loss_samples": int(args.eval_loss_samples),
            "margin": float(args.margin),
            "min_eval_words": int(args.min_eval_words),
            "generation": "greedy_argmax",
        },
        "local_eval": _compact_eval(local_eval),
        "holdout_eval": _compact_eval(holdout_eval),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "canonical_eval.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    result = evaluate(Path("/tmp/ardor-canonical-v14a2-eval"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
