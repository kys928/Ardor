#!/usr/bin/env python3
"""Single-source canonical v14a2 checkpoint/tokenizer contract.

Future v14-family experiments should import this module instead of restating model dimensions,
checkpoint identity, or tokenizer special-token IDs in trainer code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONTRACT_PATH = Path(__file__).with_name("canonical_v14a2_contract_20260909.json")
CONTRACT: dict[str, Any] = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

CANONICAL_CHECKPOINT = Path(CONTRACT["checkpoint"]["path"])
CANONICAL_CHECKPOINT_SIZE = int(CONTRACT["checkpoint"]["size_bytes"])
CANONICAL_CHECKPOINT_SHA256 = str(CONTRACT["checkpoint"]["sha256"])
MODEL_CONFIG = dict(CONTRACT["model_config"])
TOKENIZER_PATH = Path(CONTRACT["tokenizer"]["path"])
EXPECTED_VOCAB_SIZE = int(CONTRACT["tokenizer"]["vocab_size"])
EXPECTED_SPECIAL_IDS = {
    str(token): int(token_id)
    for token, token_id in CONTRACT["tokenizer"]["special_token_ids"].items()
}


def canonical_parent_reference() -> dict[str, Any]:
    """Return the exact immutable parent identity that every future v14 artifact must embed."""
    return {
        "contract": str(CONTRACT_PATH),
        "checkpoint_path": str(CANONICAL_CHECKPOINT),
        "checkpoint_size_bytes": CANONICAL_CHECKPOINT_SIZE,
        "checkpoint_sha256": CANONICAL_CHECKPOINT_SHA256,
        "model_config": dict(MODEL_CONFIG),
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_vocab_size": EXPECTED_VOCAB_SIZE,
        "tokenizer_special_ids": dict(EXPECTED_SPECIAL_IDS),
    }


def validate_static_contract() -> None:
    """Fail on accidental edits that would silently change the frozen canonical identity."""
    if MODEL_CONFIG.get("n_layers") != 33:
        raise RuntimeError(f"Canonical v14a2 must have 33 layers, got {MODEL_CONFIG.get('n_layers')!r}")
    if MODEL_CONFIG.get("n_heads") != 24:
        raise RuntimeError(f"Canonical v14a2 must have 24 heads, got {MODEL_CONFIG.get('n_heads')!r}")
    if MODEL_CONFIG.get("hidden_size") != 1536:
        raise RuntimeError("Canonical v14a2 hidden_size changed")
    if MODEL_CONFIG.get("vocab_size") != EXPECTED_VOCAB_SIZE:
        raise RuntimeError("Canonical model/tokenizer vocab sizes differ")
    expected_ids = {
        "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
        "<|user|>": 4, "<|assistant|>": 5, "<|system|>": 6, "<|eot|>": 7,
    }
    if EXPECTED_SPECIAL_IDS != expected_ids:
        raise RuntimeError(f"Canonical tokenizer special IDs changed: {EXPECTED_SPECIAL_IDS}")
    if len(CANONICAL_CHECKPOINT_SHA256) != 64:
        raise RuntimeError("Canonical checkpoint SHA-256 is malformed")


def validate_local_files(*, verify_checkpoint_sha256: bool = False) -> dict[str, Any]:
    """Validate local checkpoint/tokenizer identity; SHA streaming is optional because the file is ~12 GB."""
    validate_static_contract()
    if not CANONICAL_CHECKPOINT.is_file():
        raise FileNotFoundError(CANONICAL_CHECKPOINT)
    size = CANONICAL_CHECKPOINT.stat().st_size
    if size != CANONICAL_CHECKPOINT_SIZE:
        raise RuntimeError(
            f"Canonical checkpoint size mismatch: expected={CANONICAL_CHECKPOINT_SIZE} actual={size}"
        )
    observed_sha = None
    if verify_checkpoint_sha256:
        digest = hashlib.sha256()
        with CANONICAL_CHECKPOINT.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        observed_sha = digest.hexdigest()
        if observed_sha != CANONICAL_CHECKPOINT_SHA256:
            raise RuntimeError(
                f"Canonical checkpoint SHA mismatch: expected={CANONICAL_CHECKPOINT_SHA256} actual={observed_sha}"
            )
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(TOKENIZER_PATH)
    return {
        "checkpoint_size_bytes": size,
        "checkpoint_sha256_expected": CANONICAL_CHECKPOINT_SHA256,
        "checkpoint_sha256_observed": observed_sha,
        "sha256_verified_this_run": bool(verify_checkpoint_sha256),
        "tokenizer_path": str(TOKENIZER_PATH),
    }


validate_static_contract()
