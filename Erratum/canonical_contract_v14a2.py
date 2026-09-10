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
TOKENIZER_SIZE = int(CONTRACT["tokenizer"]["size_bytes"])
TOKENIZER_SHA256 = str(CONTRACT["tokenizer"]["sha256"])
EXPECTED_VOCAB_SIZE = int(CONTRACT["tokenizer"]["vocab_size"])
EXPECTED_SPECIAL_IDS = {
    str(token): int(token_id)
    for token, token_id in CONTRACT["tokenizer"]["special_token_ids"].items()
}


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_parent_reference() -> dict[str, Any]:
    """Return the exact immutable parent identity that every future v14 artifact must embed."""
    return {
        "contract": str(CONTRACT_PATH),
        "checkpoint_path": str(CANONICAL_CHECKPOINT),
        "checkpoint_size_bytes": CANONICAL_CHECKPOINT_SIZE,
        "checkpoint_sha256": CANONICAL_CHECKPOINT_SHA256,
        "model_config": dict(MODEL_CONFIG),
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_size_bytes": TOKENIZER_SIZE,
        "tokenizer_sha256": TOKENIZER_SHA256,
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
    if len(TOKENIZER_SHA256) != 64:
        raise RuntimeError("Canonical tokenizer SHA-256 is malformed")


def validate_local_files(*, verify_checkpoint_sha256: bool = False) -> dict[str, Any]:
    """Validate local checkpoint/tokenizer identity; tokenizer hashing is always enforced."""
    validate_static_contract()
    if not CANONICAL_CHECKPOINT.is_file():
        raise FileNotFoundError(CANONICAL_CHECKPOINT)
    checkpoint_size = CANONICAL_CHECKPOINT.stat().st_size
    if checkpoint_size != CANONICAL_CHECKPOINT_SIZE:
        raise RuntimeError(
            f"Canonical checkpoint size mismatch: expected={CANONICAL_CHECKPOINT_SIZE} actual={checkpoint_size}"
        )
    checkpoint_sha = None
    if verify_checkpoint_sha256:
        checkpoint_sha = sha256_file(CANONICAL_CHECKPOINT)
        if checkpoint_sha != CANONICAL_CHECKPOINT_SHA256:
            raise RuntimeError(
                f"Canonical checkpoint SHA mismatch: expected={CANONICAL_CHECKPOINT_SHA256} actual={checkpoint_sha}"
            )

    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(TOKENIZER_PATH)
    tokenizer_size = TOKENIZER_PATH.stat().st_size
    if tokenizer_size != TOKENIZER_SIZE:
        raise RuntimeError(
            f"Canonical tokenizer size mismatch: expected={TOKENIZER_SIZE} actual={tokenizer_size}"
        )
    tokenizer_sha = sha256_file(TOKENIZER_PATH, chunk_bytes=4 * 1024 * 1024)
    if tokenizer_sha != TOKENIZER_SHA256:
        raise RuntimeError(
            f"Canonical tokenizer SHA mismatch: expected={TOKENIZER_SHA256} actual={tokenizer_sha}"
        )

    return {
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_sha256_expected": CANONICAL_CHECKPOINT_SHA256,
        "checkpoint_sha256_observed": checkpoint_sha,
        "checkpoint_sha256_verified_this_run": bool(verify_checkpoint_sha256),
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_size_bytes": tokenizer_size,
        "tokenizer_sha256_expected": TOKENIZER_SHA256,
        "tokenizer_sha256_observed": tokenizer_sha,
        "tokenizer_sha256_verified_this_run": True,
    }


validate_static_contract()
