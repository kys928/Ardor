#!/usr/bin/env python3
"""Verify the live Network Volume tokenizer matches the v14a2/canonical model contract."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.runpod_control import bucket, s3_client

TOKENIZER_KEY = "Ardor/tokenizer_v9.json"
EXPECTED_VOCAB_SIZE = 52224
EXPECTED_SPECIAL_IDS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<|user|>": 4,
    "<|assistant|>": 5,
    "<|system|>": 6,
    "<|eot|>": 7,
}


def main() -> int:
    client = s3_client()
    volume = bucket()
    obj = client.get_object(Bucket=volume, Key=TOKENIZER_KEY)
    raw = obj["Body"].read()
    tok = json.loads(raw.decode("utf-8"))
    vocab = tok["model"]["vocab"]
    vocab_size = len(vocab)
    special_ids = {token: vocab.get(token) for token in EXPECTED_SPECIAL_IDS}
    out = {
        "passed": vocab_size == EXPECTED_VOCAB_SIZE and special_ids == EXPECTED_SPECIAL_IDS,
        "key": TOKENIZER_KEY,
        "size": len(raw),
        "etag": str(obj.get("ETag", "")).strip('"'),
        "vocab_size": vocab_size,
        "expected_vocab_size": EXPECTED_VOCAB_SIZE,
        "special_ids": special_ids,
        "expected_special_ids": EXPECTED_SPECIAL_IDS,
    }
    Path("v14a2_tokenizer_contract.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
