#!/usr/bin/env python3
"""Executable proof that repaired v14a2 satisfies the current canonical trainer contract."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The production trainer intentionally points compiler caches at /workspace.
# GitHub-hosted proof runners do not own that root. Redirect only disposable
# compiler/runtime caches; this does not alter model/checkpoint semantics.
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/ardor_torchinductor_cache"
os.environ["TRITON_CACHE_DIR"] = "/tmp/ardor_triton_cache"
os.environ["PYTHONPATH"] = str(REPO_ROOT)

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from scripts.runpod_control import s3_client, bucket
from scripts.v14a2_checkpoint_contract_probe import (
    CHECKPOINT_KEY,
    TOKENIZER_KEY,
    S3RangeReader,
    build_current_canonical_model,
    current_sha,
    shape_tuple,
    trainer,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("checkpoint_contract_proof.json"))
    args = ap.parse_args()

    # Match Hephaestus/runpod_train_entry.py: redirect only the trainer's
    # project-code import root to the checked-out repository. This does not
    # alter checkpoint, tokenizer, architecture, or resume semantics.
    trainer.ARDOR_ROOT = REPO_ROOT

    client = s3_client()
    volume = bucket()
    ck_head = client.head_object(Bucket=volume, Key=CHECKPOINT_KEY)
    tok_head = client.head_object(Bucket=volume, Key=TOKENIZER_KEY)
    tok_bytes = client.get_object(Bucket=volume, Key=TOKENIZER_KEY)["Body"].read()
    tok_path = Path("/tmp/ardor_tokenizer_v9.json")
    tok_path.write_bytes(tok_bytes)
    vocab_size = int(trainer.tokenizer_vocab_size(tok_path))
    special_ids = trainer.tokenizer_special_ids(tok_path)

    reader = S3RangeReader(client, volume, CHECKPOINT_KEY, int(ck_head["ContentLength"]))
    proof: dict[str, Any] = {
        "proof_version": 3,
        "repo_sha": current_sha(),
        "checkpoint": {
            "key": CHECKPOINT_KEY,
            "size": int(ck_head["ContentLength"]),
            "etag": str(ck_head.get("ETag", "")).strip('"'),
            "last_modified": ck_head["LastModified"].isoformat(),
        },
        "tokenizer": {
            "key": TOKENIZER_KEY,
            "size": int(tok_head["ContentLength"]),
            "etag": str(tok_head.get("ETag", "")).strip('"'),
            "vocab_size": vocab_size,
            "special_ids": special_ids,
        },
        "canonical_trainer": {
            "module": "Cerebrum.Cortex.neural_plasticity_training",
            "typed_entry": "Hephaestus/runpod_train_entry.py",
            "runner": "ardor_promptgen",
            "architecture": {
                "hidden_size": 1536,
                "n_layers": 33,
                "n_heads": 24,
                "ff_mult": 4,
                "ctx": 2048,
                "use_rope": True,
                "rope_theta": 10000.0,
                "use_compile_default": False,
            },
            "resume_contract": 'model.load_state_dict(ckpt["model"], strict=True)',
        },
        "serialization": {},
        "strict_model_contract": {},
        "optimizer_contract": {},
        "passed": False,
    }

    try:
        with FakeTensorMode():
            ckpt = torch.load(reader, map_location="cpu", weights_only=False)
            proof["serialization"]["torch_load_ok"] = True
            proof["serialization"]["top_level_type"] = type(ckpt).__name__
            if not isinstance(ckpt, dict):
                raise TypeError(f"top-level checkpoint is {type(ckpt).__name__}, expected dict")

            top_keys = sorted(str(k) for k in ckpt.keys())
            proof["serialization"].update({
                "top_level_keys": top_keys,
                "has_model_key": "model" in ckpt,
                "has_optimizer_key": "optimizer" in ckpt,
                "has_meta_key": "meta" in ckpt,
            })
            if "model" not in ckpt:
                raise KeyError('canonical trainer requires top-level key "model"')
            if not isinstance(ckpt["model"], dict):
                raise TypeError('ckpt["model"] is not a state-dict mapping')

            model, _ = build_current_canonical_model(vocab_size)
            expected = model.state_dict()
            actual = ckpt["model"]
            missing = sorted(set(expected) - set(actual))
            unexpected = sorted(set(actual) - set(expected))
            common = sorted(set(expected) & set(actual))
            shape_mismatches = [
                {"key": k, "expected": list(shape_tuple(expected[k])), "actual": list(shape_tuple(actual[k]))}
                for k in common if shape_tuple(expected[k]) != shape_tuple(actual[k])
            ]
            dtype_mismatches = [
                {"key": k, "expected": str(expected[k].dtype), "actual": str(actual[k].dtype)}
                for k in common if expected[k].dtype != actual[k].dtype
            ]
            proof["strict_model_contract"].update({
                "expected_key_count": len(expected),
                "actual_key_count": len(actual),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "shape_mismatches": shape_mismatches,
                "dtype_mismatches": dtype_mismatches,
            })

            strict_result = model.load_state_dict(actual, strict=True)
            proof["strict_model_contract"].update({
                "strict_load_ok": True,
                "strict_result": str(strict_result),
            })

            if "optimizer" in ckpt:
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
                )
                optimizer.load_state_dict(ckpt["optimizer"])
                proof["optimizer_contract"] = {
                    "present": True,
                    "load_state_dict_ok": True,
                    "param_groups": len(optimizer.param_groups),
                }
            else:
                proof["optimizer_contract"] = {
                    "present": False,
                    "load_state_dict_ok": None,
                    "allowed_by_trainer": True,
                }

            meta = ckpt.get("meta", {})
            proof["serialization"]["meta_type"] = type(meta).__name__
            proof["serialization"]["meta_keys"] = sorted(str(k) for k in meta) if isinstance(meta, dict) else []
            proof["passed"] = (
                proof["serialization"].get("has_model_key") is True
                and not missing
                and not unexpected
                and not shape_mismatches
                and proof["strict_model_contract"].get("strict_load_ok") is True
                and (
                    not proof["optimizer_contract"]["present"]
                    or proof["optimizer_contract"].get("load_state_dict_ok") is True
                )
            )
    except Exception as exc:
        proof["error"] = f"{type(exc).__name__}: {exc}"
        proof["traceback"] = traceback.format_exc()
        proof["passed"] = False
    finally:
        proof["s3_range_io"] = {
            "checkpoint_size": int(ck_head["ContentLength"]),
            "bytes_fetched": reader.bytes_fetched,
            "range_requests": reader.range_requests,
            "payload_fraction_fetched": reader.bytes_fetched / max(1, int(ck_head["ContentLength"])),
        }
        args.output.write_text(json.dumps(proof, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(json.dumps(proof, indent=2, sort_keys=True, default=str))

    return 0 if proof["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
