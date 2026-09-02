#!/usr/bin/env python3
"""Prove repaired v14a2 against the current canonical trainer's strict resume contract.

This intentionally does not launch a GPU. It uses S3 ranged reads plus
FakeTensorMode so the ~12 GB tensor payload is not downloaded or allocated.
The proof still executes the same strict state-dict load operation used by the
canonical trainer: model.load_state_dict(ckpt["model"], strict=True).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boto3
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from Cerebrum.Cortex import neural_plasticity_training as trainer


CHECKPOINT_KEY = (
    "Ardor/training/runs/v14_promoted_candidates/"
    "v14a2_local_anti_loop_margin_best_u600_REPAIRED_FULLSTATE.pt"
)
TOKENIZER_KEY = "Ardor/tokenizer_v9.json"


class S3RangeReader(io.RawIOBase):
    """Seekable read-only S3 object backed by HTTP Range requests."""

    def __init__(self, client: Any, bucket: str, key: str, size: int):
        super().__init__()
        self.client = client
        self.bucket = bucket
        self.key = key
        self.size = int(size)
        self.pos = 0
        self.bytes_fetched = 0
        self.range_requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self.pos + offset
        elif whence == io.SEEK_END:
            new_pos = self.size + offset
        else:
            raise ValueError(f"unsupported whence={whence}")
        if new_pos < 0:
            raise ValueError("negative seek position")
        self.pos = int(new_pos)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(int(n), self.size - self.pos)
        if n <= 0:
            return b""
        start = self.pos
        end = start + n - 1
        resp = self.client.get_object(
            Bucket=self.bucket,
            Key=self.key,
            Range=f"bytes={start}-{end}",
        )
        data = resp["Body"].read()
        self.pos += len(data)
        self.bytes_fetched += len(data)
        self.range_requests += 1
        return data

    def readinto(self, b: bytearray) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def endpoint() -> str:
    explicit = os.environ.get("RUNPOD_S3_ENDPOINT", "").strip()
    if explicit:
        return explicit.rstrip("/")
    dc = os.environ["RUNPOD_DATACENTER_ID"].strip().lower()
    return f"https://s3api-{dc}.runpod.io"


def current_sha() -> str:
    env_sha = os.environ.get("GITHUB_SHA", "").strip()
    if env_sha:
        return env_sha
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def shape_tuple(x: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in x.shape)


def build_current_canonical_model(vocab_size: int):
    stage_cfg = trainer.STAGE_PRESETS["sft"]
    ArdorConfig = trainer.import_project_config()
    cfg = ArdorConfig(
        vocab_size=vocab_size,
        hidden_size=1536,
        n_layers=33,
        n_heads=24,
        ff_mult=4,
        max_len=2048,
        dropout=stage_cfg.dropout,
        attn_dropout=stage_cfg.dropout,
        resid_dropout=stage_cfg.dropout,
        use_rope=True,
        rope_theta=10000.0,
    )
    cfg.validate()
    decoder_cls = trainer.import_project_decoder()
    model = trainer.build_model(decoder_cls, cfg)
    return model, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("checkpoint_contract_proof.json"))
    args = ap.parse_args()

    bucket = os.environ["RUNPOD_NETWORK_VOLUME_ID"].strip()
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint(),
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )

    ck_head = s3.head_object(Bucket=bucket, Key=CHECKPOINT_KEY)
    tok_head = s3.head_object(Bucket=bucket, Key=TOKENIZER_KEY)

    tok_bytes = s3.get_object(Bucket=bucket, Key=TOKENIZER_KEY)["Body"].read()
    tok_path = Path("/tmp/ardor_tokenizer_v9.json")
    tok_path.write_bytes(tok_bytes)
    vocab_size = int(trainer.tokenizer_vocab_size(tok_path))
    special_ids = trainer.tokenizer_special_ids(tok_path)

    reader = S3RangeReader(s3, bucket, CHECKPOINT_KEY, int(ck_head["ContentLength"]))
    proof: dict[str, Any] = {
        "proof_version": 1,
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
        mode = FakeTensorMode()
        with mode:
            ckpt = torch.load(reader, map_location="cpu", weights_only=False)
            proof["serialization"]["torch_load_ok"] = True
            proof["serialization"]["top_level_type"] = type(ckpt).__name__
            if not isinstance(ckpt, dict):
                raise TypeError(f"top-level checkpoint is {type(ckpt).__name__}, expected dict")

            top_keys = sorted(str(k) for k in ckpt.keys())
            proof["serialization"]["top_level_keys"] = top_keys
            proof["serialization"]["has_model_key"] = "model" in ckpt
            proof["serialization"]["has_optimizer_key"] = "optimizer" in ckpt
            proof["serialization"]["has_meta_key"] = "meta" in ckpt
            if "model" not in ckpt:
                raise KeyError('canonical trainer requires top-level key "model"')
            if not isinstance(ckpt["model"], dict):
                raise TypeError(f'ckpt["model"] is {type(ckpt["model"]).__name__}, expected state-dict mapping')

            model, cfg = build_current_canonical_model(vocab_size)
            expected = model.state_dict()
            actual = ckpt["model"]

            missing = sorted(set(expected.keys()) - set(actual.keys()))
            unexpected = sorted(set(actual.keys()) - set(expected.keys()))
            common = sorted(set(expected.keys()) & set(actual.keys()))
            shape_mismatches = [
                {
                    "key": k,
                    "expected": list(shape_tuple(expected[k])),
                    "actual": list(shape_tuple(actual[k])),
                }
                for k in common
                if shape_tuple(expected[k]) != shape_tuple(actual[k])
            ]
            dtype_mismatches = [
                {
                    "key": k,
                    "expected": str(expected[k].dtype),
                    "actual": str(actual[k].dtype),
                }
                for k in common
                if expected[k].dtype != actual[k].dtype
            ]

            proof["strict_model_contract"].update(
                {
                    "expected_key_count": len(expected),
                    "actual_key_count": len(actual),
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                    "shape_mismatches": shape_mismatches,
                    "dtype_mismatches": dtype_mismatches,
                }
            )

            strict_result = model.load_state_dict(actual, strict=True)
            proof["strict_model_contract"]["strict_load_ok"] = True
            proof["strict_model_contract"]["strict_result"] = str(strict_result)

            if "optimizer" in ckpt:
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=1.0e-4,
                    betas=(0.9, 0.95),
                    eps=1.0e-8,
                    weight_decay=0.0,
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
            proof["serialization"]["meta_keys"] = sorted(str(k) for k in meta.keys()) if isinstance(meta, dict) else []

            proof["passed"] = (
                proof["serialization"]["torch_load_ok"]
                and proof["serialization"]["has_model_key"]
                and not missing
                and not unexpected
                and not shape_mismatches
                and proof["strict_model_contract"]["strict_load_ok"]
                and (
                    not proof["optimizer_contract"]["present"]
                    or proof["optimizer_contract"]["load_state_dict_ok"]
                )
            )

    except Exception as exc:
        proof["error"] = f"{type(exc).__name__}: {exc}"
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
