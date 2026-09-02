#!/usr/bin/env python3
"""Diagnose the legacy payload inside repaired v14a2 without weakening canonical pass criteria."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CORTEX_ROOT = REPO_ROOT / "Cerebrum" / "Cortex"
for path in (str(REPO_ROOT), str(CORTEX_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/ardor_torchinductor_cache"
os.environ["TRITON_CACHE_DIR"] = "/tmp/ardor_triton_cache"
os.environ["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT), str(CORTEX_ROOT)))

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from scripts.runpod_control import s3_client, bucket
from scripts.v14a2_checkpoint_contract_probe import (
    CHECKPOINT_KEY, TOKENIZER_KEY, S3RangeReader,
    build_current_canonical_model, shape_tuple, trainer,
)


def main() -> int:
    trainer.ARDOR_ROOT = REPO_ROOT
    client = s3_client()
    volume = bucket()
    head = client.head_object(Bucket=volume, Key=CHECKPOINT_KEY)
    tok = client.get_object(Bucket=volume, Key=TOKENIZER_KEY)["Body"].read()
    tok_path = Path("/tmp/ardor_tokenizer_v9.json")
    tok_path.write_bytes(tok)
    vocab_size = int(trainer.tokenizer_vocab_size(tok_path))
    reader = S3RangeReader(client, volume, CHECKPOINT_KEY, int(head["ContentLength"]))

    out: dict[str, Any] = {
        "checkpoint": CHECKPOINT_KEY,
        "vocab_size": vocab_size,
        "canonical_direct_compatible": False,
        "legacy_payload_model_diagnostic": {},
        "legacy_payload_optimizer_diagnostic": {},
    }
    try:
        with FakeTensorMode():
            ckpt = torch.load(reader, map_location="cpu", weights_only=False)
            out["top_level_keys"] = sorted(str(k) for k in ckpt)
            out["canonical_direct_compatible"] = "model" in ckpt
            legacy = ckpt.get("model_state_dict")
            if not isinstance(legacy, dict):
                raise KeyError("model_state_dict missing or not a mapping")

            model, _ = build_current_canonical_model(vocab_size)
            expected = model.state_dict()
            common = sorted(set(expected) & set(legacy))
            missing = sorted(set(expected) - set(legacy))
            unexpected = sorted(set(legacy) - set(expected))
            shape_mismatches = [
                {"key": k, "expected": list(shape_tuple(expected[k])), "actual": list(shape_tuple(legacy[k]))}
                for k in common if shape_tuple(expected[k]) != shape_tuple(legacy[k])
            ]
            out["legacy_payload_model_diagnostic"] = {
                "expected_key_count": len(expected),
                "actual_key_count": len(legacy),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "shape_mismatches": shape_mismatches,
            }
            try:
                r = model.load_state_dict(legacy, strict=True)
                out["legacy_payload_model_diagnostic"]["strict_load_ok"] = True
                out["legacy_payload_model_diagnostic"]["strict_result"] = str(r)
            except Exception as exc:
                out["legacy_payload_model_diagnostic"]["strict_load_ok"] = False
                out["legacy_payload_model_diagnostic"]["strict_load_error"] = f"{type(exc).__name__}: {exc}"

            legacy_opt = ckpt.get("optimizer_state_dict")
            if isinstance(legacy_opt, dict):
                try:
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
                    opt.load_state_dict(legacy_opt)
                    out["legacy_payload_optimizer_diagnostic"] = {
                        "present": True,
                        "load_state_dict_ok": True,
                        "param_groups": len(opt.param_groups),
                    }
                except Exception as exc:
                    out["legacy_payload_optimizer_diagnostic"] = {
                        "present": True,
                        "load_state_dict_ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            else:
                out["legacy_payload_optimizer_diagnostic"] = {"present": False}
    except BaseException as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
    finally:
        out["s3_range_io"] = {"bytes_fetched": reader.bytes_fetched, "range_requests": reader.range_requests}
        Path("v14a2_payload_diagnostic.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
