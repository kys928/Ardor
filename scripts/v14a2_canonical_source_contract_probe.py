#!/usr/bin/env python3
"""Prove the ORIGINAL v14a2 payload satisfies the restored canonical model/optimizer contract."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CORTEX_ROOT = REPO_ROOT / "Cerebrum" / "Cortex"
for p in (str(CORTEX_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/ardor_torchinductor_cache"
os.environ["TRITON_CACHE_DIR"] = "/tmp/ardor_triton_cache"
os.environ["TORCH_HOME"] = "/tmp/ardor_torch_home"

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from ardor_config import ArdorConfig
from broca_decoder import ArdorDecoder
from scripts.runpod_control import s3_client, bucket
from scripts.v14a2_checkpoint_contract_probe import S3RangeReader, shape_tuple

SOURCE_KEY = "Ardor/training/runs/v14_promoted_candidates/v14a2_local_anti_loop_margin_best_u600.pt"
BASE_KEY = "Ardor/training/runs/sft_v13b_routing_stage_trainer_v10_27_guard4_heavy_four_guard_correlation_phrase_landing/checkpoints/best_model.pt"


def range_load(client: Any, volume: str, key: str):
    head = client.head_object(Bucket=volume, Key=key)
    reader = S3RangeReader(client, volume, key, int(head["ContentLength"]))
    obj = torch.load(reader, map_location="cpu", weights_only=False)
    return obj, head, reader


def main() -> int:
    client = s3_client()
    volume = bucket()
    out: dict[str, Any] = {
        "passed": False,
        "source_key": SOURCE_KEY,
        "base_key": BASE_KEY,
        "strict_model_contract": {},
        "optimizer_contract": {},
    }
    try:
        with FakeTensorMode():
            source, source_head, source_reader = range_load(client, volume, SOURCE_KEY)
            base, base_head, base_reader = range_load(client, volume, BASE_KEY)

            if not isinstance(source, dict) or not isinstance(source.get("model_state_dict"), dict):
                raise KeyError("original v14a2 requires model_state_dict")
            if not isinstance(base, dict) or not isinstance(base.get("config"), dict):
                raise KeyError("v13b base requires config")

            cfg = ArdorConfig.from_dict(dict(base["config"]))
            cfg.validate()
            model = ArdorDecoder(cfg)
            actual = source["model_state_dict"]
            expected = model.state_dict()

            common = sorted(set(expected) & set(actual))
            missing = sorted(set(expected) - set(actual))
            unexpected = sorted(set(actual) - set(expected))
            shape_mismatches = [
                {
                    "key": key,
                    "expected": list(shape_tuple(expected[key])),
                    "actual": list(shape_tuple(actual[key])),
                }
                for key in common
                if shape_tuple(expected[key]) != shape_tuple(actual[key])
            ]

            strict_result = model.load_state_dict(actual, strict=True)
            strict_ok = True

            out["model_config"] = model.model_config()
            out["architecture_assertions"] = {
                "use_rope": bool(model.use_rope),
                "position_embed_is_none": model.position_embed is None,
                "attention_use_rope": bool(model.blocks[0].attn.use_rope),
                "rope_theta": float(model.blocks[0].attn.rope_theta),
                "persistent_rope_keys": [key for key in expected if "rope" in key],
            }
            out["strict_model_contract"] = {
                "expected_key_count": len(expected),
                "actual_key_count": len(actual),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "shape_mismatches": shape_mismatches,
                "strict_load_ok": strict_ok,
                "strict_result": str(strict_result),
            }

            if isinstance(source.get("optimizer_state_dict"), dict):
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=1e-4,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.0,
                )
                optimizer.load_state_dict(source["optimizer_state_dict"])
                out["optimizer_contract"] = {
                    "present": True,
                    "load_state_dict_ok": True,
                    "param_groups": len(optimizer.param_groups),
                }
            else:
                out["optimizer_contract"] = {"present": False}

            out["source"] = {
                "etag": str(source_head.get("ETag", "")).strip('"'),
                "size": int(source_head["ContentLength"]),
                "trainer": source.get("trainer"),
                "update": source.get("update"),
            }
            out["base"] = {
                "etag": str(base_head.get("ETag", "")).strip('"'),
                "size": int(base_head["ContentLength"]),
                "config": base.get("config"),
            }
            out["s3_range_io"] = {
                "source_bytes_fetched": source_reader.bytes_fetched,
                "source_range_requests": source_reader.range_requests,
                "base_bytes_fetched": base_reader.bytes_fetched,
                "base_range_requests": base_reader.range_requests,
            }

            out["passed"] = bool(
                not missing
                and not unexpected
                and not shape_mismatches
                and strict_ok
                and out["architecture_assertions"]["use_rope"]
                and out["architecture_assertions"]["position_embed_is_none"]
                and out["architecture_assertions"]["attention_use_rope"]
                and not out["architecture_assertions"]["persistent_rope_keys"]
                and (
                    not out["optimizer_contract"].get("present")
                    or out["optimizer_contract"].get("load_state_dict_ok") is True
                )
            )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()

    Path("v14a2_canonical_source_contract.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
