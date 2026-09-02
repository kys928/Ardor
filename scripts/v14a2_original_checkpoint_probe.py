#!/usr/bin/env python3
"""Inspect original v14a2 checkpoint metadata/state without materializing tensor payload."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from scripts.runpod_control import s3_client, bucket
from scripts.v14a2_checkpoint_contract_probe import S3RangeReader

KEY = "Ardor/training/runs/v14_promoted_candidates/v14a2_local_anti_loop_margin_best_u600.pt"


def main() -> int:
    s3 = s3_client()
    b = bucket()
    head = s3.head_object(Bucket=b, Key=KEY)
    reader = S3RangeReader(s3, b, KEY, int(head["ContentLength"]))
    with FakeTensorMode():
        ckpt = torch.load(reader, map_location="cpu", weights_only=False)

    state = None
    state_key = None
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if isinstance(ckpt.get(k), dict):
                state_key = k
                state = ckpt[k]
                break

    out = {
        "key": KEY,
        "size": int(head["ContentLength"]),
        "etag": str(head.get("ETag", "")).strip('"'),
        "top_level_type": type(ckpt).__name__,
        "top_level_keys": sorted(str(k) for k in ckpt.keys()) if isinstance(ckpt, dict) else [],
        "config": ckpt.get("config") if isinstance(ckpt, dict) else None,
        "args": ckpt.get("args") if isinstance(ckpt, dict) else None,
        "trainer": ckpt.get("trainer") if isinstance(ckpt, dict) else None,
        "update": ckpt.get("update") if isinstance(ckpt, dict) else None,
        "state_key": state_key,
        "state_key_count": len(state) if isinstance(state, dict) else None,
        "has_position_embed_weight": bool(isinstance(state, dict) and "position_embed.weight" in state),
        "position_embed_shape": list(state["position_embed.weight"].shape)
        if isinstance(state, dict) and "position_embed.weight" in state else None,
        "range_requests": reader.range_requests,
        "bytes_fetched": reader.bytes_fetched,
    }
    Path("v14a2_original_checkpoint_probe.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
