#!/usr/bin/env python3
"""Independently verify the completed v14a2 canonical migration through RunPod S3."""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runpod_control import bucket, s3_client

JOB_ID = "v14a2-canonical-migration-20260903"
REPORT_PREFIX = f"ardor-control/runs/{JOB_ID}/"
SOURCE_KEY = (
    "Ardor/training/runs/v14_promoted_candidates/"
    "v14a2_local_anti_loop_margin_best_u600.pt"
)
DESTINATION_KEY = (
    "Ardor/training/runs/v14_promoted_candidates/"
    "v14a2_local_anti_loop_margin_best_u600_CANONICAL_FULLSTATE.pt"
)
EXPECTED_SOURCE_SIZE = 12_182_658_805
EXPECTED_MODEL_CONFIG = {
    "vocab_size": 52224,
    "hidden_size": 1536,
    "n_layers": 33,
    "n_heads": 24,
    "max_len": 2048,
    "ff_mult": 4,
    "dropout": 0.0,
    "attn_dropout": 0.0,
    "resid_dropout": 0.0,
    "layernorm_eps": 1e-5,
    "use_rope": True,
    "rope_theta": 10000.0,
}


def main() -> int:
    client = s3_client()
    volume = bucket()
    listing = client.list_objects_v2(Bucket=volume, Prefix=REPORT_PREFIX)
    reports = [
        item for item in listing.get("Contents", [])
        if str(item.get("Key", "")).endswith("/canonical_migration.json")
    ]
    if not reports:
        raise RuntimeError(f"No canonical migration report under {REPORT_PREFIX}")
    latest = max(reports, key=lambda item: item["LastModified"])
    report_key = str(latest["Key"])
    report_obj = client.get_object(Bucket=volume, Key=report_key)
    report = json.loads(report_obj["Body"].read().decode("utf-8"))

    destination_head = client.head_object(Bucket=volume, Key=DESTINATION_KEY)
    source_head = client.head_object(Bucket=volume, Key=SOURCE_KEY)
    destination_prefix = client.get_object(
        Bucket=volume, Key=DESTINATION_KEY, Range="bytes=0-31"
    )["Body"].read()
    destination_suffix = client.get_object(
        Bucket=volume, Key=DESTINATION_KEY, Range="bytes=-32"
    )["Body"].read()

    post = report.get("post_write_contract") or {}
    model = post.get("model") or {}
    optimizer = post.get("optimizer") or {}
    destination_size = int(destination_head["ContentLength"])
    source_size = int(source_head["ContentLength"])

    checks = {
        "worker_report_passed": report.get("passed") is True,
        "strict_model_load_ok": model.get("strict_load_ok") is True,
        "zero_missing_keys": model.get("missing_keys") == [],
        "zero_unexpected_keys": model.get("unexpected_keys") == [],
        "zero_shape_mismatches": model.get("shape_mismatches") == [],
        "optimizer_load_ok": optimizer.get("load_state_dict_ok") is True,
        "round_trip_exact_model": post.get("round_trip_exact_model") is True,
        "round_trip_exact_optimizer": post.get("round_trip_exact_optimizer") is True,
        "model_config_exact": report.get("model_config") == EXPECTED_MODEL_CONFIG,
        "destination_size_matches_report": destination_size == int(report.get("destination_size_bytes", -1)),
        "source_size_unchanged": source_size == EXPECTED_SOURCE_SIZE,
        "destination_range_readable": len(destination_prefix) > 0 and len(destination_suffix) > 0,
    }
    passed = all(checks.values())
    result = {
        "passed": passed,
        "report_key": report_key,
        "report_last_modified": latest["LastModified"].isoformat(),
        "report": report,
        "independent_s3": {
            "source_key": SOURCE_KEY,
            "source_size_bytes": source_size,
            "source_etag": str(source_head.get("ETag", "")).strip('"'),
            "destination_key": DESTINATION_KEY,
            "destination_size_bytes": destination_size,
            "destination_etag": str(destination_head.get("ETag", "")).strip('"'),
            "destination_last_modified": destination_head["LastModified"].isoformat(),
            "prefix_hex": destination_prefix.hex(),
            "suffix_hex": destination_suffix.hex(),
        },
        "checks": checks,
    }
    Path("v14a2_post_migration_proof.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
