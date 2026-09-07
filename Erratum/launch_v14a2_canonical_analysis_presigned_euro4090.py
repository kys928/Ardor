#!/usr/bin/env python3
"""Relaunch canonical v14a2 analysis over presigned S3 on exact EU-RO-1 RTX 4090.

This wrapper intentionally changes only RunPod placement and HTTP client.
The underlying presigned transport, checkpoint SHA-256 verification, runtime
pin, and scientific evaluator remain those in
launch_v14a2_canonical_analysis_presigned.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Erratum import launch_v14a2_canonical_analysis_presigned as base

EXPECTED_DATACENTER = "EU-RO-1"
EXACT_GPU = "NVIDIA GeForce RTX 4090"

base.JOB_ID = "v14a2-canonical-analysis-presigned-s3-euro4090-20260907"
base.CANDIDATE_GPUS = [EXACT_GPU]

_original_runpod_request = base.runpod_request


def _curl_post(path: str, payload: dict[str, Any]) -> Any:
    marker = "__ARDOR_HTTP_STATUS__:"
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "--request",
            "POST",
            "--url",
            f"https://rest.runpod.io/v1{path}",
            "--header",
            f"Authorization: Bearer {os.environ['RUNPOD_API_KEY']}",
            "--header",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            "--write-out",
            f"\\n{marker}%{{http_code}}",
        ],
        input=json.dumps(payload, separators=(",", ":")),
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"RunPod curl POST {path} failed rc={proc.returncode}: {proc.stderr.strip()}")
    if marker not in proc.stdout:
        raise RuntimeError(f"RunPod curl POST {path} returned no HTTP marker: {proc.stdout[:1000]!r}")
    body, raw_status = proc.stdout.rsplit(marker, 1)
    body = body.rstrip("\n")
    status = int(raw_status.strip())
    if status < 200 or status >= 300:
        raise RuntimeError(f"RunPod curl POST {path} failed: HTTP {status}: {body[:4000]}")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"RunPod curl POST {path} returned non-JSON: {body[:4000]!r}") from exc


def _placement_pinned_runpod_request(method: str, path: str, payload: Any | None = None) -> Any:
    if method == "POST" and path == "/pods":
        datacenter = base.required_env("RUNPOD_DATACENTER_ID")
        if datacenter != EXPECTED_DATACENTER:
            raise RuntimeError(
                f"Refusing unexpected datacenter for canonical analysis: {datacenter!r}; "
                f"expected {EXPECTED_DATACENTER!r}"
            )
        if not isinstance(payload, dict):
            raise RuntimeError("RunPod create payload must be a dict")
        payload = dict(payload)
        payload["gpuTypeIds"] = [EXACT_GPU]
        payload["gpuTypePriority"] = "availability"
        payload["dataCenterIds"] = [EXPECTED_DATACENTER]
        payload["dataCenterPriority"] = "availability"
        payload.pop("networkVolumeId", None)
        payload.pop("volumeMountPath", None)
        return _curl_post(path, payload)
    return _original_runpod_request(method, path, payload)


base.runpod_request = _placement_pinned_runpod_request


if __name__ == "__main__":
    raise SystemExit(base.main())
