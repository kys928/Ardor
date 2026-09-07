#!/usr/bin/env python3
"""Relaunch canonical v14a2 analysis over presigned S3 on exact EU-RO-1 RTX 4090.

This wrapper intentionally changes only RunPod placement. The underlying
presigned transport, checkpoint SHA-256 verification, runtime pin, and
scientific evaluator remain those in launch_v14a2_canonical_analysis_presigned.
"""
from __future__ import annotations

import os
from typing import Any

from Erratum import launch_v14a2_canonical_analysis_presigned as base

EXPECTED_DATACENTER = "EU-RO-1"
EXACT_GPU = "NVIDIA GeForce RTX 4090"

# New control-plane lineage for this concrete placement retry.
base.JOB_ID = "v14a2-canonical-analysis-presigned-s3-euro4090-20260907"
base.CANDIDATE_GPUS = [EXACT_GPU]

_original_runpod_request = base.runpod_request


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
    return _original_runpod_request(method, path, payload)


base.runpod_request = _placement_pinned_runpod_request


if __name__ == "__main__":
    raise SystemExit(base.main())
