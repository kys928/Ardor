#!/usr/bin/env python3
"""Launch the canonical v14a2 analysis with presigned S3 on a tiny base image.

This is an infrastructure-only fallback. It keeps the canonical checkpoint,
scientific evaluator SHA, evaluator inputs, tokenizer/data, Torch version and
cu128 index inherited from launch_v14a2_canonical_analysis_presigned. It only
replaces the prebuilt Ardor Pod image with an official Python slim image and
installs the verified runtime dependencies explicitly after Pod start.
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
EXACT_GPU = "NVIDIA RTX PRO 4500 Blackwell"
LIGHT_IMAGE = "python:3.11.16-slim-bookworm"

base.JOB_ID = "v14a2-canonical-analysis-presigned-lightweight-euro4500-20260907"
base.CANDIDATE_GPUS = [EXACT_GPU]
base.EXPECTED_IMAGE = LIGHT_IMAGE
os.environ["RUNPOD_IMAGE_NAME"] = LIGHT_IMAGE

_original_runpod_request = base.runpod_request


def _curl_post(path: str, payload: dict[str, Any]) -> Any:
    marker = "__ARDOR_HTTP_STATUS__:"
    proc = subprocess.run(
        [
            "curl", "-sS", "--request", "POST",
            "--url", f"https://rest.runpod.io/v1{path}",
            "--header", f"Authorization: Bearer {os.environ['RUNPOD_API_KEY']}",
            "--header", "Content-Type: application/json",
            "--data-binary", "@-",
            "--write-out", f"\\n{marker}%{{http_code}}",
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
        raise RuntimeError(f"RunPod curl POST {path} returned no HTTP marker")
    body, raw_status = proc.stdout.rsplit(marker, 1)
    status = int(raw_status.strip())
    body = body.rstrip("\n")
    if not 200 <= status < 300:
        raise RuntimeError(f"RunPod curl POST {path} failed: HTTP {status}: {body[:4000]}")
    return json.loads(body)


def _lightweight_start(original: str) -> str:
    staging_boundary = "PY\nrm -rf /workspace/Ardor"
    if staging_boundary not in original:
        raise RuntimeError("Could not locate staging boundary in canonical bootstrap")
    original = original.replace(
        staging_boundary,
        "PY\napt-get update >/dev/null && apt-get install -y --no-install-recommends git ca-certificates >/dev/null && rm -rf /var/lib/apt/lists/*\nrm -rf /workspace/Ardor",
        1,
    )

    old_install = (
        f"/root/.local/bin/uv pip install --python /opt/Ardor/.venv/bin/python "
        f"--index-url {base.TORCH_INDEX} --reinstall torch=={base.TORCH_VERSION}"
    )
    new_install = (
        f"python -m pip install --disable-pip-version-check --index-url {base.TORCH_INDEX} "
        f"--reinstall torch=={base.TORCH_VERSION}\n"
        "python -m pip install --disable-pip-version-check "
        "'numpy>=1.26,<3' 'tokenizers>=0.15,<1' 'tqdm>=4.66,<5'"
    )
    if old_install not in original:
        raise RuntimeError("Could not locate canonical Torch install command")
    original = original.replace(old_install, new_install, 1)
    original = original.replace("/opt/Ardor/.venv/bin/python -c", "python -c", 1)
    original = original.replace(
        "/opt/Ardor/.venv/bin/python scripts/runpod_canonical_analysis_worker.py",
        "python scripts/runpod_canonical_analysis_worker.py",
        1,
    )
    return original


def _placement_pinned_request(method: str, path: str, payload: Any | None = None) -> Any:
    if method == "POST" and path == "/pods":
        if base.required_env("RUNPOD_DATACENTER_ID") != EXPECTED_DATACENTER:
            raise RuntimeError("Refusing unexpected datacenter for canonical analysis")
        if not isinstance(payload, dict):
            raise RuntimeError("RunPod create payload must be a dict")
        payload = dict(payload)
        payload["gpuTypeIds"] = [EXACT_GPU]
        payload["gpuTypePriority"] = "availability"
        payload["dataCenterIds"] = [EXPECTED_DATACENTER]
        payload["dataCenterPriority"] = "availability"
        payload["imageName"] = LIGHT_IMAGE
        payload.pop("networkVolumeId", None)
        payload.pop("volumeMountPath", None)
        start = payload.get("dockerStartCmd")
        if not isinstance(start, list) or len(start) != 1 or not isinstance(start[0], str):
            raise RuntimeError("Unexpected canonical dockerStartCmd contract")
        payload["dockerStartCmd"] = [_lightweight_start(start[0])]
        return _curl_post(path, payload)
    return _original_runpod_request(method, path, payload)


base.runpod_request = _placement_pinned_request


if __name__ == "__main__":
    raise SystemExit(base.main())
