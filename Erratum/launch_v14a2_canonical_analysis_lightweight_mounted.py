#!/usr/bin/env python3
"""Launch canonical v14a2 analysis on a lightweight image with the Network Volume mounted.

Infrastructure-only fallback for RunPod environments where the prebuilt Ardor image
is slow to bootstrap. Scientific behavior remains pinned to the same canonical
analysis commit, checkpoint path, tokenizer/data paths, Torch version, GPU count,
RoPE/model contract, and worker. Only the container/bootstrap layer changes.
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

from Erratum import launch_v14a2_canonical_analysis as base

EXPECTED_DATACENTER = "EU-RO-1"
EXACT_GPU = "NVIDIA RTX PRO 4500 Blackwell"
LIGHT_IMAGE = "python:3.11.16-slim-bookworm"

base.JOB_ID = "v14a2-canonical-analysis-lightweight-mounted-euro4500-r2-20260907"
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


def _lightweight_start() -> str:
    return f'''set -euo pipefail
apt-get update >/dev/null
apt-get install -y --no-install-recommends git ca-certificates >/dev/null
rm -rf /var/lib/apt/lists/*
rm -rf /tmp/ArdorAnalysis
mkdir -p /tmp/ArdorAnalysis
cd /tmp/ArdorAnalysis
git init
git remote add origin https://github.com/kys928/Ardor.git
git fetch --depth 1 origin {base.AUDIT_CODE_SHA}
git checkout --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "{base.AUDIT_CODE_SHA}"
python -m py_compile Erratum/canonical_analysis_v14a2.py scripts/runpod_canonical_analysis_worker.py
python -m pip install --disable-pip-version-check --index-url {base.TORCH_INDEX} torch=={base.TORCH_VERSION}
python -m pip install --disable-pip-version-check 'numpy>=1.26,<3' 'tokenizers>=0.15,<1' 'tqdm>=4.66,<5'
python -c "import torch; assert torch.__version__.startswith('{base.TORCH_VERSION}'); assert torch.cuda.is_available(); print(f'[canonical-analysis-runtime] torch={{torch.__version__}} cuda={{torch.version.cuda}} device={{torch.cuda.get_device_name(0)}}')"
python scripts/runpod_canonical_analysis_worker.py
'''


def _lightweight_request(method: str, path: str, payload=None):
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
        payload["networkVolumeId"] = base.required_env("RUNPOD_NETWORK_VOLUME_ID")
        payload["volumeMountPath"] = "/workspace"
        payload["imageName"] = LIGHT_IMAGE
        payload["containerDiskInGb"] = 60
        payload["dockerEntrypoint"] = ["/bin/bash", "-lc"]
        payload["dockerStartCmd"] = [_lightweight_start()]
        payload.pop("containerRegistryAuthId", None)
        return _curl_post(path, payload)
    return _original_runpod_request(method, path, payload)


base.runpod_request = _lightweight_request


if __name__ == "__main__":
    raise SystemExit(base.main())
