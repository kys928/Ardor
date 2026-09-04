#!/usr/bin/env python3
"""Launch the fixed extensive analysis of the frozen canonical v14a2 parent on RunPod."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runpod_control import (
    active_key,
    delete_pod,
    internal_put_json,
    pod_cost,
    read_json_object,
    required_env,
    runpod_request,
    status_key,
    utc_now,
)

JOB_ID = "v14a2-canonical-analysis-explicit-contract-rtxpro4500-20260904"
RUNNER = "canonical_analysis_v14a2"
EXPECTED_IMAGE = "ghcr.io/kys928/ardor-runpod:latest"
AUDIT_CODE_SHA = "e052e0d3acb43643093c44990879482af0f5cde9"
TORCH_VERSION = "2.11.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
CANDIDATE_GPUS = [
    "NVIDIA RTX PRO 4500 Blackwell",
]


def main() -> int:
    allowed = {
        item.strip()
        for item in required_env("RUNPOD_ALLOWED_GPU_TYPES").split(",")
        if item.strip()
    }
    gpu_types = [gpu for gpu in CANDIDATE_GPUS if gpu in allowed]
    if gpu_types != CANDIDATE_GPUS:
        raise RuntimeError("Canonical analysis RTX PRO 4500 GPU is not allowed")

    image = required_env("RUNPOD_IMAGE_NAME")
    if image != EXPECTED_IMAGE:
        raise RuntimeError(f"Refusing unexpected canonical-analysis image: {image}")

    control_run_id = f"run-{int(time.time())}-{secrets.token_hex(4)}"
    job: dict[str, Any] = {
        "version": 1,
        "id": JOB_ID,
        "kind": "compute",
        "task": {"runner": RUNNER},
        "gpu": {"type_ids": gpu_types, "count": 1, "cloud_type": "SECURE"},
        "lifecycle": {"mode": "detached"},
        "_control_run_id": control_run_id,
    }
    encoded_job = base64.b64encode(
        json.dumps(job, sort_keys=True).encode("utf-8")
    ).decode("ascii")

    fixed_start = (
        "rm -rf /tmp/ArdorAnalysis && "
        "git init /tmp/ArdorAnalysis && "
        "cd /tmp/ArdorAnalysis && "
        "git remote add origin https://github.com/kys928/Ardor.git && "
        f"git fetch --depth 1 origin {AUDIT_CODE_SHA} && "
        "git checkout --detach FETCH_HEAD && "
        f"test \"$(git rev-parse HEAD)\" = \"{AUDIT_CODE_SHA}\" && "
        "python -m py_compile Erratum/canonical_analysis_v14a2.py scripts/runpod_canonical_analysis_worker.py && "
        f"/root/.local/bin/uv pip install --python /opt/Ardor/.venv/bin/python "
        f"--index-url {TORCH_INDEX} --reinstall torch=={TORCH_VERSION} && "
        "/opt/Ardor/.venv/bin/python -c \"import torch; "
        f"assert torch.__version__.startswith('{TORCH_VERSION}'); "
        "assert torch.cuda.is_available(); "
        "print(f'[canonical-analysis-runtime] torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}')\" && "
        "/opt/Ardor/.venv/bin/python scripts/runpod_canonical_analysis_worker.py"
    )

    payload: dict[str, Any] = {
        "name": f"ardor-{JOB_ID}"[:191],
        "computeType": "GPU",
        "cloudType": "SECURE",
        "gpuTypeIds": gpu_types,
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "dataCenterIds": [required_env("RUNPOD_DATACENTER_ID")],
        "dataCenterPriority": "availability",
        "networkVolumeId": required_env("RUNPOD_NETWORK_VOLUME_ID"),
        "volumeMountPath": "/workspace",
        "containerDiskInGb": 50,
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": [fixed_start],
        "ports": [],
        "env": {
            "ARDOR_JOB_B64": encoded_job,
            "ARDOR_AUDIT_CODE_SHA": AUDIT_CODE_SHA,
            "ARDOR_ANALYSIS_TORCH_VERSION": TORCH_VERSION,
            "ARDOR_ANALYSIS_TORCH_INDEX": TORCH_INDEX,
            "PYTHONUNBUFFERED": "1",
        },
        "interruptible": False,
        "imageName": image,
    }
    registry_auth = os.environ.get("RUNPOD_CONTAINER_REGISTRY_AUTH_ID", "").strip()
    if registry_auth:
        payload["containerRegistryAuthId"] = registry_auth

    pod_id: str | None = None
    active_name = active_key(control_run_id)
    try:
        pod = runpod_request("POST", "/pods", payload)
        if not isinstance(pod, dict) or not pod.get("id"):
            raise RuntimeError(f"RunPod create response did not contain a Pod id: {pod!r}")
        pod_id = str(pod["id"])
        record = {
            "schema_version": 1,
            "project": "Ardor",
            "job_id": JOB_ID,
            "control_run_id": control_run_id,
            "pod_id": pod_id,
            "created_at": utc_now(),
            "mode": "detached",
            "status_key": status_key(JOB_ID, control_run_id),
            "hourly_cost_usd": pod_cost(pod),
            "hourly_cap_usd": None,
            "timeout_minutes": None,
            "network_volume_id": required_env("RUNPOD_NETWORK_VOLUME_ID"),
            "datacenter_id": required_env("RUNPOD_DATACENTER_ID"),
            "image_name": image,
            "audit_code_sha": AUDIT_CODE_SHA,
            "torch_version": TORCH_VERSION,
            "torch_index": TORCH_INDEX,
            "github_sha": os.environ.get("GITHUB_SHA", "").strip() or None,
            "task": {"runner": RUNNER},
            "gpu": job["gpu"],
        }
        internal_put_json(active_name, record)
        print(json.dumps({"launch": record}, indent=2))
        return 0
    except Exception:
        if pod_id is not None and read_json_object(active_name) is None:
            try:
                delete_pod(pod_id)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
