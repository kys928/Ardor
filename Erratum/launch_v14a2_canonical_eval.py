#!/usr/bin/env python3
"""Launch the single canonical v14a2 behavioral evaluation as a detached RunPod job.

This launcher is intentionally one-purpose. It accepts no checkpoint path, evaluator,
image, GPU count, or model override from user input. The pod executes an exact pinned
Ardor code commit so the scientific architecture/evaluator contract cannot drift with
an image tag.
"""
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

JOB_ID = "v14a2-canonical-eval-20260904"
RUNNER = "canonical_eval_v14a2"
EXPECTED_IMAGE = "ghcr.io/kys928/ardor-runpod:latest"
AUDIT_CODE_SHA = "42db9328223b3d04f1b3199a65483c33c7953663"
CANDIDATE_GPUS = [
    "NVIDIA GeForce RTX 5090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX PRO 4500 Blackwell",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
]


def main() -> int:
    allowed = {
        item.strip()
        for item in required_env("RUNPOD_ALLOWED_GPU_TYPES").split(",")
        if item.strip()
    }
    gpu_types = [gpu for gpu in CANDIDATE_GPUS if gpu in allowed]
    if not gpu_types:
        raise RuntimeError("No canonical-evaluation GPU candidates are allowed")

    image = required_env("RUNPOD_IMAGE_NAME")
    if image != EXPECTED_IMAGE:
        raise RuntimeError(f"Refusing unexpected canonical-evaluation image: {image}")

    control_run_id = f"run-{int(time.time())}-{secrets.token_hex(4)}"
    job: dict[str, Any] = {
        "version": 1,
        "id": JOB_ID,
        "kind": "compute",
        "task": {"runner": RUNNER},
        "gpu": {
            "type_ids": gpu_types,
            "count": 1,
            "cloud_type": "SECURE",
        },
        "lifecycle": {"mode": "detached"},
        "_control_run_id": control_run_id,
    }
    encoded_job = base64.b64encode(
        json.dumps(job, sort_keys=True).encode("utf-8")
    ).decode("ascii")

    fixed_start = (
        "rm -rf /tmp/ArdorAudit && "
        "git init /tmp/ArdorAudit && "
        "cd /tmp/ArdorAudit && "
        "git remote add origin https://github.com/kys928/Ardor.git && "
        f"git fetch --depth 1 origin {AUDIT_CODE_SHA} && "
        "git checkout --detach FETCH_HEAD && "
        f"test \"$(git rev-parse HEAD)\" = \"{AUDIT_CODE_SHA}\" && "
        "/opt/Ardor/.venv/bin/python scripts/runpod_worker.py"
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
        cost = pod_cost(pod)
        record = {
            "schema_version": 1,
            "project": "Ardor",
            "job_id": JOB_ID,
            "control_run_id": control_run_id,
            "pod_id": pod_id,
            "created_at": utc_now(),
            "mode": "detached",
            "status_key": status_key(JOB_ID, control_run_id),
            "hourly_cost_usd": cost,
            "hourly_cap_usd": None,
            "timeout_minutes": None,
            "network_volume_id": required_env("RUNPOD_NETWORK_VOLUME_ID"),
            "datacenter_id": required_env("RUNPOD_DATACENTER_ID"),
            "image_name": image,
            "audit_code_sha": AUDIT_CODE_SHA,
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
