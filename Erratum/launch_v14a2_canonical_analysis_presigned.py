#!/usr/bin/env python3
"""Launch canonical v14a2 analysis without mounting the Network Volume.

GitHub generates short-lived presigned S3 URLs for only the exact scientific
inputs and output objects needed by this run. The GPU Pod receives no S3
credentials. The scientific analysis itself remains pinned to AUDIT_CODE_SHA.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runpod_control import (
    active_key,
    bucket,
    delete_pod,
    internal_put_json,
    pod_cost,
    read_json_object,
    required_env,
    runpod_request,
    s3_client,
    status_key,
    utc_now,
)

JOB_ID = "v14a2-canonical-analysis-presigned-s3-20260905"
RUNNER = "canonical_analysis_v14a2"
EXPECTED_IMAGE = "ghcr.io/kys928/ardor-runpod:latest"
AUDIT_CODE_SHA = "e052e0d3acb43643093c44990879482af0f5cde9"
TORCH_VERSION = "2.11.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
URL_TTL_SECONDS = 6 * 60 * 60
CHECKPOINT_SHA256 = "500252cead0b6ff9825ff9c4ef8e878c00403f866f23911133c26fff56999816"
CHECKPOINT_SIZE = 12_182_834_443

INPUTS = [
    {
        "key": "Ardor/training/runs/v14_promoted_candidates/v14a2_local_anti_loop_margin_best_u600_CANONICAL_FULLSTATE.pt",
        "path": "training/runs/v14_promoted_candidates/v14a2_local_anti_loop_margin_best_u600_CANONICAL_FULLSTATE.pt",
        "sha256": CHECKPOINT_SHA256,
        "size": CHECKPOINT_SIZE,
    },
    {"key": "Ardor/tokenizer_v9.json", "path": "tokenizer_v9.json"},
    {
        "key": "Ardor/training/scripts/ardor_v14a2_local_anti_loop_margin_trainer.py",
        "path": "training/scripts/ardor_v14a2_local_anti_loop_margin_trainer.py",
    },
    {
        "key": "Ardor/training/data/v14_v3/dataset_v3_holdout_audit.jsonl",
        "path": "training/data/v14_v3/dataset_v3_holdout_audit.jsonl",
    },
    {
        "key": "Ardor/training/data/v14_v3/dataset_v3b_route_contrastive_balanced.jsonl",
        "path": "training/data/v14_v3/dataset_v3b_route_contrastive_balanced.jsonl",
    },
]

CANDIDATE_GPUS = [
    "NVIDIA GeForce RTX 5090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX PRO 4500 Blackwell",
]


def presigned_inputs() -> list[dict[str, Any]]:
    client = s3_client()
    volume = bucket()
    rows: list[dict[str, Any]] = []
    for item in INPUTS:
        key = str(item["key"])
        head = client.head_object(Bucket=volume, Key=key)
        actual_size = int(head.get("ContentLength", 0))
        expected_size = item.get("size")
        if expected_size is not None and actual_size != int(expected_size):
            raise RuntimeError(
                f"Authoritative S3 input size mismatch for {key}: "
                f"expected {expected_size}, got {actual_size}"
            )
        rows.append(
            {
                "key": key,
                "path": str(item["path"]),
                "size": actual_size,
                "sha256": item.get("sha256"),
                "url": client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": volume, "Key": key},
                    ExpiresIn=URL_TTL_SECONDS,
                ),
            }
        )
    return rows


def presigned_put(key: str) -> str:
    return s3_client().generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket(), "Key": key},
        ExpiresIn=URL_TTL_SECONDS,
    )


def main() -> int:
    allowed = {
        item.strip()
        for item in required_env("RUNPOD_ALLOWED_GPU_TYPES").split(",")
        if item.strip()
    }
    gpu_types = [gpu for gpu in CANDIDATE_GPUS if gpu in allowed]
    if not gpu_types:
        raise RuntimeError("No presigned-analysis consumer GPU is allowed")

    image = required_env("RUNPOD_IMAGE_NAME")
    if image != EXPECTED_IMAGE:
        raise RuntimeError(f"Refusing unexpected canonical-analysis image: {image}")

    control_run_id = f"run-{int(time.time())}-{secrets.token_hex(4)}"
    run_prefix = f"ardor-control/runs/{JOB_ID}/{control_run_id}"
    status_name = status_key(JOB_ID, control_run_id)
    result_name = f"{run_prefix}/canonical_analysis.json"
    log_name = f"{run_prefix}/worker.log"

    inputs = presigned_inputs()
    inputs_b64 = base64.b64encode(
        json.dumps(inputs, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")

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

    # This bootstrap changes only artifact transport. The checked-out evaluator
    # and worker remain exactly AUDIT_CODE_SHA, and the checkpoint is content-
    # verified before evaluation.
    fixed_start = f'''set -euo pipefail
python - <<'PY'
import json, os, urllib.request
payload = {{
    "schema_version": 1,
    "job_id": os.environ["ARDOR_JOB_ID"],
    "control_run_id": os.environ["ARDOR_CONTROL_RUN_ID"],
    "runner": "{RUNNER}",
    "state": "staging",
}}
data = (json.dumps(payload, sort_keys=True) + "\\n").encode("utf-8")
req = urllib.request.Request(os.environ["ARDOR_PUT_STATUS_URL"], data=data, method="PUT")
with urllib.request.urlopen(req, timeout=120) as response:
    response.read()
PY
rm -rf /workspace/Ardor
mkdir -p /workspace/Ardor
cd /workspace/Ardor
git init
git remote add origin https://github.com/kys928/Ardor.git
git fetch --depth 1 origin {AUDIT_CODE_SHA}
git checkout --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "{AUDIT_CODE_SHA}"
python -m py_compile Erratum/canonical_analysis_v14a2.py scripts/runpod_canonical_analysis_worker.py
python - <<'PY'
import base64, hashlib, json, os, shutil, urllib.request
from pathlib import Path
root = Path("/workspace/Ardor")
rows = json.loads(base64.b64decode(os.environ["ARDOR_S3_INPUTS_B64"]).decode("utf-8"))
for row in rows:
    dest = root / row["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(row["url"], timeout=180) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    size = dest.stat().st_size
    if size != int(row["size"]):
        raise RuntimeError(f"Downloaded size mismatch for {{row['path']}}: {{size}} != {{row['size']}}")
    expected_sha = row.get("sha256")
    if expected_sha:
        digest = hashlib.sha256()
        with dest.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch for {{row['path']}}: {{actual_sha}} != {{expected_sha}}")
        print(json.dumps({{"verified": row["path"], "size": size, "sha256": actual_sha}}))
    else:
        print(json.dumps({{"staged": row["path"], "size": size}}))
PY
/root/.local/bin/uv pip install --python /opt/Ardor/.venv/bin/python --index-url {TORCH_INDEX} --reinstall torch=={TORCH_VERSION}
/opt/Ardor/.venv/bin/python -c "import torch; assert torch.__version__.startswith('{TORCH_VERSION}'); assert torch.cuda.is_available(); print(f'[canonical-analysis-runtime] torch={{torch.__version__}} cuda={{torch.version.cuda}} device={{torch.cuda.get_device_name(0)}}')"
rm -f /tmp/ardor_worker_done
python - <<'PY' &
import hashlib, json, os, time, urllib.request
from pathlib import Path
status = Path(os.environ["ARDOR_LOCAL_STATUS"])
url = os.environ["ARDOR_PUT_STATUS_URL"]
done = Path("/tmp/ardor_worker_done")
last = None
while not done.exists():
    try:
        if status.is_file():
            data = status.read_bytes()
            payload = json.loads(data.decode("utf-8"))
            if str(payload.get("state", "")) not in {{"completed", "failed"}}:
                digest = hashlib.sha256(data).hexdigest()
                if digest != last:
                    req = urllib.request.Request(url, data=data, method="PUT")
                    with urllib.request.urlopen(req, timeout=120) as response:
                        response.read()
                    last = digest
    except Exception as exc:
        print(f"[status-uploader] {{exc!r}}")
    time.sleep(5)
PY
uploader_pid=$!
set +e
/opt/Ardor/.venv/bin/python scripts/runpod_canonical_analysis_worker.py
rc=$?
set -e
touch /tmp/ardor_worker_done
wait "$uploader_pid" || true
export ARDOR_WORKER_RC="$rc"
python - <<'PY'
import json, os, urllib.request
from datetime import datetime, timezone
from pathlib import Path

def put(path: Path, url: str) -> None:
    data = path.read_bytes()
    req = urllib.request.Request(url, data=data, method="PUT")
    with urllib.request.urlopen(req, timeout=180) as response:
        response.read()

status = Path(os.environ["ARDOR_LOCAL_STATUS"])
result = Path(os.environ["ARDOR_LOCAL_RESULT"])
log = Path(os.environ["ARDOR_LOCAL_LOG"])
rc = int(os.environ["ARDOR_WORKER_RC"])
if status.is_file():
    payload = json.loads(status.read_text(encoding="utf-8"))
else:
    payload = {{}}
if str(payload.get("state", "")) not in {{"completed", "failed"}}:
    payload.update({{
        "schema_version": 1,
        "job_id": os.environ["ARDOR_JOB_ID"],
        "control_run_id": os.environ["ARDOR_CONTROL_RUN_ID"],
        "runner": "{RUNNER}",
        "state": "failed" if rc else "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "returncode": rc,
        "error": "worker_exited_without_terminal_status" if rc else None,
    }})
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
# Upload scientific result/log before exposing terminal status to the supervisor.
if result.is_file():
    put(result, os.environ["ARDOR_PUT_RESULT_URL"])
if log.is_file():
    put(log, os.environ["ARDOR_PUT_LOG_URL"])
put(status, os.environ["ARDOR_PUT_STATUS_URL"])
PY
exit "$rc"
'''

    payload: dict[str, Any] = {
        "name": f"ardor-{JOB_ID}"[:191],
        "computeType": "GPU",
        "cloudType": "SECURE",
        "gpuTypeIds": gpu_types,
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "containerDiskInGb": 60,
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": [fixed_start],
        "ports": [],
        "env": {
            "ARDOR_JOB_B64": encoded_job,
            "ARDOR_JOB_ID": JOB_ID,
            "ARDOR_CONTROL_RUN_ID": control_run_id,
            "ARDOR_AUDIT_CODE_SHA": AUDIT_CODE_SHA,
            "ARDOR_ANALYSIS_TORCH_VERSION": TORCH_VERSION,
            "ARDOR_ANALYSIS_TORCH_INDEX": TORCH_INDEX,
            "ARDOR_S3_INPUTS_B64": inputs_b64,
            "ARDOR_PUT_STATUS_URL": presigned_put(status_name),
            "ARDOR_PUT_RESULT_URL": presigned_put(result_name),
            "ARDOR_PUT_LOG_URL": presigned_put(log_name),
            "ARDOR_LOCAL_STATUS": f"/workspace/ardor-control/runs/{JOB_ID}/{control_run_id}/status.json",
            "ARDOR_LOCAL_RESULT": f"/workspace/ardor-control/runs/{JOB_ID}/{control_run_id}/canonical_analysis.json",
            "ARDOR_LOCAL_LOG": f"/workspace/ardor-control/runs/{JOB_ID}/{control_run_id}/worker.log",
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
            "status_key": status_name,
            "hourly_cost_usd": pod_cost(pod),
            "hourly_cap_usd": None,
            "timeout_minutes": None,
            "network_volume_id": required_env("RUNPOD_NETWORK_VOLUME_ID"),
            "network_volume_mounted": False,
            "artifact_transport": "presigned_s3",
            "datacenter_id": None,
            "image_name": image,
            "audit_code_sha": AUDIT_CODE_SHA,
            "torch_version": TORCH_VERSION,
            "torch_index": TORCH_INDEX,
            "github_sha": os.environ.get("GITHUB_SHA", "").strip() or None,
            "task": {"runner": RUNNER},
            "gpu": job["gpu"],
            "result_key": result_name,
            "log_key": log_name,
        }
        internal_put_json(active_name, record)
        safe_launch = dict(record)
        print(json.dumps({"launch": safe_launch}, indent=2))
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
