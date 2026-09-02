#!/usr/bin/env python3
"""GitHub-side RunPod + S3 control plane for Ardor.

Security and lifecycle:
- GitHub Actions owns RunPod API and S3 credentials.
- GPU workers receive only a validated Ardor job payload plus the mounted
  Network Volume.
- User-facing S3 operations are read-only.
- Internal S3 writes are restricted to ardor-control metadata.
- Compute defaults to detached mode, so scientific runtime is not tied to a
  GitHub-hosted runner lease.
- Unset cost/runtime variables mean no controller-imposed cap.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError as exc:
    raise SystemExit("boto3 is required: python -m pip install boto3") from exc

RUNPOD_REST = "https://rest.runpod.io/v1"
ACTIVE_PREFIX = "ardor-control/active/"
HISTORY_PREFIX = "ardor-control/history/"
RUNS_PREFIX = "ardor-control/runs/"
TERMINAL_STATES = {"completed", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def optional_float(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    return float(value) if value else None


def optional_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def safe_id(value: object, label: str) -> str:
    text = str(value or "")
    if not text or len(text) > 96:
        raise ValueError(f"{label} must be 1-96 characters")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if any(ch not in allowed for ch in text):
        raise ValueError(f"{label} contains unsafe characters")
    return text


def load_job(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Job manifest must be a JSON object")
    if payload.get("version") != 1:
        raise ValueError("Only job manifest version 1 is supported")
    safe_id(payload.get("id"), "job.id")
    if payload.get("kind") not in {"compute", "storage"}:
        raise ValueError("job.kind must be 'compute' or 'storage'")
    return payload


def runpod_request(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    allow_not_found: bool = False,
) -> Any:
    data = None
    headers = {
        "Authorization": f"Bearer {required_env('RUNPOD_API_KEY')}",
        "Accept": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{RUNPOD_REST}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if allow_not_found and exc.code == 404:
            return None
        raise RuntimeError(
            f"RunPod API {method} {path} failed: HTTP {exc.code}: {detail}"
        ) from exc


def s3_client():
    datacenter = required_env("RUNPOD_DATACENTER_ID")
    endpoint = os.environ.get("RUNPOD_S3_ENDPOINT", "").strip()
    if not endpoint:
        endpoint = f"https://s3api-{datacenter.lower()}.runpod.io/"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=datacenter.lower(),
        aws_access_key_id=required_env("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required_env("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 10, "mode": "standard"},
            connect_timeout=30,
            read_timeout=120,
        ),
    )


def bucket() -> str:
    return required_env("RUNPOD_NETWORK_VOLUME_ID")


def internal_put_json(key: str, payload: dict[str, Any]) -> None:
    if not (key.startswith(ACTIVE_PREFIX) or key.startswith(HISTORY_PREFIX)):
        raise ValueError("Internal writes are restricted to Ardor control metadata")
    s3_client().put_object(
        Bucket=bucket(),
        Key=key,
        Body=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def internal_delete(key: str) -> None:
    if not key.startswith(ACTIVE_PREFIX):
        raise ValueError("Internal deletes are restricted to active Ardor control metadata")
    s3_client().delete_object(Bucket=bucket(), Key=key)


def object_missing(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code", "")).lower()
        if code in {"404", "nosuchkey", "notfound"}:
            return True
    text = str(exc).lower()
    return (
        ("invalidargument" in text and "object not found" in text)
        or "nosuchkey" in text
        or "not found" in text
    )


def read_json_object(key: str) -> dict[str, Any] | None:
    try:
        obj = s3_client().get_object(Bucket=bucket(), Key=key)
    except Exception as exc:
        if object_missing(exc):
            return None
        raise
    raw = obj["Body"].read(2_000_000)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object at s3://{bucket()}/{key}")
    return value


def s3_key(value: object, *, allow_empty: bool = False) -> str:
    text = str(value or "").strip()
    if text.startswith("s3://"):
        raise ValueError("Use a volume-relative path, not an s3:// URL")
    if text.startswith("/workspace/"):
        text = text[len("/workspace/") :]
    text = text.lstrip("/")
    if not text and allow_empty:
        return ""
    path = PurePosixPath(text)
    if not text or ".." in path.parts or "\x00" in text:
        raise ValueError("Unsafe or empty storage path")
    return path.as_posix()


def storage_job(job: dict[str, Any]) -> None:
    spec = job.get("storage")
    if not isinstance(spec, dict):
        raise ValueError("storage job requires a storage object")
    operation = str(spec.get("operation", ""))
    if operation not in {"list", "head", "read_text", "download"}:
        raise ValueError("Storage operations are read-only: list, head, read_text, download")

    client = s3_client()
    key = s3_key(spec.get("path", ""), allow_empty=operation == "list")

    if operation == "list":
        max_items = min(max(int(spec.get("max_items", 200)), 1), 1000)
        shallow = spec.get("shallow", False)
        if not isinstance(shallow, bool):
            raise ValueError("storage.shallow must be a boolean when present")
        list_args: dict[str, Any] = {
            "Bucket": bucket(),
            "Prefix": key,
            "MaxKeys": max_items,
        }
        if shallow:
            list_args["Delimiter"] = "/"
        response = client.list_objects_v2(**list_args)
        entries = [
            {
                "key": item.get("Key"),
                "size": item.get("Size"),
                "last_modified": (
                    item.get("LastModified").isoformat()
                    if item.get("LastModified")
                    else None
                ),
                "etag": item.get("ETag"),
            }
            for item in response.get("Contents", [])
        ]
        common_prefixes = [
            item.get("Prefix")
            for item in response.get("CommonPrefixes", [])
            if item.get("Prefix")
        ]
        print(json.dumps(
            {
                "job_id": job["id"],
                "operation": operation,
                "prefix": key,
                "shallow": shallow,
                "is_truncated": bool(response.get("IsTruncated", False)),
                "common_prefixes": common_prefixes,
                "entries": entries,
            },
            indent=2,
        ))
        return

    metadata = client.head_object(Bucket=bucket(), Key=key)
    head = {
        "key": key,
        "size": int(metadata.get("ContentLength", 0)),
        "etag": metadata.get("ETag"),
        "last_modified": (
            metadata.get("LastModified").isoformat()
            if metadata.get("LastModified")
            else None
        ),
        "content_type": metadata.get("ContentType"),
    }
    if operation == "head":
        print(json.dumps(
            {"job_id": job["id"], "operation": operation, "object": head},
            indent=2,
        ))
        return

    if operation == "read_text":
        hard_cap = optional_int("RUNPOD_MAX_READ_TEXT_BYTES") or 1_048_576
        requested = int(spec.get("max_bytes", min(hard_cap, 262_144)))
        max_bytes = min(max(requested, 1), hard_cap)
        if head["size"] > max_bytes:
            raise RuntimeError(
                f"Object is {head['size']} bytes, above read_text limit {max_bytes}; "
                "use download instead"
            )
        raw = client.get_object(Bucket=bucket(), Key=key)["Body"].read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise RuntimeError("Object exceeded read_text limit while downloading")
        print(json.dumps(
            {
                "job_id": job["id"],
                "operation": operation,
                "object": head,
                "text": raw.decode(str(spec.get("encoding", "utf-8"))),
            },
            indent=2,
        ))
        return

    hard_cap = optional_int("RUNPOD_MAX_ARTIFACT_BYTES") or 536_870_912
    requested = int(spec.get("max_bytes", hard_cap))
    max_bytes = min(max(requested, 1), hard_cap)
    if head["size"] > max_bytes:
        raise RuntimeError(
            f"Object is {head['size']} bytes, above download limit {max_bytes}"
        )
    destination = Path("runpod_artifacts") / str(job["id"]) / Path(key).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket(), key, str(destination))
    print(json.dumps(
        {
            "job_id": job["id"],
            "operation": operation,
            "object": head,
            "downloaded_to": str(destination),
        },
        indent=2,
    ))


def effective_cap(limits: dict[str, Any]) -> float | None:
    values: list[float] = []
    global_cap = optional_float("RUNPOD_MAX_HOURLY_USD")
    if global_cap is not None:
        if global_cap <= 0:
            raise ValueError("RUNPOD_MAX_HOURLY_USD must be positive when set")
        values.append(global_cap)
    if limits.get("max_hourly_usd") is not None:
        value = float(limits["max_hourly_usd"])
        if value <= 0:
            raise ValueError("limits.max_hourly_usd must be positive when set")
        values.append(value)
    return min(values) if values else None


def effective_timeout(limits: dict[str, Any]) -> int | None:
    values: list[int] = []
    global_timeout = optional_int("RUNPOD_MAX_RUNTIME_MINUTES")
    if global_timeout is not None:
        if global_timeout < 1:
            raise ValueError("RUNPOD_MAX_RUNTIME_MINUTES must be positive when set")
        values.append(global_timeout)
    if limits.get("timeout_minutes") is not None:
        value = int(limits["timeout_minutes"])
        if value < 1:
            raise ValueError("limits.timeout_minutes must be positive when set")
        values.append(value)
    return min(values) if values else None


def validate_task(task: dict[str, Any]) -> None:
    runner = str(task.get("runner", ""))
    if runner == "infra_smoke":
        return
    if runner != "ardor_promptgen":
        raise ValueError("Supported task runners are 'infra_smoke' and 'ardor_promptgen'")
    stage = str(task.get("stage", ""))
    if stage not in {"lm_base", "stabilize", "sft"}:
        raise ValueError("ardor_promptgen stage must be lm_base, stabilize, or sft")
    if not str(task.get("tokenizer", "")).strip():
        raise ValueError(
            "ardor_promptgen jobs must explicitly provide task.tokenizer; "
            "the control plane will not guess tokenizer version"
        )
    if stage == "sft" and not str(task.get("sft_jsonl", "")).strip():
        raise ValueError("ardor_promptgen sft jobs require task.sft_jsonl")


def validate_compute(
    job: dict[str, Any],
) -> tuple[dict[str, Any], float | None, int | None, str, str]:
    task = job.get("task")
    gpu = job.get("gpu")
    limits = job.get("limits") or {}
    lifecycle = job.get("lifecycle") or {}
    if not isinstance(task, dict) or not isinstance(gpu, dict):
        raise ValueError("compute jobs require task and gpu objects")
    if not isinstance(limits, dict) or not isinstance(lifecycle, dict):
        raise ValueError("limits and lifecycle must be objects when present")
    validate_task(task)

    requested_types = gpu.get("type_ids")
    if not isinstance(requested_types, list) or not requested_types:
        raise ValueError("gpu.type_ids must be a non-empty list")
    requested_types = [str(value) for value in requested_types]
    allowed_types = {
        value.strip()
        for value in required_env("RUNPOD_ALLOWED_GPU_TYPES").split(",")
        if value.strip()
    }
    disallowed = [value for value in requested_types if value not in allowed_types]
    if disallowed:
        raise ValueError(f"GPU type(s) not in RUNPOD_ALLOWED_GPU_TYPES: {disallowed}")
    if int(gpu.get("count", 1)) != 1:
        raise ValueError("The v1 Ardor control plane currently permits exactly one GPU per Pod")

    cloud_type = str(gpu.get("cloud_type", "SECURE")).upper()
    if cloud_type not in {"SECURE", "COMMUNITY"}:
        raise ValueError("gpu.cloud_type must be SECURE or COMMUNITY")

    mode = str(lifecycle.get("mode", "detached")).lower()
    if mode not in {"detached", "attached"}:
        raise ValueError("lifecycle.mode must be detached or attached")
    timeout_minutes = effective_timeout(limits)
    if mode == "attached" and timeout_minutes is None:
        raise ValueError(
            "attached mode requires an explicit timeout; use detached for unbounded runs"
        )

    control_run_id = f"run-{int(time.time())}-{secrets.token_hex(4)}"
    worker_job = dict(job)
    worker_job["_control_run_id"] = control_run_id
    encoded_job = base64.b64encode(
        json.dumps(worker_job, sort_keys=True).encode("utf-8")
    ).decode("ascii")

    payload: dict[str, Any] = {
        "name": f"ardor-{job['id']}"[:191],
        "computeType": "GPU",
        "cloudType": cloud_type,
        "gpuTypeIds": requested_types,
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "dataCenterIds": [required_env("RUNPOD_DATACENTER_ID")],
        "dataCenterPriority": "availability",
        "networkVolumeId": required_env("RUNPOD_NETWORK_VOLUME_ID"),
        "volumeMountPath": "/workspace",
        "containerDiskInGb": int(gpu.get("container_disk_gb", 50)),
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": [
            "cd /opt/Ardor && uv run --frozen python scripts/runpod_worker.py"
        ],
        "ports": [],
        "env": {
            "ARDOR_JOB_B64": encoded_job,
            "PYTHONUNBUFFERED": "1",
        },
        "interruptible": bool(gpu.get("interruptible", False)),
    }

    template_id = os.environ.get("RUNPOD_TEMPLATE_ID", "").strip()
    image = os.environ.get("RUNPOD_IMAGE_NAME", "").strip()
    if template_id:
        payload["templateId"] = template_id
    elif image:
        payload["imageName"] = image
    else:
        raise RuntimeError("Set either RUNPOD_TEMPLATE_ID or RUNPOD_IMAGE_NAME")
    registry_auth = os.environ.get("RUNPOD_CONTAINER_REGISTRY_AUTH_ID", "").strip()
    if registry_auth:
        payload["containerRegistryAuthId"] = registry_auth

    return payload, effective_cap(limits), timeout_minutes, control_run_id, mode


def status_key(job_id: str, control_run_id: str) -> str:
    return f"{RUNS_PREFIX}{job_id}/{control_run_id}/status.json"


def active_key(control_run_id: str) -> str:
    return f"{ACTIVE_PREFIX}{control_run_id}.json"


def history_key(control_run_id: str) -> str:
    return f"{HISTORY_PREFIX}{control_run_id}.json"


def list_active_records() -> list[tuple[str, dict[str, Any]]]:
    client = s3_client()
    paginator = client.get_paginator("list_objects_v2")
    rows: list[tuple[str, dict[str, Any]]] = []
    for page in paginator.paginate(Bucket=bucket(), Prefix=ACTIVE_PREFIX):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key.endswith(".json"):
                continue
            value = read_json_object(key)
            if value is not None:
                rows.append((key, value))
    rows.sort(key=lambda pair: pair[0])
    return rows


def pod_cost(pod: dict[str, Any]) -> float | None:
    raw = pod.get("adjustedCostPerHr")
    if raw is None:
        raw = pod.get("costPerHr")
    if raw is None:
        return None
    return float(raw)


def delete_pod(pod_id: str) -> str:
    result = runpod_request("DELETE", f"/pods/{pod_id}", allow_not_found=True)
    return "already_absent" if result is None else "deleted"


def archive_record(
    active_key_name: str,
    record: dict[str, Any],
    *,
    reason: str,
    worker_status: dict[str, Any] | None = None,
) -> None:
    control_run_id = safe_id(record.get("control_run_id"), "control_run_id")
    archive = {
        **record,
        "archived_at": utc_now(),
        "archive_reason": reason,
        "worker_status": worker_status,
    }
    internal_put_json(history_key(control_run_id), archive)
    internal_delete(active_key_name)


def launch_compute(job: dict[str, Any]) -> None:
    payload, hourly_cap, timeout_minutes, control_run_id, mode = validate_compute(job)
    pod_id: str | None = None
    active_name = active_key(control_run_id)
    try:
        pod = runpod_request("POST", "/pods", payload)
        if not isinstance(pod, dict) or not pod.get("id"):
            raise RuntimeError(
                f"RunPod create response did not contain a Pod id: {pod!r}"
            )
        pod_id = str(pod["id"])
        cost = pod_cost(pod)

        if hourly_cap is not None:
            if cost is None:
                raise RuntimeError(
                    "A price cap is configured but RunPod did not report Pod hourly cost"
                )
            if cost > hourly_cap:
                raise RuntimeError(
                    f"Allocated Pod costs ${cost:.4f}/h, above cap ${hourly_cap:.4f}/h"
                )

        record = {
            "schema_version": 1,
            "project": "Ardor",
            "job_id": str(job["id"]),
            "control_run_id": control_run_id,
            "pod_id": pod_id,
            "created_at": utc_now(),
            "mode": mode,
            "status_key": status_key(str(job["id"]), control_run_id),
            "hourly_cost_usd": cost,
            "hourly_cap_usd": hourly_cap,
            "timeout_minutes": timeout_minutes,
            "network_volume_id": required_env("RUNPOD_NETWORK_VOLUME_ID"),
            "datacenter_id": required_env("RUNPOD_DATACENTER_ID"),
            "image_name": os.environ.get("RUNPOD_IMAGE_NAME", "").strip() or None,
            "github_sha": os.environ.get("GITHUB_SHA", "").strip() or None,
            "task": job.get("task"),
            "gpu": job.get("gpu"),
        }
        internal_put_json(active_name, record)
        print(json.dumps({"launch": record}, indent=2))

        if mode == "detached":
            return

        assert timeout_minutes is not None
        deadline = time.monotonic() + timeout_minutes * 60
        poll_seconds = max(optional_int("RUNPOD_STATUS_POLL_SECONDS") or 20, 5)
        while time.monotonic() < deadline:
            status = read_json_object(record["status_key"])
            if status is not None:
                state = str(status.get("state", ""))
                print(json.dumps(
                    {
                        "job_id": job["id"],
                        "control_run_id": control_run_id,
                        "worker_state": state,
                    }
                ))
                if state in TERMINAL_STATES:
                    cleanup = delete_pod(pod_id)
                    archive_record(
                        active_name,
                        record,
                        reason=f"worker_{state}",
                        worker_status=status,
                    )
                    if state == "failed":
                        raise RuntimeError(
                            f"Ardor worker failed: {json.dumps(status, sort_keys=True)}"
                        )
                    print(json.dumps({"pod_cleanup": cleanup}))
                    return
            time.sleep(poll_seconds)

        cleanup = delete_pod(pod_id)
        archive_record(active_name, record, reason="attached_timeout")
        raise TimeoutError(
            f"Attached job exceeded explicit timeout of {timeout_minutes} minutes; "
            f"Pod cleanup={cleanup}"
        )
    except Exception:
        if pod_id is not None:
            existing = read_json_object(active_name)
            if existing is None:
                try:
                    delete_pod(pod_id)
                except Exception:
                    pass
        raise


def reconcile_detached() -> None:
    records = list_active_records()
    print(json.dumps({"active_run_count": len(records), "checked_at": utc_now()}))
    for key, record in records:
        job_id = safe_id(record.get("job_id"), "job_id")
        control_run_id = safe_id(record.get("control_run_id"), "control_run_id")
        pod_id = safe_id(record.get("pod_id"), "pod_id")
        status = read_json_object(status_key(job_id, control_run_id))
        if status is None:
            print(json.dumps({
                "control_run_id": control_run_id,
                "pod_id": pod_id,
                "state": "status_pending",
            }))
            continue

        state = str(status.get("state", ""))
        if state not in TERMINAL_STATES:
            print(json.dumps({
                "control_run_id": control_run_id,
                "pod_id": pod_id,
                "state": state or "unknown",
            }))
            continue

        cleanup = delete_pod(pod_id)
        archive_record(
            key,
            record,
            reason=f"worker_{state}",
            worker_status=status,
        )
        print(json.dumps({
            "control_run_id": control_run_id,
            "pod_id": pod_id,
            "state": state,
            "pod_cleanup": cleanup,
            "archived": True,
        }))


def terminate_run(control_run_id: str) -> None:
    control_run_id = safe_id(control_run_id, "control_run_id")
    key = active_key(control_run_id)
    record = read_json_object(key)
    if record is None:
        raise RuntimeError(f"No active Ardor control record for {control_run_id}")
    pod_id = safe_id(record.get("pod_id"), "pod_id")
    cleanup = delete_pod(pod_id)
    status = read_json_object(str(record.get("status_key") or ""))
    archive_record(
        key,
        record,
        reason="manual_termination",
        worker_status=status,
    )
    print(json.dumps({
        "control_run_id": control_run_id,
        "pod_id": pod_id,
        "pod_cleanup": cleanup,
        "archived": True,
    }, indent=2))


def execute(path: Path) -> None:
    job = load_job(path)
    print(json.dumps({
        "job": str(path),
        "job_id": job["id"],
        "kind": job["kind"],
    }))
    if job["kind"] == "storage":
        storage_job(job)
    else:
        launch_compute(job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    execute_parser = sub.add_parser("execute", help="Validate and execute one job manifest.")
    execute_parser.add_argument("--job", type=Path, required=True)

    sub.add_parser(
        "reconcile-detached",
        help="Delete Pods whose detached Ardor workers reached terminal status.",
    )
    sub.add_parser("list-active", help="Print active detached control records.")

    terminate_parser = sub.add_parser(
        "terminate-run",
        help="Explicitly terminate and archive one detached control run.",
    )
    terminate_parser.add_argument("--control-run-id", required=True)

    args = parser.parse_args()
    if args.command == "execute":
        execute(args.job)
    elif args.command == "reconcile-detached":
        reconcile_detached()
    elif args.command == "list-active":
        print(json.dumps(
            [{"key": key, "record": record} for key, record in list_active_records()],
            indent=2,
        ))
    elif args.command == "terminate-run":
        terminate_run(args.control_run_id)


if __name__ == "__main__":
    main()
