#!/usr/bin/env python3
"""Clean up detached Ardor Pods that never reach worker startup."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from typing import Any

import runpod_control as control

DEFAULT_PROVISIONING_TIMEOUT_MINUTES = 10


def provisioning_timeout_minutes() -> int:
    raw = os.environ.get("RUNPOD_PROVISIONING_TIMEOUT_MINUTES", "").strip()
    value = int(raw) if raw else DEFAULT_PROVISIONING_TIMEOUT_MINUTES
    if value < 1:
        raise ValueError("RUNPOD_PROVISIONING_TIMEOUT_MINUTES must be positive when set")
    return value


def parse_timestamp(value: object, label: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is missing")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def pod_snapshot(pod: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "id",
        "name",
        "desiredStatus",
        "status",
        "cloudType",
        "gpuCount",
        "gpuTypeIds",
        "costPerHr",
        "adjustedCostPerHr",
        "imageName",
        "networkVolumeId",
        "volumeMountPath",
        "lastStatusChange",
        "runtime",
        "machine",
        "interruptible",
    )
    return {key: pod.get(key) for key in allowed if key in pod}


def reconcile_provisioning() -> None:
    timeout_minutes = provisioning_timeout_minutes()
    now = datetime.now(timezone.utc)
    records = control.list_active_records()
    print(json.dumps({
        "provisioning_watchdog": "check",
        "active_run_count": len(records),
        "timeout_minutes": timeout_minutes,
        "checked_at": now.isoformat(),
    }))

    for key, record in records:
        job_id = control.safe_id(record.get("job_id"), "job_id")
        control_run_id = control.safe_id(record.get("control_run_id"), "control_run_id")
        pod_id = control.safe_id(record.get("pod_id"), "pod_id")
        status_name = str(record.get("status_key") or control.status_key(job_id, control_run_id))

        worker_status = control.read_json_object(status_name)
        if worker_status is not None:
            print(json.dumps({
                "control_run_id": control_run_id,
                "pod_id": pod_id,
                "provisioning_state": "worker_started",
                "worker_state": str(worker_status.get("state", "")) or "unknown",
            }))
            continue

        created_at = parse_timestamp(record.get("created_at"), "record.created_at")
        age_seconds = max((now - created_at).total_seconds(), 0.0)
        age_minutes = age_seconds / 60.0
        pod = control.runpod_request("GET", f"/pods/{pod_id}", allow_not_found=True)

        if pod is None:
            control.archive_record(
                key,
                record,
                reason="pod_absent_before_worker_start",
                worker_status=None,
            )
            print(json.dumps({
                "control_run_id": control_run_id,
                "pod_id": pod_id,
                "provisioning_state": "pod_absent_before_worker_start",
                "age_minutes": round(age_minutes, 3),
                "archived": True,
            }))
            continue

        if not isinstance(pod, dict):
            raise RuntimeError(f"Unexpected RunPod Pod response for {pod_id}: {pod!r}")

        if age_minutes < timeout_minutes:
            print(json.dumps({
                "control_run_id": control_run_id,
                "pod_id": pod_id,
                "provisioning_state": "waiting_for_worker_start",
                "age_minutes": round(age_minutes, 3),
                "timeout_minutes": timeout_minutes,
                "pod": pod_snapshot(pod),
            }, default=str))
            continue

        # Avoid a race where the worker wrote status between the first S3 read
        # and the timeout decision.
        worker_status = control.read_json_object(status_name)
        if worker_status is not None:
            print(json.dumps({
                "control_run_id": control_run_id,
                "pod_id": pod_id,
                "provisioning_state": "worker_started_at_timeout_boundary",
                "worker_state": str(worker_status.get("state", "")) or "unknown",
            }))
            continue

        cleanup = control.delete_pod(pod_id)
        control.archive_record(
            key,
            record,
            reason="provisioning_timeout",
            worker_status=None,
        )
        print(json.dumps({
            "control_run_id": control_run_id,
            "pod_id": pod_id,
            "provisioning_state": "provisioning_timeout",
            "age_minutes": round(age_minutes, 3),
            "timeout_minutes": timeout_minutes,
            "pod_cleanup": cleanup,
            "pod": pod_snapshot(pod),
            "archived": True,
        }, default=str))


if __name__ == "__main__":
    reconcile_provisioning()
