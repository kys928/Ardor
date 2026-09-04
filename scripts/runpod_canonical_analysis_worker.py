#!/usr/bin/env python3
"""Fixed-purpose RunPod worker for the frozen canonical v14a2 bottleneck analysis."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CORTEX_ROOT = REPO_ROOT / "Cerebrum" / "Cortex"
CONTROL_ROOT = Path("/workspace/ardor-control/runs")
EXPECTED_RUNNER = "canonical_analysis_v14a2"

for path in (str(REPO_ROOT), str(CORTEX_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def safe_id(value: object, label: str) -> str:
    text = str(value or "")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if not text or len(text) > 96 or any(ch not in allowed for ch in text):
        raise ValueError(f"Unsafe {label}: {text!r}")
    return text


def load_job() -> dict[str, Any]:
    encoded = os.environ.get("ARDOR_JOB_B64", "")
    if not encoded:
        raise RuntimeError("ARDOR_JOB_B64 is not set")
    raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    job = json.loads(raw.decode("utf-8"))
    if not isinstance(job, dict):
        raise RuntimeError("Job payload must be a JSON object")
    task = job.get("task")
    if not isinstance(task, dict) or task != {"runner": EXPECTED_RUNNER}:
        raise RuntimeError(f"Analysis worker accepts only task={{'runner': {EXPECTED_RUNNER!r}}}")
    return job


def gpu_snapshot() -> dict[str, Any]:
    try:
        import torch
        return {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        return {"error": repr(exc)}


def main() -> int:
    job = load_job()
    job_id = safe_id(job.get("id"), "job.id")
    control_run_id = safe_id(job.get("_control_run_id") or job_id, "control_run_id")
    run_dir = CONTROL_ROOT / job_id / control_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    log_path = run_dir / "worker.log"
    atomic_json(run_dir / "job.json", job)

    started = {
        "schema_version": 1,
        "job_id": job_id,
        "control_run_id": control_run_id,
        "runner": EXPECTED_RUNNER,
        "state": "running",
        "started_at": utc_now(),
        "host": socket.gethostname(),
        "repo_root": str(REPO_ROOT),
        "gpu": gpu_snapshot(),
    }
    atomic_json(status_path, started)

    try:
        from Erratum.canonical_analysis_v14a2 import evaluate
        result = evaluate(run_dir)
        returncode = 0 if bool(result.get("passed")) else 1
        completed = {
            **started,
            "state": "completed" if returncode == 0 else "failed",
            "completed_at": utc_now(),
            "returncode": returncode,
            "result_path": str(run_dir / "canonical_analysis.json"),
            "log_path": str(log_path),
        }
        log_path.write_text(
            json.dumps({
                "passed": bool(result.get("passed")),
                "route_chain": result.get("route_chain"),
                "by_route": result.get("by_route"),
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        atomic_json(status_path, completed)
        return returncode
    except BaseException as exc:
        failure = {
            **started,
            "state": "failed",
            "completed_at": utc_now(),
            "returncode": None,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "log_path": str(log_path),
        }
        log_path.write_text(failure["traceback"], encoding="utf-8")
        atomic_json(status_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
