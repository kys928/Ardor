#!/usr/bin/env python3
"""One-purpose RunPod worker for the v14a6 learning-dynamics pulse/chase probe."""
from __future__ import annotations

import socket
import sys
import traceback

import runpod_worker as shared

RUNNER = "v14a6_learning_dynamics_probe"
ENTRY = shared.REPO_ROOT / "Erratum" / "ardor_v14a6_learning_dynamics_probe.py"
OUTPUT = (
    shared.PERSISTENT_ROOT
    / "training"
    / "runs"
    / "sft_v14a6_learning_dynamics_target_pulse_chase_u80"
)


def run() -> int:
    job = shared.load_job()
    job_id = shared.safe_id(job.get("id"), "job.id")
    control_run_id = shared.safe_id(
        job.get("_control_run_id") or job_id, "control_run_id"
    )
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job.task must be an object")
    if str(task.get("runner", "")) != RUNNER:
        raise ValueError(f"This worker accepts only task.runner={RUNNER!r}")
    extra = sorted(set(task) - {"runner"})
    if extra:
        raise ValueError(
            f"{RUNNER} is fixed-purpose and accepts no task fields beyond runner; "
            f"unexpected: {extra}"
        )

    run_dir = shared.CONTROL_ROOT / job_id / control_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    log_path = run_dir / "worker.log"
    shared.atomic_json(run_dir / "job.json", job)

    command = [
        sys.executable,
        str(ENTRY),
        "--verify-parent-sha256",
        "--output-dir",
        str(OUTPUT),
    ]
    started = {
        "schema_version": 1,
        "job_id": job_id,
        "control_run_id": control_run_id,
        "runner": RUNNER,
        "state": "running",
        "started_at": shared.utc_now(),
        "host": socket.gethostname(),
        "repo_root": str(shared.REPO_ROOT),
        "persistent_root": str(shared.PERSISTENT_ROOT),
        "command": command,
        "gpu": shared.gpu_snapshot(),
    }
    shared.atomic_json(status_path, started)

    try:
        returncode = shared.run_subprocess(command, log_path)
        completed = {
            **started,
            "state": "completed" if returncode == 0 else "failed",
            "completed_at": shared.utc_now(),
            "returncode": returncode,
            "log_path": str(log_path),
            "result_path": str(OUTPUT / "v14a6_learning_dynamics_decision.json"),
        }
        shared.atomic_json(status_path, completed)
        return returncode
    except BaseException as exc:
        failure = {
            **started,
            "state": "failed",
            "completed_at": shared.utc_now(),
            "returncode": None,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "log_path": str(log_path),
        }
        shared.atomic_json(status_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(run())
