#!/usr/bin/env python3
"""Fixed-purpose v14a14 80-vs-80 controlled GPU diagnostic.

The single user reviewed all 80 current targets. Train with that explicit
diagnostic policy, save the intervention checkpoint as a candidate, then freeze
anonymous generated-answer review tasks. No promotion until scored gates pass.
"""
from __future__ import annotations

import socket
import sys
import traceback
import runpod_worker as shared

RUNNER = "v14a14_stronger_supervision_controlled"
ENTRY = shared.REPO_ROOT / "Erratum" / "v14a14_controlled_eos_train.py"
REVIEW = shared.REPO_ROOT / "Erratum" / "v14a14_promotion_gate.py"


def run() -> int:
    job = shared.load_job()
    job_id = shared.safe_id(job.get("id"), "job.id")
    control_run_id = shared.safe_id(job.get("_control_run_id") or job_id, "control_run_id")
    task = job.get("task")
    if not isinstance(task, dict) or task.get("runner") != RUNNER:
        raise ValueError(f"This worker only supports task.runner={RUNNER}")
    if set(task) != {"runner"}:
        raise ValueError("No arbitrary training options allowed in v14a14 GPU experiment")
    run_dir = shared.CONTROL_ROOT / job_id / control_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "worker.log"
    status = run_dir / "status.json"
    output = (
        shared.PERSISTENT_ROOT / "training/runs"
        / f"v14a14_stronger_supervision_{control_run_id}"
    )
    command = [
        sys.executable, str(ENTRY), "--one-review-diagnostic",
        "--verify-parent-sha256", "--output-dir", str(output),
    ]
    started = {
        "schema_version": 1,
        "job_id": job_id, "control_run_id": control_run_id,
        "runner": RUNNER, "state": "running",
        "started_at": shared.utc_now(),
        "host": socket.gethostname(), "repo_root": str(shared.REPO_ROOT),
        "persistent_root": str(shared.PERSISTENT_ROOT),
        "command": command, "gpu": shared.gpu_snapshot(),
    }
    shared.atomic_json(run_dir / "job.json", job)
    shared.atomic_json(status, started)
    try:
        rc = shared.run_subprocess(command, log_file)
        if rc != 0:
            shared.atomic_json(status, {
                **started, "state": "failed", "returncode": rc,
                "completed_at": shared.utc_now(),
                "log_path": str(log_file),
            })
            return rc
        review_command = [
            sys.executable, str(REVIEW), "prepare",
            "--run-dir", str(output),
        ]
        # Human review is independent and not scored on the GPU worker.
        review_rc = shared.run_subprocess(review_command, log_file)
        completed = {
            **started,
            "state": "completed" if review_rc == 0 else "failed",
            "returncode": review_rc,
            "completed_at": shared.utc_now(),
            "log_path": str(log_file),
            "result_path": str(output / "v14a14_posttraining_diagnostic.json"),
            "candidate_manifest_path": str(
                output / "causal_80/candidate/v14a14_causal_u0040_candidate.manifest.json"
            ),
            "blind_packet_path": str(output / "blind_human_review/review_packet_blinded.md"),
            "automatic_promotion": False,
        }
        shared.atomic_json(status, completed)
        return review_rc
    except BaseException as exc:
        shared.atomic_json(status, {
            **started, "state": "failed", "returncode": None,
            "completed_at": shared.utc_now(),
            "error": repr(exc), "traceback": traceback.format_exc(),
            "log_path": str(log_file),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(run())
