#!/usr/bin/env python3
"""Control-plane entrypoint that registers current fixed-purpose experimental runners."""
from __future__ import annotations

import runpod_control

runpod_control.FIXED_PURPOSE_RUNNERS.add("v14a4_format_ab_u100")
runpod_control.FIXED_PURPOSE_RUNNERS.add("v14a4_signal_quality_audit")
runpod_control.FIXED_PURPOSE_RUNNERS.add("v14a5_signal_first_u100")

_ORIGINAL_VALIDATE_COMPUTE = runpod_control.validate_compute


def _validate_compute_with_experimental_worker(job):
    payload, hourly_cap, timeout_minutes, control_run_id, mode = _ORIGINAL_VALIDATE_COMPUTE(job)
    task = job.get("task") or {}
    if str(task.get("runner", "")) == "v14a5_signal_first_u100":
        payload["dockerStartCmd"] = [
            "cd /opt/Ardor && uv run --frozen python scripts/runpod_v14a5_worker.py"
        ]
    return payload, hourly_cap, timeout_minutes, control_run_id, mode


runpod_control.validate_compute = _validate_compute_with_experimental_worker


if __name__ == "__main__":
    runpod_control.main()
