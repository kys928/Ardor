#!/usr/bin/env python3
"""Control-plane entrypoint that registers current fixed-purpose experimental runners."""
from __future__ import annotations

import runpod_control

runpod_control.FIXED_PURPOSE_RUNNERS.add("v14a4_format_ab_u100")
runpod_control.FIXED_PURPOSE_RUNNERS.add("v14a4_signal_quality_audit")
runpod_control.FIXED_PURPOSE_RUNNERS.add("v14a5_signal_first_u100")
runpod_control.FIXED_PURPOSE_RUNNERS.add("v14a6_learning_dynamics_probe")
runpod_control.FIXED_PURPOSE_RUNNERS.add("v14a7_clip_ablation")

_ORIGINAL_VALIDATE_COMPUTE = runpod_control.validate_compute


def _cuda128_start_command(worker_script: str) -> list[str]:
    return [
        "cd /opt/Ardor && "
        "uv pip install --python .venv/bin/python "
        "--index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 && "
        ".venv/bin/python -c \"import torch; "
        "assert torch.version.cuda == '12.8'; "
        "assert torch.cuda.is_available(), 'CUDA unavailable after cu128 bootstrap'; "
        "print(torch.__version__, torch.version.cuda)\" && "
        f".venv/bin/python {worker_script}"
    ]


def _validate_compute_with_experimental_worker(job):
    payload, hourly_cap, timeout_minutes, control_run_id, mode = _ORIGINAL_VALIDATE_COMPUTE(job)
    task = job.get("task") or {}
    runner = str(task.get("runner", ""))
    if runner == "v14a5_signal_first_u100":
        payload["dockerStartCmd"] = _cuda128_start_command(
            "scripts/runpod_v14a5_worker.py"
        )
    elif runner == "v14a6_learning_dynamics_probe":
        payload["dockerStartCmd"] = _cuda128_start_command(
            "scripts/runpod_v14a6_worker.py"
        )
    elif runner == "v14a7_clip_ablation":
        payload["dockerStartCmd"] = _cuda128_start_command(
            "scripts/runpod_v14a7_worker.py"
        )
    return payload, hourly_cap, timeout_minutes, control_run_id, mode


runpod_control.validate_compute = _validate_compute_with_experimental_worker


if __name__ == "__main__":
    runpod_control.main()
