#!/usr/bin/env python3
"""Restricted Ardor worker executed inside an ephemeral RunPod Pod."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CORTEX_ROOT = REPO_ROOT / "Cerebrum" / "Cortex"
PERSISTENT_ROOT = Path("/workspace/Ardor")
CONTROL_ROOT = Path("/workspace/ardor-control/runs")
TRAIN_ENTRY = REPO_ROOT / "Hephaestus" / "runpod_train_entry.py"
ALLOWED_STAGES = {"lm_base", "stabilize", "sft"}
PATH_ARGS = {
    "resume": "--resume",
    "train_tokens": "--train_tokens",
    "train_meta": "--train_meta",
    "val_tokens": "--val_tokens",
    "val_meta": "--val_meta",
    "sft_jsonl": "--sft_jsonl",
    "tokenizer": "--tokenizer",
    "run_dir": "--run_dir",
    "gen_probe_prompts": "--gen_probe_prompts",
}
FORBIDDEN_ARCH_OVERRIDES = {"hidden_size", "n_layers", "n_heads", "ff_mult", "ctx"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def load_job() -> dict[str, Any]:
    encoded = os.environ.get("ARDOR_JOB_B64", "")
    if not encoded:
        raise RuntimeError("ARDOR_JOB_B64 is not set")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        job = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("ARDOR_JOB_B64 is not valid base64-encoded JSON") from exc
    if not isinstance(job, dict):
        raise RuntimeError("Job payload must be a JSON object")
    return job


def safe_id(value: object, label: str) -> str:
    text = str(value or "")
    if not text or len(text) > 96:
        raise ValueError(f"{label} must be 1-96 characters")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if any(ch not in allowed for ch in text):
        raise ValueError(f"{label} contains unsafe characters")
    return text


def safe_persistent_path(value: object, label: str, *, must_exist: bool = False) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} must not be empty")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    resolved = path.resolve(strict=False)
    root = PERSISTENT_ROOT.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be {root} or a child of it")
    if must_exist and not resolved.exists():
        raise ValueError(f"{label} does not exist: {resolved}")
    return resolved


def gpu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return {
            "returncode": result.returncode,
            "output": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def memory_total_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                return round(kib / 1024**2, 3)
    except Exception:
        return None
    return None


def infra_smoke(run_dir: Path) -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(CORTEX_ROOT) not in sys.path:
        sys.path.insert(0, str(CORTEX_ROOT))

    import torch
    import tokenizers
    from Cerebrum.Cortex.ardor_config import ArdorConfig
    from Cerebrum.Cortex.broca_decoder import ArdorDecoder

    if not PERSISTENT_ROOT.exists():
        raise RuntimeError(f"Persistent Ardor root is missing: {PERSISTENT_ROOT}")

    disk = shutil.disk_usage("/workspace")
    control_probe = run_dir / "write_probe.txt"
    control_probe.write_text("Ardor RunPod control-plane write probe\n", encoding="utf-8")
    probe_ok = control_probe.read_text(encoding="utf-8").startswith("Ardor RunPod")
    control_probe.unlink()

    expected = {
        "tokenizer_v9": PERSISTENT_ROOT / "tokenizer_v9.json",
        "train_tokens_20B": PERSISTENT_ROOT / "bin_dataset_20B" / "tokens.bin",
        "heldout_tokens_25M": PERSISTENT_ROOT / "bin_dataset_heldout_25M" / "tokens.bin",
        "runs": PERSISTENT_ROOT / "runs",
    }

    result = {
        "checked_at": utc_now(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "memory_total_gib": memory_total_gib(),
        "repo_root": str(REPO_ROOT),
        "persistent_root": str(PERSISTENT_ROOT),
        "workspace": {
            "path": "/workspace",
            "total_gib": round(disk.total / 1024**3, 3),
            "used_gib": round(disk.used / 1024**3, 3),
            "free_gib": round(disk.free / 1024**3, 3),
            "control_write_probe": probe_ok,
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        },
        "tokenizers_version": tokenizers.__version__,
        "ardor_imports": {
            "ArdorConfig": ArdorConfig.__name__,
            "ArdorDecoder": ArdorDecoder.__name__,
        },
        "persistent_paths": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in expected.items()
        },
        "gpu": gpu_snapshot(),
    }
    atomic_json(run_dir / "infra_smoke.json", result)

    if not probe_ok:
        raise RuntimeError("Network Volume control write/read probe failed")
    if not result["torch"]["cuda_available"]:
        raise RuntimeError("CUDA is not available inside the RunPod GPU worker")
    if int(result["torch"]["device_count"]) < 1:
        raise RuntimeError("No CUDA GPU devices are visible to PyTorch")
    return result


def build_promptgen_command(job: dict[str, Any], job_id: str, control_run_id: str) -> list[str]:
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job.task must be an object")

    stage = str(task.get("stage", ""))
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"Unsupported Ardor training stage: {stage!r}")

    forbidden = sorted(key for key in FORBIDDEN_ARCH_OVERRIDES if task.get(key) is not None)
    if forbidden:
        raise ValueError(
            "Architecture/context overrides are disabled in the v1 control plane "
            f"to protect checkpoint compatibility: {forbidden}"
        )

    tokenizer = safe_persistent_path(task.get("tokenizer"), "task.tokenizer", must_exist=True)
    default_run_dir = PERSISTENT_ROOT / "runs" / "runpod" / job_id / control_run_id
    run_dir = safe_persistent_path(task.get("run_dir") or default_run_dir, "task.run_dir")

    cmd = [
        sys.executable,
        str(TRAIN_ENTRY),
        "--stage",
        stage,
        "--tokenizer",
        str(tokenizer),
        "--run_dir",
        str(run_dir),
    ]

    for field, flag in PATH_ARGS.items():
        if field in {"tokenizer", "run_dir"}:
            continue
        value = task.get(field)
        if value is None:
            continue
        path = safe_persistent_path(value, f"task.{field}", must_exist=field != "run_dir")
        cmd.extend([flag, str(path)])

    if stage == "sft" and task.get("sft_jsonl") is None:
        raise ValueError("Ardor sft stage requires task.sft_jsonl")

    if task.get("seed") is not None:
        seed = int(task["seed"])
        if not 0 <= seed <= 2**31 - 1:
            raise ValueError("task.seed is outside the accepted range")
        cmd.extend(["--seed", str(seed)])

    if bool(task.get("use_compile", False)):
        cmd.append("--use_compile")

    return cmd


def run_subprocess(command: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["ARDOR_CODE_ROOT"] = str(REPO_ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(CORTEX_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def run() -> int:
    job = load_job()
    job_id = safe_id(job.get("id"), "job.id")
    control_run_id = safe_id(job.get("_control_run_id") or job_id, "control_run_id")
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job.task must be an object")
    runner = str(task.get("runner", ""))
    if runner not in {"ardor_promptgen", "infra_smoke"}:
        raise ValueError("Only task.runner='ardor_promptgen' or 'infra_smoke' is enabled")

    run_dir = CONTROL_ROOT / job_id / control_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    log_path = run_dir / "worker.log"
    atomic_json(run_dir / "job.json", job)

    command = (
        ["internal:infra_smoke"]
        if runner == "infra_smoke"
        else build_promptgen_command(job, job_id, control_run_id)
    )
    started = {
        "schema_version": 1,
        "job_id": job_id,
        "control_run_id": control_run_id,
        "runner": runner,
        "state": "running",
        "started_at": utc_now(),
        "host": socket.gethostname(),
        "repo_root": str(REPO_ROOT),
        "persistent_root": str(PERSISTENT_ROOT),
        "command": command,
        "gpu": gpu_snapshot(),
    }
    atomic_json(status_path, started)

    try:
        if runner == "infra_smoke":
            result = infra_smoke(run_dir)
            log_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            returncode = 0
        else:
            returncode = run_subprocess(command, log_path)

        completed = {
            **started,
            "state": "completed" if returncode == 0 else "failed",
            "completed_at": utc_now(),
            "returncode": returncode,
            "log_path": str(log_path),
            "result_path": (
                str(run_dir / "infra_smoke.json")
                if runner == "infra_smoke"
                else None
            ),
        }
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
        atomic_json(status_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(run())
