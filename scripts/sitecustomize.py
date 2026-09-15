"""Process-wide compatibility shims for Ardor control/worker scripts.

1. GitHub-hosted runners can receive Cloudflare Error 1010 from rest.runpod.io
   when Python urllib sends its default User-Agent. Keep urllib semantics
   unchanged, but give Request objects an explicit application User-Agent.
2. RunPod can restart a container a few seconds after the worker process exits.
   For the GPU worker only, keep the terminal Python process alive after normal
   interpreter shutdown so the detached supervisor can observe status.json and
   delete the Pod before the container is relaunched. Trainer subprocesses and
   GitHub control-plane processes are deliberately excluded.
"""
from __future__ import annotations

import atexit
import os
from pathlib import Path
import sys
import time
import urllib.request

_ORIGINAL_REQUEST = urllib.request.Request
_USER_AGENT = "Ardor-RunPod-Control/1.0 (+https://github.com/kys928/Ardor)"


class ArdorRequest(_ORIGINAL_REQUEST):
    def __init__(self, url, data=None, headers=None, origin_req_host=None, unverifiable=False, method=None):
        merged = dict(headers or {})
        if not any(str(key).lower() == "user-agent" for key in merged):
            merged["User-Agent"] = _USER_AGENT
        super().__init__(
            url,
            data=data,
            headers=merged,
            origin_req_host=origin_req_host,
            unverifiable=unverifiable,
            method=method,
        )


urllib.request.Request = ArdorRequest


def _is_runpod_worker_process() -> bool:
    """Return True only for the top-level GPU worker launched by RunPod."""
    if not os.environ.get("ARDOR_JOB_B64", "").strip():
        return False
    if not sys.argv:
        return False
    return Path(sys.argv[0]).name == "runpod_worker.py"


def _hold_terminal_worker_for_controller_cleanup() -> None:
    """Prevent RunPod from immediately relaunching a completed worker container."""
    print(
        "[runpod-worker] terminal process held for controller cleanup; awaiting Pod deletion",
        flush=True,
    )
    while True:
        time.sleep(3600)


if _is_runpod_worker_process():
    atexit.register(_hold_terminal_worker_for_controller_cleanup)
