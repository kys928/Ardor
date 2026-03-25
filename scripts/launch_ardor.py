from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _status(msg: str) -> None:
    print(f"[launch] {msg}", flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_state_path() -> Path:
    ardor_home = Path(os.environ.get("ARDOR_HOME", "/workspace/ArdorRuntime")).expanduser()
    return ardor_home / "runtime" / "runtime_state.json"


def _load_runtime_state() -> dict[str, Any]:
    p = _runtime_state_path()
    if not p.exists():
        raise FileNotFoundError(f"runtime state missing: {p}. Run scripts/bootstrap_runtime.py first.")
    return json.loads(p.read_text(encoding="utf-8"))


def _launch_cli(state: dict[str, Any]) -> int:
    backend_map = {"native": "native_ardor", "hf": "hf_causal_lm"}
    backend = str(state.get("backend", "native")).strip().lower()
    backend_family = backend_map.get(backend)
    if backend_family is None:
        raise ValueError(f"Unsupported backend in runtime state: {backend}")

    model_path = state.get("resolved_model_path")
    tokenizer_path = state.get("resolved_tokenizer_path")
    device = state.get("device", "cpu")

    if not model_path:
        raise ValueError("runtime state is missing resolved_model_path")

    cortex_dir = _repo_root() / "Cerebrum" / "Cortex"
    if not cortex_dir.exists():
        raise FileNotFoundError(f"Missing cortex directory: {cortex_dir}")

    sys.path.insert(0, str(cortex_dir))
    import prefrontal_cortex as pfc  # type: ignore

    core = pfc.get_global_core(
        model_path=str(model_path),
        tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
        device=str(device),
        enable_retrieval=True,
        force_reload=True,
        backend_family=backend_family,
    )

    _status("Ardor CLI ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("🗨️  > ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        print("\n🧠 Ardor:")
        answer = core.generate_text(prompt, persona_primer="")
        print(answer)
        print("\n" + "-" * 60 + "\n")
    return 0


def _launch_gui() -> int:
    gui_path = _repo_root() / "Praetor" / "GUI_Cortex.py"
    if not gui_path.exists():
        raise FileNotFoundError(f"GUI launcher not found: {gui_path}")
    return subprocess.call([sys.executable, str(gui_path)], cwd=str(_repo_root()))


def main() -> int:
    state = _load_runtime_state()
    target = str(state.get("launch_target") or os.environ.get("ARDOR_LAUNCH_TARGET", "cli")).strip().lower()

    _status(f"launch target: {target}")
    if target == "cli":
        return _launch_cli(state)
    if target == "gui":
        return _launch_gui()
    if target == "api":
        print("[launch] ERROR: ARDOR_LAUNCH_TARGET=api is not implemented yet.", file=sys.stderr)
        return 2

    print(f"[launch] ERROR: unknown launch target '{target}'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
