from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _status(msg: str) -> None:
    print(f"[launch] {msg}", flush=True)


def _fail(msg: str, code: int = 1) -> int:
    print(f"[launch] ERROR: {msg}", file=sys.stderr, flush=True)
    return code


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_state_path() -> Path:
    ardor_home = Path(os.environ.get("ARDOR_HOME", "/workspace/ArdorRuntime")).expanduser()
    return ardor_home / "runtime" / "runtime_state.json"


def _load_runtime_state() -> dict[str, Any]:
    path = _runtime_state_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"runtime state missing: {path}. Run scripts/bootstrap_runtime.py first."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse runtime state: {path}") from exc


def _require_state_value(state: dict[str, Any], key: str) -> Any:
    value = state.get(key)
    if value is None or value == "":
        raise ValueError(f"runtime state is missing required key: {key}")
    return value


def _prepare_import_paths() -> tuple[Path, Path]:
    repo_root = _repo_root()
    cortex_dir = repo_root / "Cerebrum" / "Cortex"

    if not repo_root.is_dir():
        raise FileNotFoundError(f"repo root not found: {repo_root}")
    if not cortex_dir.is_dir():
        raise FileNotFoundError(f"missing cortex directory: {cortex_dir}")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(cortex_dir) not in sys.path:
        sys.path.insert(0, str(cortex_dir))

    os.environ["PYTHONPATH"] = f"{repo_root}:{cortex_dir}:{os.environ.get('PYTHONPATH', '')}"
    return repo_root, cortex_dir


def _validate_runtime_state(state: dict[str, Any]) -> tuple[str, str, str, str | None, str, bool, bool]:
    backend = str(_require_state_value(state, "backend")).strip().lower()
    launch_target = str(_require_state_value(state, "launch_target")).strip().lower()
    model_path = str(_require_state_value(state, "resolved_model_path")).strip()
    tokenizer_path_raw = state.get("resolved_tokenizer_path")
    tokenizer_path = str(tokenizer_path_raw).strip() if tokenizer_path_raw else None
    device = str(state.get("device", "cpu")).strip() or "cpu"
    enable_dmn = bool(state.get("enable_dmn", True))
    enable_retrieval = bool(state.get("enable_retrieval", True))

    if backend not in {"native", "hf"}:
        raise ValueError(f"unsupported backend in runtime state: {backend}")

    if launch_target not in {"cli", "gui", "api"}:
        raise ValueError(f"unsupported launch target in runtime state: {launch_target}")

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device in runtime state: {device}")

    return backend, launch_target, model_path, tokenizer_path, device, enable_dmn, enable_retrieval


def _launch_cli(state: dict[str, Any]) -> int:
    backend, _, model_path, tokenizer_path, device, enable_dmn, enable_retrieval = _validate_runtime_state(state)

    backend_map = {
        "native": "native_ardor",
        "hf": "hf_causal_lm",
    }
    backend_family = backend_map[backend]

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"resolved model path does not exist: {model_path_obj}")

    if backend == "native":
        if tokenizer_path is None:
            raise ValueError("native backend requires resolved_tokenizer_path in runtime state")
        tokenizer_path_obj = Path(tokenizer_path)
        if not tokenizer_path_obj.is_file():
            raise FileNotFoundError(f"resolved tokenizer path does not exist: {tokenizer_path_obj}")

    repo_root, _ = _prepare_import_paths()

    try:
        import prefrontal_cortex as pfc  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "failed to import prefrontal_cortex. "
            "Check repo structure, backend dependencies, and import paths."
        ) from exc

    if not hasattr(pfc, "get_global_core"):
        raise AttributeError("prefrontal_cortex does not expose get_global_core()")

    _status(
        f"initializing core backend={backend} backend_family={backend_family} "
        f"device={device} dmn={enable_dmn} retrieval={enable_retrieval}"
    )

    try:
        core = pfc.get_global_core(
            model_path=str(model_path_obj),
            tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
            device=device,
            enable_retrieval=enable_retrieval,
            force_reload=True,
            backend_family=backend_family,
            enable_dmn=enable_dmn,
        )
    except TypeError as exc:
        raise RuntimeError(
            "get_global_core(...) signature does not match launcher expectations. "
            "Check backend_family/enable_dmn/tokenizer_path support."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"failed to initialize Ardor core: {exc}") from exc

    if not hasattr(core, "generate_text"):
        raise AttributeError("initialized core does not expose generate_text(...)")

    _status("Ardor CLI ready. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            prompt = input("🗨️  > ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not prompt:
            continue

        if prompt.lower() in {"exit", "quit"}:
            break

        print("\n🧠 Ardor:")
        try:
            answer = core.generate_text(prompt, persona_primer="")
        except Exception as exc:
            print(f"[launch] generation error: {exc}", file=sys.stderr, flush=True)
            print("\n" + "-" * 60 + "\n")
            continue

        print(answer)
        print("\n" + "-" * 60 + "\n")

    return 0


def _launch_gui(state: dict[str, Any]) -> int:
    repo_root, _ = _prepare_import_paths()
    gui_path = repo_root / "Praetor" / "GUI_Cortex.py"
    if not gui_path.is_file():
        raise FileNotFoundError(f"GUI launcher not found: {gui_path}")

    env = os.environ.copy()
    env["ARDOR_HOME"] = str(state.get("ardor_home") or os.environ.get("ARDOR_HOME", "/workspace/ArdorRuntime"))
    env["ARDOR_BACKEND"] = str(state.get("backend", env.get("ARDOR_BACKEND", "native")))
    env["ARDOR_LAUNCH_TARGET"] = "gui"
    env["ARDOR_DEVICE"] = str(state.get("device", env.get("ARDOR_DEVICE", "cpu")))
    env["ARDOR_ENABLE_DMN"] = "1" if bool(state.get("enable_dmn", True)) else "0"
    env["ARDOR_ENABLE_RETRIEVAL"] = "1" if bool(state.get("enable_retrieval", True)) else "0"

    return subprocess.call([sys.executable, str(gui_path)], cwd=str(repo_root), env=env)


def main() -> int:
    try:
        state = _load_runtime_state()
        target = str(
            state.get("launch_target") or os.environ.get("ARDOR_LAUNCH_TARGET", "cli")
        ).strip().lower()

        _status(f"launch target: {target}")

        if target == "cli":
            return _launch_cli(state)

        if target == "gui":
            return _launch_gui(state)

        if target == "api":
            return _fail("ARDOR_LAUNCH_TARGET=api is not implemented yet.", code=2)

        return _fail(f"unknown launch target '{target}'", code=2)

    except Exception as exc:
        return _fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())