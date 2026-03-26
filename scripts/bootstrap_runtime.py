from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def _status(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def _fail(msg: str, code: int = 1) -> int:
    print(f"[bootstrap] ERROR: {msg}", file=sys.stderr, flush=True)
    return code


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return None


def _git_branch(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


@dataclass
class RuntimeConfig:
    repo_root: Path
    ardor_home: Path
    hf_home: Path

    backend_request: str
    backend: str
    launch_target: str

    model_id: str
    model_path: Path
    tokenizer_path: Path

    device_request: str
    resolved_device: str
    gpu_name: str | None

    hf_token: str | None
    hf_local_only: bool
    hf_allow_patterns: tuple[str, ...]
    hf_revision: str | None

    enable_dmn: bool
    enable_retrieval: bool


VALID_BACKENDS = {"auto", "native", "hf"}
VALID_RESOLVED_BACKENDS = {"native", "hf"}
VALID_TARGETS = {"cli", "gui", "api"}


def _parse_csv_env(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _torch_cuda_available() -> tuple[bool, str | None]:
    try:
        import torch  # type: ignore
        ok = bool(torch.cuda.is_available())
        if ok:
            try:
                return True, torch.cuda.get_device_name(0)
            except Exception:
                return True, "unknown-cuda-device"
        return False, None
    except Exception:
        return False, None


def _resolve_device(device_request: str) -> tuple[str, str | None]:
    req = (device_request or "").strip().lower() or "auto"

    if req not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"Invalid ARDOR_DEVICE='{device_request}'. Expected 'auto', 'cpu', or 'cuda'."
        )

    cuda_ok, gpu_name = _torch_cuda_available()

    if req == "cpu":
        return "cpu", gpu_name

    if req == "cuda":
        if not cuda_ok:
            raise RuntimeError(
                "ARDOR_DEVICE=cuda requested, but CUDA is not available."
            )
        return "cuda", gpu_name

    return ("cuda", gpu_name) if cuda_ok else ("cpu", None)


def _resolve_backend(
    backend_request: str,
    resolved_device: str,
    model_id: str,
) -> str:
    req = (backend_request or "").strip().lower() or "auto"

    if req not in {"auto", "native", "hf"}:
        raise ValueError(
            f"Invalid ARDOR_BACKEND='{backend_request}'. Expected 'auto', 'native', or 'hf'."
        )

    if req in {"native", "hf"}:
        return req

    # auto policy:
    # prefer HF on CUDA-capable pods
    if resolved_device == "cuda" and model_id:
        return "hf"

    # otherwise fall back to native
    return "native"


def _env_cfg() -> RuntimeConfig:
    repo_root = _repo_root()

    ardor_home = Path(os.environ.get("ARDOR_HOME", "/workspace/ArdorRuntime")).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "/workspace/.cache/huggingface")).expanduser()

    backend_request = os.environ.get("ARDOR_BACKEND", "auto").strip().lower()
    launch_target = os.environ.get("ARDOR_LAUNCH_TARGET", "cli").strip().lower()

    model_id = os.environ.get("ARDOR_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct").strip()

    model_path = Path(
        os.environ.get(
            "ARDOR_MODEL_PATH",
            str(ardor_home / "models" / "ardor" / "model_last.pt"),
        )
    ).expanduser()

    tokenizer_path = Path(
        os.environ.get(
            "ARDOR_TOKENIZER_PATH",
            str(ardor_home / "tokenizers" / "tokenizer_v9.json"),
        )
    ).expanduser()

    device_request = os.environ.get("ARDOR_DEVICE", "auto").strip() or "auto"
    resolved_device, gpu_name = _resolve_device(device_request)

    backend = _resolve_backend(
        backend_request=backend_request,
        resolved_device=resolved_device,
        model_id=model_id,
    )

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    hf_local_only = _env_bool("ARDOR_HF_LOCAL_ONLY", False)
    hf_allow_patterns = _parse_csv_env("ARDOR_HF_ALLOW_PATTERNS")
    hf_revision = os.environ.get("ARDOR_MODEL_REVISION", "").strip() or None

    enable_dmn = _env_bool("ARDOR_ENABLE_DMN", True)
    enable_retrieval = _env_bool("ARDOR_ENABLE_RETRIEVAL", True)

    return RuntimeConfig(
        repo_root=repo_root,
        ardor_home=ardor_home,
        hf_home=hf_home,
        backend_request=backend_request,
        backend=backend,
        launch_target=launch_target,
        model_id=model_id,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device_request=device_request,
        resolved_device=resolved_device,
        gpu_name=gpu_name,
        hf_token=hf_token,
        hf_local_only=hf_local_only,
        hf_allow_patterns=hf_allow_patterns,
        hf_revision=hf_revision,
        enable_dmn=enable_dmn,
        enable_retrieval=enable_retrieval,
    )


def _ensure_dirs(cfg: RuntimeConfig) -> dict[str, Path]:
    dirs = {
        "models": cfg.ardor_home / "models",
        "tokenizers": cfg.ardor_home / "tokenizers",
        "logs": cfg.ardor_home / "logs",
        "runtime": cfg.ardor_home / "runtime",
    }
    cfg.hf_home.mkdir(parents=True, exist_ok=True)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _validate_native(cfg: RuntimeConfig) -> tuple[str, str]:
    model_path = _require_file(cfg.model_path, "Native checkpoint")
    tokenizer_path = _require_file(cfg.tokenizer_path, "Native tokenizer")

    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise RuntimeError("tokenizers is required for native runtime validation") from exc

    tok = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = int(tok.get_vocab_size())

    ckpt_vocab = None
    ckpt_arch = None

    try:
        import torch  # type: ignore

        raw = torch.load(str(model_path), map_location="cpu")
        if isinstance(raw, dict):
            meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
            config = raw.get("config") if isinstance(raw.get("config"), dict) else {}

            ckpt_vocab = (
                raw.get("vocab_size")
                or meta.get("vocab_size")
                or config.get("vocab_size")
                or config.get("vocab")
            )
            ckpt_arch = raw.get("arch") or meta.get("arch") or config.get("arch")
    except Exception as exc:
        _status(f"warning: could not inspect checkpoint metadata ({exc})")

    if ckpt_vocab is not None:
        try:
            ckpt_vocab_int = int(ckpt_vocab)
        except Exception as exc:
            raise ValueError(f"checkpoint vocab metadata is not an integer: {ckpt_vocab!r}") from exc

        if ckpt_vocab_int != vocab_size:
            raise ValueError(
                "Tokenizer/checkpoint incompatibility: "
                f"checkpoint vocab={ckpt_vocab_int}, tokenizer vocab={vocab_size}"
            )

    if ckpt_arch is not None:
        _status(f"detected native checkpoint arch: {ckpt_arch}")

    return str(model_path.resolve()), str(tokenizer_path.resolve())


def _bootstrap_hf(cfg: RuntimeConfig) -> tuple[str, str | None]:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for HF backend. "
            "Install the HF runtime extra and regenerate uv.lock."
        ) from exc

    if not cfg.model_id:
        raise ValueError("ARDOR_MODEL_ID is required when backend resolves to hf")

    kwargs: dict[str, Any] = {
        "repo_id": cfg.model_id,
        "cache_dir": str(cfg.hf_home),
        "token": cfg.hf_token,
        "local_files_only": cfg.hf_local_only,
    }

    if cfg.hf_revision:
        kwargs["revision"] = cfg.hf_revision

    if cfg.hf_allow_patterns:
        kwargs["allow_patterns"] = list(cfg.hf_allow_patterns)

    _status(
        f"resolving HF model '{cfg.model_id}' "
        f"(revision={cfg.hf_revision or 'default'}, local_only={cfg.hf_local_only}) "
        f"via cache '{cfg.hf_home}'"
    )

    try:
        model_dir = snapshot_download(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to resolve/download Hugging Face model. "
            f"repo_id={cfg.model_id!r}, revision={cfg.hf_revision or 'default'!r}, "
            f"cache_dir={str(cfg.hf_home)!r}. Original error: {exc}"
        ) from exc

    resolved_dir = Path(model_dir).resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"HF model directory was not resolved: {resolved_dir}")

    config_json = resolved_dir / "config.json"
    if not config_json.exists():
        _status(f"warning: no config.json found in resolved HF model dir: {resolved_dir}")

    return str(resolved_dir), None


def main() -> int:
    try:
        cfg = _env_cfg()
    except Exception as exc:
        return _fail(str(exc))

    if cfg.backend_request not in VALID_BACKENDS:
        return _fail(
            f"Invalid ARDOR_BACKEND='{cfg.backend_request}'. Expected one of {sorted(VALID_BACKENDS)}"
        )

    if cfg.backend not in VALID_RESOLVED_BACKENDS:
        return _fail(
            f"Resolved backend '{cfg.backend}' is invalid. Expected one of {sorted(VALID_RESOLVED_BACKENDS)}"
        )

    if cfg.launch_target not in VALID_TARGETS:
        return _fail(
            f"Invalid ARDOR_LAUNCH_TARGET='{cfg.launch_target}'. Expected one of {sorted(VALID_TARGETS)}"
        )

    os.environ["ARDOR_HOME"] = str(cfg.ardor_home)
    os.environ["HF_HOME"] = str(cfg.hf_home)
    os.environ["ARDOR_BACKEND"] = cfg.backend
    os.environ["ARDOR_LAUNCH_TARGET"] = cfg.launch_target
    os.environ["ARDOR_DEVICE"] = cfg.resolved_device
    os.environ["ARDOR_ENABLE_DMN"] = "1" if cfg.enable_dmn else "0"
    os.environ["ARDOR_ENABLE_RETRIEVAL"] = "1" if cfg.enable_retrieval else "0"
    os.environ["PYTHONPATH"] = (
        f"{cfg.repo_root}:{cfg.repo_root / 'Cerebrum' / 'Cortex'}:{os.environ.get('PYTHONPATH', '')}"
    )

    _status(f"repo root: {cfg.repo_root}")
    _status(f"runtime root: {cfg.ardor_home}")
    _status(f"hf cache root: {cfg.hf_home}")
    _status(f"backend request: {cfg.backend_request}")
    _status(f"resolved backend: {cfg.backend}")
    _status(f"launch target: {cfg.launch_target}")
    _status(f"device request: {cfg.device_request}")
    if cfg.resolved_device == "cuda":
        _status(f"resolved device: cuda ({cfg.gpu_name or 'unknown'})")
    else:
        _status("resolved device: cpu")
    _status(f"dmn enabled: {cfg.enable_dmn}")
    _status(f"retrieval enabled: {cfg.enable_retrieval}")

    _ensure_dirs(cfg)

    try:
        if cfg.backend == "native":
            if not cfg.model_path.is_file():
                raise FileNotFoundError(
                    "Auto/native backend requires a native checkpoint, but it was not found at "
                    f"{cfg.model_path}"
                )
            if not cfg.tokenizer_path.is_file():
                raise FileNotFoundError(
                    "Auto/native backend requires a tokenizer, but it was not found at "
                    f"{cfg.tokenizer_path}"
                )
            resolved_model, resolved_tokenizer = _validate_native(cfg)
            model_source = "native_checkpoint"
        else:
            resolved_model, resolved_tokenizer = _bootstrap_hf(cfg)
            model_source = f"hf:{cfg.model_id}"

        runtime_state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(cfg.repo_root),
            "git_branch": _git_branch(cfg.repo_root),
            "backend_request": cfg.backend_request,
            "backend": cfg.backend,
            "launch_target": cfg.launch_target,
            "model_source": model_source,
            "model_id": cfg.model_id if cfg.backend == "hf" else None,
            "model_revision": cfg.hf_revision if cfg.backend == "hf" else None,
            "resolved_model_path": resolved_model,
            "resolved_tokenizer_path": resolved_tokenizer,
            "resolved_hf_cache": str(cfg.hf_home.resolve()),
            "ardor_home": str(cfg.ardor_home.resolve()),
            "device_request": cfg.device_request,
            "device": cfg.resolved_device,
            "gpu_name": cfg.gpu_name,
            "enable_dmn": cfg.enable_dmn,
            "enable_retrieval": cfg.enable_retrieval,
            "hf_local_only": cfg.hf_local_only,
            "hf_allow_patterns": list(cfg.hf_allow_patterns),
            "python_executable": sys.executable,
        }

        runtime_state_path = cfg.ardor_home / "runtime" / "runtime_state.json"
        _atomic_write_json(runtime_state_path, runtime_state)

        _status(
            "ready "
            f"backend={cfg.backend} target={cfg.launch_target} "
            f"model={resolved_model} tokenizer={resolved_tokenizer or 'n/a'}"
        )
        _status(f"wrote runtime state: {runtime_state_path}")
        return 0

    except Exception as exc:
        return _fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())