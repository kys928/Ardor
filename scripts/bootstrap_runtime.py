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
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True)
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
    ardor_home: Path
    hf_home: Path
    backend: str
    launch_target: str
    model_id: str
    model_path: Path
    tokenizer_path: Path
    device: str
    hf_token: str | None


VALID_BACKENDS = {"native", "hf"}
VALID_TARGETS = {"cli", "gui", "api"}


def _env_cfg() -> RuntimeConfig:
    ardor_home = Path(os.environ.get("ARDOR_HOME", "/workspace/ArdorRuntime")).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "/workspace/.cache/huggingface")).expanduser()
    backend = os.environ.get("ARDOR_BACKEND", "native").strip().lower()
    launch_target = os.environ.get("ARDOR_LAUNCH_TARGET", "cli").strip().lower()

    model_id = os.environ.get("ARDOR_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct").strip()
    model_path = Path(
        os.environ.get("ARDOR_MODEL_PATH", str(ardor_home / "models" / "ardor" / "model_last.pt"))
    ).expanduser()
    tokenizer_path = Path(
        os.environ.get(
            "ARDOR_TOKENIZER_PATH",
            str(ardor_home / "tokenizers" / "tokenizer_v9.json"),
        )
    ).expanduser()

    device = os.environ.get("ARDOR_DEVICE", "cpu").strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    return RuntimeConfig(
        ardor_home=ardor_home,
        hf_home=hf_home,
        backend=backend,
        launch_target=launch_target,
        model_id=model_id,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        hf_token=hf_token,
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


def _validate_native(cfg: RuntimeConfig) -> tuple[str, str]:
    if not cfg.model_path.is_file():
        raise FileNotFoundError(
            f"Native backend requires a checkpoint file at ARDOR_MODEL_PATH: {cfg.model_path}"
        )
    if not cfg.tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Native backend requires a tokenizer file at ARDOR_TOKENIZER_PATH: {cfg.tokenizer_path}"
        )

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(cfg.tokenizer_path))
    vocab_size = int(tok.get_vocab_size())

    ckpt_vocab = None
    try:
        import torch

        raw = torch.load(str(cfg.model_path), map_location="cpu")
        if isinstance(raw, dict):
            meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
            config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
            ckpt_vocab = (
                raw.get("vocab_size")
                or meta.get("vocab_size")
                or config.get("vocab_size")
                or config.get("vocab")
            )
    except Exception as exc:
        _status(f"warning: could not inspect checkpoint vocab metadata ({exc})")

    if ckpt_vocab is not None and int(ckpt_vocab) != vocab_size:
        raise ValueError(
            "Tokenizer/checkpoint incompatibility: "
            f"checkpoint vocab={ckpt_vocab}, tokenizer vocab={vocab_size}"
        )

    return str(cfg.model_path.resolve()), str(cfg.tokenizer_path.resolve())


def _bootstrap_hf(cfg: RuntimeConfig) -> tuple[str, str | None]:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for ARDOR_BACKEND=hf") from exc

    if not cfg.model_id:
        raise ValueError("ARDOR_MODEL_ID is required when ARDOR_BACKEND=hf")

    _status(f"resolving HF model '{cfg.model_id}' via cache '{cfg.hf_home}'")
    model_dir = snapshot_download(
        repo_id=cfg.model_id,
        cache_dir=str(cfg.hf_home),
        token=cfg.hf_token,
    )

    if not Path(model_dir).exists():
        raise FileNotFoundError(f"HF model directory was not resolved: {model_dir}")

    return str(Path(model_dir).resolve()), None


def main() -> int:
    cfg = _env_cfg()

    if cfg.backend not in VALID_BACKENDS:
        return _fail(f"Invalid ARDOR_BACKEND='{cfg.backend}'. Expected one of {sorted(VALID_BACKENDS)}")
    if cfg.launch_target not in VALID_TARGETS:
        return _fail(
            f"Invalid ARDOR_LAUNCH_TARGET='{cfg.launch_target}'. Expected one of {sorted(VALID_TARGETS)}"
        )

    os.environ["ARDOR_HOME"] = str(cfg.ardor_home)
    os.environ["HF_HOME"] = str(cfg.hf_home)

    _status(f"runtime root: {cfg.ardor_home}")
    _status(f"hf cache root: {cfg.hf_home}")
    _ensure_dirs(cfg)

    try:
        if cfg.backend == "native":
            resolved_model, resolved_tokenizer = _validate_native(cfg)
            model_source = "native_checkpoint"
        else:
            resolved_model, resolved_tokenizer = _bootstrap_hf(cfg)
            model_source = f"hf:{cfg.model_id}"

        runtime_state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(_repo_root()),
            "backend": cfg.backend,
            "launch_target": cfg.launch_target,
            "model_source": model_source,
            "resolved_model_path": resolved_model,
            "resolved_tokenizer_path": resolved_tokenizer,
            "resolved_hf_cache": str(cfg.hf_home.resolve()),
            "ardor_home": str(cfg.ardor_home.resolve()),
            "device": cfg.device,
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
