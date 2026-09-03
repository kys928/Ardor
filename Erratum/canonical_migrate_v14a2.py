#!/usr/bin/env python3
"""One-purpose migration of the promoted ORIGINAL v14a2 checkpoint.

This tool intentionally does not accept source/destination arguments. It migrates the
single verified promoted artifact into the canonical `model` / `optimizer` / `meta`
checkpoint schema, then reopens the written file and proves strict compatibility with
the restored canonical ArdorDecoder contract.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CORTEX_ROOT = REPO_ROOT / "Cerebrum" / "Cortex"
for path in (str(CORTEX_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ardor_config import ArdorConfig
from broca_decoder import ArdorDecoder

SOURCE = Path(
    "/workspace/Ardor/training/runs/v14_promoted_candidates/"
    "v14a2_local_anti_loop_margin_best_u600.pt"
)
DESTINATION = Path(
    "/workspace/Ardor/training/runs/v14_promoted_candidates/"
    "v14a2_local_anti_loop_margin_best_u600_CANONICAL_FULLSTATE.pt"
)
REJECTED_REPAIR = Path(
    "/workspace/Ardor/training/runs/v14_promoted_candidates/"
    "v14a2_local_anti_loop_margin_best_u600_REPAIRED_FULLSTATE.pt"
)
TOKENIZER = Path("/workspace/Ardor/tokenizer_v9.json")
EXPECTED_SOURCE_SIZE = 12_182_658_805
EXPECTED_SOURCE_TRAINER = "ardor_v14a2_local_anti_loop_margin_trainer"
EXPECTED_SOURCE_UPDATE = 600
EXPECTED_MODEL_KEY_COUNT = 1060
EXPECTED_SPECIAL_IDS = {
    "pad_id": 0,
    "unk_id": 1,
    "bos_id": 2,
    "eos_id": 3,
    "user_id": 4,
    "assistant_id": 5,
    "system_id": 6,
    "eot_id": 7,
}
SPECIAL_TOKENS = {
    "pad_id": "<pad>",
    "unk_id": "<unk>",
    "bos_id": "<bos>",
    "eos_id": "<eos>",
    "user_id": "<|user|>",
    "assistant_id": "<|assistant|>",
    "system_id": "<|system|>",
    "eot_id": "<|eot|>",
}
MODEL_CONFIG = {
    "vocab_size": 52224,
    "hidden_size": 1536,
    "n_layers": 33,
    "n_heads": 24,
    "max_len": 2048,
    "ff_mult": 4,
    "dropout": 0.0,
    "attn_dropout": 0.0,
    "resid_dropout": 0.0,
    "layernorm_eps": 1e-5,
    "use_rope": True,
    "rope_theta": 10000.0,
}
BASE_CHECKPOINT = (
    "/workspace/Ardor/training/runs/"
    "sft_v13b_routing_stage_trainer_v10_27_guard4_heavy_four_guard_correlation_phrase_landing/"
    "checkpoints/best_model.pt"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def available_memory_gib() -> float | None:
    try:
        fields: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, rest = line.split(":", 1)
            fields[name] = int(rest.strip().split()[0])
        kib = fields.get("MemAvailable", fields.get("MemFree"))
        return None if kib is None else kib / 1024**2
    except Exception:
        return None


def tokenizer_contract() -> tuple[int, dict[str, int | None]]:
    raw = json.loads(TOKENIZER.read_text(encoding="utf-8"))
    vocab = raw["model"]["vocab"]
    special_ids = {name: vocab.get(token) for name, token in SPECIAL_TOKENS.items()}
    return len(vocab), special_ids


def build_model() -> ArdorDecoder:
    cfg = ArdorConfig.from_dict(MODEL_CONFIG)
    cfg.validate()
    model = ArdorDecoder(cfg)
    if not model.use_rope:
        raise RuntimeError("Canonical migration model unexpectedly has use_rope=False")
    if model.position_embed is not None:
        raise RuntimeError("Canonical migration model unexpectedly has learned position embeddings")
    if not model.blocks[0].attn.use_rope:
        raise RuntimeError("Canonical attention is not applying RoPE")
    if float(model.blocks[0].attn.rope_theta) != 10000.0:
        raise RuntimeError("Canonical attention rope_theta mismatch")
    persistent_rope_keys = [key for key in model.state_dict() if "rope" in key]
    if persistent_rope_keys:
        raise RuntimeError(f"RoPE unexpectedly created persistent state keys: {persistent_rope_keys}")
    return model


def build_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    # Same optimizer family/topology as the canonical trainer. Hyperparameters are
    # overwritten by optimizer.load_state_dict; topology compatibility is the gate.
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )


def strict_model_contract(model: torch.nn.Module, state: dict[str, Any]) -> dict[str, Any]:
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatches = [
        {
            "key": key,
            "expected": list(expected[key].shape),
            "actual": list(state[key].shape),
        }
        for key in sorted(set(expected) & set(state))
        if tuple(expected[key].shape) != tuple(state[key].shape)
    ]
    result = model.load_state_dict(state, strict=True)
    return {
        "expected_key_count": len(expected),
        "actual_key_count": len(state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "strict_load_ok": True,
        "strict_result": str(result),
    }


def assert_nested_equal(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise AssertionError(f"Tensor/non-tensor mismatch at {path}")
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            raise AssertionError(f"Tensor metadata mismatch at {path}")
        if not torch.equal(left, right):
            raise AssertionError(f"Tensor value mismatch at {path}")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise AssertionError(f"Dict/non-dict mismatch at {path}")
        if set(left) != set(right):
            raise AssertionError(f"Dictionary key mismatch at {path}")
        for key in left:
            assert_nested_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"Sequence mismatch at {path}")
        for index, (l_item, r_item) in enumerate(zip(left, right)):
            assert_nested_equal(l_item, r_item, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"Value mismatch at {path}: {left!r} != {right!r}")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return repr(value)


def verify_canonical_checkpoint(
    checkpoint: dict[str, Any],
    *,
    source_model: dict[str, Any] | None = None,
    source_optimizer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if set(checkpoint) != {"model", "optimizer", "meta"}:
        raise RuntimeError(f"Canonical top-level keys mismatch: {sorted(checkpoint)}")
    if not isinstance(checkpoint["model"], dict):
        raise RuntimeError("Canonical checkpoint model payload is not a state dict")
    if not isinstance(checkpoint["optimizer"], dict):
        raise RuntimeError("Canonical checkpoint optimizer payload is not a state dict")
    if not isinstance(checkpoint["meta"], dict):
        raise RuntimeError("Canonical checkpoint meta payload is not a dict")

    meta_cfg = checkpoint["meta"].get("model_config")
    if meta_cfg != MODEL_CONFIG:
        raise RuntimeError(f"Canonical model_config mismatch: {meta_cfg!r}")

    model = build_model()
    model_contract = strict_model_contract(model, checkpoint["model"])
    optimizer = build_optimizer(model)
    optimizer.load_state_dict(checkpoint["optimizer"])

    if source_model is not None:
        assert_nested_equal(source_model, checkpoint["model"], "model")
    if source_optimizer is not None:
        assert_nested_equal(source_optimizer, checkpoint["optimizer"], "optimizer")

    return {
        "model": model_contract,
        "optimizer": {
            "present": True,
            "load_state_dict_ok": True,
            "param_groups": len(optimizer.param_groups),
        },
        "round_trip_exact_model": source_model is not None,
        "round_trip_exact_optimizer": source_optimizer is not None,
    }


def migrate(run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "canonical_migration.json"
    available_gib = available_memory_gib()
    if available_gib is not None and available_gib < 48.0:
        raise RuntimeError(
            f"Refusing 12 GB checkpoint migration with only {available_gib:.2f} GiB available RAM"
        )

    for required in (SOURCE, TOKENIZER):
        if not required.exists():
            raise FileNotFoundError(required)
    source_size = SOURCE.stat().st_size
    if source_size != EXPECTED_SOURCE_SIZE:
        raise RuntimeError(
            f"Source size changed: expected {EXPECTED_SOURCE_SIZE}, observed {source_size}"
        )

    vocab_size, special_ids = tokenizer_contract()
    if vocab_size != MODEL_CONFIG["vocab_size"]:
        raise RuntimeError(
            f"Tokenizer/model vocab mismatch: tokenizer={vocab_size}, model={MODEL_CONFIG['vocab_size']}"
        )
    if special_ids != EXPECTED_SPECIAL_IDS:
        raise RuntimeError(
            f"Tokenizer special-token mismatch: expected {EXPECTED_SPECIAL_IDS}, observed {special_ids}"
        )

    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        raise RuntimeError("Original v14a2 checkpoint is not a dictionary")
    if source.get("trainer") != EXPECTED_SOURCE_TRAINER:
        raise RuntimeError(f"Unexpected source trainer: {source.get('trainer')!r}")
    if int(source.get("update", -1)) != EXPECTED_SOURCE_UPDATE:
        raise RuntimeError(f"Unexpected source update: {source.get('update')!r}")
    model_state = source.get("model_state_dict")
    optimizer_state = source.get("optimizer_state_dict")
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        raise RuntimeError("Original v14a2 is missing model_state_dict or optimizer_state_dict")
    if len(model_state) != EXPECTED_MODEL_KEY_COUNT:
        raise RuntimeError(f"Unexpected model key count: {len(model_state)}")
    if "position_embed.weight" in model_state:
        raise RuntimeError("Original v14a2 unexpectedly contains position_embed.weight")

    pre_model = build_model()
    pre_model_contract = strict_model_contract(pre_model, model_state)
    pre_optimizer = build_optimizer(pre_model)
    pre_optimizer.load_state_dict(optimizer_state)

    migration_meta = {
        "step": EXPECTED_SOURCE_UPDATE,
        "tokens_seen": 0,
        "stage": "legacy_v14a2_canonical_migration",
        "special_ids": special_ids,
        "model_config": dict(MODEL_CONFIG),
        "migration": {
            "schema_version": 1,
            "created_at": utc_now(),
            "source_path": str(SOURCE),
            "source_size_bytes": source_size,
            "source_trainer": source.get("trainer"),
            "source_update": source.get("update"),
            "source_model_key": "model_state_dict",
            "source_optimizer_key": "optimizer_state_dict",
            "base_checkpoint": BASE_CHECKPOINT,
            "rejected_repaired_checkpoint": str(REJECTED_REPAIR),
            "tensor_transform": "none",
            "wrapper_transform": "model_state_dict->model; optimizer_state_dict->optimizer",
            "legacy_tokens_seen_available": False,
        },
        "legacy": {
            "kind": json_safe(source.get("kind")),
            "trainer": json_safe(source.get("trainer")),
            "update": json_safe(source.get("update")),
            "created_at_unix": json_safe(source.get("created_at_unix")),
            "promotion_gate": json_safe(source.get("promotion_gate")),
        },
    }
    canonical = {
        "model": model_state,
        "optimizer": optimizer_state,
        "meta": migration_meta,
    }

    if DESTINATION.exists():
        existing = torch.load(DESTINATION, map_location="cpu", weights_only=False)
        post_contract = verify_canonical_checkpoint(
            existing,
            source_model=model_state,
            source_optimizer=optimizer_state,
        )
        report = {
            "passed": True,
            "reused_existing": True,
            "source": str(SOURCE),
            "destination": str(DESTINATION),
            "source_size_bytes": source_size,
            "destination_size_bytes": DESTINATION.stat().st_size,
            "available_memory_gib_at_start": available_gib,
            "tokenizer": {
                "path": str(TOKENIZER),
                "vocab_size": vocab_size,
                "special_ids": special_ids,
            },
            "pre_write_contract": {
                "model": pre_model_contract,
                "optimizer": {"load_state_dict_ok": True, "param_groups": len(pre_optimizer.param_groups)},
            },
            "post_write_contract": post_contract,
            "model_config": MODEL_CONFIG,
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    tmp = DESTINATION.with_suffix(DESTINATION.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    torch.save(canonical, tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, DESTINATION)

    reopened = torch.load(DESTINATION, map_location="cpu", weights_only=False)
    post_contract = verify_canonical_checkpoint(
        reopened,
        source_model=model_state,
        source_optimizer=optimizer_state,
    )

    report = {
        "passed": True,
        "reused_existing": False,
        "source": str(SOURCE),
        "destination": str(DESTINATION),
        "source_size_bytes": source_size,
        "destination_size_bytes": DESTINATION.stat().st_size,
        "available_memory_gib_at_start": available_gib,
        "tokenizer": {
            "path": str(TOKENIZER),
            "vocab_size": vocab_size,
            "special_ids": special_ids,
        },
        "pre_write_contract": {
            "model": pre_model_contract,
            "optimizer": {"load_state_dict_ok": True, "param_groups": len(pre_optimizer.param_groups)},
        },
        "post_write_contract": post_contract,
        "model_config": MODEL_CONFIG,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    del reopened, canonical, source, pre_optimizer, pre_model
    gc.collect()
    return report


def main() -> int:
    run_dir = Path(os.environ.get("ARDOR_MIGRATION_REPORT_DIR", "/workspace/ardor-control/manual-migration"))
    report = migrate(run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
