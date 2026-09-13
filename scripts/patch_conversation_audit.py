from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Expected patch anchor missing in {path}: {old[:120]!r}")
    text2 = text.replace(old, new, 1)
    path.write_text(text2, encoding="utf-8")


# 1) Repair the v14a4 clean-family prompt boundary, while allowing controlled prompt-format ablations.
diag = Path("Erratum/v14a4_diagnostics.py")
replace_once(
    diag,
    '''def _has_any(text: str, signals: Sequence[str]) -> bool:\n    low = norm(text)\n    return any(norm(signal) in low for signal in signals)\n\n\ndef family_sem_check''',
    '''def _has_any(text: str, signals: Sequence[str]) -> bool:\n    low = norm(text)\n    return any(norm(signal) in low for signal in signals)\n\n\ndef clean_eval_prompt(row: dict[str, Any]) -> str:\n    \"\"\"Return the exact answer-boundary prompt used by v14a4 continuation training.\"\"\"\n    text = str(row[\"prompt\"]).rstrip()\n    return text if text.endswith("\\n-") else text + "\\n-"\n\n\ndef family_sem_check''',
)
replace_once(
    diag,
    '''    *,\n    name: str,\n) -> dict[str, Any]:''',
    '''    *,\n    name: str,\n    prompt_transform=None,\n) -> dict[str, Any]:''',
)
replace_once(
    diag,
    '''        qclass = str(row.get("question_class", "normal"))\n        generated = history.generate(model, tok, str(row["prompt"]), device, args, special)\n        historical = history.sem_check(route, generated["text"])''',
    '''        qclass = str(row.get("question_class", "normal"))\n        eval_prompt = clean_eval_prompt(row) if prompt_transform is None else str(prompt_transform(row))\n        generated = history.generate(model, tok, eval_prompt, device, args, special)\n        historical = history.sem_check(route, generated["text"])''',
)
replace_once(
    diag,
    '''            "prompt_style": str(row["prompt_style"]),\n            "generation": generated["text"],''',
    '''            "prompt_style": str(row["prompt_style"]),\n            "eval_prompt": eval_prompt,\n            "generation": generated["text"],''',
)
replace_once(
    diag,
    '''    "family_sem_check",\n    "evaluate_clean_family",''',
    '''    "clean_eval_prompt",\n    "family_sem_check",\n    "evaluate_clean_family",''',
)


# 2) Add a no-training canonical audit that compares raw, train-matched, and runtime role-marked prompts.
audit = r'''#!/usr/bin/env python3
\"\"\"No-training audit of Ardor prompt serialization and v14a4 evaluator validity.

Loads canonical v14a2 once, evaluates the deterministic v14a4 clean holdout under:
1. the legacy raw clean prompt (known v14a4 bug),
2. the exact v14a4 training answer boundary (``\\n-``), and
3. the current runtime role-marked single-turn chat serialization.

It also records tokenizer IDs using the exact tokenizer call conventions in the trainer
(``add_special_tokens=False``) and runtime (default ``Tokenizer.encode``), plus a deterministic
32-row human-audit candidate set. It performs no optimization and writes no checkpoint.
\"\"\"
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from tokenizers import Tokenizer

from Erratum.build_v14a4_family_data import PRIORITY_ROUTES, build_holdout_rows, build_train_rows, validate
from Erratum.canonical_contract_v14a2 import EXPECTED_SPECIAL_IDS, TOKENIZER_PATH, validate_local_files
from Erratum.canonical_eval_v14a2 import _load_canonical_model, _load_historical_eval_module, _tokenizer_contract
from Erratum.v14a4_diagnostics import clean_eval_prompt, compact_clean, evaluate_clean_family

OUTPUT_DIR = Path("/workspace/Ardor/training/runs/v14a4_corrected_conversation_audit")
PROBE_DIR = Path("/workspace/Ardor/training/runs/sft_v14a4_family_balanced_semantic_landing_probe_u100")
DEFAULT_SYSTEM = "You are Ardor. Stay in-context. Be helpful. Speak naturally."


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\\n", encoding="utf-8")


def clean_history_text(text: str) -> str:
    value = (text or "").strip()
    for token in ("<|system|>", "<|user|>", "<|assistant|>", "<|eot|>"):
        value = value.replace(token, "")
    return value.strip()


def runtime_chat_prompt(
    user_text: str,
    *,
    system: str = DEFAULT_SYSTEM,
    turns: Sequence[tuple[str, str]] = (),
) -> str:
    \"\"\"Mirror the current role-token branch of ArdorCore._build_chat_prompt.\"\"\"
    parts = [f"<|system|>\\n{system.strip()}\\n<|eot|>\\n"]
    for role, message in turns:
        message = clean_history_text(message)
        if not message:
            continue
        token = "<|user|>" if role == "user" else "<|assistant|>"
        parts.append(f"{token}\\n{message}\\n<|eot|>\\n")
    user_text = clean_history_text(user_text)
    parts.append(f"<|user|>\\n{user_text}\\n<|eot|>\\n<|assistant|>\\n")
    return "".join(parts)


def token_trace(tok: Tokenizer, text: str, *, add_special_tokens: bool) -> dict[str, Any]:
    enc = tok.encode(text, add_special_tokens=add_special_tokens)
    ids = list(enc.ids)
    return {
        "text": text,
        "add_special_tokens": add_special_tokens,
        "ids": ids,
        "tokens": [tok.id_to_token(i) for i in ids],
        "length": len(ids),
    }


def configure_eval_args(history):
    args = history.build_argparser().parse_args([])
    args.device = "cuda"
    args.max_len = 768
    args.max_gen_tokens = 72
    args.eval_samples = 256
    args.eval_loss_samples = 128
    args.min_eval_words = 3
    args.save_eval_preview = 256
    args.eos_ids = [3, 7]
    args.bos_id = 2
    return args


def manual_candidates(holdout: Sequence[dict[str, Any]], outputs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): row for row in holdout}
    per_route: Counter[str] = Counter()
    chosen: list[dict[str, Any]] = []
    for output in outputs:
        route = str(output["route"])
        if per_route[route] >= 4:
            continue
        source = by_id[str(output["id"])]
        chosen.append({
            "id": str(output["id"]),
            "route": route,
            "semantic_family": str(output["semantic_family"]),
            "question_class": str(output["question_class"]),
            "prompt": str(source["prompt"]),
            "reference_chosen": str(source["chosen"]),
            "generation": str(output["generation"]),
            "heuristic_passed": bool(output["passed"]),
            "heuristic_failures": list(output["failures"]),
            "human_label": None,
            "human_reason": None,
        })
        per_route[route] += 1
    if len(chosen) != 32:
        raise RuntimeError(f"Expected 32 manual-audit candidates, got {len(chosen)}")
    return chosen


def probe_checkpoint_state() -> dict[str, Any]:
    summary_path = PROBE_DIR / "run_summary.json"
    checkpoint_path = PROBE_DIR / "checkpoints" / "best_model.pt"
    value: dict[str, Any] = {
        "probe_dir": str(PROBE_DIR),
        "summary_exists": summary_path.is_file(),
        "best_checkpoint_file_exists": checkpoint_path.is_file(),
    }
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        value.update({
            "promoted_once": bool(summary.get("promoted_once")),
            "best_update": summary.get("best_update"),
            "best_checkpoint": summary.get("best_checkpoint"),
        })
    return value


def run() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Conversation audit requires CUDA for canonical generation evaluation")

    validate_local_files(verify_checkpoint_sha256=True)
    holdout = build_holdout_rows()
    train = build_train_rows()
    dataset_summary = validate(train, holdout)

    model, strict_load, checkpoint_meta = _load_canonical_model("cuda")
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    tok_contract = _tokenizer_contract(tok)
    if not tok_contract["passed"]:
        raise RuntimeError(f"Tokenizer contract failed: {tok_contract['mismatches']}")
    observed_special = {token: tok.token_to_id(token) for token in EXPECTED_SPECIAL_IDS}
    if observed_special != EXPECTED_SPECIAL_IDS:
        raise RuntimeError(f"Tokenizer special IDs changed: {observed_special}")

    history = _load_historical_eval_module()
    args = configure_eval_args(history)
    special = history.special_ids(tok, args)
    device = torch.device("cuda")

    modes = {
        "legacy_raw": lambda row: str(row["prompt"]),
        "training_matched": clean_eval_prompt,
        "runtime_single_turn": lambda row: runtime_chat_prompt(str(row["prompt"])),
    }
    evaluated: dict[str, Any] = {}
    raw_outputs: dict[str, list[dict[str, Any]]] = {}
    for name, formatter in modes.items():
        payload = evaluate_clean_family(
            model, tok, holdout, device, history, args, special,
            name=f"canonical_v14a2_{name}", prompt_transform=formatter,
        )
        evaluated[name] = compact_clean(payload)
        raw_outputs[name] = payload["outputs"]
        write_json(OUTPUT_DIR / f"{name}_outputs.json", payload["outputs"])

    representatives: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in holdout:
        route = str(row["route"])
        if route in PRIORITY_ROUTES and route not in seen:
            representatives.append(row)
            seen.add(route)
    if len(representatives) != len(PRIORITY_ROUTES):
        raise RuntimeError("Could not select one serialization representative per priority route")

    token_audit: list[dict[str, Any]] = []
    prior_turns = (
        ("user", "I am comparing two training methods."),
        ("assistant", "Okay. I will keep that comparison in mind."),
    )
    for row in representatives:
        raw = str(row["prompt"])
        training = clean_eval_prompt(row)
        runtime_single = runtime_chat_prompt(raw)
        runtime_multi = runtime_chat_prompt(raw, turns=prior_turns)
        training_trace = token_trace(tok, training, add_special_tokens=False)
        runtime_trace = token_trace(tok, runtime_single, add_special_tokens=True)
        runtime_no_post = token_trace(tok, runtime_single, add_special_tokens=False)
        token_audit.append({
            "id": str(row["id"]),
            "route": str(row["route"]),
            "training": training_trace,
            "runtime_single": runtime_trace,
            "runtime_single_without_postprocessor": runtime_no_post,
            "runtime_multi": token_trace(tok, runtime_multi, add_special_tokens=True),
            "runtime_postprocessor_changes_ids": runtime_trace["ids"] != runtime_no_post["ids"],
            "role_token_ids": observed_special,
        })

    candidates = manual_candidates(holdout, raw_outputs["training_matched"])
    write_json(OUTPUT_DIR / "manual_audit_candidates.json", candidates)
    write_json(OUTPUT_DIR / "serialization_token_audit.json", token_audit)

    result = {
        "schema_version": 1,
        "purpose": "no-training evaluator repair + conversation serialization audit",
        "canonical_checkpoint_meta": checkpoint_meta,
        "strict_load": strict_load,
        "tokenizer": tok_contract,
        "dataset_summary": dataset_summary,
        "probe_checkpoint_state": probe_checkpoint_state(),
        "evaluation": evaluated,
        "serialization": {
            "training_format": "v14a4 prompt text + newline-dash answer boundary; trainer tokenizer encode(add_special_tokens=False)",
            "runtime_format": "system/user/history/assistant role-token conversation; runtime tokenizer encode(composed) default add_special_tokens=True",
            "default_system": DEFAULT_SYSTEM,
            "token_audit_path": str(OUTPUT_DIR / "serialization_token_audit.json"),
        },
        "manual_audit_candidates_path": str(OUTPUT_DIR / "manual_audit_candidates.json"),
    }
    write_json(OUTPUT_DIR / "audit_summary.json", result)
    return result


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
Path("Erratum/ardor_v14a4_conversation_audit.py").write_text(audit, encoding="utf-8")


# 3) Lock a fixed-purpose RunPod runner for the no-training audit.
worker = Path("scripts/runpod_worker.py")
replace_once(
    worker,
    '''V14A4_ENTRY = REPO_ROOT / "Erratum" / "ardor_v14a4_family_balanced_trainer.py"\nV14A4_PROBE_OUTPUT =''',
    '''V14A4_ENTRY = REPO_ROOT / "Erratum" / "ardor_v14a4_family_balanced_trainer.py"\nV14A4_AUDIT_ENTRY = REPO_ROOT / "Erratum" / "ardor_v14a4_conversation_audit.py"\nV14A4_PROBE_OUTPUT =''',
)
replace_once(worker, '''    "v14a4_family_probe_u100",\n}''', '''    "v14a4_family_probe_u100",\n    "v14a4_conversation_audit",\n}''')
replace_once(worker, '''    "v14a4_family_probe_u100",\n}\n\n\ndef utc_now''', '''    "v14a4_family_probe_u100",\n    "v14a4_conversation_audit",\n}\n\n\ndef utc_now''')
replace_once(
    worker,
    '''    elif runner == "v14a4_family_probe_u100":\n        command = [''',
    '''    elif runner == "v14a4_conversation_audit":\n        command = [sys.executable, str(V14A4_AUDIT_ENTRY)]\n    elif runner == "v14a4_family_probe_u100":\n        command = [''',
)

control = Path("scripts/runpod_control.py")
replace_once(control, '''    "v14a4_family_probe_u100",\n}''', '''    "v14a4_family_probe_u100",\n    "v14a4_conversation_audit",\n}''')


# 4) Tests: exact boundary, runtime serialization shape, and fixed-purpose runner lock.
tests = r'''from __future__ import annotations

import base64
import json

import pytest

from Erratum.ardor_v14a4_conversation_audit import DEFAULT_SYSTEM, runtime_chat_prompt
from Erratum.v14a4_diagnostics import clean_eval_prompt
import scripts.runpod_worker as worker


def test_clean_eval_prompt_exactly_matches_v14a4_training_answer_boundary():
    assert clean_eval_prompt({"prompt": "What is gradient clipping?"}) == "What is gradient clipping?\\n-"
    assert clean_eval_prompt({"prompt": "What is gradient clipping?\\n-"}) == "What is gradient clipping?\\n-"
    assert clean_eval_prompt({"prompt": "What is gradient clipping?   "}) == "What is gradient clipping?\\n-"


def test_runtime_chat_prompt_matches_current_role_marked_shape():
    prompt = runtime_chat_prompt("What is dropout?")
    assert prompt == (
        f"<|system|>\\n{DEFAULT_SYSTEM}\\n<|eot|>\\n"
        "<|user|>\\nWhat is dropout?\\n<|eot|>\\n<|assistant|>\\n"
    )

    multi = runtime_chat_prompt(
        "What does it refer to?",
        turns=(("user", "Remember gradient clipping."), ("assistant", "I will.")),
    )
    assert "<|user|>\\nRemember gradient clipping.\\n<|eot|>\\n" in multi
    assert "<|assistant|>\\nI will.\\n<|eot|>\\n" in multi
    assert multi.endswith("<|user|>\\nWhat does it refer to?\\n<|eot|>\\n<|assistant|>\\n")


def _job_b64(task: dict) -> str:
    payload = {"id": "test-audit", "task": task}
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def test_conversation_audit_runner_is_fixed_purpose(monkeypatch, tmp_path):
    assert "v14a4_conversation_audit" in worker.ALLOWED_RUNNERS
    assert "v14a4_conversation_audit" in worker.FIXED_PURPOSE_RUNNERS
    monkeypatch.setattr(worker, "CONTROL_ROOT", tmp_path)
    monkeypatch.setenv("ARDOR_JOB_B64", _job_b64({"runner": "v14a4_conversation_audit", "lr": 1e-3}))
    with pytest.raises(ValueError, match="accepts no task fields beyond runner"):
        worker.run()
'''
Path("tests/test_conversation_audit.py").write_text(tests, encoding="utf-8")
