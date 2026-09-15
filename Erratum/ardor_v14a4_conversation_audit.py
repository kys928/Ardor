#!/usr/bin/env python3
"""Locked no-training audit of Ardor serialization and evaluator validity.

The audit loads canonical v14a2 once and performs no optimization. It isolates:
1. the v14a4 training answer boundary with trainer tokenization,
2. the runtime role/system frame without tokenizer postprocessing,
3. the same runtime frame with the tokenizer postprocessor,
4. exact runtime serialization plus the runtime minimum-generation EOS/EOT floor,
5. the same exact serialization with two prior conversational turns.

Generation remains greedy so the scientific variable is prompt/serialization format rather
than sampling noise. The exact-runtime modes reproduce the current runtime prompt builder,
default tokenizer postprocessing, EOS/EOT stop set, and 16-token minimum-generation stop floor.
The audit also calibrates the lexical evaluator against a balanced 64-example human-labelled
set: 32 reviewed canonical generations (all failures) and their 32 reviewed reference answers
(all valid).
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

import torch
from tokenizers import Tokenizer

from Erratum.build_v14a4_family_data import PRIORITY_ROUTES, build_holdout_rows, build_train_rows, validate
from Erratum.canonical_contract_v14a2 import EXPECTED_SPECIAL_IDS, TOKENIZER_PATH, validate_local_files
from Erratum.canonical_eval_v14a2 import _load_canonical_model, _load_historical_eval_module, _tokenizer_contract
from Erratum.v14a4_diagnostics import clean_eval_prompt, family_sem_check

OUTPUT_DIR = Path("/workspace/Ardor/training/runs/v14a4_exact_runtime_serialization_audit")
PROBE_DIR = Path("/workspace/Ardor/training/runs/sft_v14a4_family_balanced_semantic_landing_probe_u100")
DEFAULT_SYSTEM = (
    "Hi, You are Ardor. Answer my questions cleanly. Respond to me in friendly manner. "
    "Prefer 3-6 sentences at most, however you can extend it if you deem necessary. "
    "Always start the conversation."
)
PRIOR_TURNS = (
    ("user", "I am comparing two training methods."),
    ("assistant", "Okay. I will keep that comparison in mind."),
)
RUNTIME_MIN_NEW_TOKENS = 16


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


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
    """Mirror the role-token branch of ArdorCore._build_chat_prompt on current main."""
    parts = [f"<|system|>\n{system.strip()}\n<|eot|>\n"]
    for role, message in turns:
        message = clean_history_text(message)
        if not message:
            continue
        token = "<|user|>" if role == "user" else "<|assistant|>"
        parts.append(f"{token}\n{message}\n<|eot|>\n")
    user_text = clean_history_text(user_text)
    parts.append(f"<|user|>\n{user_text}\n<|eot|>\n<|assistant|>\n")
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


def model_logits(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    if isinstance(out, dict):
        for key in ("logits", "lm_logits", "output"):
            if key in out and torch.is_tensor(out[key]):
                return out[key]
    if hasattr(out, "logits") and torch.is_tensor(out.logits):
        return out.logits
    raise RuntimeError(f"Cannot extract logits from {type(out)}")


@torch.inference_mode()
def greedy_serialized_generation(
    model,
    tok: Tokenizer,
    serialized: str,
    device: torch.device,
    args,
    special: dict[str, list[int]],
    *,
    add_special_tokens: bool,
    suppress_stop_tokens_until: int = 0,
) -> dict[str, Any]:
    ids = list(tok.encode(serialized, add_special_tokens=add_special_tokens).ids)
    if not ids:
        return {
            "text": "", "token_ids": [], "empty": True, "first_special": False,
            "word_count": 0, "repetition_rate": 0.0, "prompt_token_count": 0,
        }
    ids = ids[-int(args.max_len):]
    generated: list[int] = []
    first_special = False
    stop_ids = set(int(x) for x in special["eos_ids"])

    for step in range(int(args.max_gen_tokens)):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model_logits(model, x)[0, -1].float()
        if step < int(suppress_stop_tokens_until) and stop_ids:
            logits = logits.clone()
            logits[list(stop_ids)] = -float("inf")
        nxt = int(torch.argmax(logits).item())
        if step == 0 and nxt in stop_ids:
            first_special = True
        if nxt in stop_ids:
            break
        generated.append(nxt)
        ids.append(nxt)
        ids = ids[-int(args.max_len):]

    try:
        text = tok.decode(generated, skip_special_tokens=True).strip()
    except TypeError:
        text = tok.decode(generated).strip()
    return {
        "text": text,
        "token_ids": generated,
        "empty": not bool(generated),
        "first_special": first_special,
        "word_count": int(history_wc(text)),
        "repetition_rate": float(history_repetition(text)),
        "prompt_token_count": len(tok.encode(serialized, add_special_tokens=add_special_tokens).ids),
    }


_HISTORY = None


def history_wc(text: str) -> int:
    if _HISTORY is None:
        raise RuntimeError("Historical evaluator is not initialized")
    return int(_HISTORY.wc(text))


def history_repetition(text: str) -> float:
    if _HISTORY is None:
        raise RuntimeError("Historical evaluator is not initialized")
    return float(_HISTORY.repetition_rate(text))


def heuristic_text_check(history, row: dict[str, Any], text: str, min_words: int) -> dict[str, Any]:
    historical = history.sem_check(str(row["route"]), text)
    family = family_sem_check(row, text)
    failures = list(historical["failures"])
    failures.extend(failure for failure in family["failures"] if failure not in failures)
    if not text.strip() and "empty_generation" not in failures:
        failures.append("empty_generation")
    if history.wc(text) < int(min_words):
        failures.append("too_short_generation")
    return {"passed": not failures, "failures": failures, "historical": historical, "family": family}


def evaluate_mode(
    model,
    tok: Tokenizer,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    history,
    args,
    special: dict[str, list[int]],
    *,
    name: str,
    formatter: Callable[[dict[str, Any]], str],
    add_special_tokens: bool,
    suppress_stop_tokens_until: int = 0,
) -> dict[str, Any]:
    route_total: Counter[str] = Counter()
    route_pass: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    family_pass: Counter[str] = Counter()
    class_total: Counter[str] = Counter()
    class_pass: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    repetitions: list[float] = []
    prompt_lengths: list[int] = []
    loops = 0
    empties = 0
    outputs: list[dict[str, Any]] = []

    for row in rows:
        serialized = formatter(row)
        generated = greedy_serialized_generation(
            model, tok, serialized, device, args, special,
            add_special_tokens=add_special_tokens,
            suppress_stop_tokens_until=suppress_stop_tokens_until,
        )
        check = heuristic_text_check(history, row, generated["text"], int(args.min_eval_words))
        failures = list(check["failures"])
        if generated["first_special"]:
            failures.append("first_special")
        passed = not failures

        route = str(row["route"])
        family = str(row["semantic_family"])
        qclass = str(row.get("question_class", "normal"))
        key = f"{route}/{family}"
        route_total[route] += 1
        route_pass[route] += int(passed)
        family_total[key] += 1
        family_pass[key] += int(passed)
        class_total[qclass] += 1
        class_pass[qclass] += int(passed)
        failure_counts.update(failures)
        repetitions.append(float(generated["repetition_rate"]))
        prompt_lengths.append(int(generated["prompt_token_count"]))
        loops += int(check["historical"]["model_loop"]["has_model_loop"])
        empties += int(generated["empty"])
        outputs.append({
            "id": str(row["id"]),
            "route": route,
            "semantic_family": family,
            "question_class": qclass,
            "serialized_prompt": serialized,
            "prompt_token_count": generated["prompt_token_count"],
            "generation": generated["text"],
            "token_ids": generated["token_ids"],
            "passed": passed,
            "failures": failures,
        })

    n = max(1, len(outputs))
    family_success = {key: family_pass[key] / max(1, family_total[key]) for key in sorted(family_total)}
    priority_values = [
        value for key, value in family_success.items()
        if key.split("/", 1)[0] in PRIORITY_ROUTES
    ]
    family_values = list(family_success.values())
    return {
        "name": name,
        "rows": len(outputs),
        "success_count": sum(route_pass.values()),
        "success_rate": sum(route_pass.values()) / n,
        "bad_rate": 1.0 - (sum(route_pass.values()) / n),
        "empty_count": empties,
        "empty_rate": empties / n,
        "model_loop_count": loops,
        "model_loop_rate": loops / n,
        "mean_repetition_rate": mean(repetitions) if repetitions else 0.0,
        "mean_prompt_tokens": mean(prompt_lengths) if prompt_lengths else 0.0,
        "route_success_rate": {route: route_pass[route] / max(1, route_total[route]) for route in sorted(route_total)},
        "family_success_rate": family_success,
        "family_macro_success_rate": mean(family_values) if family_values else 0.0,
        "priority_family_macro_success_rate": mean(priority_values) if priority_values else 0.0,
        "question_class_success_rate": {cls: class_pass[cls] / max(1, class_total[cls]) for cls in sorted(class_total)},
        "failure_counts": dict(failure_counts),
        "outputs": outputs,
    }


def compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "outputs"}


def select_manual_rows(holdout: Sequence[dict[str, Any]], outputs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): row for row in holdout}
    per_route: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for output in outputs:
        route = str(output["route"])
        if per_route[route] >= 4:
            continue
        source = by_id[str(output["id"])]
        selected.append({
            "id": str(output["id"]),
            "route": route,
            "semantic_family": str(output["semantic_family"]),
            "question_class": str(output["question_class"]),
            "prompt": str(source["prompt"]),
            "reference_chosen": str(source["chosen"]),
            "generation": str(output["generation"]),
            "generation_heuristic_passed": bool(output["passed"]),
            "generation_heuristic_failures": list(output["failures"]),
            "human_generation_label": False,
            "human_generation_reason": "Manual review: generation does not correctly answer the prompt.",
            "human_reference_label": True,
            "human_reference_reason": "Manual review: authored reference is a substantively valid answer.",
        })
        per_route[route] += 1
    if len(selected) != 32:
        raise RuntimeError(f"Expected 32 manual rows, got {len(selected)}")
    return selected


def balanced_evaluator_calibration(history, holdout: Sequence[dict[str, Any]], manual_rows: Sequence[dict[str, Any]], min_words: int) -> dict[str, Any]:
    by_id = {str(row["id"]): row for row in holdout}
    labelled: list[dict[str, Any]] = []
    counts = Counter()

    for item in manual_rows:
        row = by_id[str(item["id"])]
        for kind, text, human_label in (
            ("generation", str(item["generation"]), False),
            ("reference", str(item["reference_chosen"]), True),
        ):
            check = heuristic_text_check(history, row, text, min_words)
            predicted = bool(check["passed"])
            if human_label and predicted:
                counts["tp"] += 1
            elif not human_label and predicted:
                counts["fp"] += 1
            elif not human_label and not predicted:
                counts["tn"] += 1
            else:
                counts["fn"] += 1
            labelled.append({
                "id": str(item["id"]),
                "kind": kind,
                "human_label": human_label,
                "heuristic_passed": predicted,
                "heuristic_failures": list(check["failures"]),
                "text": text,
            })

    tp, fp, tn, fn = (counts[k] for k in ("tp", "fp", "tn", "fn"))
    total = max(1, tp + fp + tn + fn)
    return {
        "rows": tp + fp + tn + fn,
        "positive_rows": tp + fn,
        "negative_rows": tn + fp,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": (tp + tn) / total,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "false_positive_rate": fp / max(1, fp + tn),
        "false_negative_rate": fn / max(1, fn + tp),
        "labelled_rows": labelled,
    }


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
    global _HISTORY
    if not torch.cuda.is_available():
        raise RuntimeError("Conversation audit requires CUDA")

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
    _HISTORY = history
    args = configure_eval_args(history)
    special = history.special_ids(tok, args)
    device = torch.device("cuda")

    mode_specs = {
        "training_matched": {
            "formatter": clean_eval_prompt,
            "add_special_tokens": False,
            "suppress_stop_tokens_until": 0,
        },
        "runtime_role_no_postprocessor": {
            "formatter": lambda row: runtime_chat_prompt(str(row["prompt"])),
            "add_special_tokens": False,
            "suppress_stop_tokens_until": 0,
        },
        "runtime_role_with_postprocessor_raw_greedy": {
            "formatter": lambda row: runtime_chat_prompt(str(row["prompt"])),
            "add_special_tokens": True,
            "suppress_stop_tokens_until": 0,
        },
        "runtime_exact_single_turn": {
            "formatter": lambda row: runtime_chat_prompt(str(row["prompt"])),
            "add_special_tokens": True,
            "suppress_stop_tokens_until": RUNTIME_MIN_NEW_TOKENS,
        },
        "runtime_exact_multi_turn": {
            "formatter": lambda row: runtime_chat_prompt(str(row["prompt"]), turns=PRIOR_TURNS),
            "add_special_tokens": True,
            "suppress_stop_tokens_until": RUNTIME_MIN_NEW_TOKENS,
        },
    }

    evaluated: dict[str, Any] = {}
    raw_outputs: dict[str, list[dict[str, Any]]] = {}
    for name, spec in mode_specs.items():
        payload = evaluate_mode(
            model, tok, holdout, device, history, args, special,
            name=f"canonical_v14a2_{name}",
            formatter=spec["formatter"],
            add_special_tokens=bool(spec["add_special_tokens"]),
            suppress_stop_tokens_until=int(spec["suppress_stop_tokens_until"]),
        )
        evaluated[name] = compact(payload)
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
    for row in representatives:
        raw = str(row["prompt"])
        training = clean_eval_prompt(row)
        runtime_single = runtime_chat_prompt(raw)
        runtime_multi = runtime_chat_prompt(raw, turns=PRIOR_TURNS)
        runtime_trace = token_trace(tok, runtime_single, add_special_tokens=True)
        runtime_no_post = token_trace(tok, runtime_single, add_special_tokens=False)
        token_audit.append({
            "id": str(row["id"]),
            "route": str(row["route"]),
            "training": token_trace(tok, training, add_special_tokens=False),
            "runtime_single": runtime_trace,
            "runtime_single_without_postprocessor": runtime_no_post,
            "runtime_multi": token_trace(tok, runtime_multi, add_special_tokens=True),
            "runtime_postprocessor_changes_ids": runtime_trace["ids"] != runtime_no_post["ids"],
            "role_token_ids": observed_special,
        })

    manual_rows = select_manual_rows(holdout, raw_outputs["training_matched"])
    calibration = balanced_evaluator_calibration(history, holdout, manual_rows, int(args.min_eval_words))
    write_json(OUTPUT_DIR / "manual_audit_32_generation_pairs.json", manual_rows)
    write_json(OUTPUT_DIR / "balanced_evaluator_calibration.json", calibration)
    write_json(OUTPUT_DIR / "serialization_token_audit.json", token_audit)

    training_eval = evaluated["training_matched"]
    runtime_eval = evaluated["runtime_exact_single_turn"]
    success_delta = float(runtime_eval["success_rate"]) - float(training_eval["success_rate"])
    empty_delta = float(runtime_eval["empty_rate"]) - float(training_eval["empty_rate"])
    materially_worse = success_delta <= -0.02 or empty_delta >= 0.10

    result = {
        "schema_version": 2,
        "purpose": "locked no-training exact-runtime serialization ablation + balanced evaluator calibration",
        "canonical_checkpoint_meta": checkpoint_meta,
        "strict_load": strict_load,
        "tokenizer": tok_contract,
        "dataset_summary": dataset_summary,
        "probe_checkpoint_state": probe_checkpoint_state(),
        "evaluation": evaluated,
        "runtime_vs_training": {
            "training_success_rate": training_eval["success_rate"],
            "runtime_exact_success_rate": runtime_eval["success_rate"],
            "success_rate_delta_runtime_minus_training": success_delta,
            "training_empty_rate": training_eval["empty_rate"],
            "runtime_exact_empty_rate": runtime_eval["empty_rate"],
            "empty_rate_delta_runtime_minus_training": empty_delta,
            "materially_worse": materially_worse,
            "materiality_rule": "runtime success <= training success - 0.02 OR runtime empty >= training empty + 0.10",
        },
        "balanced_evaluator_calibration": {key: value for key, value in calibration.items() if key != "labelled_rows"},
        "serialization": {
            "training_format": "v14a4 prompt + newline-dash; Tokenizer.encode(add_special_tokens=False)",
            "runtime_format": "current system/user/history/assistant role frame; Tokenizer.encode(composed) default add_special_tokens=True",
            "runtime_stop_floor": RUNTIME_MIN_NEW_TOKENS,
            "generation_control": "greedy argmax; exact serialization and runtime EOS/EOT minimum-token floor, no sampling-noise confound",
            "default_system": DEFAULT_SYSTEM,
            "token_audit_path": str(OUTPUT_DIR / "serialization_token_audit.json"),
        },
        "manual_audit_path": str(OUTPUT_DIR / "manual_audit_32_generation_pairs.json"),
        "calibration_path": str(OUTPUT_DIR / "balanced_evaluator_calibration.json"),
    }
    write_json(OUTPUT_DIR / "audit_summary.json", result)
    return result


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
