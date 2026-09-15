#!/usr/bin/env python3
"""No-training exact-runtime serialization and evaluator-calibration audit.

This follow-up audit resolves one remaining confound in the first v14a4 conversation audit:
the historical generation helper always encodes with add_special_tokens=False, while the native
Ardor runtime calls Tokenizer.encode(composed) with its default postprocessor enabled.

The same canonical v14a2 checkpoint is therefore evaluated under four controlled contexts:
  1. v14a4 training-matched prompt, no tokenizer postprocessor;
  2. current role/primer frame, no tokenizer postprocessor;
  3. current role/primer frame with the runtime tokenizer postprocessor;
  4. the same exact runtime serialization with two preceding conversation turns.

No optimizer is constructed, no backward pass is performed, and no checkpoint is written.
The script also calibrates the current lexical evaluator against a manually adjudicated balanced
64-example set: 32 real canonical generations (human-fail) and their 32 reference answers
(human-pass), all taken from the frozen first audit artifact.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

import torch
from tokenizers import Tokenizer

from Erratum.ardor_v14a4_conversation_audit import DEFAULT_SYSTEM, runtime_chat_prompt
from Erratum.build_v14a4_family_data import PRIORITY_ROUTES, build_holdout_rows, build_train_rows, validate
from Erratum.canonical_contract_v14a2 import EXPECTED_SPECIAL_IDS, TOKENIZER_PATH, validate_local_files
from Erratum.canonical_eval_v14a2 import _load_canonical_model, _load_historical_eval_module, _tokenizer_contract
from Erratum.v14a4_diagnostics import clean_eval_prompt, family_sem_check

OUTPUT_DIR = Path("/workspace/Ardor/training/runs/v14a4_exact_runtime_audit_20260915")
FIRST_AUDIT_DIR = Path("/workspace/Ardor/training/runs/v14a4_corrected_conversation_audit")
MANUAL_CANDIDATES = FIRST_AUDIT_DIR / "manual_audit_candidates.json"
FIRST_AUDIT_SUMMARY = FIRST_AUDIT_DIR / "audit_summary.json"
EXPECTED_MANUAL_IDS = {
    f"v14a4_holdout_{route}_{index:03d}"
    for route in (
        "tokenizer", "rag", "overfitting", "direct_answer", "checkpoint",
        "dropout", "gradient_clipping", "correlation",
    )
    for index in range(4)
}
PRIOR_TURNS: tuple[tuple[str, str], ...] = (
    ("user", "I am comparing two training methods."),
    ("assistant", "Okay. I will keep that comparison in mind."),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


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


@torch.inference_mode()
def generate_encoded(
    model,
    tok: Tokenizer,
    prompt: str,
    device: torch.device,
    args,
    special: dict[str, list[int]],
    *,
    add_special_tokens: bool,
) -> dict[str, Any]:
    """Mirror the historical greedy evaluator while controlling only tokenizer postprocessing."""
    ids = list(tok.encode(str(prompt), add_special_tokens=add_special_tokens).ids)
    if not ids:
        return {
            "text": "", "token_ids": [], "context_ids": [], "empty": True,
            "first_special": False, "word_count": 0, "repetition_rate": 0.0,
        }
    context_ids = ids[-int(args.max_len):]
    ids = list(context_ids)
    generated: list[int] = []
    first_special = False
    for step in range(int(args.max_gen_tokens)):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        nxt = int(torch.argmax(history_model_logits(model, x)[0, -1].float()).item())
        if step == 0 and (nxt in special["eos_ids"] or nxt in special["bos_ids"]):
            first_special = True
        if nxt in special["eos_ids"]:
            break
        generated.append(nxt)
        ids.append(nxt)
        ids = ids[-int(args.max_len):]
    text = history_clean(history_dec(tok, generated))
    return {
        "text": text,
        "token_ids": generated,
        "context_ids": context_ids,
        "empty": not bool(generated),
        "first_special": first_special,
        "word_count": history_wc(text),
        "repetition_rate": history_repetition(text),
    }


# Bound once after the frozen historical module is loaded. Keeping these as module globals makes
# generate_encoded small and keeps its generation semantics visibly identical to the old helper.
history_model_logits: Callable[..., torch.Tensor]
history_dec: Callable[..., str]
history_clean: Callable[..., str]
history_wc: Callable[..., int]
history_repetition: Callable[..., float]


@torch.inference_mode()
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
    prompt_builder: Callable[[dict[str, Any]], str],
    add_special_tokens: bool,
) -> dict[str, Any]:
    route_total: Counter[str] = Counter()
    route_pass: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    family_pass: Counter[str] = Counter()
    class_total: Counter[str] = Counter()
    class_pass: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    repetitions: list[float] = []
    loops = 0
    empties = 0
    outputs: list[dict[str, Any]] = []

    for row in rows:
        route = str(row["route"])
        family = str(row["semantic_family"])
        family_key = f"{route}/{family}"
        qclass = str(row.get("question_class", "normal"))
        prompt = str(prompt_builder(row))
        generated = generate_encoded(
            model, tok, prompt, device, args, special,
            add_special_tokens=add_special_tokens,
        )
        historical = history.sem_check(route, generated["text"])
        family_check = family_sem_check(row, generated["text"])
        failures = list(historical["failures"])
        failures.extend(f for f in family_check["failures"] if f not in failures)
        if generated["empty"] and "empty_generation" not in failures:
            failures.append("empty_generation")
        if generated["first_special"] and "first_special" not in failures:
            failures.append("first_special")
        if generated["word_count"] < int(args.min_eval_words):
            failures.append("too_short_generation")

        passed = not failures
        route_total[route] += 1
        route_pass[route] += int(passed)
        family_total[family_key] += 1
        family_pass[family_key] += int(passed)
        class_total[qclass] += 1
        class_pass[qclass] += int(passed)
        failure_counts.update(failures)
        repetitions.append(float(generated["repetition_rate"]))
        loops += int(historical["model_loop"]["has_model_loop"])
        empties += int(generated["empty"])
        outputs.append({
            "id": str(row["id"]),
            "route": route,
            "semantic_family": family,
            "question_class": qclass,
            "prompt": prompt,
            "add_special_tokens": add_special_tokens,
            "context_ids": generated["context_ids"],
            "generation": generated["text"],
            "passed": passed,
            "failures": failures,
        })

    n = max(1, len(outputs))
    family_success = {
        key: family_pass[key] / max(1, family_total[key]) for key in sorted(family_total)
    }
    priority_values = [
        value for key, value in family_success.items()
        if key.split("/", 1)[0] in PRIORITY_ROUTES
    ]
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
        "mean_repetition_rate": sum(repetitions) / max(1, len(repetitions)),
        "route_success_rate": {
            route: route_pass[route] / max(1, route_total[route]) for route in sorted(route_total)
        },
        "family_success_rate": family_success,
        "family_macro_success_rate": mean(family_success.values()) if family_success else 0.0,
        "priority_family_macro_success_rate": mean(priority_values) if priority_values else 0.0,
        "question_class_success_rate": {
            cls: class_pass[cls] / max(1, class_total[cls]) for cls in sorted(class_total)
        },
        "failure_counts": dict(failure_counts),
        "outputs": outputs,
    }


def text_heuristic(history, row: dict[str, Any], text: str, *, min_words: int) -> dict[str, Any]:
    historical = history.sem_check(str(row["route"]), text)
    family = family_sem_check(row, text)
    failures = list(historical["failures"])
    failures.extend(f for f in family["failures"] if f not in failures)
    if history.wc(text) < min_words:
        failures.append("too_short_generation")
    return {"passed": not failures, "failures": failures}


def ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def calibrate_evaluator(history, args) -> dict[str, Any]:
    if not MANUAL_CANDIDATES.is_file():
        raise FileNotFoundError(MANUAL_CANDIDATES)
    candidates = json.loads(MANUAL_CANDIDATES.read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or len(candidates) != 32:
        raise RuntimeError(f"Expected frozen 32-row manual audit, got {type(candidates)} len={len(candidates) if isinstance(candidates, list) else None}")
    ids = {str(row["id"]) for row in candidates}
    if ids != EXPECTED_MANUAL_IDS:
        raise RuntimeError(f"Frozen manual candidate IDs changed: missing={sorted(EXPECTED_MANUAL_IDS-ids)} extra={sorted(ids-EXPECTED_MANUAL_IDS)}")

    # Manual adjudication performed 2026-09-15 against the actual text in this frozen artifact:
    # all 32 model generations fail to answer their prompts correctly; all 32 reference answers
    # are semantically valid answers. The latter were individually reviewed rather than assumed
    # positive merely because they came from the dataset builder.
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for source, text, human_passed in (
            ("actual_generation", str(candidate["generation"]), False),
            ("reference_answer", str(candidate["reference_chosen"]), True),
        ):
            predicted = text_heuristic(history, candidate, text, min_words=int(args.min_eval_words))
            rows.append({
                "id": str(candidate["id"]),
                "route": str(candidate["route"]),
                "semantic_family": str(candidate["semantic_family"]),
                "source": source,
                "text": text,
                "human_passed": human_passed,
                "heuristic_passed": bool(predicted["passed"]),
                "heuristic_failures": predicted["failures"],
            })

    tp = sum(r["human_passed"] and r["heuristic_passed"] for r in rows)
    fp = sum((not r["human_passed"]) and r["heuristic_passed"] for r in rows)
    tn = sum((not r["human_passed"]) and (not r["heuristic_passed"]) for r in rows)
    fn = sum(r["human_passed"] and (not r["heuristic_passed"]) for r in rows)
    metrics = {
        "rows": len(rows),
        "human_positive": tp + fn,
        "human_negative": tn + fp,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": ratio(tp + tn, len(rows)),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
    }
    return {
        "manual_label_basis": (
            "2026-09-15 manual adjudication of the frozen 32 actual generations and their 32 "
            "reference answers: actual generations human-fail; references human-pass"
        ),
        "metrics": metrics,
        "rows": rows,
    }


def compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "outputs"}


def token_trace(tok: Tokenizer, text: str, *, add_special_tokens: bool) -> dict[str, Any]:
    enc = tok.encode(text, add_special_tokens=add_special_tokens)
    return {
        "text": text,
        "add_special_tokens": add_special_tokens,
        "ids": list(enc.ids),
        "tokens": [tok.id_to_token(i) for i in enc.ids],
        "length": len(enc.ids),
    }


def run() -> dict[str, Any]:
    global history_model_logits, history_dec, history_clean, history_wc, history_repetition

    if not torch.cuda.is_available():
        raise RuntimeError("Exact-runtime audit requires CUDA generation evaluation")
    validate_local_files(verify_checkpoint_sha256=True)
    train = build_train_rows()
    holdout = build_holdout_rows()
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
    history_model_logits = history.model_logits
    history_dec = history.dec
    history_clean = history.clean
    history_wc = history.wc
    history_repetition = history.repetition_rate
    device = torch.device("cuda")

    modes = (
        (
            "training_matched_no_specials",
            lambda row: clean_eval_prompt(row),
            False,
        ),
        (
            "runtime_role_no_specials",
            lambda row: runtime_chat_prompt(str(row["prompt"])),
            False,
        ),
        (
            "runtime_exact_specials",
            lambda row: runtime_chat_prompt(str(row["prompt"])),
            True,
        ),
        (
            "runtime_exact_two_prior_turns",
            lambda row: runtime_chat_prompt(str(row["prompt"]), turns=PRIOR_TURNS),
            True,
        ),
    )
    evaluation: dict[str, Any] = {}
    for name, builder, add_special_tokens in modes:
        payload = evaluate_mode(
            model, tok, holdout, device, history, args, special,
            name=f"canonical_v14a2_{name}",
            prompt_builder=builder,
            add_special_tokens=add_special_tokens,
        )
        evaluation[name] = compact(payload)
        write_json(OUTPUT_DIR / f"{name}_outputs.json", payload["outputs"])

    representatives: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in holdout:
        route = str(row["route"])
        if route not in seen:
            representatives.append(row)
            seen.add(route)
    token_audit: list[dict[str, Any]] = []
    for row in representatives:
        training = clean_eval_prompt(row)
        runtime_single = runtime_chat_prompt(str(row["prompt"]))
        runtime_multi = runtime_chat_prompt(str(row["prompt"]), turns=PRIOR_TURNS)
        token_audit.append({
            "id": str(row["id"]),
            "route": str(row["route"]),
            "training_no_specials": token_trace(tok, training, add_special_tokens=False),
            "runtime_role_no_specials": token_trace(tok, runtime_single, add_special_tokens=False),
            "runtime_exact_specials": token_trace(tok, runtime_single, add_special_tokens=True),
            "runtime_exact_two_prior_turns": token_trace(tok, runtime_multi, add_special_tokens=True),
        })
    write_json(OUTPUT_DIR / "exact_runtime_token_audit.json", token_audit)

    calibration = calibrate_evaluator(history, args)
    write_json(OUTPUT_DIR / "evaluator_calibration.json", calibration)

    prior_summary = None
    if FIRST_AUDIT_SUMMARY.is_file():
        prior_summary = json.loads(FIRST_AUDIT_SUMMARY.read_text(encoding="utf-8"))

    result = {
        "schema_version": 1,
        "purpose": "no-training exact runtime serialization ablation + balanced evaluator calibration",
        "strict_load": strict_load,
        "checkpoint_meta": checkpoint_meta,
        "tokenizer": tok_contract,
        "dataset_summary": dataset_summary,
        "evaluation": evaluation,
        "evaluator_calibration": calibration["metrics"],
        "manual_label_basis": calibration["manual_label_basis"],
        "serialization": {
            "training": "v14a4 prompt + newline-dash; encode(add_special_tokens=False)",
            "role_frame_only": "current system/user/assistant role frame + ROLE_PRIMER; encode(add_special_tokens=False)",
            "exact_runtime": "same role frame + ROLE_PRIMER; Tokenizer.encode default postprocessor enabled",
            "exact_runtime_multi": "exact runtime plus two prior user/assistant turns",
            "default_system": DEFAULT_SYSTEM,
            "token_audit_path": str(OUTPUT_DIR / "exact_runtime_token_audit.json"),
        },
        "first_audit_summary_loaded": prior_summary is not None,
        "first_audit_evaluation": prior_summary.get("evaluation") if isinstance(prior_summary, dict) else None,
        "output_dir": str(OUTPUT_DIR),
    }
    write_json(OUTPUT_DIR / "audit_summary.json", result)
    return result


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
