#!/usr/bin/env python3
"""v14a13: frozen 28-pair prompt/target audit. CPU only; no checkpoint loading.

Rebuild exactly the training prompts/answers used by v14a12 from the current
source builder, verify the frozen chosen-text digests, and emit a reviewable
row-level audit. Heuristic flags are triage, NEVER factual pass/fail labels.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from Erratum import build_v14a4_family_data as data

ROOT = Path(__file__).resolve().parent
V14A12 = ROOT / "ardor_v14a12_supervision_quality_eos.py"
CONTROL_DIGEST = "3171dd19ce90bd0740e2a78c2924f42c0a3c786f340114c97ab92212ba19bad6"
CORRECTED_DIGEST = "2770dd683a7a2124033a6d69ea749af63aed769905e7a13e1177234917f3edc7"
# Frozen selection order from v14a12 training_target_manifest.json on the volume.
TRAIN_IDS = (
    "0448", "0406", "0385", "0420", "0001", "0134", "0085", "0078",
    "0359", "0366", "0002", "0086", "0227", "0115", "0220", "0402",
    "0396", "0340", "0025", "0347", "0257", "0376", "0110", "0152",
    "0300", "0223", "0069", "0391",
)
IDS = tuple(f"v14a4_train_gradient_clipping_{i}" for i in TRAIN_IDS)


def load_authored_targets() -> dict[str, tuple[str, ...]]:
    """Parse the checked-in source, avoiding torch/model imports."""
    module = ast.parse(V14A12.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "TARGETS":
                return ast.literal_eval(node.value)
    raise RuntimeError("v14a12 TARGETS definition not found")


def digest(rows: list[dict]) -> str:
    payload = [{"id": str(r["id"]), "chosen": str(r["chosen"])} for r in rows]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def rebuild() -> tuple[list[dict], list[dict]]:
    full = {r["id"]: r for r in data.build_train_rows()}
    if len(IDS) != 28 or len(set(IDS)) != 28 or not all(k in full for k in IDS):
        raise RuntimeError("Frozen v14a12 selection does not match current builder")
    # v14a4 and v14a5 behavior normalization: the latter only repairs
    # dropout/overfitting_relationship, never gradient_clipping.
    original = [{
        "id": k,
        "route": full[k]["route"],
        "semantic_family": full[k]["semantic_family"],
        "prompt": full[k]["prompt"].rstrip() + "\n-",
        "chosen": full[k]["chosen"].strip(),
        "source": "behavior",
        "question_class": full[k]["question_class"],
        "prompt_style": full[k]["prompt_style"],
        "answer_style": full[k]["answer_style"],
        "collision_route": full[k].get("collision_route"),
    } for k in IDS]
    if Counter(r["semantic_family"] for r in original) != Counter({k: 4 for k in data.FAMILIES["gradient_clipping"]}):
        raise RuntimeError("Expected four training rows per gradient-clipping family")
    targets = load_authored_targets()
    assigned = {}
    for family, candidates in targets.items():
        pool = sorted((r for r in original if r["semantic_family"] == family), key=lambda r: r["id"])
        if len(pool) != 4 or len(candidates) != 4:
            raise RuntimeError(f"Bad family selection: {family}")
        for row, answer in zip(pool, candidates):
            if row["collision_route"]:
                answer += " " + data.COLLISION_DISTINCTIONS[("gradient_clipping", row["collision_route"])]
            assigned[row["id"]] = answer
    corrected = [{**r, "chosen": assigned[r["id"]]} for r in original]
    observed = (digest(original), digest(corrected))
    expected = (CONTROL_DIGEST, CORRECTED_DIGEST)
    if observed != expected:
        raise RuntimeError(f"Source/selection drift: observed {observed}, expected {expected}")
    return original, corrected


def normalized_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def flags(row: dict, other_rows: list[dict]) -> list[str]:
    result = []
    same = [r["id"] for r in other_rows if normalized_prompt(r["prompt"]) == normalized_prompt(row["prompt"])]
    if len(same) > 1:
        result.append("duplicate_prompt:" + ",".join(sorted(same)))
    if row["semantic_family"] == "learning_rate_comparison" and row["collision_route"]:
        result.append("mixed_learning_rate_and_collision_task")
    if row["question_class"] == "false_premise_correction" and not re.search(
        r"\b(no|not|does not|do not|backwards|false|incorrect|cannot|can destabilize)\b",
        row["chosen"], re.I,
    ):
        result.append("manual_check_explicit_premise_rejection")
    if row["semantic_family"] == "learning_rate_comparison" and not re.search(
        r"learning.rate", row["chosen"], re.I,
    ):
        result.append("manual_check_learning_rate_distinction")
    if row["semantic_family"] == "norm_threshold" and not re.search(
        r"(if|when|only|conditional|exceed|above)", row["chosen"], re.I,
    ):
        result.append("manual_check_conditional_norm_bound")
    return result


def build_report() -> dict:
    control, corrected = rebuild()
    cases = []
    for old, new in zip(control, corrected):
        cases.append({
            "id": old["id"], "family": old["semantic_family"],
            "question_class": old["question_class"],
            "collision_route": old["collision_route"],
            "prompt": old["prompt"],
            "original_answer": old["chosen"],
            "corrected_answer": new["chosen"],
            "original_flags": flags(old, control),
            "corrected_flags": flags(new, corrected),
            "requires_human_adjudication": True,
            "human_review_status": "pending",
        })
    c = Counter(flag.split(":")[0] for x in cases for flag in x["corrected_flags"])
    return {
        "version": "v14a13-audit-1",
        "source": "current repo's v14a4 generator + v14a12 TARGETS, frozen S3 ID list",
        "control_digest_sha256": CONTROL_DIGEST,
        "corrected_digest_sha256": CORRECTED_DIGEST,
        "case_count": len(cases),
        "unique_prompt_count": len({normalized_prompt(x["prompt"]) for x in cases}),
        "triage_flag_counts": dict(c),
        "automatic_flags_are_not_factual_scores": True,
        "cases": cases,
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# v14a13: frozen prompt-to-target audit",
        "",
        "Diagnostic only. No GPU or checkpoint loading. Factual adequacy is NOT",
        "inferred from keywords or presence/absence of heuristic flags.",
        "",
        f"Pairs: {report['case_count']}; distinct normalized prompts: {report['unique_prompt_count']}.",
        f"Flags: {json.dumps(report['triage_flag_counts'], sort_keys=True)}",
        "",
    ]
    for i, c in enumerate(report["cases"], 1):
        lines.extend([
            f"## {i:02d}. {c['id']} ({c['family']})",
            f"Question class: {c['question_class']}; collision: {c['collision_route']}",
            "",
            "**Prompt**",
            "",
            "    " + c["prompt"].replace("\n", "\n    "),
            "",
            "**Original target**",
            "",
            c["original_answer"],
            "",
            "**Corrected target**",
            "",
            c["corrected_answer"],
            "",
            "**Triage flags**: " + (", ".join(c["corrected_flags"]) or "none"),
            "",
            "**Human factual review**: pending",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("v14a13_audit_output"))
    args = parser.parse_args()
    report = build_report()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt_target_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "prompt_target_audit.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(json.dumps({
        "cases": report["case_count"],
        "unique_prompts": report["unique_prompt_count"],
        "triage": report["triage_flag_counts"],
        "sha256_verified": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
