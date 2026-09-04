#!/usr/bin/env python3
"""Read-only audit of the exact v14 route-supervision datasets on the RunPod volume."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any

from scripts.runpod_control import bucket, s3_client

DATASETS = {
    "holdout": "Ardor/training/data/v14_v3/dataset_v3_holdout_audit.jsonl",
    "balanced": "Ardor/training/data/v14_v3/dataset_v3b_route_contrastive_balanced.jsonl",
    "targeted_collisions": "Ardor/training/data/v14_v3/dataset_v14b3_targeted_route_collisions.jsonl",
    "local_margin": "Ardor/training/data/v14_v3/dataset_v14a2_local_anti_loop_margin.jsonl",
}
ROUTES = [
    "tokenizer", "rag", "overfitting", "direct_answer",
    "checkpoint", "dropout", "gradient_clipping", "correlation",
]
SIGNALS = {
    "correlation": ["correlation", "variables", "relationship", "association", "causation", "positive", "negative"],
    "direct_answer": ["answer", "main", "first", "direct", "point", "clear", "question"],
    "tokenizer": ["token", "ids", "vocabulary", "text", "split", "encode"],
    "rag": ["retriev", "context", "document", "external", "evidence", "ground"],
    "overfitting": ["overfit", "memor", "training data", "generaliz", "new data", "validation", "unseen"],
    "checkpoint": ["checkpoint", "save", "saved", "state", "weights", "resume", "best", "final"],
    "dropout": ["dropout", "mask", "random", "disable", "regular", "activation"],
    "gradient_clipping": ["gradient", "clip", "norm", "large", "explode", "update", "stabil"],
}


def norm_text(value: Any) -> str:
    s = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return s.removesuffix(" -").removesuffix("-").strip()


def parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return [str(value)]


def read_jsonl(key: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obj = s3_client().get_object(Bucket=bucket(), Key=key)
    h = hashlib.sha256()
    raw = bytearray()
    while True:
        chunk = obj["Body"].read(8 * 1024 * 1024)
        if not chunk:
            break
        h.update(chunk)
        raw.extend(chunk)
    rows = []
    for i, line in enumerate(bytes(raw).decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected object at {key}:{i}")
        rows.append(value)
    return rows, {"path": key, "bytes": len(raw), "sha256": h.hexdigest(), "rows": len(rows)}


def quantiles(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "p90": None, "max": None}
    vals = sorted(values)
    p90 = vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))]
    return {"min": vals[0], "median": median(vals), "mean": mean(vals), "p90": p90, "max": vals[-1]}


def route_of(row: dict[str, Any]) -> str:
    return str(row.get("route") or row.get("positive_group") or "unknown")


def anchor_of(row: dict[str, Any]) -> str:
    return str(row.get("anchor_context") or row.get("prompt") or row.get("text") or "").strip()


def chosen_of(row: dict[str, Any]) -> str:
    return str(row.get("chosen") or row.get("answer") or row.get("response") or "").strip()


def semantic_signal(route: str, text: str) -> bool:
    low = norm_text(text)
    return any(x in low for x in SIGNALS.get(route, []))


def summarize(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    keys = Counter()
    routes = Counter()
    row_types = Counter()
    hard_negs = Counter()
    anchor_lengths: list[int] = []
    chosen_lengths: list[int] = []
    chosen_present = 0
    chosen_signal = Counter()
    chosen_total = Counter()
    unique_anchors: set[str] = set()
    unique_chosen: set[str] = set()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_field_counts = Counter()

    for row in rows:
        keys.update(row.keys())
        route = route_of(row)
        routes[route] += 1
        row_types[str(row.get("row_type", ""))] += 1
        anchor = anchor_of(row)
        chosen = chosen_of(row)
        if anchor:
            unique_anchors.add(norm_text(anchor))
            anchor_lengths.append(len(anchor.split()))
        if chosen:
            chosen_present += 1
            unique_chosen.add(norm_text(chosen))
            chosen_lengths.append(len(chosen.split()))
            chosen_total[route] += 1
            chosen_signal[route] += int(semantic_signal(route, chosen))
        for hn in parse_list(row.get("hard_negatives")):
            hard_negs[f"{route}->{hn}"] += 1
        for key in row:
            low = key.lower()
            if "source" in low or "holdout" in low or "split" in low:
                if row.get(key) not in (None, "", [], {}):
                    source_field_counts[key] += 1
        if route in ROUTES and len(examples[route]) < 2:
            examples[route].append({
                "id": str(row.get("id", "")),
                "row_type": str(row.get("row_type", "")),
                "anchor": anchor[:400],
                "chosen": chosen[:300],
                "hard_negatives": parse_list(row.get("hard_negatives")),
                "source_fields": {k: row[k] for k in row if ("source" in k.lower() or "holdout" in k.lower() or "split" in k.lower()) and row.get(k) not in (None, "", [], {})},
            })

    return {
        "name": name,
        "top_level_key_counts": dict(keys.most_common()),
        "route_counts": dict(routes),
        "row_type_counts": dict(row_types),
        "rows_with_chosen": chosen_present,
        "anchor_word_lengths": quantiles(anchor_lengths),
        "chosen_word_lengths": quantiles(chosen_lengths),
        "unique_normalized_anchors": len(unique_anchors),
        "unique_normalized_chosen": len(unique_chosen),
        "chosen_semantic_signal_rate_by_route": {
            r: chosen_signal[r] / max(1, chosen_total[r]) for r in sorted(chosen_total)
        },
        "hard_negative_edges": dict(hard_negs.most_common(50)),
        "source_field_counts": dict(source_field_counts),
        "examples": dict(examples),
    }


def overlap(train: list[dict[str, Any]], holdout: list[dict[str, Any]], name: str) -> dict[str, Any]:
    holdout_anchor = {norm_text(anchor_of(r)) for r in holdout if anchor_of(r)}
    holdout_text = {norm_text(r.get("text", "")) for r in holdout if r.get("text")}
    holdout_ids = {str(r.get("id")) for r in holdout if r.get("id") is not None}

    exact_anchor = []
    normalized_text = []
    id_overlap = []
    source_holdout_refs = []
    source_id_hits = []
    for row in train:
        rid = str(row.get("id", ""))
        a = norm_text(anchor_of(row))
        if a and a in holdout_anchor:
            exact_anchor.append(rid)
        if a and a in holdout_text:
            normalized_text.append(rid)
        if rid and rid in holdout_ids:
            id_overlap.append(rid)
        for key, val in row.items():
            low = key.lower()
            if "holdout" in low and val not in (None, "", [], {}):
                source_holdout_refs.append({"id": rid, "field": key, "value": val})
            if ("source" in low or low.endswith("_id")) and str(val) in holdout_ids:
                source_id_hits.append({"id": rid, "field": key, "value": val})
    return {
        "dataset": name,
        "exact_normalized_anchor_overlap_count": len(exact_anchor),
        "anchor_vs_holdout_text_overlap_count": len(normalized_text),
        "id_overlap_count": len(id_overlap),
        "rows_with_holdout_named_source_fields": len(source_holdout_refs),
        "source_id_matches_holdout_id_count": len(source_id_hits),
        "examples": {
            "exact_anchor": exact_anchor[:20],
            "text_overlap": normalized_text[:20],
            "id_overlap": id_overlap[:20],
            "holdout_source_refs": source_holdout_refs[:20],
            "source_id_hits": source_id_hits[:20],
        },
    }


def main() -> None:
    all_rows: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, Any] = {}
    for name, key in DATASETS.items():
        rows, meta = read_jsonl(key)
        all_rows[name] = rows
        metadata[name] = meta

    result = {
        "schema_version": 1,
        "purpose": "Audit supervision signal and holdout leakage before behavior-first route training",
        "datasets": metadata,
        "summaries": {name: summarize(rows, name) for name, rows in all_rows.items()},
        "holdout_overlap": {
            "balanced": overlap(all_rows["balanced"], all_rows["holdout"], "balanced"),
            "targeted_collisions": overlap(all_rows["targeted_collisions"], all_rows["holdout"], "targeted_collisions"),
        },
    }
    out = Path("route_data_audit.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
