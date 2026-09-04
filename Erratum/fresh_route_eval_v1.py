#!/usr/bin/env python3
"""Deterministic, evaluation-only route prompts for post-v14a2 promotion.

This module deliberately contains prompts only, no chosen continuations. Training code
must not import or consume this dataset. Evaluators should call assert_no_exact_overlap
against every training dataset before scoring it.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

ROUTES = [
    "tokenizer", "rag", "overfitting", "direct_answer",
    "checkpoint", "dropout", "gradient_clipping", "correlation",
]

FRAMES = [
    "Explain {topic} to a new machine-learning engineer.",
    "A teammate asks about {topic}. What is the key mechanism?",
    "In one concise paragraph, describe {topic}.",
    "What concept is responsible for {topic}, and what does it do?",
]

TOPICS = {
    "tokenizer": [
        "turning ordinary text into the integer symbols a language model consumes",
        "mapping a sentence through a fixed vocabulary before model computation",
        "breaking text into subword-sized units before transformer layers process it",
        "converting model-readable IDs back and forth from human text",
        "vocabulary lookup at the boundary between text and a language model",
        "handling pieces of words before they become neural-network inputs",
        "text segmentation and integer-ID mapping as one preprocessing mechanism",
        "why raw written text needs an encoding step before the decoder sees it",
    ],
    "rag": [
        "using freshly retrieved documents to support an answer before generation",
        "adding search results to model context at answer time",
        "grounding a generated response in external evidence fetched for the query",
        "bringing relevant passages into the prompt before the model writes its answer",
        "combining document retrieval with language generation for factual support",
        "answering from query-specific external context rather than only stored parameters",
        "fetching useful source material first and then generating from that material",
        "augmenting a model response with evidence retrieved at inference time",
    ],
    "overfitting": [
        "training performance improving while results on unseen examples get worse",
        "a model learning the training examples too specifically to generalize well",
        "a widening gap between fit on training data and performance on new data",
        "memorizing accidental training patterns instead of learning reusable structure",
        "excellent training scores paired with weak validation behavior",
        "why excessive fit to seen examples can damage performance on new inputs",
        "a model depending on quirks of its training set rather than robust patterns",
        "poor generalization caused by learning the training sample too closely",
    ],
    "direct_answer": [
        "responding with the requested result before giving background explanation",
        "keeping the main conclusion at the front of a response",
        "answering a simple question without burying the result in setup",
        "putting the useful conclusion before optional context",
        "making the first part of a response immediately resolve the user's question",
        "giving the main point first and adding only necessary supporting detail",
        "avoiding a long preamble when the user asked for a straightforward result",
        "structuring a response so the requested answer is immediately visible",
    ],
    "checkpoint": [
        "capturing model parameters and optimizer state during a long training run",
        "preserving a recoverable training state before continuing optimization",
        "saving a model state so training can resume after interruption",
        "keeping a particular trained state for later evaluation or restoration",
        "recording model weights at a known point in an experiment",
        "retaining the best observed training state instead of only the final one",
        "making a durable resume point during model optimization",
        "storing enough training state to reproduce or continue a run later",
    ],
    "dropout": [
        "randomly disabling some activations during training but not normal inference",
        "injecting temporary activation masking as a neural-network regularizer",
        "training with randomly omitted hidden units to reduce co-adaptation",
        "using random activation removal to improve generalization",
        "why some neural activations are intentionally zeroed on training steps",
        "a regularization method that samples a different active subnetwork during training",
        "stochastic masking of hidden activations as part of model training",
        "temporarily turning off parts of a network during optimization to resist over-reliance",
    ],
    "gradient_clipping": [
        "limiting gradient norm before the optimizer applies an update",
        "preventing an unusually large gradient from causing a destructive training step",
        "capping backpropagated update signals when their magnitude becomes extreme",
        "stabilizing optimization by bounding gradient size",
        "controlling exploding gradients immediately before parameter updates",
        "rescaling a gradient when its norm exceeds a chosen threshold",
        "keeping optimizer steps stable when backpropagation produces very large gradients",
        "placing a maximum effective size on gradients during neural-network training",
    ],
    "correlation": [
        "two measured variables changing together without establishing that one causes the other",
        "a statistical association between variables that may be positive or negative",
        "quantifying whether two quantities tend to vary together",
        "distinguishing an observed relationship between variables from causal evidence",
        "describing co-variation between measurements without claiming a mechanism",
        "a relationship where changes in one variable are associated with changes in another",
        "why statistical association alone is insufficient to prove causation",
        "measuring the direction and strength of how two variables move together",
    ],
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip()


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in ROUTES:
        topics = TOPICS[route]
        if len(topics) != 8:
            raise RuntimeError(f"Expected eight fresh topics for {route}, got {len(topics)}")
        for topic_idx, topic in enumerate(topics):
            for frame_idx, frame in enumerate(FRAMES):
                prompt = frame.format(topic=topic)
                rows.append({
                    "id": f"fresh_route_eval_v1_{route}_{topic_idx:02d}_{frame_idx:02d}",
                    "route": route,
                    "prompt": prompt,
                    "split": "fresh_eval_only",
                    "dataset": "fresh_route_eval_v1",
                })
    if len(rows) != 256:
        raise RuntimeError(f"Expected 256 fresh eval rows, got {len(rows)}")
    return rows


def canonical_jsonl(rows: Iterable[dict[str, Any]] | None = None) -> str:
    use = list(build_rows() if rows is None else rows)
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in use)


def dataset_sha256() -> str:
    return hashlib.sha256(canonical_jsonl().encode("utf-8")).hexdigest()


def assert_no_exact_overlap(reference_prompts: Iterable[str]) -> None:
    fresh = {normalize(row["prompt"]) for row in build_rows()}
    refs = {normalize(str(x)) for x in reference_prompts if str(x).strip()}
    overlap = sorted(fresh & refs)
    if overlap:
        raise RuntimeError(f"Fresh route eval has {len(overlap)} exact normalized overlaps: {overlap[:10]}")


if __name__ == "__main__":
    rows = build_rows()
    print(json.dumps({
        "rows": len(rows),
        "routes": {route: sum(row["route"] == route for row in rows) for route in ROUTES},
        "sha256": dataset_sha256(),
    }, indent=2, sort_keys=True))
