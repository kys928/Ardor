#!/usr/bin/env python3
"""Build the behavior-first v14a3 continuation dataset from explicit semantic templates.

The new rows are deliberately answer-level supervision, not eight repeated canonical definitions.
Every chosen answer lands the defining route semantics in sentence one, then contrasts the nearest
observed collision when the row is a hard pair. Existing balanced/targeted datasets remain replay.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
import re
from pathlib import Path
from typing import Any

ROUTES = [
    "tokenizer", "rag", "overfitting", "direct_answer",
    "checkpoint", "dropout", "gradient_clipping", "correlation",
]
PRIORITY = ["gradient_clipping", "dropout", "rag", "tokenizer"]
COLLISIONS = [
    ("dropout", "overfitting"),
    ("dropout", "correlation"),
    ("gradient_clipping", "overfitting"),
    ("gradient_clipping", "checkpoint"),
    ("rag", "direct_answer"),
    ("rag", "tokenizer"),
    ("tokenizer", "direct_answer"),
]
SIGNALS = {
    "correlation": ["correlation", "variables", "relationship", "association", "causation"],
    "direct_answer": ["direct answer", "main point", "point first", "answer first"],
    "tokenizer": ["token", "token ids", "vocabulary", "encode text"],
    "rag": ["retriev", "external context", "external evidence", "grounded evidence"],
    "overfitting": ["overfit", "training data", "generaliz", "unseen", "new data"],
    "checkpoint": ["checkpoint", "saved model", "saved state", "resume training"],
    "dropout": ["dropout", "randomly", "mask", "activations"],
    "gradient_clipping": ["gradient clipping", "gradient norm", "large gradients", "clip gradients"],
}
PROMPT_FRAMES = [
    "Explain the mechanism described here in one or two sentences: {cue}",
    "What concept is this describing, and what does it do? {cue}",
    "Give the direct technical explanation for this situation: {cue}",
    "A student asks what this mechanism means: {cue}. Answer clearly.",
    "Identify the relevant ML concept from the mechanism, then explain it: {cue}",
    "Without extra setup, explain what is happening here: {cue}",
    "Which concept fits this description? State the concept and its role: {cue}",
    "Explain this in plain technical language: {cue}",
]
CUES = {
    "gradient_clipping": [
        "the optimizer is about to receive an unusually large gradient, so its magnitude is bounded before the update",
        "a training step would be unstable because the gradient norm is huge, so the gradient is capped first",
        "the learning signal is kept but its excessive magnitude is limited before parameters are changed",
        "exploding gradients are controlled by restricting how large the gradient can be at the optimizer step",
        "the gradient direction is retained while an excessive norm is reduced before updating weights",
        "training stability is protected by imposing a maximum gradient magnitude before an update",
        "a very large backward-pass signal is rescaled or capped before AdamW applies it",
        "parameter updates are prevented from becoming extreme by limiting the gradient norm",
    ],
    "dropout": [
        "some activations are randomly turned off during training and restored normally at inference",
        "a changing random subset of hidden units is masked on each training pass to reduce co-adaptation",
        "training injects random activation masks so the network cannot rely on the same units every time",
        "neurons are probabilistically disabled only while training as a regularization mechanism",
        "random masks zero parts of the activation vector during training to encourage robustness",
        "the model trains with randomly missing activations so representations do not depend on one fixed pathway",
        "a regularizer repeatedly removes a random fraction of activations during training",
        "the forward pass uses stochastic activation masking during training but not normal inference",
    ],
    "rag": [
        "the system searches an external knowledge source and places relevant retrieved context into the prompt before answering",
        "generation is preceded by retrieval of documents that can ground the response in outside evidence",
        "the model first fetches relevant external passages and then conditions its answer on those passages",
        "outside context is retrieved at query time and supplied to the generator before it writes the answer",
        "the answer uses evidence selected from an external store rather than relying only on model weights",
        "a retriever finds useful documents and the language model generates with those documents as context",
        "relevant external information is fetched before generation to ground the final response",
        "the generation step is augmented with context returned by a search or retrieval component",
    ],
    "tokenizer": [
        "raw text is converted into vocabulary-based token identifiers before the neural network processes it",
        "a text string is segmented and mapped to integer IDs from the model vocabulary",
        "words or subwords are encoded as token IDs that the embedding layer can consume",
        "the model cannot ingest the string directly, so text pieces are mapped to vocabulary indices first",
        "input text is split into token units and converted into numerical vocabulary IDs",
        "a preprocessing component maps text fragments to integers and can decode those integers back to text",
        "the string becomes a sequence of token IDs before it reaches the transformer",
        "text segments are matched against a vocabulary so the model receives numeric token indices",
    ],
    "overfitting": [
        "training performance keeps improving while performance on new unseen examples gets worse",
        "the model has fit details of the training set so closely that it generalizes poorly",
        "the system memorizes training-specific patterns instead of learning behavior that transfers to new data",
        "loss is excellent on training examples but validation behavior degrades because the fit is too specific",
        "the model matches the training data extremely well and performs worse outside that data",
        "training examples are fitted too closely, hurting generalization to unseen cases",
        "the learned solution captures noise or training-specific detail rather than reusable structure",
        "the gap between training and validation grows because the model has become too specialized to the training set",
    ],
    "correlation": [
        "two variables change together statistically, but that relationship alone does not establish that one causes the other",
        "measurements show an association between variables without proving a causal mechanism",
        "the variables are statistically related even though causation cannot be inferred from that fact alone",
        "one variable tends to vary with another, which is an association rather than proof of cause",
        "the data show a relationship in how two variables move, but no causal conclusion follows automatically",
        "there is statistical dependence between variables without evidence that either one produces the other",
        "a pattern links the variables in the observations, while the direction of cause remains unknown",
        "the variables co-vary in the dataset, which indicates association rather than causal proof",
    ],
    "checkpoint": [
        "the model and training state are saved so evaluation or training can continue from that point later",
        "weights and relevant optimizer state are persisted at a particular training step for later recovery",
        "a saved model state captures training progress so a run can be resumed instead of restarted",
        "the current parameters are written to a recoverable model snapshot for evaluation or continuation",
        "training state is stored as a snapshot that can later restore the model to this step",
        "a model snapshot preserves the learned weights and possibly optimizer state for later use",
        "the run writes a saved state that can be loaded again to resume or evaluate the model",
        "the current training state is serialized so the exact stage can be recovered later",
    ],
    "direct_answer": [
        "the response gives the requested main point immediately instead of adding unnecessary setup first",
        "the answer leads with the core result and keeps supporting explanation secondary",
        "the main point is stated first so the user does not need to search through preamble",
        "the response answers the question immediately and only then adds useful context",
        "the requested conclusion appears at the start rather than after unrelated framing",
        "the reply prioritizes the central answer before optional explanation",
        "the user gets the main answer first, with extra detail only where it helps",
        "the response begins with the substantive answer rather than delaying it with setup",
    ],
}
LANDINGS = {
    "gradient_clipping": [
        "Gradient clipping limits excessively large gradients before the optimizer updates the parameters.",
        "This is gradient clipping: the gradient magnitude is capped or rescaled before the optimizer step.",
        "The mechanism is gradient clipping, which prevents very large gradients from producing unstable updates.",
        "Gradient clipping keeps an update stable by restricting the norm or magnitude of the gradient first.",
        "Here, gradient clipping is controlling an oversized backward-pass signal before weights change.",
        "This uses gradient clipping to bound large gradients while preserving a usable update direction.",
    ],
    "dropout": [
        "Dropout randomly masks some activations during training as a regularization method.",
        "This is dropout: a random subset of activations is disabled on each training pass.",
        "The mechanism is dropout, which stochastically zeros activations while the model is training.",
        "Dropout regularizes the network by randomly removing activation pathways during training.",
        "Here, dropout is injecting random activation masks so the model cannot depend on one fixed pathway.",
        "This training-time random masking of activations is dropout.",
    ],
    "rag": [
        "RAG retrieves relevant external context before generation so the answer can use grounded evidence.",
        "This is retrieval-augmented generation: external passages are fetched before the model writes the answer.",
        "The mechanism is RAG, where retrieved outside information is supplied to the generator as context.",
        "RAG grounds generation by retrieving useful documents or evidence before answering.",
        "Here, retrieval augments the language model with external context at query time.",
        "This is RAG because a retriever supplies relevant external information before generation.",
    ],
    "tokenizer": [
        "A tokenizer converts text into vocabulary-based token IDs that the model can process.",
        "This is tokenization: text is segmented and mapped to integer IDs from the model vocabulary.",
        "The tokenizer encodes raw text as token identifiers before those IDs reach the transformer.",
        "Tokenization maps text pieces to vocabulary indices so they can be embedded numerically.",
        "Here, the tokenizer is turning a string into a sequence of model-readable token IDs.",
        "This preprocessing step is the tokenizer converting text fragments into vocabulary IDs.",
    ],
    "overfitting": [
        "Overfitting happens when a model fits the training data too closely and generalizes worse to unseen data.",
        "This is overfitting: training-specific patterns are learned so strongly that performance on new examples suffers.",
        "The failure mode is overfitting, where excellent training fit comes at the expense of generalization.",
        "Overfitting means the model has become too specialized to its training examples to transfer reliably.",
        "Here, the widening train-versus-validation gap indicates overfitting rather than healthy learning.",
        "This describes overfitting because the model is fitting training detail instead of reusable structure.",
    ],
    "correlation": [
        "Correlation means variables are statistically associated, but that association alone does not prove causation.",
        "This is correlation: the variables co-vary in the data without establishing a causal relationship.",
        "The relationship is a correlation, which describes association rather than proof that one variable causes the other.",
        "Correlation captures statistical dependence between variables, not a demonstrated causal mechanism.",
        "Here, the data show correlation because the variables move together without causal evidence.",
        "This describes a correlation: an observed statistical relationship whose cause remains unresolved.",
    ],
    "checkpoint": [
        "A checkpoint is a saved model or training state that can later be evaluated or used to resume training.",
        "This is a checkpoint: a recoverable snapshot of the model's training state at a particular step.",
        "The saved artifact is a checkpoint, preserving model progress for later loading, evaluation, or continuation.",
        "A checkpoint stores the learned state so the run can return to this stage instead of starting over.",
        "Here, the serialized model state is a checkpoint that records a recoverable point in training.",
        "This saved training snapshot is a checkpoint used to restore or evaluate the model later.",
    ],
    "direct_answer": [
        "A direct answer gives the requested main point first and avoids unnecessary setup before it.",
        "This is a direct answer: the response leads immediately with the substantive result.",
        "The response style is direct answering, where the core answer appears before optional explanation.",
        "A direct answer prioritizes the conclusion the user asked for instead of delaying it with preamble.",
        "Here, the reply is direct because it states the main point immediately and keeps context secondary.",
        "This uses a direct-answer style by putting the requested result at the start of the response.",
    ],
}
ELABORATIONS = {
    "gradient_clipping": ["It changes update magnitude control, not which units are randomly active.", "Its purpose is optimizer stability when gradients would otherwise become extreme.", "The operation is applied to gradients, usually after backpropagation and before the optimizer step.", "It does not mean the model has memorized the training set; it bounds the update signal."],
    "dropout": ["The random mask is a training-time regularizer, not a cap on gradient magnitude.", "Its goal is to reduce brittle co-adaptation and improve generalization.", "The randomness acts on activations during training rather than on the optimizer's gradient norm.", "At ordinary inference those stochastic masks are normally disabled."],
    "rag": ["The defining step is retrieval of outside information before generation.", "That external evidence is different from merely answering concisely from the model's internal weights.", "Its grounding comes from retrieved context, not from text-to-token preprocessing.", "The retriever and generator therefore form distinct stages of the answer pipeline."],
    "tokenizer": ["This conversion happens before semantic processing or external retrieval.", "The important output is a sequence of vocabulary IDs, not retrieved evidence.", "It defines how strings become numeric model inputs and how token IDs map back to text.", "Its job is representation of text at the vocabulary boundary, not answer style."],
    "overfitting": ["The key symptom is poor generalization, not random activation masking itself.", "It is a learned generalization failure rather than an optimizer gradient-control operation.", "Validation or unseen-data performance is what distinguishes it from simply fitting training examples well.", "Regularizers may reduce it, but the failure mode itself is the train-to-unseen performance gap."],
    "correlation": ["The crucial distinction is that statistical association does not establish cause.", "This describes a property of variables in data, not stochastic masking inside a neural network.", "A causal claim would require evidence beyond the observed co-variation.", "The signal is the relationship between variables rather than a training regularizer."],
    "checkpoint": ["It preserves state; it does not itself constrain the gradient used by the optimizer.", "The defining property is recoverability of a particular training stage.", "Loading it restores saved parameters and, when stored, optimizer state.", "It is a persistent snapshot rather than a mechanism that changes the current update."],
    "direct_answer": ["The distinction is response organization, not retrieval of new external evidence.", "It changes how the answer is presented rather than how text is tokenized.", "Supporting detail can follow, but it should not hide the requested result.", "The route is about answer structure, not a separate knowledge-acquisition stage."],
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def route_signal(route: str, first_sentence: str) -> bool:
    low = norm(first_sentence)
    return any(signal in low for signal in SIGNALS[route])


def answer_for(route: str, index: int, hard_negative: str | None = None) -> str:
    first = LANDINGS[route][index % len(LANDINGS[route])]
    second = ELABORATIONS[route][(index // len(LANDINGS[route])) % len(ELABORATIONS[route])]
    if hard_negative:
        second = second + f" This is specifically contrasted with {hard_negative.replace('_', ' ')} in this example."
    return first + " " + second


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    serial = 0
    for pair_index, (left, right) in enumerate(COLLISIONS):
        for route, negative in ((left, right), (right, left)):
            for i, frame in enumerate(PROMPT_FRAMES):
                cue = CUES[route][i % len(CUES[route])]
                rows.append({
                    "id": f"v14a3_hard_{serial:04d}",
                    "row_type": "behavior_hard_collision",
                    "route": route,
                    "hard_negative": negative,
                    "collision_pair": [left, right],
                    "prompt": frame.format(cue=cue),
                    "chosen": answer_for(route, i + pair_index * 8, negative),
                    "source": "v14a3_behavior_first_authored",
                })
                serial += 1
    # Extra route-balanced semantic landing rows keep every route explicit; the trainer controls priority weighting.
    for route in ROUTES:
        count = 16 if route in PRIORITY else 8
        for i in range(count):
            frame = PROMPT_FRAMES[(i + 3) % len(PROMPT_FRAMES)]
            cue = CUES[route][(i * 3 + 1) % len(CUES[route])]
            rows.append({
                "id": f"v14a3_landing_{serial:04d}",
                "row_type": "behavior_semantic_landing",
                "route": route,
                "hard_negative": None,
                "collision_pair": None,
                "prompt": frame.format(cue=cue),
                "chosen": answer_for(route, i + 5),
                "source": "v14a3_behavior_first_authored",
            })
            serial += 1
    return rows


def validate(rows: list[dict[str, Any]], holdout: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts = Counter(str(r["route"]) for r in rows)
    if set(route_counts) != set(ROUTES):
        raise RuntimeError(f"v14a3 data missing routes: {sorted(set(ROUTES) - set(route_counts))}")
    holdout_prompts = {norm(str(r.get("prompt") or r.get("text") or "")) for r in holdout}
    overlaps = [r["id"] for r in rows if norm(str(r["prompt"])) in holdout_prompts]
    if overlaps:
        raise RuntimeError(f"Holdout prompt overlap detected: {overlaps[:10]}")
    bad_first = []
    for row in rows:
        first = str(row["chosen"]).split(".", 1)[0] + "."
        if not route_signal(str(row["route"]), first):
            bad_first.append(str(row["id"]))
    if bad_first:
        raise RuntimeError(f"Rows without first-sentence route signal: {bad_first[:20]}")
    seen_edges = {(str(r["route"]), str(r["hard_negative"])) for r in rows if r.get("hard_negative")}
    missing_edges = []
    for a, b in COLLISIONS:
        for edge in ((a, b), (b, a)):
            if edge not in seen_edges:
                missing_edges.append(edge)
    if missing_edges:
        raise RuntimeError(f"Missing directed collision coverage: {missing_edges}")
    chosen = [norm(str(r["chosen"])) for r in rows]
    unique = len(set(chosen))
    duplicate_fraction = 1.0 - unique / max(1, len(chosen))
    if duplicate_fraction > 0.15:
        raise RuntimeError(
            f"Chosen-answer duplicate fraction {duplicate_fraction:.3f} exceeds 0.15; add wording diversity"
        )
    by_route_unique = {
        route: len({norm(str(r["chosen"])) for r in rows if r["route"] == route})
        for route in ROUTES
    }
    return {
        "rows": len(rows),
        "route_counts": dict(route_counts),
        "unique_chosen": unique,
        "chosen_duplicate_fraction": duplicate_fraction,
        "unique_chosen_by_route": by_route_unique,
        "holdout_exact_prompt_overlap": len(overlaps),
        "directed_collision_edges": sorted([list(x) for x in seen_edges]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=Path, default=Path("/workspace/Ardor/training/data/v14_v3/dataset_v3_holdout_audit.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/Ardor/training/data/v14_v3/dataset_v14a3_behavior_first.jsonl"))
    args = parser.parse_args()
    holdout = read_jsonl(args.holdout)
    rows = build_rows()
    summary = validate(rows, holdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
