#!/usr/bin/env python3
"""Build the v14a4 family-balanced semantic-landing train and clean holdout datasets.

The dataset is deliberately organized by route x semantic_family. Training rows use varied
prompt and answer forms. Collision comparisons are a small explicit subset instead of boilerplate
inside ordinary answers. False-premise questions are labeled and require correction-first answers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ROUTES = [
    "tokenizer", "rag", "overfitting", "direct_answer",
    "checkpoint", "dropout", "gradient_clipping", "correlation",
]
PRIORITY_ROUTES = ["gradient_clipping", "dropout", "rag", "tokenizer"]

TARGET_TRAIN_COUNTS = {
    "gradient_clipping": 450,
    "dropout": 225,
    "rag": 150,
    "tokenizer": 150,
    "checkpoint": 50,
    "correlation": 50,
    "direct_answer": 50,
    "overfitting": 50,
}
HOLDOUT_PER_ROUTE = 32

FAMILIES = {
    "gradient_clipping": [
        "definition_mechanism", "exploding_gradients", "norm_threshold",
        "optimizer_stability", "backprop_stability", "learning_rate_comparison",
        "false_premise_correction",
    ],
    "dropout": [
        "definition_mechanism", "train_vs_eval", "dropout_probability",
        "coadaptation", "regularization_generalization", "overfitting_relationship",
        "false_premise_correction",
    ],
    "rag": [
        "retrieval_sequence", "external_grounding", "query_matching", "context_injection",
        "answer_synthesis", "unsupported_claims", "model_memory_vs_retrieval", "limitations",
    ],
    "tokenizer": [
        "text_to_ids", "vocabulary", "subword_boundaries", "special_tokens",
        "encoding_decoding", "model_compatibility", "checkpoint_loading_mismatch",
        "vocabulary_version_mismatch",
    ],
    "checkpoint": ["saved_state", "resume_training", "evaluation_snapshot", "optimizer_state"],
    "correlation": ["association_not_causation", "positive_negative", "confounding", "statistical_relationship"],
    "direct_answer": ["point_first", "concise_structure", "context_after_answer", "no_preamble"],
    "overfitting": ["train_validation_gap", "memorization", "unseen_generalization", "regularization_relation"],
}

COLLISIONS = {
    "gradient_clipping": ["overfitting", "checkpoint"],
    "dropout": ["overfitting", "correlation"],
    "rag": ["direct_answer", "tokenizer"],
    "tokenizer": ["direct_answer", "rag"],
    "overfitting": ["dropout", "gradient_clipping"],
    "checkpoint": ["gradient_clipping"],
    "direct_answer": ["rag", "tokenizer"],
    "correlation": ["dropout"],
}

ROUTE_SIGNALS = {
    "gradient_clipping": ["gradient clipping", "clip gradient", "gradient norm", "gradient magnitude", "exploding gradient", "cap the gradient", "rescale the gradient"],
    "dropout": ["dropout", "activation mask", "masked activation", "randomly disable", "training-time noise", "co-adaptation", "coadaptation"],
    "rag": ["retriev", "external context", "external source", "ground", "document", "search result", "outside evidence"],
    "tokenizer": ["tokenizer", "tokenization", "token id", "vocabulary", "subword", "special token", "encode", "decode"],
    "checkpoint": ["checkpoint", "saved state", "saved model", "resume training", "snapshot"],
    "correlation": ["correlation", "association", "co-vary", "covary", "statistical relationship", "causation"],
    "direct_answer": ["direct answer", "main point", "answer first", "point first", "no preamble", "conclusion first"],
    "overfitting": ["overfitting", "overfit", "memorization", "generalization", "validation", "unseen data", "train-validation"],
}

FAMILY_SIGNALS = {
    ("gradient_clipping", "definition_mechanism"): ["gradient clipping", "clip", "cap", "rescale"],
    ("gradient_clipping", "exploding_gradients"): ["exploding gradient", "very large gradient", "unstable update"],
    ("gradient_clipping", "norm_threshold"): ["norm", "threshold", "maximum gradient", "bound"],
    ("gradient_clipping", "optimizer_stability"): ["optimizer", "update", "stability", "stable"],
    ("gradient_clipping", "backprop_stability"): ["backprop", "backward", "gradient", "stability"],
    ("gradient_clipping", "learning_rate_comparison"): ["learning rate", "gradient", "clip", "different"],
    ("gradient_clipping", "false_premise_correction"): ["exploding gradient", "destabil", "does not help", "don't help", "do not help"],
    ("dropout", "definition_mechanism"): ["dropout", "mask", "activation", "random"],
    ("dropout", "train_vs_eval"): ["training", "evaluation", "inference", "disabled"],
    ("dropout", "dropout_probability"): ["probability", "dropout rate", "fraction", "chance"],
    ("dropout", "coadaptation"): ["co-adaptation", "coadaptation", "rely on the same", "dependency"],
    ("dropout", "regularization_generalization"): ["regularization", "generalization", "robust"],
    ("dropout", "overfitting_relationship"): ["overfitting", "generalization", "training-time mask"],
    ("dropout", "false_premise_correction"): ["evaluation", "inference", "disabled", "does not add", "doesn't add"],
    ("rag", "retrieval_sequence"): ["retrieve", "before generation", "then generate", "first"],
    ("rag", "external_grounding"): ["external", "ground", "evidence", "source"],
    ("rag", "query_matching"): ["query", "match", "relevant", "retrieve"],
    ("rag", "context_injection"): ["context", "prompt", "inject", "supply"],
    ("rag", "answer_synthesis"): ["synthes", "combine", "answer", "retrieved"],
    ("rag", "unsupported_claims"): ["unsupported", "hallucination", "ground", "evidence"],
    ("rag", "model_memory_vs_retrieval"): ["memory", "weights", "retrieval", "external"],
    ("rag", "limitations"): ["limitation", "bad retrieval", "irrelevant", "missing evidence", "cannot guarantee"],
    ("tokenizer", "text_to_ids"): ["text", "token id", "integer", "encode"],
    ("tokenizer", "vocabulary"): ["vocabulary", "token id", "mapping"],
    ("tokenizer", "subword_boundaries"): ["subword", "boundary", "split", "segment"],
    ("tokenizer", "special_tokens"): ["special token", "bos", "eos", "pad", "role token"],
    ("tokenizer", "encoding_decoding"): ["encode", "decode", "token id", "text"],
    ("tokenizer", "model_compatibility"): ["compatible", "compatibility", "same tokenizer", "model vocabulary"],
    ("tokenizer", "checkpoint_loading_mismatch"): ["checkpoint", "loading", "token id", "mismatch"],
    ("tokenizer", "vocabulary_version_mismatch"): ["vocabulary", "version", "mismatch", "token id"],
    ("checkpoint", "saved_state"): ["checkpoint", "saved state", "snapshot"],
    ("checkpoint", "resume_training"): ["resume", "continue training", "checkpoint"],
    ("checkpoint", "evaluation_snapshot"): ["evaluate", "snapshot", "checkpoint"],
    ("checkpoint", "optimizer_state"): ["optimizer state", "momentum", "checkpoint", "resume"],
    ("correlation", "association_not_causation"): ["association", "causation", "correlation"],
    ("correlation", "positive_negative"): ["positive", "negative", "correlation"],
    ("correlation", "confounding"): ["confound", "third variable", "causation"],
    ("correlation", "statistical_relationship"): ["statistical", "relationship", "variables"],
    ("direct_answer", "point_first"): ["main point", "first", "direct answer"],
    ("direct_answer", "concise_structure"): ["concise", "direct", "answer"],
    ("direct_answer", "context_after_answer"): ["context", "after", "answer first"],
    ("direct_answer", "no_preamble"): ["preamble", "setup", "direct"],
    ("overfitting", "train_validation_gap"): ["training", "validation", "gap"],
    ("overfitting", "memorization"): ["memor", "training data", "overfit"],
    ("overfitting", "unseen_generalization"): ["unseen", "generalization", "new data"],
    ("overfitting", "regularization_relation"): ["regularization", "overfitting", "generalization"],
}

# Each family has independent train and holdout cues. Holdout cues are never used for training.
TRAIN_CUES = {
    ("gradient_clipping", "definition_mechanism"): [
        "an oversized gradient is capped before parameters are updated",
        "the backward signal is rescaled when its magnitude is too large",
        "the optimizer receives a bounded gradient instead of the raw extreme value",
        "a training step limits the gradient before changing the weights",
    ],
    ("gradient_clipping", "exploding_gradients"): [
        "gradient values grow so large that an update can become unstable",
        "the backward pass produces exploding gradients that threaten training stability",
        "very large gradients would create an extreme parameter update",
        "training becomes unstable because gradients blow up in magnitude",
    ],
    ("gradient_clipping", "norm_threshold"): [
        "the global gradient norm is compared with a maximum threshold and reduced if needed",
        "a maximum norm bounds how large the gradient vector may be",
        "the gradient is rescaled only when its norm exceeds the allowed limit",
        "training enforces a threshold on gradient magnitude before the optimizer step",
    ],
    ("gradient_clipping", "optimizer_stability"): [
        "the optimizer should not take a huge step when the gradient spikes",
        "update magnitude is controlled before AdamW changes the parameters",
        "an unstable optimizer step is avoided by bounding the incoming gradient",
        "the parameter update is kept stable even when the raw gradient is extreme",
    ],
    ("gradient_clipping", "backprop_stability"): [
        "a large signal produced by backpropagation is bounded before it reaches the optimizer",
        "backward propagation creates a gradient spike that must be controlled",
        "the backward-pass gradient is limited so training can continue stably",
        "a backpropagated signal is too large and is rescaled before the update",
    ],
    ("gradient_clipping", "learning_rate_comparison"): [
        "the gradient itself is bounded rather than globally shrinking every update with a smaller learning rate",
        "one method caps unusually large gradients while another changes the learning-rate scale for all steps",
        "the goal is to control occasional gradient spikes without simply lowering the learning rate everywhere",
        "the operation acts on the gradient norm, which is different from choosing a smaller optimizer learning rate",
    ],
    ("gradient_clipping", "false_premise_correction"): [
        "exploding gradients are claimed to help training remain stable",
        "a teammate says very large gradients prevent unstable updates",
        "someone argues that exploding gradients are desirable because they speed learning",
        "the statement says an unbounded backward signal protects the optimizer",
    ],
    ("dropout", "definition_mechanism"): [
        "a random subset of activations is zeroed on each training pass",
        "training masks different hidden activations from batch to batch",
        "some units are randomly disabled while the model is training",
        "a stochastic mask removes part of the activation vector during training",
    ],
    ("dropout", "train_vs_eval"): [
        "random activation masking is used during training but disabled for ordinary evaluation",
        "the network sees stochastic masks in training and the full network at inference",
        "a training-only random mask disappears when the model switches to evaluation mode",
        "evaluation uses the full activations even though training randomly drops some of them",
    ],
    ("dropout", "dropout_probability"): [
        "each eligible activation has a configured chance of being masked during training",
        "a dropout rate controls the fraction of activations randomly removed",
        "the regularizer uses a probability to decide which activations are zeroed",
        "changing the dropout probability changes how aggressively training masks units",
    ],
    ("dropout", "coadaptation"): [
        "random masks stop the same hidden units from always depending on one another",
        "the network is discouraged from relying on one fixed combination of features",
        "changing which activations survive each pass reduces brittle co-adaptation",
        "features must remain useful even when some neighboring activations are missing",
    ],
    ("dropout", "regularization_generalization"): [
        "training-time random masking acts as regularization and can improve robustness on new examples",
        "stochastic activation removal discourages brittle features and supports generalization",
        "the network is regularized by learning under randomly missing activations",
        "random masking makes the model less dependent on a single internal pathway",
    ],
    ("dropout", "overfitting_relationship"): [
        "random activation masks can reduce overfitting by making memorized feature dependencies less reliable",
        "the regularizer helps prevent the model from fitting one rigid pattern in the training set",
        "dropout can reduce overfitting because features must work under changing masks",
        "training with randomly absent activations can improve performance beyond the training examples",
    ],
    ("dropout", "false_premise_correction"): [
        "evaluation mode is claimed to add dropout noise for better predictions",
        "someone says dropout becomes more random during inference than during training",
        "a teammate claims evaluation mode should keep masking activations to improve accuracy",
        "the premise says ordinary inference intentionally injects dropout noise",
    ],
    ("rag", "retrieval_sequence"): [
        "the system retrieves relevant material first and generates only after that context is available",
        "a retriever runs before the language model writes the answer",
        "document search happens first, then the generator conditions on the results",
        "generation follows an external retrieval step instead of starting from model weights alone",
    ],
    ("rag", "external_grounding"): [
        "the answer is grounded in evidence fetched from an external source at query time",
        "outside documents are supplied so claims can be tied to retrieved evidence",
        "the generator uses external material rather than relying only on parameters",
        "retrieved sources provide evidence that grounds the generated response",
    ],
    ("rag", "query_matching"): [
        "the retrieval component matches the user query to relevant passages",
        "a search stage chooses documents whose content is relevant to the incoming question",
        "the system uses the query to select which external chunks should be retrieved",
        "relevance matching determines what context is returned before generation",
    ],
    ("rag", "context_injection"): [
        "retrieved passages are inserted into the model context before the answer is generated",
        "external evidence is added to the prompt or context window for the generator",
        "the generation input is augmented with text returned by retrieval",
        "selected documents become part of the context the language model conditions on",
    ],
    ("rag", "answer_synthesis"): [
        "the generator combines information from retrieved passages into a coherent answer",
        "retrieved evidence is synthesized into the final response instead of copied blindly",
        "the model forms an answer using several pieces of external context",
        "generation integrates the selected evidence into one response",
    ],
    ("rag", "unsupported_claims"): [
        "external evidence can reduce unsupported claims by grounding generation in retrieved material",
        "retrieved sources give the model evidence to use instead of inventing unsupported details",
        "grounding the response in documents can lower hallucination risk",
        "the system uses retrieved evidence to support claims made in the answer",
    ],
    ("rag", "model_memory_vs_retrieval"): [
        "external retrieval at query time is different from information stored implicitly in model weights",
        "the system fetches new context instead of depending only on parameter memory",
        "retrieved documents are outside the model and are not the same thing as memorized training knowledge",
        "query-time search supplies evidence that was not necessarily encoded in the model's parameters",
    ],
    ("rag", "limitations"): [
        "poor retrieval can still give the generator irrelevant or missing evidence",
        "retrieval does not guarantee correctness when the search stage returns bad context",
        "RAG can fail if useful evidence is absent from the external store",
        "grounding quality depends on whether retrieval finds relevant trustworthy material",
    ],
    ("tokenizer", "text_to_ids"): [
        "raw text is segmented and mapped to integer token identifiers before the transformer sees it",
        "the model receives token IDs rather than the original character string",
        "text pieces are converted into numeric IDs that index the embedding table",
        "a preprocessing step turns the input string into a sequence of vocabulary IDs",
    ],
    ("tokenizer", "vocabulary"): [
        "each token piece maps to an entry in the tokenizer vocabulary",
        "the vocabulary defines which integer ID corresponds to each token",
        "token IDs only have meaning relative to the vocabulary mapping used by the model",
        "the tokenizer looks up text pieces in its vocabulary to produce IDs",
    ],
    ("tokenizer", "subword_boundaries"): [
        "a word may be split into several subword tokens depending on the tokenizer rules",
        "token boundaries can fall inside words rather than matching whitespace boundaries",
        "subword segmentation determines which pieces of a string become separate tokens",
        "different segmentation rules can map the same visible text to different token sequences",
    ],
    ("tokenizer", "special_tokens"): [
        "special IDs mark roles or boundaries such as BOS, EOS, padding, user, and assistant",
        "the tokenizer reserves special tokens for sequence boundaries and conversation roles",
        "special token IDs carry structural meaning in addition to ordinary text pieces",
        "BOS, EOS, padding, and role markers must map to the expected reserved IDs",
    ],
    ("tokenizer", "encoding_decoding"): [
        "encoding maps text to token IDs while decoding maps token IDs back to text",
        "the tokenizer supports both string-to-ID conversion and reconstruction from IDs",
        "text is encoded into vocabulary indices and those indices can later be decoded",
        "the same tokenizer defines the forward text-to-ID map and the reverse ID-to-text map",
    ],
    ("tokenizer", "model_compatibility"): [
        "a model expects the exact token-ID mapping used when its embeddings were trained",
        "tokenizer compatibility matters because embedding rows correspond to specific vocabulary IDs",
        "using a different tokenizer can send the model different IDs for the same text",
        "the tokenizer and model vocabulary must agree on what each token ID means",
    ],
    ("tokenizer", "checkpoint_loading_mismatch"): [
        "loading a checkpoint with a different token-ID mapping can attach learned embedding rows to the wrong text pieces",
        "a checkpoint assumes the tokenizer vocabulary that was used when its embedding matrix was trained",
        "tokenizer mismatch during model loading can make otherwise valid weights interpret IDs incorrectly",
        "checkpoint compatibility includes preserving the vocabulary and special-token IDs expected by the saved model",
    ],
    ("tokenizer", "vocabulary_version_mismatch"): [
        "two tokenizer versions can assign different IDs or boundaries even when the visible input text is identical",
        "changing the vocabulary version can invalidate the token-ID semantics expected by trained weights",
        "a new tokenizer vocabulary may reorder or replace token IDs used by an older model",
        "tokenizer version drift can alter segmentation and ID assignments across runs",
    ],
    ("checkpoint", "saved_state"): [
        "the model parameters are serialized as a recoverable training snapshot",
        "a saved artifact records model state at a particular step",
        "training writes a model snapshot that can be loaded later",
        "a persistent copy of the learned state is stored for recovery",
    ],
    ("checkpoint", "resume_training"): [
        "training can continue later by loading a saved checkpoint instead of starting over",
        "the run resumes from previously saved state",
        "a checkpoint restores progress so optimization can continue from that point",
        "saved training state provides a restart point after interruption",
    ],
    ("checkpoint", "evaluation_snapshot"): [
        "a saved checkpoint can be loaded to evaluate exactly that stage of the model",
        "evaluation uses a stored snapshot rather than whatever weights exist later",
        "a checkpoint preserves a specific model version for testing",
        "the saved state provides a reproducible point for evaluation",
    ],
    ("checkpoint", "optimizer_state"): [
        "a full training checkpoint may include optimizer moments as well as model weights",
        "optimizer state is stored so resumed training preserves momentum-like statistics",
        "resuming faithfully can require saved AdamW state in addition to parameters",
        "the checkpoint can carry both learned weights and the optimizer's internal state",
    ],
    ("correlation", "association_not_causation"): [
        "two variables move together statistically without proving one causes the other",
        "an observed association does not establish a causal mechanism",
        "co-variation in data is evidence of relationship, not automatic causation",
        "the variables are related statistically but cause and effect remain unproven",
    ],
    ("correlation", "positive_negative"): [
        "a positive relationship means variables tend to move together while a negative one means they move oppositely",
        "the sign of a correlation describes the direction of statistical co-variation",
        "positive and negative correlations indicate different directions of association",
        "correlation can be positive or negative depending on how the variables vary together",
    ],
    ("correlation", "confounding"): [
        "a third variable can create an apparent relationship without direct causation",
        "confounding can explain why two variables correlate even when neither causes the other",
        "an unobserved factor may influence both variables and produce their association",
        "correlation can arise because both measurements respond to another variable",
    ],
    ("correlation", "statistical_relationship"): [
        "correlation summarizes how variables vary together in observed data",
        "the measure describes statistical association between variables",
        "a correlation quantifies relationship in the data rather than a causal process",
        "the variables show a statistical pattern of co-variation",
    ],
    ("direct_answer", "point_first"): [
        "the response states the requested conclusion before secondary explanation",
        "the main point appears immediately at the start of the answer",
        "the user gets the substantive result before extra context",
        "the answer leads with what was actually asked",
    ],
    ("direct_answer", "concise_structure"): [
        "the response stays focused on the requested result and removes unnecessary framing",
        "a concise answer delivers the key information without wandering setup",
        "the reply is short enough to be clear while still answering the question",
        "the structure prioritizes relevant content over filler",
    ],
    ("direct_answer", "context_after_answer"): [
        "useful context can follow after the main answer has already been stated",
        "explanation is secondary to the conclusion and comes afterward",
        "the reply gives the result first and supporting detail second",
        "context is allowed, but it should not delay the answer",
    ],
    ("direct_answer", "no_preamble"): [
        "the response avoids an unnecessary preamble before answering",
        "the user should not need to read setup before reaching the result",
        "the answer begins directly rather than with generic framing",
        "irrelevant introduction is removed so the requested point comes first",
    ],
    ("overfitting", "train_validation_gap"): [
        "training performance improves while validation performance gets worse",
        "the gap between training and validation grows because the fit is too specific",
        "low training error is paired with deteriorating performance on held-out data",
        "the model keeps fitting training examples while validation quality declines",
    ],
    ("overfitting", "memorization"): [
        "the model memorizes training-specific details instead of reusable structure",
        "training examples are fitted so closely that the learned behavior becomes brittle",
        "the model captures noise and idiosyncrasies of the training set",
        "memorized patterns replace more general rules that would transfer",
    ],
    ("overfitting", "unseen_generalization"): [
        "performance drops on unseen examples despite strong training results",
        "the learned behavior does not transfer reliably to new data",
        "generalization suffers outside the examples used for fitting",
        "the model performs well on familiar samples but poorly on new ones",
    ],
    ("overfitting", "regularization_relation"): [
        "regularization can reduce overfitting by discouraging an excessively specific fit",
        "techniques such as dropout may improve generalization and therefore reduce overfitting",
        "regularization pushes the model away from memorizing training-only detail",
        "the goal of regularization is to improve behavior beyond the training set",
    ],
}

HOLDOUT_CUES = {
    ("gradient_clipping", "definition_mechanism"): ["the update uses a bounded backward signal rather than the raw spike", "a large gradient is restrained just before parameter change"],
    ("gradient_clipping", "exploding_gradients"): ["gradient explosion makes a training step numerically dangerous", "the backward signal grows until updates become unstable"],
    ("gradient_clipping", "norm_threshold"): ["a maximum gradient norm is enforced before optimization", "the vector norm is reduced only after crossing a preset ceiling"],
    ("gradient_clipping", "optimizer_stability"): ["an optimizer step is protected from a sudden gradient spike", "the update is prevented from jumping too far because of one extreme gradient"],
    ("gradient_clipping", "backprop_stability"): ["safe backpropagation requires controlling an excessive backward signal", "the backward pass is stable only after a huge gradient is bounded"],
    ("gradient_clipping", "learning_rate_comparison"): ["capping exceptional gradients is being compared with lowering the learning rate on every step", "one control changes gradient magnitude only when it is extreme, unlike globally reducing the step scale"],
    ("gradient_clipping", "false_premise_correction"): ["exploding gradients supposedly make optimization safer", "someone claims unbounded gradients are what prevent unstable parameter updates"],
    ("dropout", "definition_mechanism"): ["masked neurons change randomly from one training pass to the next", "training-time noise removes a random subset of activations"],
    ("dropout", "train_vs_eval"): ["the random masks disappear when the model enters evaluation mode", "inference uses the full activation path even though training masks units"],
    ("dropout", "dropout_probability"): ["a configured masking chance decides how many activations are usually removed", "the dropout rate controls the expected fraction of hidden units that vanish during training"],
    ("dropout", "coadaptation"): ["robust feature use is encouraged by preventing the same units from always appearing together", "co-adaptation is reduced because internal features cannot depend on a fixed set of partners"],
    ("dropout", "regularization_generalization"): ["stochastic masking acts as noise that can improve behavior on unseen data", "randomly missing activations force more robust internal representations"],
    ("dropout", "overfitting_relationship"): ["random activation masking is used to make memorization less reliable", "the model is less able to overfit one fixed feature pathway when masks keep changing"],
    ("dropout", "false_premise_correction"): ["evaluation mode supposedly adds useful dropout noise", "someone claims inference should randomly mask more activations than training"],
    ("rag", "retrieval_sequence"): ["generation waits until a search stage has returned relevant passages", "the answer pipeline performs retrieval first and language generation second"],
    ("rag", "external_grounding"): ["source-backed answers depend on evidence fetched outside the model", "the response is tied to external material retrieved for the current query"],
    ("rag", "query_matching"): ["query matching selects which passages from the external store are useful", "the search component ranks chunks by relevance to the question"],
    ("rag", "context_injection"): ["retrieved passages are placed into the generator's working context", "external text is added to the prompt context before decoding"],
    ("rag", "answer_synthesis"): ["the final response synthesizes multiple retrieved pieces of evidence", "answer construction combines the relevant facts returned by retrieval"],
    ("rag", "unsupported_claims"): ["source retrieval is used to lower unsupported claims in the generated answer", "external evidence is supplied so the model has less reason to invent facts"],
    ("rag", "model_memory_vs_retrieval"): ["query-time search is being contrasted with knowledge stored in model parameters", "retrieved context is external and temporary rather than parameter memory"],
    ("rag", "limitations"): ["the retriever returns irrelevant passages and the final answer suffers", "no useful source is found, so retrieval augmentation cannot provide grounding"],
    ("tokenizer", "text_to_ids"): ["the string becomes numerical identifiers before entering the embedding layer", "plain text must be turned into discrete IDs before transformer processing"],
    ("tokenizer", "vocabulary"): ["the meaning of an integer token depends on the vocabulary table", "a vocabulary lookup assigns numeric IDs to text pieces"],
    ("tokenizer", "subword_boundaries"): ["one visible word breaks into several model tokens", "token boundaries cut through a word because segmentation works on subwords"],
    ("tokenizer", "special_tokens"): ["role and sequence-boundary markers occupy reserved tokenizer IDs", "special markers such as EOS and padding must keep their expected IDs"],
    ("tokenizer", "encoding_decoding"): ["the same mapping turns strings into IDs and IDs back into strings", "text-to-token encoding and token-to-text decoding use the tokenizer vocabulary"],
    ("tokenizer", "model_compatibility"): ["tokenizer compatibility is required because the model embedding rows expect a fixed ID mapping", "a different tokenizer sends changed IDs into weights trained for the original vocabulary"],
    ("tokenizer", "checkpoint_loading_mismatch"): ["token IDs matter during model loading because the checkpoint embeddings assume their original mapping", "a checkpoint is loaded with a tokenizer whose IDs no longer match the saved embedding semantics"],
    ("tokenizer", "vocabulary_version_mismatch"): ["a new tokenizer version assigns different IDs to familiar text pieces", "vocabulary drift changes both segmentation and integer assignments across runs"],
    ("checkpoint", "saved_state"): ["a recoverable snapshot of learned parameters is written to storage", "the run preserves model state at a particular step"],
    ("checkpoint", "resume_training"): ["optimization continues from a previously saved state after interruption", "training restarts from stored progress rather than initialization"],
    ("checkpoint", "evaluation_snapshot"): ["a specific saved stage of the model is loaded for testing", "evaluation targets a frozen snapshot from an earlier training step"],
    ("checkpoint", "optimizer_state"): ["resumed AdamW needs its saved moments as well as the model weights", "the training snapshot preserves optimizer internals for faithful continuation"],
    ("correlation", "association_not_causation"): ["two measurements are associated but no causal direction is established", "the variables move together without proof that one produces the other"],
    ("correlation", "positive_negative"): ["one relationship moves in the same direction and another in opposite directions", "the sign of the association indicates whether variables rise together or inversely"],
    ("correlation", "confounding"): ["a hidden third factor could explain the observed relationship", "both variables may respond to another cause that creates their apparent association"],
    ("correlation", "statistical_relationship"): ["the data show systematic co-variation between two variables", "a statistic summarizes how strongly two quantities vary together"],
    ("direct_answer", "point_first"): ["the response gives the requested result in its opening sentence", "the conclusion appears before explanation"],
    ("direct_answer", "concise_structure"): ["the reply stays focused and removes unnecessary framing", "only the information needed to answer the question is foregrounded"],
    ("direct_answer", "context_after_answer"): ["supporting explanation comes only after the core answer", "the result is stated first and context follows second"],
    ("direct_answer", "no_preamble"): ["the reply skips generic setup and immediately addresses the question", "no introductory filler appears before the substantive answer"],
    ("overfitting", "train_validation_gap"): ["training accuracy rises while held-out accuracy deteriorates", "the model looks better on training data as validation quality falls"],
    ("overfitting", "memorization"): ["training-specific details are remembered instead of a reusable rule", "the model fits noise and quirks of the seen examples"],
    ("overfitting", "unseen_generalization"): ["performance collapses on new examples despite excellent training results", "the learned pattern does not transfer outside the training set"],
    ("overfitting", "regularization_relation"): ["a regularizer is introduced to reduce an overly specific fit", "the training setup adds constraints intended to improve generalization"],
}

TRAIN_PROMPT_STYLES = {
    "identify": "Which ML concept best matches this situation? {cue}",
    "mechanism": "Explain the mechanism at work here: {cue}",
    "beginner": "Explain this to a beginner without losing the technical point: {cue}",
    "diagnose": "Diagnose the training or inference behavior described here: {cue}",
    "definition_first": "Start with the relevant concept, then explain this case: {cue}",
    "why": "Why does this mechanism matter in the following case? {cue}",
    "operational": "Describe the operation being performed here: {cue}",
    "troubleshoot": "A teammate sees this behavior and asks what is happening: {cue}",
    "concise": "Give a concise technical explanation of this example: {cue}",
    "first_principles": "Explain this from first principles: {cue}",
    "name_and_role": "Name the concept and state its role in this example: {cue}",
    "plain_language": "Put this into plain technical language: {cue}",
}
HOLDOUT_PROMPT_STYLES = {
    "unseen_explain": "In one clear answer, what is going on when {cue}?",
    "unseen_teammate": "A teammate asks about this behavior: {cue}. How would you explain it?",
    "unseen_meaning": "What should I understand from this example: {cue}?",
    "unseen_name_reason": "Name the relevant idea and give the reason it matters: {cue}",
    "unseen_technical": "Explain the technical meaning behind this situation: {cue}",
    "unseen_short": "What is the shortest accurate explanation for this case: {cue}",
}

ANSWER_STYLES = [
    "definition_first", "mechanism_first", "causal", "process", "troubleshooting",
    "concise", "two_sentence", "contrastive", "operational", "result_first",
]
HOLDOUT_ANSWER_STYLES = [
    "reference_definition", "reference_mechanism", "reference_causal", "reference_process",
    "reference_concise", "reference_two_sentence",
]

FACTS = {
    ("gradient_clipping", "definition_mechanism"): ("Gradient clipping", "limits an excessively large gradient before the optimizer updates parameters", "the gradient is capped or rescaled before the optimizer step", "optimizer updates remain bounded instead of reacting to the full spike"),
    ("gradient_clipping", "exploding_gradients"): ("Gradient clipping", "controls exploding gradients that would otherwise produce unstable updates", "very large backward-pass gradients are bounded before optimization", "training is less likely to jump or diverge from one extreme update"),
    ("gradient_clipping", "norm_threshold"): ("Gradient clipping", "enforces a bound on gradient magnitude or norm", "the gradient norm is compared with a threshold and rescaled when it exceeds that limit", "the update keeps its useful direction while excessive magnitude is reduced"),
    ("gradient_clipping", "optimizer_stability"): ("Gradient clipping", "protects the optimizer from rare gradient spikes", "the incoming gradient is bounded before AdamW applies the parameter update", "one extreme backward signal cannot create an equally extreme optimizer step"),
    ("gradient_clipping", "backprop_stability"): ("Gradient clipping", "stabilizes training when backpropagation produces an oversized gradient", "the backward-pass signal is limited before it reaches the optimizer", "large backpropagated signals are prevented from destabilizing the next update"),
    ("gradient_clipping", "learning_rate_comparison"): ("Gradient clipping", "bounds unusually large gradients rather than globally lowering the learning rate", "clipping acts on gradient magnitude when it exceeds a limit, while learning rate scales optimizer steps more generally", "it targets gradient spikes without requiring every update to use a smaller learning-rate scale"),
    ("gradient_clipping", "false_premise_correction"): ("Exploding gradients", "do not help training stability; they are the problem gradient clipping is meant to control", "clipping bounds the oversized gradient before the optimizer step", "the stable behavior comes from limiting the exploding gradient, not from allowing it to grow"),
    ("dropout", "definition_mechanism"): ("Dropout", "randomly masks a subset of activations during training", "a different stochastic activation mask is applied across training passes", "the network cannot depend on every hidden pathway being present each time"),
    ("dropout", "train_vs_eval"): ("Dropout", "uses random activation masking during training and is normally disabled for evaluation or inference", "training samples stochastic masks while evaluation uses the full activation path", "predictions are not intentionally perturbed by dropout noise in ordinary evaluation mode"),
    ("dropout", "dropout_probability"): ("The dropout rate", "sets the probability that eligible activations are masked during training", "the configured probability controls how aggressively units are dropped", "higher or lower masking strength changes the amount of regularization"),
    ("dropout", "coadaptation"): ("Dropout", "reduces brittle co-adaptation between hidden features", "randomly missing activations force features to remain useful without the same partners always present", "the model relies less on one fixed internal combination of units"),
    ("dropout", "regularization_generalization"): ("Dropout", "acts as a training-time regularizer that can improve generalization", "stochastic masks inject structured noise into hidden activations during training", "representations can become more robust to variation outside the training examples"),
    ("dropout", "overfitting_relationship"): ("Dropout", "can reduce overfitting by making fixed memorized feature dependencies less reliable", "training repeatedly removes random activations", "the model is encouraged to learn patterns that survive beyond one rigid training pathway"),
    ("dropout", "false_premise_correction"): ("Evaluation-mode dropout noise", "is not the normal behavior; dropout masks are generally disabled during evaluation and inference", "the stochastic masks are a training-time regularizer", "ordinary evaluation uses the full network rather than adding extra dropout randomness"),
    ("rag", "retrieval_sequence"): ("RAG", "retrieves relevant external context before generation", "a retrieval stage runs first and the generator then conditions on the returned passages", "the final answer can use information fetched for the current query"),
    ("rag", "external_grounding"): ("RAG", "grounds generation in external sources or evidence", "retrieved documents are supplied to the model as context", "claims can be tied to information outside the model's parameters"),
    ("rag", "query_matching"): ("RAG retrieval", "matches the incoming query to relevant external passages", "the retriever ranks or selects chunks based on relevance to the question", "the generator receives context chosen for this specific query"),
    ("rag", "context_injection"): ("RAG", "adds retrieved passages to the generator's context before decoding", "external text is inserted into the prompt or context window", "generation can condition directly on the selected evidence"),
    ("rag", "answer_synthesis"): ("RAG answer synthesis", "combines retrieved evidence into a coherent generated response", "the generator integrates relevant information from one or more retrieved passages", "the output can summarize or reconcile evidence rather than merely copy a source"),
    ("rag", "unsupported_claims"): ("RAG grounding", "can reduce unsupported claims by giving generation external evidence to rely on", "retrieval supplies relevant sources before the answer is written", "the model has evidence available instead of relying only on uncertain parameter memory"),
    ("rag", "model_memory_vs_retrieval"): ("RAG retrieval", "is different from model memory because it fetches external information at query time", "retrieved passages come from an outside store rather than being recalled only from model weights", "the available evidence can change without retraining the model"),
    ("rag", "limitations"): ("RAG", "is limited by retrieval quality and source coverage", "irrelevant, missing, or unreliable retrieved context can still mislead generation", "retrieval augmentation does not guarantee a correct answer when the evidence stage fails"),
    ("tokenizer", "text_to_ids"): ("A tokenizer", "converts raw text into token IDs that the model can process", "the input string is segmented and mapped to integer vocabulary identifiers", "those IDs index embeddings before transformer computation begins"),
    ("tokenizer", "vocabulary"): ("A tokenizer vocabulary", "defines the mapping between text pieces and token IDs", "each recognized token or subword is assigned a vocabulary index", "the integer IDs only have the intended meaning under that mapping"),
    ("tokenizer", "subword_boundaries"): ("Tokenization", "can split visible words into smaller subword pieces", "segmentation rules choose token boundaries that do not have to match whitespace", "the same text can become a different ID sequence under a different tokenizer"),
    ("tokenizer", "special_tokens"): ("Special tokenizer tokens", "reserve IDs for structural markers such as BOS, EOS, padding, and conversation roles", "the tokenizer maps those markers to fixed expected IDs", "model input structure depends on those reserved identifiers staying consistent"),
    ("tokenizer", "encoding_decoding"): ("A tokenizer", "encodes text into token IDs and decodes token IDs back into text", "the vocabulary and segmentation rules define both directions of the mapping", "the same tokenizer contract keeps string and ID representations consistent"),
    ("tokenizer", "model_compatibility"): ("Tokenizer compatibility", "means the tokenizer's ID mapping matches the vocabulary expected by the model weights", "the embedding table was trained with particular token IDs assigned to particular text pieces", "changing that mapping can make valid model weights interpret input IDs incorrectly"),
    ("tokenizer", "checkpoint_loading_mismatch"): ("A tokenizer-checkpoint mismatch", "can break model loading semantics even when the weight tensors themselves load", "the checkpoint embedding rows assume the token-ID mapping used during training", "different IDs can connect learned embeddings to the wrong text pieces"),
    ("tokenizer", "vocabulary_version_mismatch"): ("Tokenizer version drift", "can change token boundaries or ID assignments across runs", "a different vocabulary version may segment the same text differently or remap IDs", "the model can receive incompatible integer sequences despite identical visible input"),
    ("checkpoint", "saved_state"): ("A checkpoint", "is a saved snapshot of model or training state", "parameters are serialized at a particular step", "the same stage can be restored later"),
    ("checkpoint", "resume_training"): ("A checkpoint", "provides a recovery point from which training can resume", "saved model and relevant training state are loaded before optimization continues", "the run does not need to restart from initialization"),
    ("checkpoint", "evaluation_snapshot"): ("A checkpoint", "preserves a specific model state for reproducible evaluation", "evaluation loads the saved parameters from that step", "later training changes do not alter the stored snapshot"),
    ("checkpoint", "optimizer_state"): ("A full training checkpoint", "can include optimizer state as well as model weights", "AdamW moments and other optimizer internals are serialized with the parameters", "resumed optimization can continue with the same accumulated state"),
    ("correlation", "association_not_causation"): ("Correlation", "means variables are statistically associated without proving causation", "the data show co-variation between the variables", "a causal mechanism requires evidence beyond the observed association"),
    ("correlation", "positive_negative"): ("Correlation", "can be positive or negative depending on the direction of co-variation", "positive values indicate variables tend to move together while negative values indicate opposite movement", "the sign describes association direction rather than causal direction"),
    ("correlation", "confounding"): ("A correlation", "can be produced by a confounding variable rather than direct causation", "a third factor may influence both observed variables", "the association alone cannot identify the true causal structure"),
    ("correlation", "statistical_relationship"): ("Correlation", "describes a statistical relationship between variables", "it summarizes how the variables vary together in observed data", "the measure characterizes association rather than a model-training mechanism"),
    ("direct_answer", "point_first"): ("A direct answer", "states the requested main point first", "the response opens with the substantive result", "supporting explanation does not hide the answer"),
    ("direct_answer", "concise_structure"): ("A direct answer", "keeps the response focused and concise", "irrelevant framing is removed while the requested result is preserved", "the user can identify the answer immediately"),
    ("direct_answer", "context_after_answer"): ("A direct answer", "puts useful context after the core answer", "the conclusion is delivered before secondary explanation", "extra detail supports rather than delays the response"),
    ("direct_answer", "no_preamble"): ("A direct answer", "avoids unnecessary preamble before addressing the question", "the response begins with the requested information", "generic setup does not delay the substantive point"),
    ("overfitting", "train_validation_gap"): ("Overfitting", "appears when training performance improves while validation or held-out performance deteriorates", "the model fits the training examples more specifically than the underlying general pattern", "the train-to-validation gap grows"),
    ("overfitting", "memorization"): ("Overfitting", "can involve memorizing training-specific detail instead of learning reusable structure", "the model fits noise or idiosyncrasies of seen examples", "performance does not transfer reliably"),
    ("overfitting", "unseen_generalization"): ("Overfitting", "means strong training fit is paired with poor generalization to unseen data", "the learned behavior is too specialized to the training distribution", "new examples expose the weakness of the fit"),
    ("overfitting", "regularization_relation"): ("Regularization", "can reduce overfitting by discouraging an excessively specific fit to training data", "the training procedure adds pressure for behavior that generalizes beyond seen examples", "validation or unseen-data performance can improve"),
}

DETAIL_VARIANTS = [
    "The key distinction is where the mechanism acts in the pipeline.",
    "That behavior should be identified from its operation, not from generic training vocabulary.",
    "The important signal is the mechanism itself rather than a memorized phrase.",
    "This remains true even when the prompt uses a different surface description.",
    "The explanation should preserve the causal direction instead of merely naming related terms.",
    "The concept can be recognized from what changes and when that change occurs.",
    "The same mechanism may appear under several paraphrases, but its functional role is unchanged.",
    "A correct answer should land this semantic distinction before adding optional detail.",
]

COLLISION_DISTINCTIONS = {
    ("gradient_clipping", "overfitting"): "Gradient clipping acts on gradient magnitude during an update, whereas overfitting is a failure to generalize beyond training data.",
    ("gradient_clipping", "checkpoint"): "Gradient clipping changes the current gradient before optimization; a checkpoint merely stores model or training state.",
    ("dropout", "overfitting"): "Dropout is a training-time regularization mechanism; overfitting is the generalization failure it may help reduce.",
    ("dropout", "correlation"): "Dropout randomly masks neural activations, while correlation describes a statistical relationship between variables.",
    ("rag", "direct_answer"): "RAG changes how evidence is acquired before generation; direct answering changes how the response is organized.",
    ("rag", "tokenizer"): "RAG retrieves external context, while a tokenizer converts text to and from model token IDs.",
    ("tokenizer", "direct_answer"): "A tokenizer maps text and token IDs; direct answering is a response-organization style.",
    ("tokenizer", "rag"): "A tokenizer represents text as IDs, whereas RAG retrieves external information before generation.",
    ("overfitting", "dropout"): "Overfitting is a generalization failure, while dropout is a regularizer that can help reduce it.",
    ("overfitting", "gradient_clipping"): "Overfitting concerns generalization, while gradient clipping bounds gradients during optimization.",
    ("checkpoint", "gradient_clipping"): "A checkpoint stores recoverable state, while gradient clipping modifies an oversized gradient before an update.",
    ("direct_answer", "rag"): "A direct answer organizes the response around the main point, while RAG retrieves external evidence.",
    ("direct_answer", "tokenizer"): "A direct answer concerns response structure, while tokenization maps strings to model IDs.",
    ("correlation", "dropout"): "Correlation describes statistical association, while dropout randomly masks neural activations during training.",
}

FALSE_PREMISE_FAMILIES = {
    ("gradient_clipping", "false_premise_correction"),
    ("dropout", "false_premise_correction"),
}

def norm(text: str) -> str:
    text = re.sub(r"[-‐‑‒–—]+", " ", str(text).strip().lower())
    return re.sub(r"\s+", " ", text)

def _question_class(route: str, family: str, collision: str | None) -> str:
    if (route, family) in FALSE_PREMISE_FAMILIES:
        return "false_premise_correction"
    if collision:
        return "collision_comparison"
    return "normal"

def _answer(route: str, family: str, style: str, detail_index: int, collision: str | None, split: str) -> str:
    label, definition, mechanism, effect = FACTS[(route, family)]
    detail = DETAIL_VARIANTS[detail_index % len(DETAIL_VARIANTS)]
    if (route, family) in FALSE_PREMISE_FAMILIES:
        if route == "gradient_clipping":
            variants = [
                f"They do not. {definition.capitalize()}. {mechanism.capitalize()}, so {effect}.",
                f"That premise is backwards: {definition}. {mechanism.capitalize()}.",
                f"No—{definition}. {effect.capitalize()}.",
                f"Exploding gradients are not beneficial here. {mechanism.capitalize()}, which is why gradient clipping is used.",
                f"The stable behavior comes from clipping, not from the exploding gradient itself. {definition.capitalize()}.",
                f"An unbounded gradient does not protect training. {mechanism.capitalize()} and {effect}.",
            ]
        else:
            variants = [
                f"It does not. {definition.capitalize()}. {mechanism.capitalize()}.",
                f"That premise is backwards: {definition}. {effect.capitalize()}.",
                f"No—ordinary evaluation does not add dropout noise. {mechanism.capitalize()}.",
                f"Dropout randomness belongs to training, not normal evaluation. {definition.capitalize()}.",
                f"Evaluation mode normally removes the stochastic mask. {effect.capitalize()}.",
                f"Inference is not made noisier by dropout. {mechanism.capitalize()} and {effect}.",
            ]
        base = variants[(detail_index + (1 if split == "holdout" else 0)) % len(variants)]
        return base + " " + detail

    if collision:
        distinction = COLLISION_DISTINCTIONS[(route, collision)]
        variants = [
            f"{label} is the relevant concept: {definition}. {distinction}",
            f"{distinction} In this case, {mechanism}.",
            f"The two ideas differ in function. {distinction}",
            f"{label} fits this case because {definition}. By contrast, {distinction.split(';',1)[-1].strip() if ';' in distinction else distinction}",
            f"This case is about {label.lower()}, not {collision.replace('_',' ')}. {distinction}",
            f"{mechanism.capitalize()}. That is why this is {label.lower()}; {distinction}",
        ]
        return variants[detail_index % len(variants)] + " " + detail

    templates = {
        "definition_first": f"{label} {definition}. {detail}",
        "mechanism_first": f"{mechanism.capitalize()}. This is {label.lower()}, and {effect}.",
        "causal": f"{effect.capitalize()} because {mechanism}. The relevant concept is {label.lower()}.",
        "process": f"The process is {label.lower()}: first, {mechanism}; as a result, {effect}.",
        "troubleshooting": f"The behavior points to {label.lower()}. It {definition}, so {effect}.",
        "concise": f"{label}: {definition}.",
        "two_sentence": f"{label} {definition}. {effect.capitalize()}.",
        "contrastive": f"{label} is identified by this mechanism: {mechanism}. {detail}",
        "operational": f"Operationally, {mechanism}. That is {label.lower()}, whose role is that {effect}.",
        "result_first": f"{effect.capitalize()}. The mechanism responsible is {label.lower()}: {definition}.",
        "reference_definition": f"{label} {definition}. {effect.capitalize()}.",
        "reference_mechanism": f"{mechanism.capitalize()}. The concept is {label.lower()}.",
        "reference_causal": f"{effect.capitalize()} because {mechanism}; this is {label.lower()}.",
        "reference_process": f"This is {label.lower()}: {mechanism}, which means {effect}.",
        "reference_concise": f"{label}: {definition}.",
        "reference_two_sentence": f"{label} {definition}. The key result is that {effect}.",
    }
    base = templates[style]
    if detail not in base:
        base = base + " " + detail
    return base

def _collision_for(route: str, row_index: int, family: str) -> str | None:
    if (route, family) in FALSE_PREMISE_FAMILIES:
        return None
    if row_index % 8 != 0:
        return None
    options = COLLISIONS.get(route, [])
    if not options:
        return None
    return options[(row_index // 8) % len(options)]

def build_train_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    style_names = list(TRAIN_PROMPT_STYLES)
    for route in ROUTES:
        target = TARGET_TRAIN_COUNTS[route]
        families = FAMILIES[route]
        family_seen: Counter[str] = Counter()
        for i in range(target):
            family = families[i % len(families)]
            local_i = family_seen[family]
            family_seen[family] += 1
            cue_options = TRAIN_CUES[(route, family)]
            cue = cue_options[local_i % len(cue_options)]
            collision = _collision_for(route, i, family)
            if collision:
                prompt_style = "collision_compare"
                prompt = (
                    f"Compare {route.replace('_',' ')} with {collision.replace('_',' ')} in this case: "
                    f"{cue}. Which concept actually explains the mechanism and why?"
                )
                answer_style = "contrastive"
            else:
                prompt_style = style_names[(local_i + i) % len(style_names)]
                prompt = TRAIN_PROMPT_STYLES[prompt_style].format(cue=cue)
                answer_style = ANSWER_STYLES[local_i % len(ANSWER_STYLES)]
            qclass = _question_class(route, family, collision)
            if qclass == "false_premise_correction":
                prompt_style = "false_premise"
                prompt = f"Correct the premise before answering: {cue}. What is actually true?"
                answer_style = "correction_first"
            chosen = _answer(route, family, answer_style, (local_i // len(ANSWER_STYLES)) * 3 + local_i, collision, "train")
            rows.append({
                "id": f"v14a4_train_{route}_{i:04d}",
                "split": "train",
                "route": route,
                "semantic_family": family,
                "prompt_style": prompt_style,
                "answer_style": answer_style,
                "question_class": qclass,
                "collision_route": collision,
                "prompt": prompt,
                "chosen": chosen,
            })
    return rows

def build_holdout_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    style_names = list(HOLDOUT_PROMPT_STYLES)
    for route in ROUTES:
        families = FAMILIES[route]
        family_seen: Counter[str] = Counter()
        for i in range(HOLDOUT_PER_ROUTE):
            family = families[i % len(families)]
            local_i = family_seen[family]
            family_seen[family] += 1
            cues = HOLDOUT_CUES[(route, family)]
            cue = cues[local_i % len(cues)]
            collision = None
            if (route, family) not in FALSE_PREMISE_FAMILIES and i % 8 == 3:
                options = COLLISIONS.get(route, [])
                if options:
                    collision = options[(i // 8) % len(options)]
            if collision:
                prompt_style = "unseen_collision"
                prompt = (
                    f"Distinguish {route.replace('_',' ')} from {collision.replace('_',' ')} here: "
                    f"{cue}. Which mechanism is actually being described?"
                )
                answer_style = "reference_two_sentence"
            else:
                prompt_style = style_names[(local_i + i) % len(style_names)]
                prompt = HOLDOUT_PROMPT_STYLES[prompt_style].format(cue=cue)
                answer_style = HOLDOUT_ANSWER_STYLES[(local_i * 2 + i) % len(HOLDOUT_ANSWER_STYLES)]
            qclass = _question_class(route, family, collision)
            if qclass == "false_premise_correction":
                prompt_style = "unseen_false_premise"
                prompt = f"A claim says: {cue}. Correct it, then explain the actual mechanism."
                answer_style = "correction_first"
            chosen = _answer(route, family, answer_style, 1000 + local_i + i, collision, "holdout")
            rows.append({
                "id": f"v14a4_holdout_{route}_{i:03d}",
                "split": "holdout",
                "route": route,
                "semantic_family": family,
                "prompt_style": prompt_style,
                "answer_style": answer_style,
                "question_class": qclass,
                "collision_route": collision,
                "prompt": prompt,
                "chosen": chosen,
            })
    return rows

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def sha256_rows(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return h.hexdigest()

def validate(train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], historical_holdout: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    required = {
        "id", "split", "route", "semantic_family", "prompt_style", "answer_style",
        "question_class", "collision_route", "prompt", "chosen",
    }
    for split_name, rows in (("train", train_rows), ("holdout", holdout_rows)):
        ids = set()
        for row in rows:
            missing = required - set(row)
            if missing:
                raise RuntimeError(f"{split_name} row missing fields {sorted(missing)}: {row.get('id')}")
            if row["id"] in ids:
                raise RuntimeError(f"duplicate id: {row['id']}")
            ids.add(row["id"])
            route = str(row["route"])
            family = str(row["semantic_family"])
            if route not in ROUTES or family not in FAMILIES[route]:
                raise RuntimeError(f"invalid route/family: {route}/{family}")
            if row["split"] != split_name:
                raise RuntimeError(f"wrong split on {row['id']}")
            if not str(row["prompt"]).strip() or not str(row["chosen"]).strip():
                raise RuntimeError(f"empty prompt/chosen: {row['id']}")
            qclass = str(row["question_class"])
            if qclass not in {"normal", "collision_comparison", "false_premise_correction"}:
                raise RuntimeError(f"bad question_class on {row['id']}: {qclass}")
            if qclass == "false_premise_correction" and (route, family) not in FALSE_PREMISE_FAMILIES:
                raise RuntimeError(f"false premise outside declared family: {row['id']}")
            if qclass == "collision_comparison" and not row["collision_route"]:
                raise RuntimeError(f"collision row missing collision_route: {row['id']}")
            if qclass == "normal" and row["collision_route"] is not None:
                raise RuntimeError(f"normal row has collision_route: {row['id']}")

    train_counts = Counter(r["route"] for r in train_rows)
    if dict(train_counts) != {r: TARGET_TRAIN_COUNTS[r] for r in ROUTES}:
        raise RuntimeError(f"train route counts mismatch: {dict(train_counts)}")
    holdout_counts = Counter(r["route"] for r in holdout_rows)
    if any(holdout_counts[r] != HOLDOUT_PER_ROUTE for r in ROUTES):
        raise RuntimeError(f"holdout route counts mismatch: {dict(holdout_counts)}")

    train_prompts = {norm(r["prompt"]) for r in train_rows}
    holdout_prompts = {norm(r["prompt"]) for r in holdout_rows}
    overlap = train_prompts & holdout_prompts
    if overlap:
        raise RuntimeError(f"train/clean-holdout prompt overlap: {len(overlap)}")

    if historical_holdout is not None:
        hist_prompts = {
            norm(str(r.get("prompt") or r.get("text") or r.get("anchor_context") or ""))
            for r in historical_holdout
            if str(r.get("prompt") or r.get("text") or r.get("anchor_context") or "").strip()
        }
        hist_overlap = train_prompts & hist_prompts
        if hist_overlap:
            raise RuntimeError(f"v14a4 train/historical holdout exact prompt overlap: {len(hist_overlap)}")

    family_counts: dict[str, dict[str, int]] = {}
    unique_answer_ratio: dict[str, float] = {}
    collision_fraction: dict[str, float] = {}
    false_premise_fraction: dict[str, float] = {}
    for route in ROUTES:
        route_rows = [r for r in train_rows if r["route"] == route]
        counts = Counter(r["semantic_family"] for r in route_rows)
        family_counts[route] = dict(counts)
        if max(counts.values()) - min(counts.values()) > 1:
            raise RuntimeError(f"family imbalance for {route}: {dict(counts)}")
        ratio = len({norm(r["chosen"]) for r in route_rows}) / max(1, len(route_rows))
        unique_answer_ratio[route] = ratio
        minimum = 0.65 if route in PRIORITY_ROUTES else 0.45
        if ratio < minimum:
            raise RuntimeError(f"chosen diversity too low for {route}: {ratio:.3f} < {minimum:.3f}")
        non_false = [r for r in route_rows if r["question_class"] != "false_premise_correction"]
        coll = sum(r["question_class"] == "collision_comparison" for r in non_false) / max(1, len(non_false))
        collision_fraction[route] = coll
        if COLLISIONS.get(route) and not 0.08 <= coll <= 0.16:
            raise RuntimeError(f"collision fraction outside 8-16% for {route}: {coll:.3f}")
        false_premise_fraction[route] = sum(r["question_class"] == "false_premise_correction" for r in route_rows) / max(1, len(route_rows))

    holdout_family_counts: dict[str, dict[str, int]] = {}
    for route in ROUTES:
        counts = Counter(r["semantic_family"] for r in holdout_rows if r["route"] == route)
        holdout_family_counts[route] = dict(counts)
        if set(counts) != set(FAMILIES[route]):
            raise RuntimeError(f"holdout missing families for {route}: {dict(counts)}")
        if min(counts.values()) < 4:
            raise RuntimeError(f"holdout family has <4 paraphrases for {route}: {dict(counts)}")

    return {
        "schema_version": 1,
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "train_sha256": sha256_rows(train_rows),
        "holdout_sha256": sha256_rows(holdout_rows),
        "train_route_counts": dict(train_counts),
        "holdout_route_counts": dict(holdout_counts),
        "train_family_counts": family_counts,
        "holdout_family_counts": holdout_family_counts,
        "unique_chosen_ratio_by_route": unique_answer_ratio,
        "collision_fraction_by_route": collision_fraction,
        "false_premise_fraction_by_route": false_premise_fraction,
        "exact_train_clean_holdout_prompt_overlap": 0,
        "exact_train_historical_holdout_prompt_overlap": 0 if historical_holdout is not None else None,
    }

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", type=Path)
    parser.add_argument("--holdout-output", type=Path)
    args = parser.parse_args()
    train = build_train_rows()
    holdout = build_holdout_rows()
    summary = validate(train, holdout)
    if args.train_output:
        write_jsonl(args.train_output, train)
    if args.holdout_output:
        write_jsonl(args.holdout_output, holdout)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
