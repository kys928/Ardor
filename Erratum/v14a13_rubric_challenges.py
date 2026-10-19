"""Independent-from-training-output rubric challenges, grounded in PyTorch docs.

This is a predeclared *hand-authored calibration battery*, not an independent
human experiment and not an automated semantic classifier. Expected labels are
specified without calling evaluator.applicable() or evaluator.evaluate().

Reference facts:
https://docs.pytorch.org/docs/main/generated/torch.nn.utils.clip_grads_with_norm_.html
https://docs.pytorch.org/docs/main/generated/torch.nn.utils.clip_grad_norm_.html
https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
"""
from __future__ import annotations

KEYS = {
    "definition_mechanism": (
        "gradient_clipping_identified", "post_backprop_pre_optimizer",
        "large_gradient_bounded",
    ),
    "exploding_gradients": ("explosion_harms_stability", "clipping_mitigates_spike"),
    "norm_threshold": (
        "norm_is_compared_to_limit", "conditional_rescaling",
        "optimizer_follows_clipping",
    ),
    "optimizer_stability": (
        "spike_precedes_update", "gradient_bounded_before_optimizer",
        "mitigates_update_risk",
    ),
    "backprop_stability": (
        "backprop_computes_gradients", "clipping_after_backward",
        "next_update_mitigation",
    ),
    "learning_rate_comparison": (
        "clipping_conditional_gradient_control", "learning_rate_controls_step_scale",
        "mechanisms_distinguished",
    ),
    "false_premise_correction": (
        "explicitly_reject_false_premise", "explosion_is_risk",
        "clipping_is_protection",
    ),
}

# Each positive example deliberately supplies different, local evidence for
# each required proposition. Negative examples include lexical traps or
# contradictions and have their own predeclared factual failure rationale.
CHALLENGES = [
    {
        "id": "def_good", "family": "definition_mechanism", "gold": True,
        "response": "Gradient clipping reduces an oversized computed gradient after backpropagation and before the optimizer step.",
        "support": {
            "gradient_clipping_identified": "Gradient clipping",
            "post_backprop_pre_optimizer": "after backpropagation and before the optimizer step",
            "large_gradient_bounded": "reduces an oversized computed gradient",
        },
        "reason": "Mechanism, magnitude, and placement are all stated.",
    },
    {
        "id": "def_lexical_trap", "family": "definition_mechanism", "gold": False,
        "response": "Gradient clipping gradient norm optimizer update.",
        "support": {"gradient_clipping_identified": "Gradient clipping"},
        "reason": "Keyword pile does not state the causal mechanism or the correct timing.",
    },
    {
        "id": "explode_good", "family": "exploding_gradients", "gold": True,
        "response": "Exploding gradients can destabilize optimization. Gradient clipping reduces an oversized gradient before the optimizer uses it.",
        "support": {
            "explosion_harms_stability": "Exploding gradients can destabilize optimization",
            "clipping_mitigates_spike": "Gradient clipping reduces an oversized gradient before the optimizer uses it",
        },
        "reason": "Explains problem and corrective operation.",
    },
    {
        "id": "explode_wrong_direction", "family": "exploding_gradients", "gold": False,
        "response": "Exploding gradients make optimization more stable; clipping amplifies them before the optimizer.",
        "contradicted": {
            "explosion_harms_stability": "Exploding gradients make optimization more stable",
            "clipping_mitigates_spike": "clipping amplifies them",
        },
        "reason": "Reverses both causal claims.",
    },
    {
        "id": "norm_good", "family": "norm_threshold", "gold": True,
        "response": "The total gradient norm is compared to a maximum. Only if it exceeds that maximum are gradients scaled down; smaller gradients remain unchanged. The optimizer then uses the adjusted gradient.",
        "support": {
            "norm_is_compared_to_limit": "norm is compared to a maximum",
            "conditional_rescaling": "Only if it exceeds that maximum are gradients scaled down; smaller gradients remain unchanged",
            "optimizer_follows_clipping": "The optimizer then uses the adjusted gradient",
        },
        "reason": "Correctly distinguishes conditional max-norm cap from a target norm.",
    },
    {
        "id": "norm_false_target", "family": "norm_threshold", "gold": False,
        "response": "The norm is compared to a maximum. Every gradient is enlarged or reduced until its norm equals the maximum. The optimizer uses that modified gradient.",
        "support": {
            "norm_is_compared_to_limit": "norm is compared to a maximum",
            "optimizer_follows_clipping": "The optimizer uses that modified gradient",
        },
        "contradicted": {"conditional_rescaling": "Every gradient is enlarged or reduced until its norm equals the maximum"},
        "reason": "Claims smaller gradients are amplified, contrary to norm-clipping semantics.",
    },
    {
        "id": "optimizer_good", "family": "optimizer_stability", "gold": True,
        "response": "A spike in a computed gradient can destabilize the next update. Clipping bounds the gradient before the optimizer uses it, reducing the risk of an unstable update.",
        "support": {
            "spike_precedes_update": "A spike in a computed gradient can destabilize the next update",
            "gradient_bounded_before_optimizer": "Clipping bounds the gradient before the optimizer uses it",
            "mitigates_update_risk": "reducing the risk of an unstable update",
        },
        "reason": "Accurate concise mechanism without gratuitous explicit disclaimer.",
    },
    {
        "id": "optimizer_overclaim", "family": "optimizer_stability", "gold": False,
        "response": "A gradient spike can destabilize an update. Clipping limits the gradient before the optimizer step and reduces update risk. Therefore, it guarantees every update is safe.",
        "support": {
            "spike_precedes_update": "A gradient spike can destabilize an update",
            "gradient_bounded_before_optimizer": "Clipping limits the gradient before the optimizer step",
            "mitigates_update_risk": "reduces update risk",
        },
        "contradictions": [{
            "evidence": "it guarantees every update is safe",
            "description": "Absolute guarantee contradicts bounded-risk explanation.",
        }],
        "reason": "All required positives occur, but an additional unsupported guarantee must veto.",
    },
    {
        "id": "backprop_good", "family": "backprop_stability", "gold": True,
        "response": "Backpropagation computes gradients. Clipping acts on these computed gradients after the backward pass but before the optimizer, reducing risk from an oversized gradient during the next update.",
        "support": {
            "backprop_computes_gradients": "Backpropagation computes gradients",
            "clipping_after_backward": "after the backward pass but before the optimizer",
            "next_update_mitigation": "reducing risk from an oversized gradient during the next update",
        },
        "reason": "Specifies the true timing without requiring a tangential caveat.",
    },
    {
        "id": "backprop_wrong_timing", "family": "backprop_stability", "gold": False,
        "response": "Clipping happens before gradients are computed and prevents every numerical error in backpropagation.",
        "contradicted": {
            "backprop_computes_gradients": "before gradients are computed",
            "clipping_after_backward": "before gradients are computed",
            "next_update_mitigation": "prevents every numerical error in backpropagation",
        },
        "reason": "Wrong timing and overclaim.",
    },
    {
        "id": "lr_good", "family": "learning_rate_comparison", "gold": True,
        "response": "Norm clipping reduces a computed gradient only above a threshold. The learning rate controls the optimizer's update scale more generally; clipping itself does not change that setting.",
        "support": {
            "clipping_conditional_gradient_control": "Norm clipping reduces a computed gradient only above a threshold",
            "learning_rate_controls_step_scale": "The learning rate controls the optimizer's update scale more generally",
            "mechanisms_distinguished": "clipping itself does not change that setting",
        },
        "reason": "Distinguishes what is changed, when, and by which control.",
    },
    {
        "id": "lr_wrong_equivalence", "family": "learning_rate_comparison", "gold": False,
        "response": "When the norm gets large, clipping lowers the global learning rate; these are exactly the same mechanism.",
        "contradicted": {
            "clipping_conditional_gradient_control": "clipping lowers the global learning rate",
            "learning_rate_controls_step_scale": "these are exactly the same mechanism",
            "mechanisms_distinguished": "these are exactly the same mechanism",
        },
        "reason": "Keyword presence hides the false equivalence.",
    },
    {
        "id": "premise_good", "family": "false_premise_correction", "gold": True,
        "response": "No—exploding gradients do not keep training safe. Extreme gradients can destabilize updates. Gradient clipping reduces an oversized gradient before the optimizer step.",
        "support": {
            "explicitly_reject_false_premise": "No—exploding gradients do not keep training safe",
            "explosion_is_risk": "Extreme gradients can destabilize updates",
            "clipping_is_protection": "Gradient clipping reduces an oversized gradient before the optimizer step",
        },
        "reason": "Direct premise rejection plus correction and causal mechanism.",
    },
    {
        "id": "premise_agrees", "family": "false_premise_correction", "gold": False,
        "response": "Yes, exploding gradients safeguard training. Clipping can be applied before the optimizer update.",
        "support": {},
        "contradicted": {
            "explicitly_reject_false_premise": "Yes, exploding gradients safeguard training",
            "explosion_is_risk": "exploding gradients safeguard training",
        },
        "reason": "Endorses dangerous false premise despite mentioning clipping.",
    },
    {
        "id": "collision_good", "family": "definition_mechanism",
        "question_class": "collision_comparison", "gold": True,
        "response": "Gradient clipping reduces an oversized computed gradient after backpropagation and before the optimizer step. A checkpoint only saves training state; it does not alter the computed gradient.",
        "support": {
            "gradient_clipping_identified": "Gradient clipping",
            "post_backprop_pre_optimizer": "after backpropagation and before the optimizer step",
            "large_gradient_bounded": "reduces an oversized computed gradient",
            "contrasted_collision": "A checkpoint only saves training state; it does not alter the computed gradient",
        },
        "reason": "Correctly distinguishes mechanism from checkpoint.",
    },
    {
        "id": "collision_keywords_only", "family": "definition_mechanism",
        "question_class": "collision_comparison", "gold": False,
        "response": "Gradient clipping reduces an oversized computed gradient after backpropagation and before the optimizer step. Checkpoint is another training word.",
        "support": {
            "gradient_clipping_identified": "Gradient clipping",
            "post_backprop_pre_optimizer": "after backpropagation and before the optimizer step",
            "large_gradient_bounded": "reduces an oversized computed gradient",
        },
        "reason": "Correct clipping alone is inadequate for an explicit collision question.",
    },
]


def annotations_for_challenge(challenge: dict) -> tuple[dict, dict]:
    """Build fixed gold annotations using a manually authored, frozen key map.

    No evaluator code is invoked here. This isolates the challenge labels
    from the scoring implementation, but the author remains one AI reviewer.
    """
    family = challenge["family"]
    keys = list(KEYS[family])
    if challenge.get("question_class") == "collision_comparison":
        keys.append("contrasted_collision")
    supported = challenge.get("support", {})
    contradicted = challenge.get("contradicted", {})
    if set(supported) & set(contradicted):
        raise RuntimeError("Same fact cannot be both supported and contradicted")
    labels = {}
    for key in keys:
        if key in supported:
            labels[key] = {"status": "supported", "evidence": supported[key]}
        elif key in contradicted:
            labels[key] = {"status": "contradicted", "evidence": contradicted[key]}
        else:
            labels[key] = {"status": "missing", "evidence": ""}
    case = {
        "id": challenge["id"], "family": family,
        "question_class": challenge.get("question_class", "normal"),
    }
    annotation = {
        "case_id": challenge["id"],
        "reviewer": "source-grounded-challenge-author",
        "response": challenge["response"],
        "propositions": labels,
        "contradictions": challenge.get("contradictions", []),
    }
    return case, annotation
