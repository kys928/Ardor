from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from Erratum import ardor_v14a9_holdout_generalization_diagnostic as exp


def example_row(family="definition_mechanism"):
    return {
        "id": "holdout_sample",
        "prompt": "What is happening to the gradient?",
        "chosen": "Gradient clipping caps oversized gradients.",
        "source": "heldout_reference",
        "route": "gradient_clipping",
        "semantic_family": family,
        "question_class": "normal",
        "prompt_style": "unseen_explain",
    }


def test_first_sentence_bounds_and_no_period():
    assert exp.first_sentence("Gradient clipping works. It helps.") == ("Gradient clipping works.", True)
    assert exp.first_sentence("No clear terminator") == ("No clear terminator", False)
    assert exp.first_sentence("") == ("", False)


def test_first_sentence_rubric_is_independent_from_full_generation():
    row = example_row()
    result = exp.first_sentence_rubric(
        row, "Gradient clipping caps the oversized gradient. Bad bad bad bad."
    )
    assert result["passes_keyword_semantic_rubric"]
    assert result["has_sentence_boundary"]
    assert result["has_route_signal"]
    assert result["has_family_signal"]
    assert result["first_sentence"] == "Gradient clipping caps the oversized gradient."


def test_first_sentence_requires_correction_for_false_premise():
    row = example_row("false_premise_correction")
    row["question_class"] = "false_premise_correction"
    good = exp.first_sentence_rubric(
        row, "Exploding gradients do not help training stability. Gradient clipping bounds them."
    )
    bad = exp.first_sentence_rubric(
        row, "Exploding gradients improve stability. Gradient clipping follows."
    )
    assert good["passes_keyword_semantic_rubric"]
    assert not bad["passes_keyword_semantic_rubric"]
    assert "false_premise_not_corrected" in bad["failure_reasons"]


def test_generation_repetition_scored_separately_from_first_sentence():
    row = example_row()
    outputs = [
        {
            "id": "holdout_sample", "semantic_family": "definition_mechanism",
            "question_class": "normal", "passed": False,
            "failures": ["high_repetition=0.900"],
            "generation": (
                "Gradient clipping limits the gradient. "
                "The phrase repeats repeats repeats. The phrase repeats repeats repeats."
            ),
        }
    ]
    class FakeHistory:
        @staticmethod
        def repetition_rate(text):
            return 0.9
    out = exp.generation_diagnostics(outputs, [row], FakeHistory)
    assert out["first_sentence_keyword_semantic_success_rate"] == 1.0
    assert out["full_answer_original_rubric_success_count"] == 0
    assert out["high_repetition_count"] == 1
    assert out["mean_trigram_repetition_rate"] == 0.9


def test_reference_boundary_eos_prob_is_after_chosen(monkeypatch):
    seen = []
    class Tok:
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=[1, 2] if "Gradient clipping caps" not in text else [3, 4])
    class Model:
        def eval(self):
            pass
        def __call__(self, x):
            seen.append(x.shape[-1])
            logits = torch.zeros((1, x.shape[1], 10), dtype=torch.float)
            logits[..., 3] = 2.0
            logits[..., 7] = 1.0
            return logits
    monkeypatch.setattr(exp.base, "encode", lambda tok, text: tok.encode(text).ids)
    out = exp.reference_boundary_termination_probs(
        Model(), Tok(), [example_row()], torch.device("cpu"), max_len=100,
    )
    assert seen == [4]
    assert out["mean_p_eos_after_reference"] > out["mean_p_eot_after_reference"]
    assert out["mean_p_either_termination_after_reference"] < 1.0
    assert out["rows"] == 1


def test_heldout_nll_uses_answer_boundary_and_token_weighting(monkeypatch):
    seen = []
    def fake_continuation_ce(model, tok, rows, device, **kwargs):
        assert all(r["source"] == "heldout_reference" for r in rows)
        assert all(r["prompt"].endswith("\n-") for r in rows)
        seen.append([r["id"] for r in rows])
        return torch.tensor(2.25), {"tokens": 12, "top1_rate": 0.5}
    monkeypatch.setattr(exp.base, "continuation_ce", fake_continuation_ce)
    class FakeModel:
        def eval(self):
            pass
    out = exp.heldout_continuation_nll(
        FakeModel(), object(), [example_row()], torch.device("cpu"),
        max_len=128, batch_rows=4
    )
    assert out["reference_continuation_nll"] == 2.25
    assert out["reference_continuation_top1"] == 0.5
    assert out["tokens"] == 12
    assert seen == [["holdout_sample"], ["holdout_sample"]]
