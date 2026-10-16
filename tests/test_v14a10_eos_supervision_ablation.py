from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from Erratum import ardor_v14a10_eos_supervision_ablation as exp
from Erratum import ardor_v14a9_holdout_generalization_diagnostic as audit


def example(prompt_ids=(11, 12, 13), answer_ids=(21, 22)):
    return {
        "id": "r1", "prompt": "p", "chosen": "a",
        "route": "gradient_clipping", "semantic_family": "norm_threshold",
        "source": "behavior", "_pids": list(prompt_ids), "_aids": list(answer_ids),
    }


def patch_tokenization(monkeypatch):
    monkeypatch.setattr(
        exp.base, "encode",
        lambda tok, text: [11, 12, 13] if text == "p" else [21, 22],
    )


def test_eos_batch_has_one_extra_label_and_correct_mask(monkeypatch):
    patch_tokenization(monkeypatch)
    eos_id = exp.base.EXPECTED_SPECIAL_IDS["<eos>"]
    pad_id = exp.base.EXPECTED_SPECIAL_IDS["<pad>"]
    x, y, meta = exp.eos_supervised_batch(
        [example()], object(), max_len=20, pad_id=pad_id, eos_id=eos_id,
        device=torch.device("cpu"),
    )
    assert x.shape == y.shape == (1, 5)
    assert x.tolist()[0] == [11, 12, 13, 21, 22]
    assert y.tolist()[0] == [-100, -100, 21, 22, eos_id]
    assert meta["eos_labels"] == 1
    assert meta["families"] == {"gradient_clipping/norm_threshold": 1}
    assert all(label != eos_id for label in y.tolist()[0][2:4])


def test_eos_batch_truncation_preserves_last_answer_token(monkeypatch):
    patch_tokenization(monkeypatch)
    x, y, _ = exp.eos_supervised_batch(
        [example()], object(), max_len=4, pad_id=0, eos_id=3,
        device=torch.device("cpu"),
    )
    assert x.tolist()[0] == [13, 21, 22]
    assert y.tolist()[0] == [21, 22, 3]


def test_eos_ce_counts_exactly_one_extra_label_and_backprops(monkeypatch):
    patch_tokenization(monkeypatch)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(1, 1, 32))
        def forward(self, x):
            return self.logits.expand(x.shape[0], x.shape[1], -1)
    m = Model()
    loss, info = exp.eos_continuation_ce(
        m, object(), [example()], torch.device("cpu"), max_len=20, pad_id=0,
    )
    assert info["tokens"] == 3
    assert info["eos_labels"] == 1
    loss.backward()
    assert m.logits.grad is not None and bool(m.logits.grad.abs().sum() > 0)


def test_objective_arm_choice_frozen(monkeypatch):
    assert exp.ARMS == ("no_eos", "explicit_eos")
    assert exp.PEAK_LR == 2e-5
    assert exp.PROBE_UPDATES == (0, 20, 40)
    assert exp.TOTAL_UPDATES == 40
    assert exp.GRAD_CLIP == 0.5
    assert exp.END_TOKEN == "<eos>"
    assert exp.base.EXPECTED_SPECIAL_IDS[exp.END_TOKEN] == 3


def fake_result(arm, *, eos=1e-6, reps=26, repetition=0.62, nll=1.25, top1=0.78):
    return {
        "arm": arm,
        "final_train_target_nll": audit.V14A8_EXPECTED_U40_NLL,
        "final_train_target_top1": audit.V14A8_EXPECTED_U40_TOP1,
        "final_semantic_macro": audit.V14A8_EXPECTED_U40_SEMANTIC_MACRO,
        "batch_sequence_ids": ["r1", "r2", "r1"],
        "final_generalization": {
            "holdout_reference_nll": nll, "holdout_reference_top1": top1,
            "first_sentence_keyword_rubric_success_rate": 0.125,
            "mean_eos_eot_p_at_reference_boundary": eos,
            "mean_trigram_repetition_rate": repetition,
            "high_repetition_count": reps,
            "full_answer_rubric_success_count": 3,
        },
    }


def test_verdict_requires_heldout_preservation_not_just_eos():
    original = fake_result("no_eos")
    restored = fake_result("explicit_eos", eos=0.4, reps=2, repetition=0.1, nll=1.24)
    verdict = exp.verdict(original, restored)
    assert verdict["interpretation"].endswith("without_heldout_loss")
    assert all(verdict["checks"].values())

    degraded = fake_result("explicit_eos", eos=0.5, reps=2, repetition=0.1, nll=2.2)
    verdict = exp.verdict(original, degraded)
    assert verdict["interpretation"] == "eos_helped_generation_but_heldout_learning_regressed"
    assert not verdict["checks"]["heldout_reference_learning_preserved"]


def test_verdict_detects_missing_eos_and_bad_control():
    control = fake_result("no_eos")
    intervention = fake_result("explicit_eos", eos=0.0001, reps=25, repetition=0.6)
    assert exp.verdict(control, intervention)["interpretation"] == (
        "explicit_eos_supervision_did_not_restore_termination"
    )
    control["final_train_target_nll"] = 2.0
    with pytest.raises(RuntimeError, match="baseline did not replicate"):
        exp.verdict(control, intervention)


def test_verdict_requires_identical_batch_order():
    original = fake_result("no_eos")
    changed = fake_result("explicit_eos")
    changed["batch_sequence_ids"] = ["r2", "r1"]
    with pytest.raises(RuntimeError, match="Batch order mismatch"):
        exp.verdict(original, changed)
