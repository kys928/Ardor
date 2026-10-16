from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest

from Erratum import ardor_v14a12_supervision_quality_eos as exp


def example_rows():
    rows = []
    for family in exp.TARGETS:
        for index in range(4):
            question_class = "false_premise_correction" if family == "false_premise_correction" else "normal"
            rows.append({
                "id": f"v14a4_train_gradient_clipping_{family}_{index}",
                "route": "gradient_clipping", "semantic_family": family,
                "source": "behavior", "question_class": question_class,
                "collision_route": None, "prompt": "Original authored prompt",
                "chosen": f"Original chosen target for {family} {index}.",
            })
    return rows


def test_exact_28_train_ids_metadata_and_order_frozen():
    original = example_rows()
    frozen = deepcopy(original)
    modified = exp.corrected_training_rows(original)
    assert original == frozen
    assert len(modified) == 28
    assert [r["id"] for r in modified] == [r["id"] for r in original]
    assert Counter(r["semantic_family"] for r in modified) == {x: 4 for x in exp.TARGETS}
    assert all(a["chosen"] != b["chosen"] for a, b in zip(original, modified))
    for first, second in zip(original, modified):
        assert {k: v for k, v in first.items() if k != "chosen"} == {
            k: v for k, v in second.items() if k != "chosen"
        }
    assert exp.chosen_digest(original) != exp.chosen_digest(modified)
    assert exp.corrected_training_rows(original) == modified


def test_all_corrected_targets_are_substantive_and_no_meta_filler():
    corrected = exp.corrected_training_rows(example_rows())
    for row in corrected:
        answer = row["chosen"].lower()
        assert len(answer.split()) >= 19
        assert "gradient" in answer
        assert not any(phrase in answer for phrase in exp.PROHIBITED_FILLER)
        if row["semantic_family"] == "norm_threshold":
            assert any(word in answer for word in ("norm", "threshold"))
            assert any(word in answer for word in ("exceed", "above", "oversized"))
        if row["semantic_family"] == "learning_rate_comparison":
            assert "learning rate" in answer
        if row["semantic_family"] == "false_premise_correction":
            assert any(x in answer for x in ("no.", "premise is backwards", "no:", "do not"))


def test_collision_question_gets_contrast_without_prompt_edit():
    rows = example_rows()
    for row in rows:
        if row["semantic_family"] == "optimizer_stability":
            row["question_class"] = "collision_comparison"
            row["collision_route"] = "overfitting"
            break
    corrected = exp.corrected_training_rows(rows)
    entry = next(r for r in corrected if r["question_class"] == "collision_comparison")
    assert "overfitting" in entry["chosen"].lower()
    assert "gradient" in entry["chosen"].lower()
    assert entry["prompt"] == "Original authored prompt"


def test_bad_training_rows_are_rejected():
    original = example_rows()
    with pytest.raises(ValueError, match="28 unique"):
        exp.corrected_training_rows(original[:-1])
    copy_rows = deepcopy(original)
    copy_rows[-1]["id"] = copy_rows[-2]["id"]
    with pytest.raises(ValueError, match="28 unique"):
        exp.corrected_training_rows(copy_rows)
    changed_route = deepcopy(original)
    changed_route[0]["route"] = "dropout"
    with pytest.raises(ValueError, match="gradient-clipping"):
        exp.corrected_training_rows(changed_route)


def fake_result(*, nll=1.25, top1=0.78, eos_p=0.92, repetitions=1, rubric=13, batches=None):
    return {
        "batch_sequence_ids": batches or ["r0", "r4", "r1"],
        "final_generalization": {
            "holdout_reference_nll": nll,
            "holdout_reference_top1": top1,
            "first_sentence_keyword_rubric_success_rate": 0.1875,
            "mean_eos_eot_p_at_reference_boundary": eos_p,
            "mean_trigram_repetition_rate": 0.08,
            "high_repetition_count": repetitions,
            "full_answer_rubric_success_count": rubric,
        },
    }


def test_control_reproduction_gates_compare_stored_s3_numbers():
    exact = fake_result(
        nll=exp.HISTORICAL_EXPLICIT_EOS["holdout_reference_nll"],
        top1=exp.HISTORICAL_EXPLICIT_EOS["holdout_reference_top1"],
        eos_p=exp.HISTORICAL_EXPLICIT_EOS["mean_eos_eot_p_at_reference_boundary"],
    )
    assert all(exp.control_reproduced(exact).values())
    bad = fake_result(nll=2.0, eos_p=0.15)
    assert not all(exp.control_reproduced(bad).values())


def test_ablation_checks_preserve_eos_and_require_manual_semantic_verification():
    before = fake_result()
    after = fake_result(nll=1.27, top1=0.782, eos_p=0.91, repetitions=1, rubric=21)
    decision = exp.comparison_decision(before, after)
    assert decision["all_automatic_retention_checks_passed"]
    assert decision["manual_factual_adequacy_adjudication"] == "pending"
    assert decision["checkpoint_saved"] is False
    assert decision["promotion_attempted"] is False
    assert decision["not_blind_holdout_warning"]
    worsened = fake_result(nll=1.57, eos_p=0.2, repetitions=10)
    assert not exp.comparison_decision(before, worsened)["all_automatic_retention_checks_passed"]
    after["batch_sequence_ids"] = ["r3", "r2"]
    with pytest.raises(RuntimeError, match="Batch row ID order"):
        exp.comparison_decision(before, after)


def test_architecture_and_eos_controls_are_reused():
    assert exp.EOS_ID == exp.base.EXPECTED_SPECIAL_IDS["<eos>"] == 3
    assert exp.eos.ARMS == ("no_eos", "explicit_eos")
    assert exp.eos.PEAK_LR == 2e-5
    assert exp.eos.TOTAL_UPDATES == 40
    assert exp.eos.GRAD_CLIP == 0.5
    assert exp.ARMS == ("original_eos_control", "corrected_eos")
