from __future__ import annotations

import pytest

from Erratum import ardor_v14a8_lr_ablation as exp


def fake_result(name: str, nll_gain: float, top1_gain: float, learned: bool) -> dict:
    return {
        "arm": name,
        "exact_nll_gain": nll_gain,
        "exact_top1_gain": top1_gain,
        "token_learning_supported": learned,
        "final_exact_nll": (5.196875697090512 if name == "lr_2e-8" else 5.228102450522165 - nll_gain),
        "final_exact_top1": (0.27301587507365244 if name == "lr_2e-8" else 0.2666666661463087 + top1_gain),
    }


def test_exact_v14a7_pulse_controls_unchanged():
    assert exp.PULSE_UPDATES == 40
    assert exp.PROBE_UPDATES == (0, 1, 5, 10, 20, 30, 40)
    assert exp.V14A6_LR_HORIZON == 100
    assert exp.FIXED_GRAD_CLIP == 0.5


def test_lr_is_single_changing_arm_dimension():
    assert exp.ARMS == (
        {"name": "lr_2e-8", "lr": 2e-8},
        {"name": "lr_2e-7", "lr": 2e-7},
        {"name": "lr_2e-6", "lr": 2e-6},
        {"name": "lr_2e-5", "lr": 2e-5},
    )


def test_larger_lr_restores_token_learning():
    result = exp.build_decision([
        fake_result("lr_2e-8", 0.03122675, 0.0063492, False),
        fake_result("lr_2e-7", 0.08, 0.03, True),
        fake_result("lr_2e-6", 0.3, 0.08, True),
        fake_result("lr_2e-5", 0.5, 0.09, True),
    ])
    assert result["primary_interpretation"] == "lr_scale_bottleneck_supported"
    assert result["recovered_higher_lr_arms"] == ["lr_2e-7", "lr_2e-6", "lr_2e-5"]
    assert result["best_nll_gain_arm"] == "lr_2e-5"


def test_lr_scale_not_sufficient_when_every_arm_fails():
    result = exp.build_decision([
        fake_result("lr_2e-8", 0.03122675, 0.0063492, False),
        fake_result("lr_2e-7", 0.04, 0.007, False),
        fake_result("lr_2e-6", 0.045, 0.008, False),
        fake_result("lr_2e-5", 0.048, 0.009, False),
    ])
    assert result["primary_interpretation"] == "lr_scale_alone_did_not_restore_learning"


def test_guard_rejects_nonreproducing_v14a7_baseline():
    rows = [
        fake_result("lr_2e-8", 0.03122675, 0.0063492, False),
        fake_result("lr_2e-7", 0.04, 0.007, False),
        fake_result("lr_2e-6", 0.045, 0.008, False),
        fake_result("lr_2e-5", 0.048, 0.009, False),
    ]
    rows[0]["final_exact_nll"] = 5.3
    with pytest.raises(RuntimeError, match="did not reproduce v14a7"):
        exp.build_decision(rows)
