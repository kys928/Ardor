from __future__ import annotations

from Erratum import ardor_v14a7_clip_ablation as v14a7


def fake_result(arm: str, learned: bool, nll_gain: float, top1_gain: float):
    return {
        "arm": arm,
        "token_learning_supported": learned,
        "exact_nll_gain": nll_gain,
        "exact_top1_gain": top1_gain,
    }


def test_clip_ablation_arms_are_single_factor_ladder():
    assert [row["name"] for row in v14a7.ARMS] == [
        "clip_0p5",
        "clip_5",
        "clip_50",
        "no_clip",
    ]
    assert [row["max_norm"] for row in v14a7.ARMS] == [0.5, 5.0, 50.0, None]


def test_decision_supports_clip_bottleneck_when_looser_arm_recovers():
    decision = v14a7.build_decision(
        [
            fake_result("clip_0p5", False, 0.03, 0.006),
            fake_result("clip_5", True, 0.08, 0.025),
            fake_result("clip_50", True, 0.10, 0.030),
            fake_result("no_clip", True, 0.11, 0.031),
        ]
    )
    assert decision["primary_interpretation"] == "clip_0p5_update_bottleneck_supported"
    assert decision["recovered_looser_arms"] == ["clip_5", "clip_50", "no_clip"]


def test_decision_rejects_clip_as_primary_when_all_arms_fail():
    decision = v14a7.build_decision(
        [
            fake_result("clip_0p5", False, 0.03, 0.006),
            fake_result("clip_5", False, 0.035, 0.007),
            fake_result("clip_50", False, 0.036, 0.008),
            fake_result("no_clip", False, 0.037, 0.008),
        ]
    )
    assert (
        decision["primary_interpretation"]
        == "clip_threshold_not_primary_bottleneck_supported"
    )


def test_v14a7_pulse_matches_v14a6_probe_schedule():
    assert v14a7.PULSE_UPDATES == 40
    assert v14a7.PROBE_UPDATES == (0, 1, 5, 10, 20, 30, 40)
