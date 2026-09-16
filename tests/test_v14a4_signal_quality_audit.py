from __future__ import annotations

import base64
import json

import pytest

from Erratum.ardor_v14a4_signal_quality_audit import diagnosis_for_failure, similarity
import scripts.runpod_worker as worker


def test_similarity_prefers_semantically_matching_phrase():
    prompt = "Why does gradient clipping help when gradients explode?"
    close = "How can gradient clipping stabilize exploding gradients?"
    far = "What does a tokenizer vocabulary contain?"
    assert similarity(prompt, close) > similarity(prompt, far)


def test_diagnosis_defaults_to_model_learning_when_signal_is_clean_and_exposed():
    profile = {
        "actual_u100_behavior_exposures": 8,
        "family_signal_rate": 1.0,
        "route_signal_rate": 1.0,
        "chosen_diversity": 0.9,
        "generic_answer_rate": 0.0,
        "actual_u100_replay_exposures_same_route": 8,
        "legacy_replay_conflict_rate": 0.0,
        "legacy_replay_family_dilution_score": 0.1,
    }
    matches = [{"prompt_similarity": 0.7, "reference_answer_similarity": 0.6, "collision_route": None}]
    diagnosis, scores, reasons = diagnosis_for_failure(profile, matches, matches)
    assert diagnosis == "model_learning_failure"
    assert scores["coverage_failure"] < 0.55
    assert scores["supervision_failure"] < 0.55
    assert scores["collision_failure"] < 0.55
    assert reasons


def test_diagnosis_detects_low_coverage():
    profile = {
        "actual_u100_behavior_exposures": 2,
        "family_signal_rate": 1.0,
        "route_signal_rate": 1.0,
        "chosen_diversity": 0.9,
        "generic_answer_rate": 0.0,
        "actual_u100_replay_exposures_same_route": 4,
        "legacy_replay_conflict_rate": 0.0,
        "legacy_replay_family_dilution_score": 0.0,
    }
    matches = [{"prompt_similarity": 0.15, "reference_answer_similarity": 0.12, "collision_route": None}]
    diagnosis, _, _ = diagnosis_for_failure(profile, matches, [])
    assert diagnosis == "coverage_failure"


def _job_b64(task: dict) -> str:
    payload = {"id": "signal-audit-test", "task": task}
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def test_signal_quality_runner_is_fixed_purpose(monkeypatch, tmp_path):
    assert "v14a4_signal_quality_audit" in worker.ALLOWED_RUNNERS
    assert "v14a4_signal_quality_audit" in worker.FIXED_PURPOSE_RUNNERS
    monkeypatch.setattr(worker, "CONTROL_ROOT", tmp_path)
    monkeypatch.setenv(
        "ARDOR_JOB_B64",
        _job_b64({"runner": "v14a4_signal_quality_audit", "unexpected": True}),
    )
    with pytest.raises(ValueError, match="accepts no task fields beyond runner"):
        worker.run()
