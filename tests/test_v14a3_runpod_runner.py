from __future__ import annotations

from pathlib import Path

import pytest

from scripts.runpod_control import FIXED_PURPOSE_RUNNERS, validate_task
from scripts.runpod_worker import ALLOWED_RUNNERS, V14A3_ENTRY


def test_control_plane_accepts_only_fixed_v14a3_runner_shape():
    assert "v14a3_behavior_first" in FIXED_PURPOSE_RUNNERS
    validate_task({"runner": "v14a3_behavior_first"})
    with pytest.raises(ValueError):
        validate_task({"runner": "v14a3_behavior_first", "lr": 1e-3})


def test_worker_pins_v14a3_entry_and_parent_sha_verification():
    assert "v14a3_behavior_first" in ALLOWED_RUNNERS
    assert V14A3_ENTRY.name == "ardor_v14a3_behavior_first_trainer.py"
    assert V14A3_ENTRY.parent.name == "Erratum"
    source = Path(__file__).resolve().parents[1].joinpath("scripts/runpod_worker.py").read_text(encoding="utf-8")
    assert 'runner == "v14a3_behavior_first"' in source
    assert '"--verify-parent-sha256"' in source
