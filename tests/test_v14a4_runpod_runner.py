from __future__ import annotations

from pathlib import Path

import pytest

from scripts.runpod_control import FIXED_PURPOSE_RUNNERS, validate_task
from scripts.runpod_worker import (
    ALLOWED_RUNNERS,
    V14A4_ENTRY,
    V14A4_PROBE_OUTPUT,
)


def test_control_plane_locks_v14a4_probe_runner_shape():
    assert "v14a4_family_probe_u100" in FIXED_PURPOSE_RUNNERS
    validate_task({"runner": "v14a4_family_probe_u100"})
    with pytest.raises(ValueError):
        validate_task({"runner": "v14a4_family_probe_u100", "max_updates": 600})
    with pytest.raises(ValueError):
        validate_task({"runner": "v14a4_family_probe_u100", "lr": 1e-3})


def test_worker_pins_v14a4_probe_to_reviewed_u100_command():
    assert "v14a4_family_probe_u100" in ALLOWED_RUNNERS
    assert V14A4_ENTRY.name == "ardor_v14a4_family_balanced_trainer.py"
    assert V14A4_ENTRY.parent.name == "Erratum"
    assert str(V14A4_PROBE_OUTPUT).endswith("sft_v14a4_family_balanced_semantic_landing_probe_u100")

    source = Path(__file__).resolve().parents[1].joinpath("scripts/runpod_worker.py").read_text(encoding="utf-8")
    block = source.split('elif runner == "v14a4_family_probe_u100":', 1)[1].split("    else:", 1)[0]
    assert '"--verify-parent-sha256"' in block
    assert '"--max-updates", "100"' in block
    assert '"--eval-every", "25"' in block
    assert '"--eval-updates", "1", "10"' in block
    assert '"--no-stop-on-regression"' in block
    assert 'str(V14A4_PROBE_OUTPUT)' in block
    assert '"--lr"' not in block
    assert '"--seed"' not in block
