from __future__ import annotations

from pathlib import Path


def test_control_entry_registers_v14a5_and_routes_to_isolated_worker():
    root = Path(__file__).resolve().parents[1]
    source = root.joinpath("scripts/runpod_control_entry.py").read_text(encoding="utf-8")
    assert 'FIXED_PURPOSE_RUNNERS.add("v14a5_signal_first_u100")' in source
    assert 'str(task.get("runner", "")) == "v14a5_signal_first_u100"' in source
    assert "scripts/runpod_v14a5_worker.py" in source


def test_v14a5_worker_pins_reviewed_u100_command():
    root = Path(__file__).resolve().parents[1]
    source = root.joinpath("scripts/runpod_v14a5_worker.py").read_text(encoding="utf-8")
    assert 'RUNNER = "v14a5_signal_first_u100"' in source
    assert '"ardor_v14a5_signal_first_trainer.py"' in source
    assert '"--verify-parent-sha256"' in source
    assert '"--max-updates", "100"' in source
    assert '"--eval-every", "25"' in source
    assert '"--eval-updates", "1", "10"' in source
    assert '"--no-stop-on-regression"' in source
    assert '"--lr"' not in source
    assert '"--seed"' not in source
