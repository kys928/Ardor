from __future__ import annotations

from pathlib import Path


def test_control_entry_registers_v14a5_v14a6_v14a7_v14a8_with_cuda():
    root = Path(__file__).resolve().parents[1]
    source = root.joinpath("scripts/runpod_control_entry.py").read_text(encoding="utf-8")
    assert 'FIXED_PURPOSE_RUNNERS.add("v14a5_signal_first_u100")' in source
    assert 'FIXED_PURPOSE_RUNNERS.add("v14a6_learning_dynamics_probe")' in source
    assert 'FIXED_PURPOSE_RUNNERS.add("v14a7_clip_ablation")' in source
    assert 'FIXED_PURPOSE_RUNNERS.add("v14a8_lr_ablation")' in source
    assert 'runner == "v14a5_signal_first_u100"' in source
    assert 'runner == "v14a6_learning_dynamics_probe"' in source
    assert 'runner == "v14a7_clip_ablation"' in source
    assert 'runner == "v14a8_lr_ablation"' in source
    assert "https://download.pytorch.org/whl/cu128" in source
    assert "torch==2.7.1" in source
    assert "torch.version.cuda == '12.8'" in source
    assert "torch.cuda.is_available()" in source
    assert "scripts/runpod_v14a5_worker.py" in source
    assert "scripts/runpod_v14a6_worker.py" in source
    assert "scripts/runpod_v14a7_worker.py" in source
    assert "scripts/runpod_v14a8_worker.py" in source
    assert "uv run --frozen python scripts/runpod_v14a5_worker.py" not in source
    assert "uv run --frozen python scripts/runpod_v14a6_worker.py" not in source
    assert "uv run --frozen python scripts/runpod_v14a7_worker.py" not in source


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


def test_v14a6_worker_pins_reviewed_pulse_chase_command():
    root = Path(__file__).resolve().parents[1]
    source = root.joinpath("scripts/runpod_v14a6_worker.py").read_text(encoding="utf-8")
    assert 'RUNNER = "v14a6_learning_dynamics_probe"' in source
    assert '"ardor_v14a6_learning_dynamics_probe.py"' in source
    assert '"--verify-parent-sha256"' in source
    assert '"v14a6_learning_dynamics_decision.json"' in source
    assert '"--lr"' not in source
    assert '"--seed"' not in source


def test_v14a7_worker_pins_reviewed_clip_ablation_command():
    root = Path(__file__).resolve().parents[1]
    source = root.joinpath("scripts/runpod_v14a7_worker.py").read_text(encoding="utf-8")
    assert 'RUNNER = "v14a7_clip_ablation"' in source
    assert '"ardor_v14a7_clip_ablation.py"' in source
    assert '"--verify-parent-sha256"' in source
    assert '"v14a7_clip_ablation_decision.json"' in source
    assert '"--lr"' not in source
    assert '"--seed"' not in source


def test_v14a8_worker_pins_reviewed_lr_ablation():
    root = Path(__file__).resolve().parents[1]
    source = root.joinpath("scripts/runpod_v14a8_worker.py").read_text(encoding="utf-8")
    assert 'RUNNER = "v14a8_lr_ablation"' in source
    assert '"ardor_v14a8_lr_ablation.py"' in source
    assert '"--verify-parent-sha256"' in source
    assert '"v14a8_lr_ablation_decision.json"' in source
    assert '"--lr"' not in source
    assert '"--seed"' not in source


def test_control_entry_registers_v14a9_fixed_replay_runner():
    root = Path(__file__).resolve().parents[1]
    source = root.joinpath("scripts/runpod_control_entry.py").read_text(encoding="utf-8")
    assert 'FIXED_PURPOSE_RUNNERS.add("v14a9_holdout_generalization")' in source
    assert 'runner == "v14a9_holdout_generalization"' in source
    assert 'scripts/runpod_v14a9_worker.py' in source


def test_v14a9_worker_accepts_no_custom_training_controls():
    root = Path(__file__).resolve().parents[1]
    source = root.joinpath("scripts/runpod_v14a9_worker.py").read_text(encoding="utf-8")
    assert 'RUNNER = "v14a9_holdout_generalization"' in source
    assert 'ardor_v14a9_holdout_generalization_diagnostic.py' in source
    assert 'v14a9_generalization_decision.json' in source
    assert '"--verify-parent-sha256"' in source
    assert '"--seed"' not in source


def test_v14a10_fixed_purpose_runner_registration():
    repo = Path(__file__).resolve().parents[1]
    entry = (repo / "scripts/runpod_control_entry.py").read_text(encoding="utf-8")
    worker = (repo / "scripts/runpod_v14a10_worker.py").read_text(encoding="utf-8")
    assert 'FIXED_PURPOSE_RUNNERS.add("v14a10_eos_supervision_ablation")' in entry
    assert 'runner == "v14a10_eos_supervision_ablation"' in entry
    assert 'scripts/runpod_v14a10_worker.py' in entry
    assert 'RUNNER = "v14a10_eos_supervision_ablation"' in worker
    assert '"ardor_v14a10_eos_supervision_ablation.py"' in worker
    assert 'v14a10_eos_ablation_decision.json' in worker
    assert '"--verify-parent-sha256"' in worker
    assert '"--seed"' not in worker
