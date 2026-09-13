#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


worker_path = ROOT / "scripts" / "runpod_worker.py"
worker = worker_path.read_text(encoding="utf-8")
worker = replace_once(
    worker,
    'V14A3_ENTRY = REPO_ROOT / "Erratum" / "ardor_v14a3_behavior_first_trainer.py"\nALLOWED_STAGES',
    'V14A3_ENTRY = REPO_ROOT / "Erratum" / "ardor_v14a3_behavior_first_trainer.py"\n'
    'V14A4_ENTRY = REPO_ROOT / "Erratum" / "ardor_v14a4_family_balanced_trainer.py"\n'
    'V14A4_PROBE_OUTPUT = PERSISTENT_ROOT / "training" / "runs" / "sft_v14a4_family_balanced_semantic_landing_probe_u100"\n'
    'ALLOWED_STAGES',
    "worker v14a4 entry",
)
worker = replace_once(
    worker,
    '    "canonical_eval_v14a2",\n    "v14a3_behavior_first",\n}\nFIXED_PURPOSE_RUNNERS',
    '    "canonical_eval_v14a2",\n    "v14a3_behavior_first",\n    "v14a4_family_probe_u100",\n}\nFIXED_PURPOSE_RUNNERS',
    "worker allowed runner",
)
worker = replace_once(
    worker,
    '    "canonical_eval_v14a2",\n    "v14a3_behavior_first",\n}\n\n\ndef utc_now',
    '    "canonical_eval_v14a2",\n    "v14a3_behavior_first",\n    "v14a4_family_probe_u100",\n}\n\n\ndef utc_now',
    "worker fixed runner",
)
worker = replace_once(
    worker,
    '    elif runner == "v14a3_behavior_first":\n        command = [sys.executable, str(V14A3_ENTRY), "--verify-parent-sha256"]\n    else:',
    '    elif runner == "v14a3_behavior_first":\n'
    '        command = [sys.executable, str(V14A3_ENTRY), "--verify-parent-sha256"]\n'
    '    elif runner == "v14a4_family_probe_u100":\n'
    '        command = [\n'
    '            sys.executable,\n'
    '            str(V14A4_ENTRY),\n'
    '            "--verify-parent-sha256",\n'
    '            "--max-updates", "100",\n'
    '            "--eval-every", "25",\n'
    '            "--eval-updates", "1", "10",\n'
    '            "--no-stop-on-regression",\n'
    '            "--output-dir", str(V14A4_PROBE_OUTPUT),\n'
    '        ]\n'
    '    else:',
    "worker fixed command",
)
worker_path.write_text(worker, encoding="utf-8")

control_path = ROOT / "scripts" / "runpod_control.py"
control = control_path.read_text(encoding="utf-8")
control = replace_once(
    control,
    '    "canonical_eval_v14a2",\n    "v14a3_behavior_first",\n}\n\n\ndef utc_now',
    '    "canonical_eval_v14a2",\n    "v14a3_behavior_first",\n    "v14a4_family_probe_u100",\n}\n\n\ndef utc_now',
    "control fixed runner",
)
control_path.write_text(control, encoding="utf-8")

test_path = ROOT / "tests" / "test_v14a4_runpod_runner.py"
test_path.write_text('''from __future__ import annotations\n\nfrom pathlib import Path\n\nimport pytest\n\nfrom scripts.runpod_control import FIXED_PURPOSE_RUNNERS, validate_task\nfrom scripts.runpod_worker import (\n    ALLOWED_RUNNERS,\n    V14A4_ENTRY,\n    V14A4_PROBE_OUTPUT,\n)\n\n\ndef test_control_plane_locks_v14a4_probe_runner_shape():\n    assert "v14a4_family_probe_u100" in FIXED_PURPOSE_RUNNERS\n    validate_task({"runner": "v14a4_family_probe_u100"})\n    with pytest.raises(ValueError):\n        validate_task({"runner": "v14a4_family_probe_u100", "max_updates": 600})\n    with pytest.raises(ValueError):\n        validate_task({"runner": "v14a4_family_probe_u100", "lr": 1e-3})\n\n\ndef test_worker_pins_v14a4_probe_to_reviewed_u100_command():\n    assert "v14a4_family_probe_u100" in ALLOWED_RUNNERS\n    assert V14A4_ENTRY.name == "ardor_v14a4_family_balanced_trainer.py"\n    assert V14A4_ENTRY.parent.name == "Erratum"\n    assert str(V14A4_PROBE_OUTPUT).endswith("sft_v14a4_family_balanced_semantic_landing_probe_u100")\n\n    source = Path(__file__).resolve().parents[1].joinpath("scripts/runpod_worker.py").read_text(encoding="utf-8")\n    block = source.split('elif runner == "v14a4_family_probe_u100":', 1)[1].split("    else:", 1)[0]\n    assert '"--verify-parent-sha256"' in block\n    assert '"--max-updates", "100"' in block\n    assert '"--eval-every", "25"' in block\n    assert '"--eval-updates", "1", "10"' in block\n    assert '"--no-stop-on-regression"' in block\n    assert 'str(V14A4_PROBE_OUTPUT)' in block\n    assert '"--lr"' not in block\n    assert '"--seed"' not in block\n''', encoding="utf-8")

print("patched locked v14a4 family probe runner")
