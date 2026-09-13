from pathlib import Path

# The current runtime's normal generate_text() default persona is ROLE_PRIMER, not
# _build_chat_prompt's empty-persona fallback. Make the audit mirror that exact default.
audit_path = Path("Erratum/ardor_v14a4_conversation_audit.py")
audit = audit_path.read_text(encoding="utf-8")
old_system = 'DEFAULT_SYSTEM = "You are Ardor. Stay in-context. Be helpful. Speak naturally."'
new_system = (
    'DEFAULT_SYSTEM = "Hi, You are Ardor. Answer my questions cleanly. Respond to me in friendly manner. '
    'Prefer 3-6 sentences at most, however you can extend it if you deem necessary. Always start the conversation."'
)
if old_system not in audit:
    raise RuntimeError("Generated audit fallback-system anchor missing")
audit_path.write_text(audit.replace(old_system, new_system, 1), encoding="utf-8")

# Rewrite the tiny focused test directly so string-escaping in the staging generator
# cannot turn a real newline into a literal backslash-n expectation.
test_path = Path("tests/test_conversation_audit.py")
test_path.write_text('''from __future__ import annotations

import base64
import json

import pytest

from Erratum.ardor_v14a4_conversation_audit import DEFAULT_SYSTEM, runtime_chat_prompt
from Erratum.v14a4_diagnostics import clean_eval_prompt
import scripts.runpod_worker as worker


def test_clean_eval_prompt_exactly_matches_v14a4_training_answer_boundary():
    expected = "What is gradient clipping?" + chr(10) + "-"
    assert clean_eval_prompt({"prompt": "What is gradient clipping?"}) == expected
    assert clean_eval_prompt({"prompt": expected}) == expected
    assert clean_eval_prompt({"prompt": "What is gradient clipping?   "}) == expected


def test_runtime_chat_prompt_matches_current_role_marked_shape():
    assert DEFAULT_SYSTEM == (
        "Hi, You are Ardor. Answer my questions cleanly. Respond to me in friendly manner. "
        "Prefer 3-6 sentences at most, however you can extend it if you deem necessary. "
        "Always start the conversation."
    )
    nl = chr(10)
    prompt = runtime_chat_prompt("What is dropout?")
    assert prompt == (
        f"<|system|>{nl}{DEFAULT_SYSTEM}{nl}<|eot|>{nl}"
        f"<|user|>{nl}What is dropout?{nl}<|eot|>{nl}<|assistant|>{nl}"
    )

    multi = runtime_chat_prompt(
        "What does it refer to?",
        turns=(("user", "Remember gradient clipping."), ("assistant", "I will.")),
    )
    assert f"<|user|>{nl}Remember gradient clipping.{nl}<|eot|>{nl}" in multi
    assert f"<|assistant|>{nl}I will.{nl}<|eot|>{nl}" in multi
    assert multi.endswith(f"<|user|>{nl}What does it refer to?{nl}<|eot|>{nl}<|assistant|>{nl}")


def _job_b64(task: dict) -> str:
    payload = {"id": "test-audit", "task": task}
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def test_conversation_audit_runner_is_fixed_purpose(monkeypatch, tmp_path):
    assert "v14a4_conversation_audit" in worker.ALLOWED_RUNNERS
    assert "v14a4_conversation_audit" in worker.FIXED_PURPOSE_RUNNERS
    monkeypatch.setattr(worker, "CONTROL_ROOT", tmp_path)
    monkeypatch.setenv("ARDOR_JOB_B64", _job_b64({"runner": "v14a4_conversation_audit", "lr": 1e-3}))
    with pytest.raises(ValueError, match="accepts no task fields beyond runner"):
        worker.run()
''', encoding="utf-8")
