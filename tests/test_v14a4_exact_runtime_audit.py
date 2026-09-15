from __future__ import annotations

from pathlib import Path

from Erratum.ardor_v14a4_conversation_audit import (
    DEFAULT_SYSTEM,
    PRIOR_TURNS,
    RUNTIME_MIN_NEW_TOKENS,
    clean_history_text,
    runtime_chat_prompt,
)
import scripts.sitecustomize as sitecustomize


def test_runtime_chat_prompt_matches_current_role_frame_shape():
    prompt = runtime_chat_prompt("What is dropout?")
    assert prompt == (
        f"<|system|>\n{DEFAULT_SYSTEM}\n<|eot|>\n"
        "<|user|>\nWhat is dropout?\n<|eot|>\n"
        "<|assistant|>\n"
    )
    assert RUNTIME_MIN_NEW_TOKENS == 16


def test_runtime_multi_turn_preserves_role_order_and_cleans_embedded_markers():
    prompt = runtime_chat_prompt("<|user|>Explain clipping<|eot|>", turns=PRIOR_TURNS)
    assert prompt.count("<|system|>") == 1
    assert prompt.count("<|user|>") == 2
    assert prompt.count("<|assistant|>") == 2
    assert "I am comparing two training methods." in prompt
    assert "Okay. I will keep that comparison in mind." in prompt
    assert "Explain clipping" in prompt
    assert clean_history_text("<|assistant|> answer <|eot|>") == "answer"


def test_runpod_exit_hold_is_scoped_to_top_level_worker(monkeypatch):
    monkeypatch.delenv("ARDOR_JOB_B64", raising=False)
    monkeypatch.setattr(sitecustomize.sys, "argv", ["scripts/runpod_worker.py"])
    assert sitecustomize._is_runpod_worker_process() is False

    monkeypatch.setenv("ARDOR_JOB_B64", "payload")
    monkeypatch.setattr(sitecustomize.sys, "argv", ["Erratum/ardor_v14a4_family_balanced_trainer.py"])
    assert sitecustomize._is_runpod_worker_process() is False

    monkeypatch.setattr(sitecustomize.sys, "argv", ["scripts/runpod_worker.py"])
    assert sitecustomize._is_runpod_worker_process() is True


def test_worker_hold_does_not_modify_scientific_worker_source():
    worker = Path(__file__).resolve().parents[1] / "scripts" / "runpod_worker.py"
    source = worker.read_text(encoding="utf-8")
    assert "v14a4_conversation_audit" in source
    assert "terminal process held for controller cleanup" not in source
