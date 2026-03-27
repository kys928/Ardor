"""Tiny usage snippet for decoder backends in training/probe scripts."""

from pathlib import Path

import torch

from Cerebrum.Cortex.backends.factory import load_backend


def run_native_example():
    backend = load_backend(
        model_path="./Models/native_decoder.pt",
        tokenizer_path="./tokenizer.json",
        device="cpu",
        repo_root=Path("."),
        backend_type="native",
    )
    tokenizer = backend.get_tokenizer()
    sample_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    logits = backend.forward_logits(sample_ids)
    print("native", tokenizer is not None, tuple(logits.shape), backend.describe().get("backend_type"))


def run_hf_example():
    backend = load_backend(
        model_path="./tiny-hf-model-dir",
        tokenizer_path=None,
        device="cpu",
        repo_root=Path("."),
        backend_type="hf",
    )
    tokenizer = backend.get_tokenizer()
    sample_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(sample_ids)
    logits = backend.forward_logits(sample_ids, attention_mask=attention_mask)
    print("hf", tokenizer is not None, tuple(logits.shape), backend.describe().get("backend_type"))


if __name__ == "__main__":
    # Intended as a reference snippet; paths above must be replaced with real local artifacts.
    pass
