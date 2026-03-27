import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORTEX = ROOT / "Cerebrum" / "Cortex"
LANG = ROOT / "Cerebrum" / "LanguageProcessing"
for path in (str(ROOT), str(CORTEX), str(LANG)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from Cerebrum.Cortex.ardor_config import ArdorConfig
from Cerebrum.Cortex.backends.factory import load_backend
from Cerebrum.Cortex.broca_decoder import ArdorDecoder
import Cerebrum.Cortex.prefrontal_cortex as pfc


class _StubBackend:
    def __init__(self, backend_type: str = "native"):
        self._backend_type = backend_type

    def describe(self):
        return {
            "backend_type": self._backend_type,
            "layers": 2,
            "heads": 2,
            "hidden": 8,
            "max_len": 16,
            "vocab": 10,
            "mismatch": {"missing": [], "unexpected": []},
        }

    def get_tokenizer(self):
        return object()

    def tokenizer_path(self):
        return None

    def get_vocab_size(self):
        return 10

    def unwrap_model(self):
        return object()

    def forward_logits(self, input_ids, attention_mask=None):
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], 10)

    def encode_text(self, text):
        return torch.randn(1, 8)

    def generate(self, prompt, **decode_cfg):
        return "ok"

    def get_hidden_size(self):
        return 8



def _write_native_tokenizer(path: Path, vocab_size: int) -> str:
    vocab = {f"t{i}": i for i in range(vocab_size)}
    vocab["<unk>"] = vocab_size
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(path))
    return str(path)


def _build_native_checkpoint(path: Path, vocab_size: int = 32, hidden_size: int = 16) -> ArdorDecoder:
    cfg = ArdorConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=2,
        n_heads=4,
        max_len=32,
        dropout=0.1,
    )
    model = ArdorDecoder(cfg)
    torch.save(model.state_dict(), str(path))
    return model


def _build_tiny_hf_local_dir(path: Path):
    tr = pytest.importorskip("transformers")
    path.mkdir(parents=True, exist_ok=True)

    vocab_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"] + [f"t{i}" for i in range(28)]
    vocab = {tok: i for i, tok in enumerate(vocab_tokens)}
    tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer_obj.pre_tokenizer = Whitespace()

    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )
    fast_tok.save_pretrained(str(path))

    cfg = GPT2Config(
        vocab_size=len(vocab),
        n_embd=16,
        n_layer=2,
        n_head=4,
        n_positions=32,
        bos_token_id=vocab["[BOS]"],
        eos_token_id=vocab["[EOS]"],
    )
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(str(path))

    # ensure config exists for auto detection messaging assertions
    assert (path / "config.json").exists()
    return tr


def test_ardorcore_boot_path_uses_backend_factory(monkeypatch):
    calls = {"count": 0}

    def _fake_loader(**kwargs):
        calls["count"] += 1
        return _StubBackend("native")

    monkeypatch.setattr(pfc, "load_backend", _fake_loader)
    monkeypatch.setattr(pfc, "load_retrieval_backend", lambda *args, **kwargs: None)

    core = pfc.ArdorCore(
        model_path="dummy.pt",
        tokenizer_path=None,
        device="cpu",
        enable_retrieval=False,
        aeternum=object(),
    )

    assert calls["count"] == 1
    assert core.backend is not None
    assert core.schema["backend_type"] == "native"


def test_unified_logits_contract_native_and_hf(tmp_path):
    native_ckpt = tmp_path / "native.pt"
    _build_native_checkpoint(native_ckpt)
    native_tok = tmp_path / "native_tokenizer.json"
    _write_native_tokenizer(native_tok, vocab_size=32)

    native_backend = load_backend(
        model_path=str(native_ckpt),
        tokenizer_path=str(native_tok),
        device="cpu",
        repo_root=ROOT,
        backend_type="native",
    )
    input_ids = torch.randint(0, 32, (2, 6), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    native_logits = native_backend.forward_logits(input_ids, attention_mask=attention_mask)
    assert tuple(native_logits.shape) == (2, 6, 32)

    hf_dir = tmp_path / "hf_local"
    _build_tiny_hf_local_dir(hf_dir)
    hf_backend = load_backend(
        model_path=str(hf_dir),
        tokenizer_path=None,
        device="cpu",
        repo_root=ROOT,
        backend_type="hf",
    )
    hf_logits = hf_backend.forward_logits(input_ids, attention_mask=attention_mask)
    assert isinstance(hf_logits, torch.Tensor)
    assert hf_logits.shape[0] == 2
    assert hf_logits.shape[1] == 6


def test_training_probe_style_smoke_native_and_hf(tmp_path):
    native_ckpt = tmp_path / "native.pt"
    _build_native_checkpoint(native_ckpt)
    native_tok = tmp_path / "native_tokenizer.json"
    _write_native_tokenizer(native_tok, vocab_size=32)

    hf_dir = tmp_path / "hf_local"
    _build_tiny_hf_local_dir(hf_dir)

    for model_path, tokenizer_path, backend_type in (
        (str(native_ckpt), str(native_tok), "native"),
        (str(hf_dir), None, "hf"),
    ):
        backend = load_backend(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device="cpu",
            repo_root=ROOT,
            backend_type=backend_type,
        )
        tokenizer = backend.get_tokenizer()
        logits = backend.forward_logits(torch.tensor([[0, 1, 2]], dtype=torch.long), attention_mask=torch.ones(1, 3, dtype=torch.long))
        model = backend.unwrap_model()

        assert tokenizer is not None
        assert isinstance(logits, torch.Tensor)
        assert model is not None


def test_native_strict_vs_partial_load(tmp_path):
    ckpt = tmp_path / "native.pt"
    model = _build_native_checkpoint(ckpt)
    sd = model.state_dict()
    sd.pop("norm.weight")
    broken_ckpt = tmp_path / "native_broken.pt"
    torch.save(sd, broken_ckpt)
    tok_path = tmp_path / "tok.json"
    _write_native_tokenizer(tok_path, vocab_size=32)

    with pytest.raises(RuntimeError, match="Strict native checkpoint load failed"):
        load_backend(
            model_path=str(broken_ckpt),
            tokenizer_path=str(tok_path),
            device="cpu",
            repo_root=ROOT,
            backend_type="native",
            allow_partial_load=False,
        )

    backend = load_backend(
        model_path=str(broken_ckpt),
        tokenizer_path=str(tok_path),
        device="cpu",
        repo_root=ROOT,
        backend_type="native",
        allow_partial_load=True,
    )
    desc = backend.describe()
    assert desc["strict_loaded"] is False
    assert desc["partial_loaded"] is True
    assert "norm.weight" in desc["missing_keys"]
    assert desc["checkpoint_path"].endswith("native_broken.pt")


def test_backend_auto_detection_and_ambiguous_failure(tmp_path):
    native_ckpt = tmp_path / "native.pt"
    _build_native_checkpoint(native_ckpt)
    native_tok = tmp_path / "native_tokenizer.json"
    _write_native_tokenizer(native_tok, vocab_size=32)

    hf_dir = tmp_path / "hf_local"
    _build_tiny_hf_local_dir(hf_dir)

    native_backend = load_backend(
        model_path=str(native_ckpt),
        tokenizer_path=str(native_tok),
        device="cpu",
        repo_root=ROOT,
        backend_type="auto",
    )
    hf_backend = load_backend(
        model_path=str(hf_dir),
        tokenizer_path=None,
        device="cpu",
        repo_root=ROOT,
        backend_type="auto",
    )
    assert native_backend.describe()["backend_type"] == "native"
    assert hf_backend.describe()["backend_type"] == "hf"

    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir(parents=True, exist_ok=True)
    (ambiguous / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    with pytest.raises(ValueError, match="Could not detect backend"):
        load_backend(
            model_path=str(ambiguous),
            tokenizer_path=None,
            device="cpu",
            repo_root=ROOT,
            backend_type="auto",
        )


def test_legacy_metadata_compatibility_via_ardorcore(monkeypatch):
    monkeypatch.setattr(pfc, "load_backend", lambda **kwargs: _StubBackend("native"))
    monkeypatch.setattr(pfc, "load_retrieval_backend", lambda *args, **kwargs: None)

    core = pfc.ArdorCore(
        model_path="dummy.pt",
        tokenizer_path=None,
        device="cpu",
        enable_retrieval=False,
        aeternum=object(),
    )

    assert core.layers == 2
    assert core.heads == 2
    assert core.hidden == 8
    assert core.model_ctx_len == 16
    assert core.vocab_size == 10


def test_native_only_guardrail_for_hf_backend(monkeypatch):
    monkeypatch.setattr(pfc, "load_backend", lambda **kwargs: _StubBackend("hf"))
    monkeypatch.setattr(pfc, "load_retrieval_backend", lambda *args, **kwargs: None)

    core = pfc.ArdorCore(
        model_path="dummy_hf_dir",
        tokenizer_path=None,
        device="cpu",
        enable_retrieval=False,
        aeternum=object(),
    )

    with pytest.raises(RuntimeError, match="only available for native backend"):
        core._native_token_embed_mean("hello")
