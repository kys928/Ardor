from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from backends.base import ModelBackend


class HFCausalLMBackend(ModelBackend):
    def __init__(self, model_path: str, device: str, tokenizer_path: Optional[str] = None):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError("transformers is required for HFCausalLMBackend") from e

        self.device = device
        self.model_path = model_path
        self.tokenizer_path_hint = tokenizer_path
        tok_source = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
        self._cfg = getattr(self.model, "config", None)
        self._tokenizer_same_source = bool((tokenizer_path is None) or (tokenizer_path == model_path))

    def _extract_logits(self, out: Any) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "logits") and isinstance(out.logits, torch.Tensor):
            return out.logits
        if isinstance(out, dict) and isinstance(out.get("logits"), torch.Tensor):
            return out["logits"]
        if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            return out[0]
        raise TypeError(f"HF backend forward produced unsupported output type: {type(out)!r}")

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        kwargs: Dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        with torch.no_grad():
            out = self.model(**kwargs)
        return self._extract_logits(out)

    def generate(self, prompt: str, **decode_cfg: Any) -> str:
        max_new_tokens = int(decode_cfg.get("max_new_tokens", 120))
        temperature = float(decode_cfg.get("temperature", 0.7))
        top_p = float(decode_cfg.get("top_p", 0.9))
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=max(temperature, 1e-4),
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

    def encode_text(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=min(self.get_context_len() or 1024, 1024)).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            hs = out.hidden_states[-1]
            pooled = hs.mean(dim=1)
            pooled = F.normalize(pooled, dim=-1)
        return pooled

    def get_vocab_size(self) -> int:
        return int(getattr(self._cfg, "vocab_size", 0) or 0)

    def get_context_len(self) -> Optional[int]:
        value = int(getattr(self._cfg, "max_position_embeddings", 0) or getattr(self._cfg, "max_sequence_length", 0) or 0)
        return value or None

    def get_hidden_size(self) -> Optional[int]:
        value = int(getattr(self._cfg, "hidden_size", 0) or getattr(self._cfg, "n_embd", 0) or 0)
        return value or None

    def get_device(self) -> torch.device | str:
        return self.device

    def get_tokenizer(self):
        return self.tokenizer

    def describe(self) -> Dict[str, Any]:
        c = self._cfg.to_dict() if self._cfg is not None and hasattr(self._cfg, "to_dict") else {}
        try:
            dtype = str(next(self.model.parameters()).dtype)
        except StopIteration:
            dtype = "unknown"
        return {
            "backend_type": "hf",
            "arch": type(self.model).__name__,
            "model_type": c.get("model_type", type(self.model).__name__),
            "vocab": self.get_vocab_size(),
            "hidden": self.get_hidden_size(),
            "max_len": self.get_context_len(),
            "layers": c.get("num_hidden_layers"),
            "heads": c.get("num_attention_heads"),
            "hf_config": c,
            "mismatch": {"missing": [], "unexpected": []},
            "strict_loaded": True,
            "partial_loaded": False,
            "missing_keys": [],
            "unexpected_keys": [],
            "checkpoint_path": self.model_path,
            "tokenizer_path": self.tokenizer_path_hint,
            "tokenizer_and_model_same_source": self._tokenizer_same_source,
            "device": str(self.device),
            "dtype": dtype,
        }

    def unwrap_model(self):
        return self.model

    def tokenizer_path(self) -> Optional[str]:
        return self.tokenizer_path_hint

    def supports_retrieval(self) -> bool:
        return True

    def supports_hidden_state_export(self) -> bool:
        return True

    def close(self) -> None:
        return None
