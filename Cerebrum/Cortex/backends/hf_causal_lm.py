from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from backends.base import ModelBackend


class HFCausalLMBackend(ModelBackend):
    def __init__(self, model_path: str, device: str):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError("transformers is required for HFCausalLMBackend") from e

        self.device = device
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
        self._cfg = getattr(self.model, "config", None)

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
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=min(self.context_len() or 1024, 1024)).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            hs = out.hidden_states[-1]
            pooled = hs.mean(dim=1)
            pooled = F.normalize(pooled, dim=-1)
        return pooled

    def vocab_size(self) -> int:
        return int(getattr(self._cfg, "vocab_size", 0) or 0)

    def context_len(self) -> int:
        return int(getattr(self._cfg, "max_position_embeddings", 0) or getattr(self._cfg, "max_sequence_length", 0) or 0)

    def hidden_size(self) -> int:
        return int(getattr(self._cfg, "hidden_size", 0) or getattr(self._cfg, "n_embd", 0) or 0)

    def schema(self) -> Dict[str, Any]:
        c = self._cfg.to_dict() if self._cfg is not None and hasattr(self._cfg, "to_dict") else {}
        return {
            "arch": type(self.model).__name__,
            "vocab": self.vocab_size(),
            "hidden": self.hidden_size(),
            "max_len": self.context_len(),
            "layers": c.get("num_hidden_layers"),
            "heads": c.get("num_attention_heads"),
            "hf_config": c,
            "mismatch": {"missing": [], "unexpected": []},
        }

    def tokenizer_path(self) -> Optional[str]:
        return None

    def supports_retrieval(self) -> bool:
        return True

    def supports_hidden_state_export(self) -> bool:
        return True

    def close(self) -> None:
        return None
