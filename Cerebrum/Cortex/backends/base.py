from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class ModelBackend(ABC):
    """Unified decoder backend contract used by ArdorCore and training/probe scripts."""

    @abstractmethod
    def forward_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return logits with shape [batch, seq, vocab]."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, **decode_cfg: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_vocab_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_context_len(self) -> Optional[int]:
        raise NotImplementedError

    @abstractmethod
    def get_hidden_size(self) -> Optional[int]:
        raise NotImplementedError

    @abstractmethod
    def get_device(self) -> torch.device | str:
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def unwrap_model(self):
        raise NotImplementedError

    @abstractmethod
    def tokenizer_path(self) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def supports_retrieval(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def supports_hidden_state_export(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    # Backward-compatible aliases used by existing code.
    def vocab_size(self) -> int:
        return self.get_vocab_size()

    def context_len(self) -> Optional[int]:
        return self.get_context_len()

    def hidden_size(self) -> Optional[int]:
        return self.get_hidden_size()

    def schema(self) -> Dict[str, Any]:
        return self.describe()
