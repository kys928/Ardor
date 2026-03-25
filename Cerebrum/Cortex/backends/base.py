from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class ModelBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, **decode_cfg: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def context_len(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def hidden_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def schema(self) -> Dict[str, Any]:
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
