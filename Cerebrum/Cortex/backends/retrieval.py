from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backends.base import ModelBackend


class RetrievalBackend:
    parietal_dim: int = 384

    def __init__(self, model_backend: ModelBackend, device: str, max_items: int = 2000):
        self.backend = model_backend
        self.device = device
        self.max_items = int(max_items)
        self.index_texts: List[str] = []
        self.index_emb: Optional[torch.Tensor] = None
        hidden = max(1, int(self.backend.hidden_size()))
        self.to_generation_space = nn.Linear(self.parietal_dim, hidden, bias=False).to(device)
        self._to_parietal_fallback: Optional[nn.Linear] = None

    def _project_to_parietal(self, vec: torch.Tensor) -> torch.Tensor:
        if vec.ndim == 1:
            vec = vec.unsqueeze(0)
        if vec.shape[-1] == self.parietal_dim:
            return F.normalize(vec, dim=-1)
        if self._to_parietal_fallback is None or self._to_parietal_fallback.in_features != int(vec.shape[-1]):
            self._to_parietal_fallback = nn.Linear(int(vec.shape[-1]), self.parietal_dim, bias=False).to(self.device)
        return F.normalize(self._to_parietal_fallback(vec.to(self.device)), dim=-1)

    def encode(self, text: str) -> torch.Tensor:
        vec = self.backend.encode_text(text)
        return self._project_to_parietal(vec)

    def build_index(self, texts: List[str]) -> None:
        self.index_texts = list(texts)[-self.max_items:]
        if not self.index_texts:
            self.index_emb = None
            return
        embs = [self.encode(t) for t in self.index_texts]
        self.index_emb = torch.cat(embs, dim=0)

    def topk(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if self.index_emb is None or len(self.index_texts) == 0:
            return []
        q = self.encode(query)
        sims = (q @ self.index_emb.T).squeeze(0)
        vals, idx = torch.topk(sims, k=min(k, sims.numel()))
        out: List[Dict[str, Any]] = []
        for j, i in enumerate(idx.tolist()):
            out.append({"trace": self.index_texts[i][:220], "score": vals[j].detach().item()})
        return out

    def project_to_generation_space(self, x: torch.Tensor) -> torch.Tensor:
        y = self.to_generation_space(x.to(self.device))
        return F.normalize(y, dim=-1)


def load_retrieval_backend(model_backend: ModelBackend, device: str, enabled: bool) -> Optional[RetrievalBackend]:
    if not enabled:
        return None
    return RetrievalBackend(model_backend, device=device)
