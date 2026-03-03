from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from Cerebrum.Cortex.ardor_config import ArdorConfig


class ParietalMemory:
    """
    Episodic vector store (HIGH-INTEGRITY):
      - Retrieval/index space is FIXED at 384-d (parietal_dim)
      - Provides a projection bridge to decoder hidden space (config.hidden_size)

    Rules:
      - Store/embed USER PROMPTS (or safe summaries) ONLY
      - Inject ONLY safe summary text (never raw assistant output)
    """

    parietal_dim: int = 384

    def __init__(
        self,
        tok: Tokenizer,
        device: str,
        *,
        broca_model: nn.Module,
        encoder: Optional[nn.Module],
        memory_jsonl: Optional[str],
        config: ArdorConfig,
        max_items: int = 2000,
        max_len: int = 192,
        embed_field: str = "prompt",  # "prompt" or "summary"
    ):
        self.tok = tok
        self.device = device
        self.broca = broca_model
        self.encoder = encoder
        self.memory_jsonl = Path(memory_jsonl) if memory_jsonl else None
        self.max_items = int(max_items)
        self.max_len = int(max_len)
        self.embed_field = embed_field if embed_field in ("prompt", "summary") else "prompt"

        # 384 -> decoder hidden bridge
        self.to_decoder_space = nn.Linear(self.parietal_dim, config.hidden_size, bias=False).to(device)

        # Optional hidden -> 384 fallback (only used if encoder is None)
        self._to_parietal_fallback: Optional[nn.Linear] = None
        try:
            broca_hidden = int(getattr(self.broca, "token_embed").weight.shape[1])
            if broca_hidden != self.parietal_dim:
                self._to_parietal_fallback = nn.Linear(broca_hidden, self.parietal_dim, bias=False).to(device)
        except Exception:
            pass

        self.episodes: List[Dict[str, Any]] = []
        self.index_emb: Optional[torch.Tensor] = None

        if self.memory_jsonl and self.memory_jsonl.exists():
            self.rebuild_from_jsonl(self.memory_jsonl, max_items=self.max_items)

    # ──────────────────────────────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────────────────────────────
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encs = [self.tok.encode(t) for t in texts]
        ids_list = [e.ids[: self.max_len] for e in encs]

        B = len(ids_list)
        L = max(1, max(len(x) for x in ids_list))
        pad_id = 0

        ids = torch.full((B, L), pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((B, L), dtype=torch.float32, device=self.device)

        for i, seq in enumerate(ids_list):
            n = len(seq)
            if n:
                ids[i, :n] = torch.tensor(seq, dtype=torch.long, device=self.device)
                mask[i, :n] = 1.0
        return ids, mask

    def _encoder_forward_pooled(self, encoder: nn.Module, ids: torch.Tensor) -> torch.Tensor:
        out = encoder(ids, return_pooled=True)
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            return out[1]
        if isinstance(out, dict) and "pooled" in out:
            return out["pooled"]
        # fallback: mean pool last hidden
        if isinstance(out, torch.Tensor):
            return out.mean(dim=1)
        raise TypeError(f"Unsupported encoder output type: {type(out)}")

    def encode_batch(self, prompts: List[str]) -> torch.Tensor:
        ids, mask = self._tokenize_batch(prompts)
        with torch.no_grad():
            if self.encoder is not None:
                pooled = self._encoder_forward_pooled(self.encoder, ids)
                # If encoder isn't 384-d, project down (rare, but safe)
                if pooled.shape[1] != self.parietal_dim:
                    if self._to_parietal_fallback is None or self._to_parietal_fallback.in_features != pooled.shape[1]:
                        self._to_parietal_fallback = nn.Linear(int(pooled.shape[1]), self.parietal_dim, bias=False).to(self.device)
                    pooled = self._to_parietal_fallback(pooled)
                return F.normalize(pooled, dim=1)

            # Fallback: pooled token embeddings from broca, then (if needed) down-project to 384
            tok_emb = self.broca.token_embed(ids)  # [B,L,H]
            denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            pooled = (tok_emb * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B,H]
            if pooled.shape[1] != self.parietal_dim:
                if self._to_parietal_fallback is None:
                    self._to_parietal_fallback = nn.Linear(int(pooled.shape[1]), self.parietal_dim, bias=False).to(self.device)
                pooled = self._to_parietal_fallback(pooled)
            return F.normalize(pooled, dim=1)

    def encode(self, text: str) -> torch.Tensor:
        return self.encode_batch([text])

    def project_to_decoder(self, x384: torch.Tensor) -> torch.Tensor:
        # x384: [B,384] or [1,384]
        x = self.to_decoder_space(x384)
        return F.normalize(x, dim=-1)

    # ──────────────────────────────────────────────────────────────────
    # JSONL rebuild + ingest
    # ──────────────────────────────────────────────────────────────────
    def rebuild_from_jsonl(self, path: Path, *, max_items: int = 2000) -> None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        lines = lines[-max_items:]

        embed_texts: List[str] = []
        episodes: List[Dict[str, Any]] = []

        for ln in lines:
            try:
                obj = json.loads(ln)
            except Exception:
                continue

            p = str(obj.get("prompt", "")).strip()
            s = str(obj.get("summary", "")).strip()
            ts = float(obj.get("ts", 0.0) or 0.0)
            if not p:
                continue

            snippet = (s if s else f"User asked: {p}").strip() or f"User asked: {p}"
            emb_src = (s if (self.embed_field == "summary" and s) else p).strip() or p

            episodes.append({"snippet": snippet, "ts": ts})
            embed_texts.append(emb_src)

        if not embed_texts:
            self.episodes, self.index_emb = [], None
            return

        vecs: List[torch.Tensor] = []
        bs = 32
        for i in range(0, len(embed_texts), bs):
            vecs.append(self.encode_batch(embed_texts[i:i+bs]))
        self.episodes = episodes
        self.index_emb = torch.cat(vecs, dim=0)

    def ingest_episode(self, *, prompt: str, summary: str, embed_text: Optional[str] = None,
                       prompt_vec: Optional[torch.Tensor] = None, ts: Optional[float] = None) -> None:
        p = (prompt or "").strip()
        if not p:
            return
        snip = (summary or "").strip() or f"User asked: {p}"
        if ts is None:
            ts = time.time()

        self.episodes.append({"snippet": snip, "ts": float(ts)})

        src = (embed_text or (p if self.embed_field == "prompt" else (summary or p))).strip() or p
        if prompt_vec is None:
            prompt_vec = self.encode(src)

        if self.index_emb is None:
            self.index_emb = prompt_vec.detach().to(self.device)
        else:
            self.index_emb = torch.cat([self.index_emb, prompt_vec.detach().to(self.device)], dim=0)

    def topk(self, query: str, k: int = 4) -> List[str]:
        if self.index_emb is None or not self.episodes:
            return []
        q = self.encode(query)  # [1,384]
        sims = (q @ self.index_emb.T).squeeze(0)  # [N]
        top = torch.topk(sims, k=min(k, sims.numel())).indices.tolist()
        return [self.episodes[i]["snippet"] for i in top]
