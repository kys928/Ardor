from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

from backends.base import ModelBackend
from loaders.native_checkpoint import load_native_decoder
from loaders.native_tokenizer import load_tokenizer_matching_vocab
from loaders.native_encoder import load_encoder_cached, encoder_forward_pooled


class NativeArdorBackend(ModelBackend):
    def __init__(self, model_path: str, tokenizer_path: Optional[str], device: str, repo_root: Path, encoder_ckpt: Optional[str] = None):
        self.repo_root = repo_root
        self.device = device
        self.model_path = model_path
        self.model, self._schema, want_vocab, meta = load_native_decoder(model_path, device)

        requested_tok = tokenizer_path if (tokenizer_path and os.path.isfile(tokenizer_path)) else None
        self.tokenizer, self._tokenizer_path = load_tokenizer_matching_vocab(repo_root, requested_tok, want_vocab, meta)
        try:
            name = type(self.tokenizer.model).__name__.lower()
            if name == "bpe" and getattr(self.tokenizer, "decoder", None) is None:
                self.tokenizer.decoder = ByteLevel()
        except Exception:
            pass

        self.encoder = load_encoder_cached(encoder_ckpt, device, getattr(self.model, "cfg", None)) if encoder_ckpt else None

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids)
        if isinstance(out, (list, tuple)) and len(out) > 0:
            out = out[0]
        if isinstance(out, dict) and "logits" in out:
            out = out["logits"]
        return out

    def generate(self, prompt: str, **decode_cfg: Any) -> str:
        decode_fn = decode_cfg.get("decode_fn")
        if callable(decode_fn):
            return str(decode_fn(decode_cfg.get("temperature", 0.7), decode_cfg.get("top_p", 0.9)))
        return ""

    def encode_text(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text).ids[:1024] or [self.tokenizer.token_to_id("<pad>") or 0]
        x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.encoder is not None:
                pooled = encoder_forward_pooled(self.encoder, x)
            else:
                emb = self.model.token_embed(x)
                pooled = emb.mean(dim=1)
            pooled = F.normalize(pooled, dim=-1)
        return pooled

    def vocab_size(self) -> int:
        return int(self._schema.get("vocab") or self.model.token_embed.weight.shape[0])

    def context_len(self) -> int:
        return int(self._schema.get("max_len") or getattr(self.model, "max_len", 0) or 0)

    def hidden_size(self) -> int:
        return int(self._schema.get("hidden") or getattr(self.model, "hidden_size", 0) or 0)

    def schema(self) -> Dict[str, Any]:
        return dict(self._schema)

    def tokenizer_path(self) -> Optional[str]:
        return self._tokenizer_path

    def supports_retrieval(self) -> bool:
        return True

    def supports_hidden_state_export(self) -> bool:
        return True

    def close(self) -> None:
        return None
