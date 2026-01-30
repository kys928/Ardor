import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Set  # Python 3.7-compatible type hinting

class RelevanceLoss(nn.Module):
    """
    Penalizes generations that ignore key prompt terms.
    Designed for integration alongside standard LM loss.
    """

    def __init__(self, tokenizer, stopwords=None, top_k=20, lambda_weight=0.05):
        """
        tokenizer     — HuggingFace / tokenizers.Tokenizer instance
        stopwords     — set of lowercase stopwords
        top_k         — how many predicted tokens to consider per step
        lambda_weight — scaling factor for this loss
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.stopwords = stopwords if stopwords else set()
        self.top_k = top_k
        self.lambda_weight = lambda_weight

    def _keywords(self, text: str) -> Set[str]:  # changed from set[str] to Set[str]
        """Extract lowercase keywords, excluding stopwords."""
        toks = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
        return {t for t in toks if t not in self.stopwords}

    def _token_ids_from_keywords(self, keywords: Set[str]) -> Set[int]:  # changed from set[int] to Set[int]
        """Convert keywords to token IDs (using first subtoken ID)."""
        ids = set()
        for kw in keywords:
            enc = self.tokenizer.encode(kw).ids
            if enc:
                ids.add(enc[0])  # Only consider the first token for match
        return ids

    def forward(self, logits: torch.Tensor, prompt_text: str) -> torch.Tensor:
        """
        logits      — [B, T, V] raw predictions from the model
        prompt_text — str, the prompt/question given to the model
        """
        if not prompt_text.strip():
            return torch.tensor(0.0, device=logits.device)

        keywords = self._keywords(prompt_text)
        if not keywords:
            return torch.tensor(0.0, device=logits.device)

        keyword_ids = self._token_ids_from_keywords(keywords)
        if not keyword_ids:
            return torch.tensor(0.0, device=logits.device)

        # Get top-k predictions at each position
        topk_ids = torch.topk(logits, self.top_k, dim=-1).indices  # [B, T, top_k]

        # Check if any of the keyword IDs appear in top-k
        keyword_tensor = torch.tensor(list(keyword_ids), device=logits.device)
        matches = (topk_ids.unsqueeze(-1) == keyword_tensor).any(dim=-1)  # [B, T]

        # Penalize positions where none of the keywords appear
        miss_ratio = (~matches).float().mean()  # fraction of positions with no keyword in top-k

        return self.lambda_weight * miss_ratio
