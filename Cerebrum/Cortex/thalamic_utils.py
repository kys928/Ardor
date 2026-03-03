# thalamic_utils.py
import torch

def causal_mask(T: int, device=None, dtype=torch.bool) -> torch.Tensor:
    """
    Standard AR (causal) mask: shape [1, 1, T, T], True = keep, False = block.
    """
    m = torch.ones(T, T, dtype=dtype, device=device).tril()
    return m.unsqueeze(0).unsqueeze(0)
