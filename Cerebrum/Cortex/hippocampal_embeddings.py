# import torch
# import torch.nn as nn
#
# class Embeddings(nn.Module):
#     def __init__(self, vocab_size, hidden_dim, max_len=512):
#         super().__init__()
#         self.token_embed = nn.Embedding(vocab_size, hidden_dim)
#         self.position_embed = nn.Embedding(max_len, hidden_dim)
#
#     def forward(self, x):
#         positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
#         return self.token_embed(x) + self.position_embed(positions)
