"""Text CNN with three convolution widths and masked max pooling."""

import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=128,
                 dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, hidden_dim, width) for width in (3, 4, 5))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(3 * hidden_dim, num_classes)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = []
        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))
            # A short response gets one padded window; all extra windows are ignored.
            valid = (lengths - convolution.kernel_size[0] + 1).clamp(min=1)
            positions = torch.arange(features.size(2), device=tokens.device)
            mask = positions[None, :] >= valid[:, None]
            features = features.masked_fill(mask[:, None, :], float("-inf"))
            pooled.append(features.max(dim=2).values)
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))
