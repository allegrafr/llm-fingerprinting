"""One-layer unidirectional LSTM with masked mean pooling."""

import torch
from torch import nn


class TextLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=128,
                 dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens, lengths):
        states, _ = self.lstm(self.embedding(tokens))
        # Later right-padding cannot affect earlier states in a forward-only LSTM.
        mask = torch.arange(tokens.size(1), device=tokens.device)[None, :] < lengths[:, None]
        pooled = (states * mask[:, :, None]).sum(dim=1) / lengths[:, None]
        return self.classifier(self.dropout(pooled))
