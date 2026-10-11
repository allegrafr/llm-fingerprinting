"""Shared text preprocessing. Fit Vocabulary on training rows only."""

import csv
import re
from collections import Counter

import torch
from torch.utils.data import Dataset


TOKEN_PATTERN = re.compile(r"\n|\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    # Preserve case, punctuation, and newlines as potential style signals.
    return TOKEN_PATTERN.findall(text)


def text_for(row, mode="output"):
    if mode == "output":
        return row["generation"]
    if mode == "input":
        return row["prompt"]
    if mode == "both":
        return row["prompt"] + "\n" + row["generation"]
    raise ValueError(f"Unknown input mode: {mode}")


def read_dataset(path):
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "source_id", "domain", "model", "prompt", "generation", "split"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("Expected a nonempty processed RAID dataset")
    if {r["split"] for r in rows} != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test splits")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate generation IDs")
    return rows


def check_splits(train, validation, test):
    splits = [train, validation, test]
    if any(not rows for rows in splits):
        raise ValueError("All three splits must be nonempty")
    labels = {r["model"] for r in train}
    if any({r["model"] for r in rows} != labels for rows in splits):
        raise ValueError("Every split must contain the same model labels")
    for field in ("id", "source_id", "prompt", "generation"):
        values = [{" ".join(r[field].split()).casefold() for r in rows} for rows in splits]
        if any(values[i] & values[j] for i in range(3) for j in range(i)):
            raise ValueError(f"Cross-split leakage in {field}")


class Vocabulary:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id

    @classmethod
    def fit(cls, texts, max_vocab=20000, min_frequency=2):
        if max_vocab < 2 or min_frequency < 1:
            raise ValueError("Invalid vocabulary settings")
        counts = Counter(token for text in texts for token in tokenize(text))
        tokens = sorted((t for t, n in counts.items() if n >= min_frequency),
                        key=lambda t: (-counts[t], t))
        mapping = {"<PAD>": 0, "<UNK>": 1}
        for token in tokens[:max_vocab - 2]:
            mapping[token] = len(mapping)
        return cls(mapping)

    def encode(self, text, max_length):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        return [self.token_to_id.get(t, 1) for t in tokenize(text)[:max_length]] or [1]

    def __len__(self):
        return len(self.token_to_id)


class TextDataset(Dataset):
    def __init__(self, rows, vocabulary, labels, max_length, mode="output"):
        self.items = [(vocabulary.encode(text_for(r, mode), max_length), labels[r["model"]])
                      for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_batch(items):
    sequences, labels = zip(*items)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    tokens = torch.zeros(len(items), max(5, int(lengths.max())), dtype=torch.long)
    for index, sequence in enumerate(sequences):
        tokens[index, :len(sequence)] = torch.tensor(sequence)
    return tokens, lengths, torch.tensor(labels, dtype=torch.long)
