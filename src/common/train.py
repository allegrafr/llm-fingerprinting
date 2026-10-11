"""Reproducible training with validation-only checkpoint selection."""

import hashlib
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import TextDataset, Vocabulary, check_splits, collate_batch, text_for, tokenize
from .evaluate import evaluate, save_evaluation
from .models.cnn import TextCNN
from .models.rnn import TextLSTM


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 15
    patience: int = 4
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    embedding_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.3
    max_length: int = 256
    max_vocab: int = 20000
    min_frequency: int = 2
    device: str = "auto"
    input_mode: str = "output"
    cpu_threads: int = 4

    def validate(self):
        for name in ("epochs", "patience", "batch_size", "embedding_dim", "hidden_dim",
                     "min_frequency", "cpu_threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_length < 5 or self.max_vocab < 2:
            raise ValueError("max_length >= 5 and max_vocab >= 2 are required")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer/dropout settings")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    return torch.device(requested)


def make_model(name, vocab_size, num_classes, config):
    models = {"cnn": TextCNN, "lstm": TextLSTM}
    return models[name](vocab_size, num_classes, config.embedding_dim,
                        config.hidden_dim, config.dropout)


def train_experiment(name, train_rows, validation_rows, test_rows, directory, config,
                     provenance=None):
    config.validate()
    check_splits(train_rows, validation_rows, test_rows)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Refusing to overwrite results: {directory}. Choose a new output directory.")
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = select_device(config.device)
    labels = sorted({r["model"] for r in train_rows})
    label_to_id = {label: index for index, label in enumerate(labels)}
    vocabulary = Vocabulary.fit((text_for(r, config.input_mode) for r in train_rows),
                                config.max_vocab, config.min_frequency)
    generator = torch.Generator().manual_seed(config.seed)

    def loader(rows, shuffle=False):
        dataset = TextDataset(rows, vocabulary, label_to_id, config.max_length, config.input_mode)
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                          collate_fn=collate_batch, num_workers=0,
                          generator=generator if shuffle else None)

    train_loader, val_loader = loader(train_rows, True), loader(validation_rows)
    model = make_model(name, len(vocabulary), len(labels), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    source_files = sorted(Path(__file__).parent.rglob("*.py"))
    metadata = {
        "model": name, "config": asdict(config), "device_used": str(device),
        "labels": labels, "vocabulary_size": len(vocabulary),
        "parameters": sum(p.numel() for p in model.parameters()),
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "code_sha256": {str(p.relative_to(Path(__file__).parent)):
                        hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "provenance": provenance or {},
        "splits": {split: {"rows": len(rows), "source_ids": sorted({r['source_id'] for r in rows}),
                            "truncated_rows": sum(len(tokenize(text_for(r, config.input_mode)))
                                                  > config.max_length for r in rows)}
                   for split, rows in (("train", train_rows), ("validation", validation_rows),
                                       ("test", test_rows))},
        "checkpoint_selection": "maximum validation macro F1; lower validation loss breaks ties",
        "reproducibility_note": "Seeds fixed; numerical results may vary across devices and library versions.",
    }
    (directory / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (directory / "vocabulary.json").write_text(json.dumps(vocabulary.token_to_id, indent=2) + "\n")
    print(f"{name}: device={device}, vocabulary={len(vocabulary)}, parameters={metadata['parameters']}", flush=True)
    history, best_score, stale = [], (-1.0, float("-inf")), 0
    start = time.perf_counter()
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for tokens, lengths, targets in train_loader:
            tokens, lengths, targets = tokens.to(device), lengths.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, lengths)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += len(targets)
        val, _, _, _ = evaluate(model, val_loader, device, labels)
        history.append({"epoch": epoch, "train_loss": loss_sum / count, "train_accuracy": correct / count,
                        "validation_loss": val["loss"], "validation_accuracy": val["accuracy"],
                        "validation_macro_f1": val["macro_f1"]})
        (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"{name} epoch {epoch:02d}: train loss={loss_sum/count:.4f}, "
              f"val accuracy={val['accuracy']:.3f}, val macro F1={val['macro_f1']:.3f}", flush=True)
        score = (val["macro_f1"], -val["loss"])
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "model": name, "config": asdict(config), "labels": labels,
                        "vocabulary": vocabulary.token_to_id, "epoch": epoch}, directory / "best.pt")
        else:
            stale += 1
        if stale >= config.patience:
            break
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    result, truth, predictions, probabilities = evaluate(model, loader(test_rows), device, labels)
    result.update({"best_epoch": best_epoch, "best_validation_macro_f1": best_score[0],
                   "elapsed_seconds": time.perf_counter() - start, "seed": config.seed})
    save_evaluation(directory, test_rows, result, truth, predictions, probabilities, history)
    print(f"{name} TEST: accuracy={result['accuracy']:.4f}, macro F1={result['macro_f1']:.4f}, "
          f"best epoch={best_epoch}", flush=True)
    return result
