"""Metrics and figures shared by the research-question runners."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def metrics(y_true, y_pred, labels):
    indices = list(range(len(labels)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=indices, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=indices, target_names=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=indices).tolist(),
        "label_order": labels,
        "n": len(y_true),
    }


def evaluate(model, loader, device, labels):
    model.eval()
    truth, predictions, probabilities = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for tokens, lengths, targets in loader:
            logits = model(tokens.to(device), lengths.to(device))
            total_loss += torch.nn.functional.cross_entropy(
                logits, targets.to(device), reduction="sum").item()
            truth.extend(targets.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            probabilities.extend(logits.softmax(dim=1).cpu().tolist())
    result = metrics(truth, predictions, labels)
    result["loss"] = total_loss / len(truth)
    return result, truth, predictions, probabilities


def save_evaluation(directory, rows, result, truth, predictions, probabilities, history):
    labels = result["label_order"]
    result["by_domain"] = {}
    for domain in sorted({r["domain"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["domain"] == domain]
        result["by_domain"][domain] = metrics(
            [truth[i] for i in indices], [predictions[i] for i in indices], labels)
    (directory / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (directory / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "source_id", "domain", "true_model", "predicted_model"]
                        + [f"p_{label}" for label in labels])
        for row, target, prediction, probs in zip(rows, truth, predictions, probabilities):
            writer.writerow([row["id"], row["source_id"], row["domain"], labels[target],
                             labels[prediction], *probs])

    figure, ax = plt.subplots(figsize=(7, 6))
    matrix = result["confusion_matrix"]
    plot = ax.imshow(matrix, cmap="Blues")
    figure.colorbar(plot, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
           yticklabels=labels, xlabel="Predicted model", ylabel="True model", title="Test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > max(map(max, matrix)) / 2 else "black")
    figure.tight_layout()
    figure.savefig(directory / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [h["epoch"] for h in history]
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train")
    axes[0].plot(epochs, [h["validation_loss"] for h in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].plot(epochs, [h["validation_accuracy"] for h in history], label="Validation accuracy")
    axes[1].plot(epochs, [h["validation_macro_f1"] for h in history], label="Validation macro F1")
    axes[1].axhline(1 / len(labels), color="gray", linestyle="--", label="Chance accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(figure)
