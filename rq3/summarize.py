"""Aggregate saved predictions and compare identical RQ1/RQ3 test examples."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.evaluate import metrics


def read_predictions(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate prediction IDs: {path}")
    return rows


def score_predictions(rows, labels):
    mapping = {label: i for i, label in enumerate(labels)}
    return metrics([mapping[r["true_model"]] for r in rows],
                   [mapping[r["predicted_model"]] for r in rows], labels)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def compare_predictions(current, reference, expected_rows, labels):
    """Require exactly the original RQ1 test IDs before comparing scores."""
    reference_by_id = {r["id"]: r for r in reference}
    current_by_id = {r["id"]: r for r in current}
    if len(reference_by_id) != len(reference) or len(current_by_id) != len(current):
        raise ValueError("Duplicate prediction IDs")
    expected = {r["id"]: r for r in expected_rows}
    if set(reference_by_id) != set(expected) or not set(expected) <= current_by_id.keys():
        raise ValueError("RQ1 predictions must cover exactly the shared test IDs")
    matched = [current_by_id[r["id"]] for r in expected_rows]
    baseline = [reference_by_id[r["id"]] for r in expected_rows]
    for row in expected_rows:
        for prediction in (reference_by_id[row["id"]], current_by_id[row["id"]]):
            if any(prediction[k] != row[k] for k in ("source_id", "domain")) or prediction["true_model"] != row["model"]:
                raise ValueError("Prediction metadata does not match the shared dataset")
    rq1, rq3 = score_predictions(baseline, labels), score_predictions(matched, labels)
    return {"rq1": rq1, "rq3": rq3,
            "accuracy_change_percentage_points": 100 * (rq3["accuracy"] - rq1["accuracy"]),
            "macro_f1_change": rq3["macro_f1"] - rq1["macro_f1"]}, matched


def summarize(output_dir, rows, models, domains, reference_dir=None):
    output_dir = Path(output_dir)
    labels = sorted({r["model"] for r in rows})
    expected_test = [r for r in rows if r["split"] == "test"]
    aggregates, folds, per_class, comparisons = {}, [], [], []
    for model in models:
        predictions = []
        fold_scores = []
        for domain in domains:
            directory = output_dir / domain / model
            saved = json.loads((directory / "test_metrics.json").read_text())
            fold_predictions = read_predictions(directory / "test_predictions.csv")
            expected_ids = {r["id"] for r in rows if r["domain"] == domain}
            if {r["id"] for r in fold_predictions} != expected_ids:
                raise ValueError(f"Incomplete held-out predictions: {domain}/{model}")
            computed = score_predictions(fold_predictions, labels)
            if computed["confusion_matrix"] != saved["confusion_matrix"]:
                raise ValueError("Saved metrics and predictions disagree")
            predictions.extend(fold_predictions)
            fold_scores.append(computed)
            folds.append({"model": model, "held_out_domain": domain, "test_responses": computed["n"],
                          "accuracy": computed["accuracy"], "macro_f1": computed["macro_f1"],
                          "best_epoch": saved["best_epoch"]})
            for label in labels:
                report = computed["classification_report"][label]
                per_class.append({"classifier": model, "held_out_domain": domain,
                                  "generator": label, **{k: report[k] for k in ("precision", "recall", "f1-score", "support")}})
        if len({r["id"] for r in predictions}) != len(rows):
            raise ValueError("Every response must have exactly one out-of-domain prediction")
        aggregate = score_predictions(predictions, labels)
        aggregate["mean_domain_accuracy"] = float(np.mean([r["accuracy"] for r in fold_scores]))
        aggregate["mean_domain_macro_f1"] = float(np.mean([r["macro_f1"] for r in fold_scores]))
        aggregate["note"] = "Pooled predictions come from four independently trained domain-held-out models."
        write_csv(output_dir / f"{model}_out_of_domain_predictions.csv", predictions)
        if reference_dir is not None:
            reference = read_predictions(Path(reference_dir) / model / "test_predictions.csv")
            comparison, matched = compare_predictions(predictions, reference, expected_test, labels)
            aggregate["matched_rq1_comparison"] = comparison
            write_csv(output_dir / f"{model}_matched_test_predictions.csv", matched)
            for domain in domains:
                current_domain = [r for r in matched if r["domain"] == domain]
                reference_domain = [r for r in reference if r["domain"] == domain]
                old, new = score_predictions(reference_domain, labels), score_predictions(current_domain, labels)
                comparisons.append({"model": model, "domain": domain, "responses": len(current_domain),
                                    "rq1_accuracy": old["accuracy"], "rq3_accuracy": new["accuracy"],
                                    "rq1_macro_f1": old["macro_f1"], "rq3_macro_f1": new["macro_f1"],
                                    "accuracy_change_percentage_points": 100 * (new["accuracy"] - old["accuracy"])})
        aggregates[model] = aggregate
    write_csv(output_dir / "fold_summary.csv", folds)
    write_csv(output_dir / "per_generator_by_domain.csv", per_class)
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregates, indent=2) + "\n")
    if comparisons:
        write_csv(output_dir / "matched_rq1_comparison.csv", comparisons)

    figure, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(domains))
    width = 0.7 / len(models)
    for i, model in enumerate(models):
        values = [next(r["accuracy"] for r in folds if r["model"] == model and r["held_out_domain"] == d) for d in domains]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width, label=model.upper())
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3)
    ax.axhline(0.25, color="gray", linestyle="--", label="Chance accuracy")
    ax.set(xticks=x, xticklabels=[d.title() for d in domains], ylim=(0, 1),
           ylabel="Test accuracy", xlabel="Domain excluded from training and validation",
           title="RQ3: performance on unseen domains (400 responses per domain)")
    ax.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy_by_domain.png", dpi=180)
    plt.close(figure)
    return aggregates
