"""RQ3: train fresh CNN/LSTM classifiers while holding out each entire domain."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from common.data import read_dataset
from common.train import TrainConfig, train_experiment
from rq3.splits import make_domain_fold
from rq3.summarize import read_predictions, compare_predictions, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/raid_balanced.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "rq3/results/seed42")
    parser.add_argument("--rq1-results", type=Path, help="Optional saved RQ1 run for a matched-test comparison")
    parser.add_argument("--model", choices=("cnn", "lstm", "both"), default="both")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    config = TrainConfig(seed=args.seed, epochs=args.epochs, patience=args.patience,
                         batch_size=args.batch_size, max_length=args.max_length, device=args.device)
    config.validate()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is nonempty; choose a new --output-dir")
    rows = read_dataset(args.data)
    domains = sorted({r["domain"] for r in rows})
    models = ("cnn", "lstm") if args.model == "both" else (args.model,)
    folds = {domain: make_domain_fold(rows, domain) for domain in domains}
    dataset_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    reference_hashes = {}
    if args.rq1_results:
        expected = [r for r in rows if r["split"] == "test"]
        for model in models:
            directory = args.rq1_results / model
            reference_config = json.loads((directory / "config.json").read_text())
            if reference_config["provenance"]["dataset_sha256"] != dataset_hash:
                parser.error("RQ1 reference used a different dataset")
            if any(reference_config["config"][k] != v for k, v in asdict(config).items() if k != "device"):
                parser.error("RQ1 comparison requires matching hyperparameters and seed")
            predictions = read_predictions(directory / "test_predictions.csv")
            compare_predictions(predictions, predictions, expected, sorted({r["model"] for r in rows}))
            reference_hashes[model] = hashlib.sha256((directory / "test_predictions.csv").read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py"))}
    protocol = {"research_question": "RQ3", "dataset_sha256": dataset_hash,
                "rq3_code_sha256": script_hashes, "config": asdict(config), "domains": domains,
                "models": models, "rq1_prediction_sha256": reference_hashes,
                "split_policy": "Other domains: original train and validation rows; held-out domain: all rows. Other-domain test rows unused.",
                "limitation": "RQ1 trains on 1280 responses; each RQ3 fold trains on 960. Matched evaluation controls test IDs, not training size."}
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for domain, (train, validation, test) in folds.items():
        for model in models:
            print(f"\nHeld-out domain: {domain}; classifier: {model}; "
                  f"train/validation/test: {len(train)}/{len(validation)}/{len(test)}", flush=True)
            train_experiment(model, train, validation, test, args.output_dir / domain / model,
                             config, {**protocol, "held_out_domain": domain, "input": "generation only"})
    results = summarize(args.output_dir, rows, models, domains, args.rq1_results)
    for model, result in results.items():
        print(f"{model}: pooled out-of-domain accuracy={result['accuracy']:.4f}, "
              f"macro F1={result['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
