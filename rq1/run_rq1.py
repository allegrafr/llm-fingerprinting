"""RQ1: classify the generator from output text, using the saved source splits."""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.data import read_dataset
from common.train import TrainConfig, train_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/raid_balanced.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "rq1/results/seed42")
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
    rows = read_dataset(args.data)
    splits = [[r for r in rows if r["split"] == split] for split in ("train", "validation", "test")]
    models = ("cnn", "lstm") if args.model == "both" else (args.model,)
    # Fail before training either model if any requested result would be overwritten.
    for model in models:
        target = args.output_dir / model
        if target.exists() and any(target.iterdir()):
            parser.error(f"Results already exist at {target}; choose another --output-dir")
    provenance = {"dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "research_question": "RQ1", "input": "generation only"}
    summary = []
    for model in models:
        result = train_experiment(model, *splits, args.output_dir / model, config, provenance)
        summary.append({"model": model, "seed": args.seed, "accuracy": result["accuracy"],
                        "macro_f1": result["macro_f1"], "best_epoch": result["best_epoch"],
                        "test_responses": result["n"]})
    with (args.output_dir / f"summary_{args.model}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
