"""Small end-to-end run, including restoration of the saved checkpoint."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.data import TextDataset, Vocabulary, collate_batch
from common.evaluate import evaluate, metrics
from common.train import TrainConfig, make_model, train_experiment


class TrainingTests(unittest.TestCase):
    def test_metrics_include_unpredicted_classes(self):
        result = metrics([0, 1, 2, 3], [0, 0, 0, 0], ["a", "b", "c", "d"])
        self.assertEqual(result["accuracy"], 0.25)
        self.assertAlmostEqual(result["macro_f1"], 0.1)
        self.assertEqual(result["classification_report"]["d"]["recall"], 0.0)

    def test_saved_checkpoint_reproduces_predictions(self):
        splits = []
        for split in ("train", "validation", "test"):
            splits.append([{"id": f"{split}-{label}", "source_id": f"{split}-{label}",
                            "domain": "toy", "model": label, "prompt": f"{split} prompt {label}",
                            "generation": f"{split} style {label} {label} {label}"}
                           for label in ("a", "b", "c", "d")])
        config = TrainConfig(epochs=1, device="cpu", embedding_dim=8, hidden_dim=8,
                             max_length=8, dropout=0, min_frequency=1)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "run"
            result = train_experiment("cnn", *splits, directory, config)
            checkpoint = torch.load(directory / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            vocabulary = Vocabulary(checkpoint["vocabulary"])
            self.assertNotIn("validation", vocabulary.token_to_id)
            self.assertNotIn("test", vocabulary.token_to_id)
            labels = checkpoint["labels"]
            model = make_model("cnn", len(vocabulary), len(labels), config)
            model.load_state_dict(checkpoint["state_dict"])
            dataset = TextDataset(splits[2], vocabulary, {v: i for i, v in enumerate(labels)}, 8)
            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_batch)
            actual, _, predicted, _ = evaluate(model, loader, torch.device("cpu"), labels)
            with (directory / "test_predictions.csv").open() as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual([labels[i] for i in predicted], [r["predicted_model"] for r in saved])
            self.assertEqual(actual["confusion_matrix"], result["confusion_matrix"])
            self.assertEqual(json.loads((directory / "test_metrics.json").read_text())["n"], 4)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                train_experiment("cnn", *splits, directory, config)


if __name__ == "__main__":
    unittest.main()
