"""Regression tests for domain exclusion and matched RQ1 comparisons."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from common.data import Vocabulary, read_dataset
from rq3.splits import make_domain_fold
from rq3.summarize import compare_predictions, score_predictions, summarize, write_csv


class RQ3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_dataset(ROOT / "data/processed/raid_balanced.csv")
        cls.domains = sorted({r["domain"] for r in cls.rows})
        cls.labels = sorted({r["model"] for r in cls.rows})

    def test_entire_domain_excluded_and_tested_once(self):
        tested = []
        for domain in self.domains:
            train, validation, test = make_domain_fold(self.rows, domain)
            self.assertEqual([len(s) for s in (train, validation, test)], [960, 120, 400])
            self.assertNotIn(domain, {r["domain"] for r in train + validation})
            self.assertEqual({r["domain"] for r in test}, {domain})
            self.assertEqual({r["split"] for r in train}, {"train"})
            self.assertEqual({r["split"] for r in validation}, {"validation"})
            tested.extend(r["id"] for r in test)
        self.assertEqual(len(tested), len(set(tested)))
        self.assertEqual(set(tested), {r["id"] for r in self.rows})

    def test_split_sources_cannot_cross_domains(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["domain"] = "invalid"
        with self.assertRaisesRegex(ValueError, "Every source"):
            make_domain_fold(rows, self.domains[0])

    def test_incomplete_model_groups_rejected(self):
        with self.assertRaisesRegex(ValueError, "Every source"):
            make_domain_fold(self.rows[1:], self.domains[0])

    def test_held_out_only_token_never_enters_vocabulary(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["domain"] == "poetry":
                row["generation"] += " heldoutuniquetoken heldoutuniquetoken"
        train, _, _ = make_domain_fold(rows, "poetry")
        vocabulary = Vocabulary.fit(r["generation"] for r in train)
        self.assertNotIn("heldoutuniquetoken", vocabulary.token_to_id)

    @staticmethod
    def predictions(rows, correct=True):
        return [{"id": r["id"], "source_id": r["source_id"], "domain": r["domain"],
                 "true_model": r["model"], "predicted_model": r["model"] if correct else "chatgpt"}
                for r in rows]

    def test_comparison_joins_by_id_and_rejects_missing_ids(self):
        expected = [r for r in self.rows if r["split"] == "test"]
        current = self.predictions(self.rows, correct=False)
        reference = self.predictions(expected)
        result, matched = compare_predictions(current[::-1], reference, expected, self.labels)
        self.assertEqual(result["rq1"]["accuracy"], 1)
        self.assertEqual(result["rq3"]["accuracy"], 0.25)
        self.assertEqual(result["accuracy_change_percentage_points"], -75)
        self.assertEqual(len(matched), 160)
        with self.assertRaisesRegex(ValueError, "exactly"):
            compare_predictions(current, reference[:-1], expected, self.labels)
        reference[0]["true_model"] = "invalid"
        with self.assertRaisesRegex(ValueError, "metadata"):
            compare_predictions(current, reference, expected, self.labels)

    def test_summary_aggregates_four_folds_and_matched_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, reference_dir = root / "rq3", root / "rq1"
            for domain in self.domains:
                directory = output / domain / "cnn"
                directory.mkdir(parents=True)
                predictions = self.predictions([r for r in self.rows if r["domain"] == domain], False)
                write_csv(directory / "test_predictions.csv", predictions)
                (directory / "test_metrics.json").write_text(json.dumps({
                    **score_predictions(predictions, self.labels), "best_epoch": 1}))
            (reference_dir / "cnn").mkdir(parents=True)
            write_csv(reference_dir / "cnn/test_predictions.csv", self.predictions(
                [r for r in self.rows if r["split"] == "test"]))
            result = summarize(output, self.rows, ["cnn"], self.domains, reference_dir)["cnn"]
            self.assertEqual(result["n"], 1600)
            self.assertEqual(result["accuracy"], 0.25)
            self.assertEqual(result["mean_domain_accuracy"], 0.25)
            self.assertEqual(result["matched_rq1_comparison"]["rq3"]["n"], 160)
            self.assertTrue((output / "accuracy_by_domain.png").is_file())


if __name__ == "__main__":
    unittest.main()
