"""Regression checks for matched sampling and evaluation leakage."""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_dataset.py"
spec = importlib.util.spec_from_file_location("build_dataset", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fixtures():
    groups = {}
    for domain in builder.DOMAINS:
        for index in range(20):
            source = f"{domain}-{index}"
            groups[domain, source] = [
                {"id": f"{source}-{model}", "source_id": source, "domain": domain,
                 "model": model, "prompt": f"Prompt {source}",
                 "generation": f"Response {source} {model}", "attack": "none",
                 "decoding": "sampling", "repetition_penalty": "no"}
                for model in builder.MODELS
            ]
    return groups


class DatasetTests(unittest.TestCase):
    def test_grouped_balance_and_order_independence(self):
        groups = fixtures()
        rows, _, _ = builder.select_rows(groups, 10, 42)
        builder.validate(rows, 10)
        reversed_rows, _, _ = builder.select_rows(dict(reversed(list(groups.items()))), 10, 42)
        self.assertEqual(rows, reversed_rows)
        self.assertEqual(len(rows), 160)
        changed, _, _ = builder.select_rows(groups, 10, 43)
        self.assertNotEqual(rows, changed)

    def test_normalized_prompt_and_output_duplicates_removed(self):
        groups = fixtures()
        for row in groups["news", "news-1"]:
            row["prompt"] = "  PROMPT   news-0  "
        groups["reviews", "reviews-1"][0]["generation"] = " RESPONSE news-2 chatgpt "
        rows, eligible, excluded = builder.select_rows(groups, 10, 42)
        builder.validate(rows, 10)
        self.assertEqual(sum(excluded.values()), 2)
        self.assertEqual(sum(eligible.values()), 78)

    def test_insufficient_data_fails(self):
        groups = fixtures()
        groups = {k: v for k, v in groups.items() if k[0] != "poetry"}
        with self.assertRaisesRegex(ValueError, "poetry"):
            builder.select_rows(groups, 10, 42)

    def test_leakage_is_rejected(self):
        rows, _, _ = builder.select_rows(fixtures(), 10, 42)
        train = next(r for r in rows if r["split"] == "train")
        test = next(r for r in rows if r["split"] == "test")
        test["generation"] = train["generation"]
        with self.assertRaisesRegex(ValueError, "Cross-split leakage"):
            builder.validate(rows, 10)

    def test_reader_filters_and_requires_matching_prompts(self):
        rows = [row.copy() for group in fixtures().values() for row in group]
        rows[0]["attack"] = "paraphrase"  # Makes one source incomplete.
        rows[4]["prompt"] = "Different model-specific prompt"
        rows[8]["generation"] = "   "  # Makes another source incomplete.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=builder.FIELDS[:-1])
                writer.writeheader()
                writer.writerows(rows)
            groups, audit = builder.read_groups(path)
        self.assertEqual(len(groups), 77)
        self.assertEqual(audit["incomplete_sources_removed"], 2)
        self.assertEqual(audit["mismatched_prompt_sources_removed"], 1)


if __name__ == "__main__":
    unittest.main()
