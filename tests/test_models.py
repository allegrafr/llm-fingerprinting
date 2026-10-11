"""Check leakage safeguards, padding behavior, and ability to learn a toy signal."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.data import Vocabulary, check_splits, collate_batch
from common.models.cnn import TextCNN
from common.models.rnn import TextLSTM


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_vocabulary_does_not_learn_unseen_words(self):
        vocabulary = Vocabulary.fit(["Hello Hello!\nworld"], min_frequency=1)
        self.assertEqual(vocabulary.encode("unseen", 10), [1])
        self.assertNotIn("unseen", vocabulary.token_to_id)
        self.assertIn("!", vocabulary.token_to_id)
        self.assertIn("\n", vocabulary.token_to_id)
        self.assertEqual(vocabulary.encode("", 10), [1])
        self.assertEqual(len(vocabulary.encode("Hello " * 100, 5)), 5)

    def test_collation_tracks_real_lengths(self):
        tokens, lengths, labels = collate_batch([([2], 0), ([3, 4, 5], 1)])
        self.assertEqual(tuple(tokens.shape), (2, 5))
        self.assertEqual(lengths.tolist(), [1, 3])
        self.assertEqual(tokens[0].tolist(), [2, 0, 0, 0, 0])

    def test_extra_padding_does_not_change_predictions(self):
        for model_type in (TextCNN, TextLSTM):
            with self.subTest(model=model_type.__name__):
                model = model_type(20, 4, embedding_dim=8, hidden_dim=8, dropout=0).eval()
                short, lengths, _ = collate_batch([([2], 0), ([3, 4, 5, 6, 7, 8], 1)])
                long = torch.nn.functional.pad(short, (0, 10))
                with torch.no_grad():
                    torch.testing.assert_close(model(short, lengths), model(long, lengths),
                                               rtol=1e-5, atol=1e-6)

    def test_models_can_learn_a_small_signal(self):
        tokens, lengths, targets = collate_batch([
            ([2] * 6, 0), ([3] * 6, 1), ([4] * 6, 2), ([5] * 6, 3)])
        for model_type in (TextCNN, TextLSTM):
            with self.subTest(model=model_type.__name__):
                model = model_type(10, 4, embedding_dim=8, hidden_dim=8, dropout=0)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
                for _ in range(40):
                    optimizer.zero_grad()
                    loss = torch.nn.functional.cross_entropy(model(tokens, lengths), targets)
                    loss.backward()
                    optimizer.step()
                self.assertEqual(model(tokens, lengths).argmax(1).tolist(), targets.tolist())
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.equal(model.embedding.weight[0], torch.zeros(8)))

    def test_split_guard_rejects_shared_sources(self):
        def row(prefix):
            return {"id": prefix, "source_id": prefix, "prompt": prefix,
                    "generation": prefix, "model": "a"}
        train, validation, test = [row("train")], [row("val")], [row("test")]
        check_splits(train, validation, test)
        test[0]["source_id"] = "train"
        with self.assertRaisesRegex(ValueError, "source_id"):
            check_splits(train, validation, test)


if __name__ == "__main__":
    unittest.main()
