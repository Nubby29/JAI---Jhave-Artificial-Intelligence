# JAI Version: 0.4.0
"""Tests for JAI's first complete decoder-style Transformer."""

import unittest

from brain.transformer import TransformerLanguageModel


class TestTransformerLanguageModel(unittest.TestCase):
    def test_probability_distribution(self) -> None:
        model = TransformerLanguageModel(5, model_size=8, context_size=4, seed=1)
        probabilities = model.predict_proba([0, 1, 2])
        self.assertAlmostEqual(sum(probabilities), 1.0, places=7)
        self.assertEqual(len(probabilities), 5)

    def test_causal_attention(self) -> None:
        model = TransformerLanguageModel(5, model_size=8, context_size=4, seed=1)
        _, weights = model.forward([0, 1, 2, 3])
        for row_index, row in enumerate(weights):
            for column_index, weight in enumerate(row):
                if column_index > row_index:
                    self.assertEqual(weight, 0.0)

    def test_prediction_is_valid_token(self) -> None:
        model = TransformerLanguageModel(4, model_size=8, context_size=4, seed=2)
        self.assertIn(model.predict_next([0, 1]), range(4))

    def test_parameter_count_is_positive(self) -> None:
        model = TransformerLanguageModel(4, model_size=8)
        self.assertGreater(model.parameter_count(), 0)


if __name__ == "__main__":
    unittest.main()
