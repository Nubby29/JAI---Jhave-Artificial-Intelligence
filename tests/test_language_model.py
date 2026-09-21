# JAI Version: 0.2.0
"""Tests for JAI language-model learning."""

import unittest

from brain.language_model import LanguageModel
from training.trainer import train_language_model


class TestLanguageModel(unittest.TestCase):
    def test_probability_distribution(self) -> None:
        model = LanguageModel(5, embedding_size=4, seed=1)
        probabilities = model.predict_proba([0, 1])
        self.assertAlmostEqual(sum(probabilities), 1.0, places=7)
        self.assertEqual(len(probabilities), 5)

    def test_training_reduces_next_token_loss(self) -> None:
        model = LanguageModel(4, embedding_size=8, context_size=2, seed=7)
        sequences = [[0, 1, 2, 3, 2, 3], [0, 1, 2, 3]]
        history = train_language_model(model, sequences, learning_rate=0.3, epochs=250)
        self.assertLess(history[-1], history[0])

    def test_prediction_is_valid_token(self) -> None:
        model = LanguageModel(4, embedding_size=4, seed=2)
        self.assertIn(model.predict_next([0]), range(4))


if __name__ == "__main__":
    unittest.main()
