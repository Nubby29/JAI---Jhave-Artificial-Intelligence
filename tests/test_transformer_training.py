# JAI Version: 0.4.1
"""Tests for JAI Transformer training and generation."""

import unittest

from brain.transformer import TransformerLanguageModel
from training.trainer import train_transformer_language_model


class TestTransformerTraining(unittest.TestCase):
    def test_training_reduces_loss(self) -> None:
        model = TransformerLanguageModel(4, model_size=4, context_size=3, seed=7)
        sequences = [[0, 1, 2, 3, 2, 3], [0, 1, 2, 3]]
        history = train_transformer_language_model(
            model, sequences, learning_rate=0.08, epochs=80
        )
        self.assertLess(history[-1], history[0])

    def test_generation_returns_valid_tokens(self) -> None:
        model = TransformerLanguageModel(5, model_size=4, context_size=3, seed=2)
        generated = model.generate([0, 1], max_new_tokens=5)
        self.assertEqual(len(generated), 7)
        self.assertTrue(all(token in range(5) for token in generated))

    def test_generation_can_stop(self) -> None:
        model = TransformerLanguageModel(3, model_size=4, context_size=3, seed=2)
        generated = model.generate([0], max_new_tokens=5, stop_token_id=0)
        self.assertEqual(generated, [0, 0])


if __name__ == "__main__":
    unittest.main()
