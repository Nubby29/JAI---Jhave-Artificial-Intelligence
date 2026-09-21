# JAI Version: 0.3.0
"""Tests for JAI attention foundations."""

import unittest

from brain.attention import SelfAttention
from brain.transformer import TransformerBlock


class TestAttention(unittest.TestCase):
    def test_attention_weights_are_probability_distributions(self) -> None:
        attention = SelfAttention(3)
        inputs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        _, weights = attention.forward(inputs)

        self.assertEqual(len(weights), 3)
        for row in weights:
            self.assertEqual(len(row), 3)
            self.assertAlmostEqual(sum(row), 1.0, places=7)
            self.assertTrue(all(weight >= 0.0 for weight in row))

    def test_attention_output_shape(self) -> None:
        attention = SelfAttention(4)
        inputs = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        outputs, _ = attention.forward(inputs)
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(len(vector) == 4 for vector in outputs))

    def test_invalid_input_size(self) -> None:
        attention = SelfAttention(3)
        with self.assertRaises(ValueError):
            attention.forward([[1.0, 2.0]])

    def test_transformer_block_residual_connection(self) -> None:
        block = TransformerBlock(2)
        inputs = [[1.0, 0.0], [0.0, 1.0]]
        outputs, _ = block.forward(inputs)
        self.assertEqual(len(outputs), 2)
        self.assertGreater(outputs[0][0], inputs[0][0])
        self.assertGreater(outputs[1][1], inputs[1][1])


if __name__ == "__main__":
    unittest.main()
