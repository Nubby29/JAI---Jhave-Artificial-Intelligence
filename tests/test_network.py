# JAI Version: 0.0.2
"""Tests for JAI's first multi-neuron network."""

import unittest

from brain.network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):
    def test_forward_shape(self) -> None:
        network = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
        output = network.forward([0.5, -0.5])
        self.assertEqual(len(output), 1)
        self.assertGreaterEqual(output[0], 0.0)
        self.assertLessEqual(output[0], 1.0)

    def test_parameter_count(self) -> None:
        network = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
        self.assertEqual(network.parameter_count(), 13)

    def test_wrong_input_size(self) -> None:
        network = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
        with self.assertRaises(ValueError):
            network.forward([1.0])
