# JAI Version: 0.0.3
"""Tests for JAI network learning."""

import unittest

from brain.network import NeuralNetwork
from training.trainer import train_network


class TestNetworkTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [1.0]),
        ]

    def test_training_reduces_loss(self) -> None:
        network = NeuralNetwork(2, 4, 1, seed=7)
        history = train_network(network, self.samples, learning_rate=2.0, epochs=2000)
        self.assertLess(history[-1], history[0])

    def test_training_improves_predictions(self) -> None:
        network = NeuralNetwork(2, 4, 1, seed=7)
        before = sum(
            (network.forward(x)[0] - y[0]) ** 2 for x, y in self.samples
        )
        train_network(network, self.samples, learning_rate=2.0, epochs=2000)
        after = sum(
            (network.forward(x)[0] - y[0]) ** 2 for x, y in self.samples
        )
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
