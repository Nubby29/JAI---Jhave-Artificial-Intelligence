# JAI Version: 0.0.1
"""Basic tests for the first JAI neuron."""

import unittest

from brain.activations import sigmoid
from brain.neuron import Neuron


class TestNeuron(unittest.TestCase):
    def test_sigmoid_midpoint(self) -> None:
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_forward_uses_weight_and_bias(self) -> None:
        neuron = Neuron(weight=1.0, bias=0.0)
        self.assertAlmostEqual(neuron.forward(0.0), 0.5)

    def test_linear_output(self) -> None:
        neuron = Neuron(weight=2.0, bias=1.0)
        self.assertAlmostEqual(neuron.linear_output(3.0), 7.0)


if __name__ == "__main__":
    unittest.main()
