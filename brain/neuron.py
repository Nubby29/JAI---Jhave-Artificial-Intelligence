# JAI Version: 0.0.1
"""The first artificial neuron in JAI."""

from brain.activations import sigmoid


class Neuron:
    """A single trainable neuron."""

    def __init__(self, weight: float = 0.0, bias: float = 0.0) -> None:
        self.weight = weight
        self.bias = bias

    def forward(self, x: float) -> float:
        """Calculate the neuron's output."""
        return sigmoid(self.weight * x + self.bias)

    def linear_output(self, x: float) -> float:
        """Return the neuron's pre-activation value."""
        return self.weight * x + self.bias
