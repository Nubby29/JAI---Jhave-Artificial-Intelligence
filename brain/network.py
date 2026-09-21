# JAI Version: 0.0.2
"""A small fully connected neural network built from JAI neurons."""

import random

from brain.activations import sigmoid


class NeuralNetwork:
    """A feed-forward network with one hidden layer.

    Architecture:
        inputs -> hidden neurons -> output neurons
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seed: int = 42,
    ) -> None:
        if input_size < 1 or hidden_size < 1 or output_size < 1:
            raise ValueError("Layer sizes must be positive.")

        rng = random.Random(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_weights = [
            [rng.uniform(-1.0, 1.0) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.hidden_biases = [0.0 for _ in range(hidden_size)]

        self.output_weights = [
            [rng.uniform(-1.0, 1.0) for _ in range(hidden_size)]
            for _ in range(output_size)
        ]
        self.output_biases = [0.0 for _ in range(output_size)]

    def forward(self, inputs: list[float]) -> list[float]:
        """Run inputs through the hidden and output layers."""
        if len(inputs) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} inputs, got {len(inputs)}."
            )

        hidden = []
        for weights, bias in zip(self.hidden_weights, self.hidden_biases):
            z = sum(w * x for w, x in zip(weights, inputs)) + bias
            hidden.append(sigmoid(z))

        outputs = []
        for weights, bias in zip(self.output_weights, self.output_biases):
            z = sum(w * h for w, h in zip(weights, hidden)) + bias
            outputs.append(sigmoid(z))

        return outputs

    def parameter_count(self) -> int:
        """Return the number of trainable parameters."""
        return (
            self.hidden_size * self.input_size
            + self.hidden_size
            + self.output_size * self.hidden_size
            + self.output_size
        )
