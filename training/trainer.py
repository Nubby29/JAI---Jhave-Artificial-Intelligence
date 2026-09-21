# JAI Version: 0.0.3
"""Backpropagation trainer for JAI's first neural network."""

from brain.activations import sigmoid
from brain.network import NeuralNetwork


def train_network(network: NeuralNetwork, samples: list[tuple[list[float], list[float]]], learning_rate: float = 1.0, epochs: int = 1000) -> list[float]:
    """Train a one-hidden-layer network using backpropagation."""
    if not samples:
        raise ValueError("Training samples cannot be empty.")
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive.")

    history: list[float] = []

    for _ in range(epochs):
        total_loss = 0.0
        hwg = [[0.0] * network.input_size for _ in range(network.hidden_size)]
        hbg = [0.0] * network.hidden_size
        owg = [[0.0] * network.hidden_size for _ in range(network.output_size)]
        obg = [0.0] * network.output_size

        for inputs, targets in samples:
            if len(inputs) != network.input_size or len(targets) != network.output_size:
                raise ValueError("Sample dimensions do not match the network.")

            hidden = []
            for weights, bias in zip(network.hidden_weights, network.hidden_biases):
                hidden.append(sigmoid(sum(w * x for w, x in zip(weights, inputs)) + bias))

            outputs = []
            for weights, bias in zip(network.output_weights, network.output_biases):
                outputs.append(sigmoid(sum(w * h for w, h in zip(weights, hidden)) + bias))

            total_loss += sum((p - t) ** 2 for p, t in zip(outputs, targets)) / network.output_size

            od = [
                2.0 * (p - t) * p * (1.0 - p)
                for p, t in zip(outputs, targets)
            ]

            for oi, delta in enumerate(od):
                for hi, h in enumerate(hidden):
                    owg[oi][hi] += delta * h
                obg[oi] += delta

            hd = []
            for hi, h in enumerate(hidden):
                downstream = sum(
                    od[oi] * network.output_weights[oi][hi]
                    for oi in range(network.output_size)
                )
                hd.append(downstream * h * (1.0 - h))

            for hi, delta in enumerate(hd):
                for ii, x in enumerate(inputs):
                    hwg[hi][ii] += delta * x
                hbg[hi] += delta

        count = len(samples)
        for hi in range(network.hidden_size):
            for ii in range(network.input_size):
                network.hidden_weights[hi][ii] -= learning_rate * hwg[hi][ii] / count
            network.hidden_biases[hi] -= learning_rate * hbg[hi] / count

        for oi in range(network.output_size):
            for hi in range(network.hidden_size):
                network.output_weights[oi][hi] -= learning_rate * owg[oi][hi] / count
            network.output_biases[oi] -= learning_rate * obg[oi] / count

        history.append(total_loss / count)

    return history
