# JAI Version: 0.0.1
"""Training loop for JAI's first artificial neuron."""

from brain.neuron import Neuron
from training.loss import mean_squared_error


def train_neuron(
    neuron: Neuron,
    samples: list[tuple[float, float]],
    learning_rate: float = 0.5,
    epochs: int = 1000,
) -> list[float]:
    """Train a sigmoid neuron and return the loss after every epoch.

    The neuron learns a binary classification boundary from (x, target) pairs.
    """
    history: list[float] = []

    for _ in range(epochs):
        total_loss = 0.0
        weight_gradient = 0.0
        bias_gradient = 0.0

        for x, target in samples:
            z = neuron.linear_output(x)
            prediction = neuron.forward(x)
            error = prediction - target
            total_loss += mean_squared_error(prediction, target)

            # d(sigmoid(z))/dz = prediction * (1 - prediction)
            dz = 2.0 * error * prediction * (1.0 - prediction)
            weight_gradient += dz * x
            bias_gradient += dz

        count = len(samples)
        neuron.weight -= learning_rate * weight_gradient / count
        neuron.bias -= learning_rate * bias_gradient / count
        history.append(total_loss / count)

    return history
