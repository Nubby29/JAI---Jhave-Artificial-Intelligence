# JAI Version: 0.0.3
"""JAI — the first learning neural network."""

from brain.network import NeuralNetwork
from training.trainer import train_network


def main() -> None:
    jai = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, seed=7)

    samples = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [1.0]),
    ]

    initial_loss = sum(
        (jai.forward(inputs)[0] - targets[0]) ** 2
        for inputs, targets in samples
    ) / len(samples)

    history = train_network(jai, samples, learning_rate=2.0, epochs=2000)

    print("JAI 0.0.3 — Learning Through Backpropagation")
    print("Architecture: 2 inputs -> 4 hidden -> 1 output")
    print(f"Trainable parameters: {jai.parameter_count()}")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {history[-1]:.6f}")

    print("\nLearned responses:")
    for inputs, targets in samples:
        prediction = jai.forward(inputs)[0]
        print(
            f"inputs={inputs} | target={targets[0]:.1f} | "
            f"prediction={prediction:.4f}"
        )


if __name__ == "__main__":
    main()
