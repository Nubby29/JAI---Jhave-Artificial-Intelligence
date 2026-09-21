# JAI Version: 0.0.1
"""JAI — the first executable seed."""

from brain.neuron import Neuron
from training.trainer import train_neuron


def main() -> None:
    # A tiny dataset: JAI learns whether a number is on the
    # positive side of the boundary.
    samples = [
        (-2.0, 0.0),
        (-1.0, 0.0),
        (1.0, 1.0),
        (2.0, 1.0),
    ]

    jai = Neuron()
    history = train_neuron(jai, samples, learning_rate=1.0, epochs=1000)

    print("JAI 0.0.1 — The Seed")
    print(f"Learned weight: {jai.weight:.4f}")
    print(f"Learned bias:   {jai.bias:.4f}")
    print(f"Final loss:     {history[-1]:.6f}")

    for x, target in samples:
        prediction = jai.forward(x)
        print(
            f"x={x:>4.1f} | target={target:.1f} | "
            f"prediction={prediction:.4f}"
        )


if __name__ == "__main__":
    main()
