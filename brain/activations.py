# JAI Version: 0.0.1
"""Activation functions for the JAI neural foundation."""

import math


def sigmoid(x: float) -> float:
    """Return the sigmoid activation of x."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)
