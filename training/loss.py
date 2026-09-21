# JAI Version: 0.0.1
"""Loss functions used by the first JAI trainer."""


def mean_squared_error(prediction: float, target: float) -> float:
    """Return the squared error for one prediction."""
    return (prediction - target) ** 2
