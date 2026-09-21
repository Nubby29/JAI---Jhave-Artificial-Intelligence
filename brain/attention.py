# JAI Version: 0.3.0
"""Self-attention foundations built from scratch for JAI."""

import math


class SelfAttention:
    """Single-head scaled dot-product self-attention."""

    def __init__(self, model_size: int) -> None:
        if model_size < 1:
            raise ValueError("model_size must be positive.")
        self.model_size = model_size
        self.query_weights = self._identity()
        self.key_weights = self._identity()
        self.value_weights = self._identity()

    def _identity(self) -> list[list[float]]:
        return [
            [1.0 if row == column else 0.0 for column in range(self.model_size)]
            for row in range(self.model_size)
        ]

    def _project(self, vectors: list[list[float]], weights: list[list[float]]) -> list[list[float]]:
        return [
            [
                sum(vector[k] * weights[k][j] for k in range(self.model_size))
                for j in range(self.model_size)
            ]
            for vector in vectors
        ]

    @staticmethod
    def _softmax(values: list[float]) -> list[float]:
        maximum = max(values)
        exponentials = [math.exp(value - maximum) for value in values]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def forward(self, inputs: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
        """Return (attended_vectors, attention_weights)."""
        if not inputs:
            raise ValueError("inputs cannot be empty.")
        if any(len(vector) != self.model_size for vector in inputs):
            raise ValueError("Every input vector must match model_size.")

        queries = self._project(inputs, self.query_weights)
        keys = self._project(inputs, self.key_weights)
        values = self._project(inputs, self.value_weights)
        attention_weights = []
        outputs = []
        scale = math.sqrt(self.model_size)

        for query in queries:
            scores = [
                sum(query[k] * key[k] for k in range(self.model_size)) / scale
                for key in keys
            ]
            weights = self._softmax(scores)
            attention_weights.append(weights)
            outputs.append([
                sum(weights[token] * values[token][j] for token in range(len(values)))
                for j in range(self.model_size)
            ])

        return outputs, attention_weights
