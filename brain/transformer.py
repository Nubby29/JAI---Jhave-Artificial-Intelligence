# JAI Version: 0.4.0
"""A small decoder-style Transformer language model built from scratch."""

import math
import random


class TransformerLanguageModel:
    """Decoder-only Transformer with embeddings, attention, FFN, and softmax."""

    def __init__(
        self,
        vocabulary_size: int,
        model_size: int = 16,
        context_size: int = 8,
        seed: int = 42,
    ) -> None:
        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least 2.")
        if model_size < 1 or context_size < 1:
            raise ValueError("model_size and context_size must be positive.")

        self.vocabulary_size = vocabulary_size
        self.model_size = model_size
        self.context_size = context_size
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(model_size)

        self.embeddings = [
            [rng.uniform(-scale, scale) for _ in range(model_size)]
            for _ in range(vocabulary_size)
        ]
        self.query_weights = self._matrix(rng, scale)
        self.key_weights = self._matrix(rng, scale)
        self.value_weights = self._matrix(rng, scale)
        self.output_weights = self._matrix(rng, scale)

        hidden_size = model_size * 2
        self.ffn_in = [
            [rng.uniform(-scale, scale) for _ in range(hidden_size)]
            for _ in range(model_size)
        ]
        self.ffn_out = [
            [rng.uniform(-scale, scale) for _ in range(model_size)]
            for _ in range(hidden_size)
        ]
        self.output_biases = [0.0] * vocabulary_size

    def _matrix(self, rng: random.Random, scale: float) -> list[list[float]]:
        return [
            [rng.uniform(-scale, scale) for _ in range(self.model_size)]
            for _ in range(self.model_size)
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

    @staticmethod
    def _relu(values: list[float]) -> list[float]:
        return [max(0.0, value) for value in values]

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        mean = sum(vector) / len(vector)
        variance = sum((value - mean) ** 2 for value in vector) / len(vector)
        return [(value - mean) / math.sqrt(variance + 1e-8) for value in vector]

    def _position_encoding(self, position: int) -> list[float]:
        encoding = []
        for i in range(self.model_size):
            angle = position / (10000 ** (2 * (i // 2) / self.model_size))
            encoding.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
        return encoding

    def forward(self, token_ids: list[int]) -> tuple[list[float], list[list[float]]]:
        """Return next-token probabilities and the causal attention matrix."""
        if not token_ids:
            raise ValueError("token_ids cannot be empty.")
        token_ids = token_ids[-self.context_size:]
        if any(token_id < 0 or token_id >= self.vocabulary_size for token_id in token_ids):
            raise ValueError("Token ID is outside the vocabulary.")

        hidden = []
        start = len(token_ids) - len(token_ids)
        for position, token_id in enumerate(token_ids, start=start):
            hidden.append([
                self.embeddings[token_id][j] + self._position_encoding(position)[j]
                for j in range(self.model_size)
            ])

        queries = self._project(hidden, self.query_weights)
        keys = self._project(hidden, self.key_weights)
        values = self._project(hidden, self.value_weights)
        attention = []
        attended = []

        for i, query in enumerate(queries):
            scores = []
            for j, key in enumerate(keys):
                if j > i:
                    scores.append(float("-inf"))
                else:
                    scores.append(
                        sum(query[k] * key[k] for k in range(self.model_size))
                        / math.sqrt(self.model_size)
                    )
            weights = self._softmax(scores)
            attention.append(weights)
            attended.append([
                sum(weights[t] * values[t][j] for t in range(len(values)))
                for j in range(self.model_size)
            ])

        residual = [
            self._normalize([
                hidden[i][j] + attended[i][j] for j in range(self.model_size)
            ])
            for i in range(len(hidden))
        ]

        ffn = []
        for vector in residual:
            expanded = [
                sum(vector[k] * self.ffn_in[k][j] for k in range(self.model_size))
                for j in range(self.model_size * 2)
            ]
            activated = self._relu(expanded)
            ffn.append([
                sum(activated[k] * self.ffn_out[k][j] for k in range(self.model_size * 2))
                for j in range(self.model_size)
            ])

        final = self._normalize([
            residual[-1][j] + ffn[-1][j] for j in range(self.model_size)
        ])
        logits = [
            sum(self.output_weights[token][j] * final[j] for j in range(self.model_size))
            + self.output_biases[token]
            for token in range(self.vocabulary_size)
        ]
        return self._softmax(logits), attention

    def predict_proba(self, context: list[int]) -> list[float]:
        return self.forward(context)[0]

    def predict_next(self, context: list[int]) -> int:
        probabilities = self.predict_proba(context)
        return max(range(self.vocabulary_size), key=probabilities.__getitem__)

    def parameter_count(self) -> int:
        hidden_size = self.model_size * 2
        return (
            self.vocabulary_size * self.model_size
            + 4 * self.model_size * self.model_size
            + self.model_size * hidden_size
            + hidden_size * self.model_size
            + self.vocabulary_size * self.model_size
            + self.vocabulary_size
        )
