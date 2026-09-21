# JAI Version: 0.2.0
"""A tiny neural next-token language model built from scratch."""

import math
import random


class LanguageModel:
    """Embedding + linear softmax model for next-token prediction."""

    def __init__(self, vocabulary_size: int, embedding_size: int = 16, context_size: int = 4, seed: int = 42) -> None:
        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least 2.")
        if embedding_size < 1 or context_size < 1:
            raise ValueError("embedding_size and context_size must be positive.")
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.context_size = context_size
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(embedding_size)
        self.embeddings = [[rng.uniform(-scale, scale) for _ in range(embedding_size)] for _ in range(vocabulary_size)]
        self.output_weights = [[rng.uniform(-scale, scale) for _ in range(embedding_size)] for _ in range(vocabulary_size)]
        self.output_biases = [0.0] * vocabulary_size

    def _context(self, token_ids: list[int]) -> list[float]:
        if not token_ids:
            raise ValueError("Context cannot be empty.")
        context = token_ids[-self.context_size:]
        if any(token_id < 0 or token_id >= self.vocabulary_size for token_id in context):
            raise ValueError("Token ID is outside the vocabulary.")
        return [sum(self.embeddings[token_id][j] for token_id in context) / len(context) for j in range(self.embedding_size)]

    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        maximum = max(logits)
        exps = [math.exp(value - maximum) for value in logits]
        total = sum(exps)
        return [value / total for value in exps]

    def predict_proba(self, context: list[int]) -> list[float]:
        hidden = self._context(context)
        logits = [sum(w * h for w, h in zip(weights, hidden)) + bias for weights, bias in zip(self.output_weights, self.output_biases)]
        return self._softmax(logits)

    def predict_next(self, context: list[int]) -> int:
        probabilities = self.predict_proba(context)
        return max(range(self.vocabulary_size), key=probabilities.__getitem__)

    def parameter_count(self) -> int:
        return self.vocabulary_size * self.embedding_size * 2 + self.vocabulary_size
