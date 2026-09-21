# JAI Version: 0.2.0
"""Training utilities for JAI's tiny language model."""

import math

from brain.language_model import LanguageModel


def train_language_model(model: LanguageModel, sequences: list[list[int]], learning_rate: float = 0.05, epochs: int = 100) -> list[float]:
    """Train next-token prediction with cross-entropy and SGD."""
    if not sequences:
        raise ValueError("Training sequences cannot be empty.")
    if learning_rate <= 0 or epochs < 1:
        raise ValueError("learning_rate and epochs must be positive.")

    history = []
    for _ in range(epochs):
        total_loss = 0.0
        examples = 0
        eg = [[0.0] * model.embedding_size for _ in range(model.vocabulary_size)]
        owg = [[0.0] * model.embedding_size for _ in range(model.vocabulary_size)]
        obg = [0.0] * model.vocabulary_size

        for sequence in sequences:
            if len(sequence) < 2:
                continue
            for position in range(1, len(sequence)):
                context = sequence[max(0, position - model.context_size):position]
                target = sequence[position]
                probabilities = model.predict_proba(context)
                total_loss += -math.log(max(probabilities[target], 1e-12))
                examples += 1
                delta = probabilities[:]
                delta[target] -= 1.0
                hidden = model._context(context)
                for token_id in range(model.vocabulary_size):
                    for j in range(model.embedding_size):
                        owg[token_id][j] += delta[token_id] * hidden[j]
                    obg[token_id] += delta[token_id]
                for token_id in context:
                    for j in range(model.embedding_size):
                        eg[token_id][j] += sum(delta[k] * model.output_weights[k][j] for k in range(model.vocabulary_size)) / len(context)

        if examples == 0:
            raise ValueError("Sequences must contain at least two tokens.")

        scale = learning_rate / examples
        for token_id in range(model.vocabulary_size):
            for j in range(model.embedding_size):
                model.embeddings[token_id][j] -= scale * eg[token_id][j]
                model.output_weights[token_id][j] -= scale * owg[token_id][j]
            model.output_biases[token_id] -= scale * obg[token_id]
        history.append(total_loss / examples)

    return history
