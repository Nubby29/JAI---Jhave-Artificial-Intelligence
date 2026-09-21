# JAI Version: 0.4.1
"""Training utilities for JAI's Transformer."""

import math

from brain.language_model import LanguageModel
from brain.transformer import TransformerLanguageModel


def train_language_model(model: LanguageModel, sequences: list[list[int]], learning_rate: float = 0.05, epochs: int = 100) -> list[float]:
    """Train the original tiny language model with cross-entropy and SGD."""
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
                        eg[token_id][j] += sum(
                            delta[k] * model.output_weights[k][j]
                            for k in range(model.vocabulary_size)
                        ) / len(context)
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


def train_transformer_language_model(
    model: TransformerLanguageModel,
    sequences: list[list[int]],
    learning_rate: float = 0.01,
    epochs: int = 100,
) -> list[float]:
    """Train the decoder Transformer end-to-end using analytical backpropagation."""
    if not sequences:
        raise ValueError("Training sequences cannot be empty.")
    if learning_rate <= 0 or epochs < 1:
        raise ValueError("learning_rate and epochs must be positive.")

    history = []
    for _ in range(epochs):
        total_loss = 0.0
        examples = 0
        gradients = _zero_gradients(model)
        for sequence in sequences:
            if len(sequence) < 2:
                continue
            for position in range(1, len(sequence)):
                context = sequence[max(0, position - model.context_size):position]
                target = sequence[position]
                loss, example_gradients = model.loss_and_gradients(context, target)
                total_loss += loss
                examples += 1
                _add_gradients(gradients, example_gradients)
        if examples == 0:
            raise ValueError("Sequences must contain at least two tokens.")
        model.apply_gradients(gradients, learning_rate / examples)
        history.append(total_loss / examples)
    return history


def _zero_gradients(model):
    hidden_size = model.model_size * 2
    return {
        "embeddings": [[0.0] * model.model_size for _ in range(model.vocabulary_size)],
        "query_weights": [[0.0] * model.model_size for _ in range(model.model_size)],
        "key_weights": [[0.0] * model.model_size for _ in range(model.model_size)],
        "value_weights": [[0.0] * model.model_size for _ in range(model.model_size)],
        "output_weights": [[0.0] * model.model_size for _ in range(model.vocabulary_size)],
        "output_biases": [0.0] * model.vocabulary_size,
        "ffn_in": [[0.0] * hidden_size for _ in range(model.model_size)],
        "ffn_out": [[0.0] * model.model_size for _ in range(hidden_size)],
    }


def _add_gradients(total, current):
    for name in ("embeddings", "query_weights", "key_weights", "value_weights", "output_weights", "ffn_in", "ffn_out"):
        for i in range(len(total[name])):
            for j in range(len(total[name][i])):
                total[name][i][j] += current[name][i][j]
    for i in range(len(total["output_biases"])):
        total["output_biases"][i] += current["output_biases"][i]
