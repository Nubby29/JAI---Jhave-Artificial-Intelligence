# JAI Version: 0.4.1
"""A trainable decoder-style Transformer built from scratch for JAI."""

import math
import random


class TransformerLanguageModel:
    """Small decoder-only Transformer with analytical backpropagation."""

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

        self.embeddings = [[rng.uniform(-scale, scale) for _ in range(model_size)]
                           for _ in range(vocabulary_size)]
        self.query_weights = self._matrix(rng, scale)
        self.key_weights = self._matrix(rng, scale)
        self.value_weights = self._matrix(rng, scale)
        self.output_weights = self._matrix(rng, scale)

        hidden_size = model_size * 2
        self.ffn_in = [[rng.uniform(-scale, scale) for _ in range(hidden_size)]
                       for _ in range(model_size)]
        self.ffn_out = [[rng.uniform(-scale, scale) for _ in range(model_size)]
                        for _ in range(hidden_size)]
        self.output_biases = [0.0] * vocabulary_size

    def _matrix(self, rng: random.Random, scale: float) -> list[list[float]]:
        return [[rng.uniform(-scale, scale) for _ in range(self.model_size)]
                for _ in range(self.model_size)]

    def _project(self, vectors, weights):
        return [[sum(vector[k] * weights[k][j] for k in range(self.model_size))
                 for j in range(self.model_size)] for vector in vectors]

    @staticmethod
    def _softmax(values):
        maximum = max(values)
        exponentials = [math.exp(value - maximum) for value in values]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    @staticmethod
    def _relu(values):
        return [max(0.0, value) for value in values]

    @staticmethod
    def _normalize(vector):
        mean = sum(vector) / len(vector)
        variance = sum((value - mean) ** 2 for value in vector) / len(vector)
        return [(value - mean) / math.sqrt(variance + 1e-8) for value in vector]

    @staticmethod
    def _normalize_backward(vector, upstream):
        """Backpropagate through parameter-free layer normalization."""
        n = len(vector)
        mean = sum(vector) / n
        centered = [value - mean for value in vector]
        variance = sum(value * value for value in centered) / n
        inv_std = 1.0 / math.sqrt(variance + 1e-8)
        normalized = [value * inv_std for value in centered]
        sum_upstream = sum(upstream)
        sum_product = sum(upstream[i] * normalized[i] for i in range(n))
        return [
            inv_std / n * (n * upstream[i] - sum_upstream - normalized[i] * sum_product)
            for i in range(n)
        ]

    def _position_encoding(self, position):
        encoding = []
        for i in range(self.model_size):
            angle = position / (10000 ** (2 * (i // 2) / self.model_size))
            encoding.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
        return encoding

    def _validate_context(self, token_ids):
        if not token_ids:
            raise ValueError("token_ids cannot be empty.")
        context = token_ids[-self.context_size:]
        if any(token_id < 0 or token_id >= self.vocabulary_size for token_id in context):
            raise ValueError("Token ID is outside the vocabulary.")
        return context

    def _forward_cache(self, token_ids):
        token_ids = self._validate_context(token_ids)
        hidden = []
        positions = []
        for position, token_id in enumerate(token_ids):
            position_vector = self._position_encoding(position)
            positions.append(position_vector)
            hidden.append([self.embeddings[token_id][j] + position_vector[j]
                           for j in range(self.model_size)])

        queries = self._project(hidden, self.query_weights)
        keys = self._project(hidden, self.key_weights)
        values = self._project(hidden, self.value_weights)

        attention = []
        attended = []
        scale = math.sqrt(self.model_size)
        for i, query in enumerate(queries):
            scores = [
                (sum(query[k] * key[k] for k in range(self.model_size)) / scale)
                if j <= i else float("-inf")
                for j, key in enumerate(keys)
            ]
            weights = self._softmax(scores)
            attention.append(weights)
            attended.append([
                sum(weights[t] * values[t][j] for t in range(len(values)))
                for j in range(self.model_size)
            ])

        residual_inputs = [
            [hidden[i][j] + attended[i][j] for j in range(self.model_size)]
            for i in range(len(hidden))
        ]
        residual = [self._normalize(vector) for vector in residual_inputs]

        expanded = []
        activated = []
        ffn = []
        for vector in residual:
            current_expanded = [
                sum(vector[k] * self.ffn_in[k][j] for k in range(self.model_size))
                for j in range(self.model_size * 2)
            ]
            current_activated = self._relu(current_expanded)
            current_ffn = [
                sum(current_activated[k] * self.ffn_out[k][j]
                    for k in range(self.model_size * 2))
                for j in range(self.model_size)
            ]
            expanded.append(current_expanded)
            activated.append(current_activated)
            ffn.append(current_ffn)

        final_input = [residual[-1][j] + ffn[-1][j] for j in range(self.model_size)]
        final = self._normalize(final_input)
        logits = [
            sum(self.output_weights[token][j] * final[j] for j in range(self.model_size))
            + self.output_biases[token]
            for token in range(self.vocabulary_size)
        ]
        probabilities = self._softmax(logits)
        return {
            "token_ids": token_ids,
            "hidden": hidden,
            "queries": queries,
            "keys": keys,
            "values": values,
            "attention": attention,
            "residual_inputs": residual_inputs,
            "residual": residual,
            "expanded": expanded,
            "activated": activated,
            "ffn": ffn,
            "final_input": final_input,
            "final": final,
            "logits": logits,
            "probabilities": probabilities,
        }

    def forward(self, token_ids):
        """Return next-token probabilities and the causal attention matrix."""
        cache = self._forward_cache(token_ids)
        return cache["probabilities"], cache["attention"]

    def loss_and_gradients(self, context, target):
        """Return cross-entropy loss and analytical gradients for one example."""
        cache = self._forward_cache(context)
        if target < 0 or target >= self.vocabulary_size:
            raise ValueError("Target token is outside the vocabulary.")

        d_embeddings = [[0.0] * self.model_size for _ in range(self.vocabulary_size)]
        d_q = [[0.0] * self.model_size for _ in range(self.model_size)]
        d_k = [[0.0] * self.model_size for _ in range(self.model_size)]
        d_v = [[0.0] * self.model_size for _ in range(self.model_size)]
        d_ow = [[0.0] * self.model_size for _ in range(self.vocabulary_size)]
        d_ob = [0.0] * self.vocabulary_size
        d_ffn_in = [[0.0] * (self.model_size * 2) for _ in range(self.model_size)]
        d_ffn_out = [[0.0] * self.model_size for _ in range(self.model_size * 2)]

        probabilities = cache["probabilities"]
        loss = -math.log(max(probabilities[target], 1e-12))
        d_logits = probabilities[:]
        d_logits[target] -= 1.0

        d_final = [0.0] * self.model_size
        for token in range(self.vocabulary_size):
            d_ob[token] += d_logits[token]
            for j in range(self.model_size):
                d_ow[token][j] += d_logits[token] * cache["final"][j]
                d_final[j] += d_logits[token] * self.output_weights[token][j]

        d_final_input = self._normalize_backward(cache["final_input"], d_final)
        d_residual = [[0.0] * self.model_size for _ in cache["residual"]]
        d_ffn = [[0.0] * self.model_size for _ in cache["ffn"]]
        last = len(d_residual) - 1
        for j in range(self.model_size):
            d_residual[last][j] += d_final_input[j]
            d_ffn[last][j] += d_final_input[j]

        d_expanded = [[0.0] * (self.model_size * 2) for _ in cache["expanded"]]
        for i in range(len(cache["ffn"])):
            for hidden_index in range(self.model_size * 2):
                for j in range(self.model_size):
                    d_ffn_out[hidden_index][j] += (
                        cache["activated"][i][hidden_index] * d_ffn[i][j]
                    )
                    d_activated = sum(
                        d_ffn[i][j] * self.ffn_out[hidden_index][j]
                        for j in range(self.model_size)
                    )
                    if cache["expanded"][i][hidden_index] > 0.0:
                        d_expanded[i][hidden_index] += d_activated
            for j in range(self.model_size):
                for hidden_index in range(self.model_size * 2):
                    d_ffn_in[j][hidden_index] += (
                        cache["residual"][i][j] * d_expanded[i][hidden_index]
                    )
                    d_residual[i][j] += (
                        d_expanded[i][hidden_index] * self.ffn_in[j][hidden_index]
                    )

        d_residual_input = [[0.0] * self.model_size for _ in cache["residual_inputs"]]
        for i in range(len(cache["residual"])):
            d_residual_input[i] = self._normalize_backward(
                cache["residual_inputs"][i], d_residual[i]
            )

        d_hidden = [row[:] for row in d_residual_input]
        d_attended = [row[:] for row in d_residual_input]

        d_queries = [[0.0] * self.model_size for _ in cache["queries"]]
        d_keys = [[0.0] * self.model_size for _ in cache["keys"]]
        d_values = [[0.0] * self.model_size for _ in cache["values"]]
        scale = math.sqrt(self.model_size)

        for i in range(len(cache["attention"])):
            d_attention = [0.0] * len(cache["attention"][i])
            for j in range(len(cache["values"])):
                for dimension in range(self.model_size):
                    d_values[j][dimension] += (
                        cache["attention"][i][j] * d_attended[i][dimension]
                    )
                    d_attention[j] += (
                        d_attended[i][dimension] * cache["values"][j][dimension]
                    )

            weighted_sum = sum(
                cache["attention"][i][j] * d_attention[j]
                for j in range(len(cache["attention"][i]))
            )
            for j in range(i + 1):
                d_score = cache["attention"][i][j] * (d_attention[j] - weighted_sum)
                for dimension in range(self.model_size):
                    d_queries[i][dimension] += d_score * cache["keys"][j][dimension] / scale
                    d_keys[j][dimension] += d_score * cache["queries"][i][dimension] / scale

        def projection_backward(inputs, weights, upstream, d_weights):
            d_inputs = [[0.0] * self.model_size for _ in inputs]
            for i in range(len(inputs)):
                for source in range(self.model_size):
                    for destination in range(self.model_size):
                        d_weights[source][destination] += (
                            inputs[i][source] * upstream[i][destination]
                        )
                        d_inputs[i][source] += (
                            upstream[i][destination] * weights[source][destination]
                        )
            return d_inputs

        d_hidden_from_q = projection_backward(
            cache["hidden"], self.query_weights, d_queries, d_q
        )
        d_hidden_from_k = projection_backward(
            cache["hidden"], self.key_weights, d_keys, d_k
        )
        d_hidden_from_v = projection_backward(
            cache["hidden"], self.value_weights, d_values, d_v
        )

        for i, token_id in enumerate(cache["token_ids"]):
            for j in range(self.model_size):
                d_hidden[i][j] += (
                    d_hidden_from_q[i][j] + d_hidden_from_k[i][j] + d_hidden_from_v[i][j]
                )
                d_embeddings[token_id][j] += d_hidden[i][j]

        return loss, {
            "embeddings": d_embeddings,
            "query_weights": d_q,
            "key_weights": d_k,
            "value_weights": d_v,
            "output_weights": d_ow,
            "output_biases": d_ob,
            "ffn_in": d_ffn_in,
            "ffn_out": d_ffn_out,
        }

    def apply_gradients(self, gradients, learning_rate):
        """Apply a complete gradient dictionary to model parameters."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        for token in range(self.vocabulary_size):
            for j in range(self.model_size):
                self.embeddings[token][j] -= learning_rate * gradients["embeddings"][token][j]
                self.output_weights[token][j] -= learning_rate * gradients["output_weights"][token][j]
            self.output_biases[token] -= learning_rate * gradients["output_biases"][token]
        for name, parameter in (
            ("query_weights", self.query_weights),
            ("key_weights", self.key_weights),
            ("value_weights", self.value_weights),
            ("ffn_in", self.ffn_in),
            ("ffn_out", self.ffn_out),
        ):
            gradient = gradients[name]
            for i in range(len(parameter)):
                for j in range(len(parameter[i])):
                    parameter[i][j] -= learning_rate * gradient[i][j]

    def predict_proba(self, context):
        return self.forward(context)[0]

    def predict_next(self, context):
        probabilities = self.predict_proba(context)
        return max(range(self.vocabulary_size), key=probabilities.__getitem__)

    def generate(self, context, max_new_tokens=20, stop_token_id=None, temperature=0.0):
        """Generate token IDs greedily, or sample when temperature is positive."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")
        if temperature < 0:
            raise ValueError("temperature cannot be negative.")
        generated = self._validate_context(context)[:]
        rng = random.Random()
        for _ in range(max_new_tokens):
            probabilities = self.predict_proba(generated)
            if temperature == 0.0:
                next_id = max(range(self.vocabulary_size), key=probabilities.__getitem__)
            else:
                scaled = [math.log(max(p, 1e-12)) / temperature for p in probabilities]
                scaled = self._softmax(scaled)
                draw = rng.random()
                cumulative = 0.0
                next_id = self.vocabulary_size - 1
                for index, probability in enumerate(scaled):
                    cumulative += probability
                    if draw <= cumulative:
                        next_id = index
                        break
            generated.append(next_id)
            if stop_token_id is not None and next_id == stop_token_id:
                break
        return generated

    def parameter_count(self):
        hidden_size = self.model_size * 2
        return (
            self.vocabulary_size * self.model_size
            + 4 * self.model_size * self.model_size
            + self.model_size * hidden_size
            + hidden_size * self.model_size
            + self.vocabulary_size * self.model_size
            + self.vocabulary_size
        )
