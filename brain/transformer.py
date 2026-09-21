# JAI Version: 0.3.0
"""A minimal Transformer-style encoder foundation for JAI."""

from brain.attention import SelfAttention


class TransformerBlock:
    """Single-head self-attention block with a residual connection."""

    def __init__(self, model_size: int) -> None:
        if model_size < 1:
            raise ValueError("model_size must be positive.")
        self.model_size = model_size
        self.attention = SelfAttention(model_size)

    def forward(self, inputs: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
        attended, weights = self.attention.forward(inputs)
        outputs = [
            [inputs[i][j] + attended[i][j] for j in range(self.model_size)]
            for i in range(len(inputs))
        ]
        return outputs, weights
