# JAI Version: 0.3.0
"""JAI — attention and Transformer foundations."""

from brain.transformer import TransformerBlock
from tokenizer.tokenizer import Tokenizer


def main() -> None:
    texts = ["JAI learns from examples.", "JAI remembers important words."]
    tokenizer = Tokenizer()
    tokenizer.build_vocabulary(texts)

    # One-hot-like vectors make the attention mechanism easy to inspect.
    model_size = len(tokenizer.tokens)
    token_ids = tokenizer.encode("JAI learns from examples.")
    inputs = []
    for token_id in token_ids:
        vector = [0.0] * model_size
        vector[token_id] = 1.0
        inputs.append(vector)

    block = TransformerBlock(model_size)
    _, weights = block.forward(inputs)

    print("JAI 0.3.0 — Attention / Transformer Foundations")
    print(f"Vocabulary size: {model_size}")
    print(f"Sequence length: {len(inputs)}")
    print("Attention matrix:")
    for row in weights:
        print("  " + " ".join(f"{value:.3f}" for value in row))
    print("\nEach row shows how much that token attends to every token.")
    print("JAI now has the basic mechanism for context-dependent relationships.")


if __name__ == "__main__":
    main()
