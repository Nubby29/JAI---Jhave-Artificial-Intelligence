# JAI Version: 0.4.0
"""JAI — the first complete decoder-style Transformer foundation."""

from brain.transformer import TransformerLanguageModel
from tokenizer.tokenizer import Tokenizer


def main() -> None:
    texts = [
        "JAI learns from examples.",
        "JAI learns from data.",
        "JAI thinks about words.",
        "JAI predicts the next token.",
    ]
    tokenizer = Tokenizer()
    tokenizer.build_vocabulary(texts)

    model = TransformerLanguageModel(
        len(tokenizer.tokens),
        model_size=16,
        context_size=8,
        seed=7,
    )

    prompt = tokenizer.encode("JAI learns", add_bos=True)
    probabilities, attention = model.forward(prompt)

    print("JAI 0.4.0 — First Decoder-Style Transformer")
    print(f"Vocabulary size: {len(tokenizer.tokens)}")
    print(f"Model size: {model.model_size}")
    print(f"Context size: {model.context_size}")
    print(f"Model parameters: {model.parameter_count()}")
    print(f"Attention rows: {len(attention)}")
    next_id = max(range(len(probabilities)), key=probabilities.__getitem__)
    print(f"Prompt: {tokenizer.decode(prompt, skip_special=True)!r}")
    print(f"Predicted next token: {tokenizer.id_to_token[next_id]!r}")


if __name__ == "__main__":
    main()
