# JAI Version: 0.5.1
"""JAI — a trainable decoder-style Transformer."""

from brain.transformer import TransformerLanguageModel
from tokenizer.tokenizer import Tokenizer
from training.trainer import train_transformer_language_model


def main() -> None:
    texts = [
        "JAI learns from examples.",
        "JAI learns from data.",
        "JAI thinks about words.",
        "JAI predicts the next token.",
    ]
    tokenizer = Tokenizer()
    tokenizer.build_vocabulary(texts)
    sequences = [tokenizer.encode(text, add_bos=True, add_eos=True) for text in texts]

    model = TransformerLanguageModel(
        len(tokenizer.tokens),
        model_size=8,
        context_size=8,
        seed=7,
    )
    history = train_transformer_language_model(
        model, sequences, learning_rate=0.03, epochs=120
    )

    prompt = tokenizer.encode("JAI learns", add_bos=True)
    probabilities, attention = model.forward(prompt)
    next_id = max(range(len(probabilities)), key=probabilities.__getitem__)

    print("JAI 0.4.1 — Transformer Backpropagation")
    print(f"Vocabulary size: {len(tokenizer.tokens)}")
    print(f"Model parameters: {model.parameter_count()}")
    print(f"Initial loss: {history[0]:.4f}")
    print(f"Final loss: {history[-1]:.4f}")
    print(f"Prompt: {tokenizer.decode(prompt, skip_special=True)!r}")
    print(f"Predicted next token: {tokenizer.id_to_token[next_id]!r}")
    print(f"Attention rows: {len(attention)}")


if __name__ == "__main__":
    main()
