# JAI Version: 0.2.0
"""JAI — the first tiny language model."""

from brain.language_model import LanguageModel
from tokenizer.tokenizer import Tokenizer
from training.trainer import train_language_model


def main() -> None:
    texts = ["JAI learns.", "JAI learns.", "JAI learns.", "JAI thinks.", "JAI thinks."]
    tokenizer = Tokenizer()
    tokenizer.build_vocabulary(texts)
    sequences = [tokenizer.encode(text, add_bos=True, add_eos=True) for text in texts]

    model = LanguageModel(len(tokenizer.tokens), embedding_size=12, context_size=3, seed=7)
    history = train_language_model(model, sequences, learning_rate=0.25, epochs=300)

    print("JAI 0.2.0 — The First Tiny Language Model")
    print(f"Vocabulary size: {len(tokenizer.tokens)}")
    print(f"Model parameters: {model.parameter_count()}")
    print(f"Initial loss: {history[0]:.6f}")
    print(f"Final loss:   {history[-1]:.6f}")

    prompt = tokenizer.encode("JAI", add_bos=True)
    next_id = model.predict_next(prompt)
    print("Prompt: JAI")
    print(f"Predicted next token: {tokenizer.id_to_token[next_id]!r}")


if __name__ == "__main__":
    main()
