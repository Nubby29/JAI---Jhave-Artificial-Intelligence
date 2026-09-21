# JAI Version: 0.14.7
"""JAI — instant-start interactive chat with optional model training."""

import sys
from pathlib import Path

from brain.transformer import TransformerLanguageModel
from chat.corpus import DIALOGUES
from chat.session import ChatSession
from memory.manager import MemoryManager
from tokenizer.tokenizer import Tokenizer
from training.trainer import train_transformer_language_model


CHECKPOINT = Path("jai_chat_model.json")
VOCABULARY = Path("jai_chat_vocab.json")
MEMORY_ROOT = Path("memory")


def build_chat_model(train: bool = False):
    """Load a trained model instantly, or train only when explicitly requested."""
    texts = [f"User: {user}\nJAI: {assistant}" for user, assistant in DIALOGUES]
    tokenizer = Tokenizer()
    tokenizer.build_vocabulary(texts)

    if CHECKPOINT.exists() and VOCABULARY.exists():
        try:
            loaded_tokenizer = Tokenizer.load(VOCABULARY)
            model = TransformerLanguageModel.load(CHECKPOINT)
            if len(loaded_tokenizer.tokens) == model.vocabulary_size:
                return model, loaded_tokenizer, False
        except (ValueError, KeyError, OSError):
            pass

    if not train:
        model = TransformerLanguageModel(
            len(tokenizer.tokens),
            model_size=8,
            context_size=32,
            seed=7,
        )
        return model, tokenizer, True

    sequences = [
        tokenizer.encode(text, add_bos=True, add_eos=True)
        for text in texts
    ]
    model = TransformerLanguageModel(
        len(tokenizer.tokens),
        model_size=8,
        context_size=32,
        seed=7,
    )
    print("Training JAI's language model...")
    history = train_transformer_language_model(
        model,
        sequences,
        learning_rate=0.025,
        epochs=80,
    )
    model.save(CHECKPOINT)
    tokenizer.save(VOCABULARY)
    print(f"Training complete: {history[0]:.4f} -> {history[-1]:.4f}")
    return model, tokenizer, False


def main() -> None:
    train = "--train" in sys.argv[1:]
    memory = MemoryManager(MEMORY_ROOT)
    model, tokenizer, bootstrap = build_chat_model(train=train)
    chat = ChatSession(
        model,
        tokenizer,
        memory=memory,
        max_new_tokens=32,
        temperature=0.35,
        bootstrap=bootstrap,
    )

    print("JAI 0.14.7 — Interactive Chat + Permanent Memory")
    if bootstrap:
        print("Chat-ready mode: no trained checkpoint found.")
        print("Run 'python main.py --train' once to train the language model.")
    else:
        print("Trained language model loaded.")
    print("Type 'exit' to stop.")
    stats = memory.stats()
    print(
        f"Memory loaded: {stats['encounters']} encounters, "
        f"{stats['records']} organized records."
    )
    print()

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nJAI: Goodbye!")
            break

        if message.lower() in {"exit", "quit"}:
            print("JAI: Goodbye!")
            break

        print(f"JAI: {chat.reply(message)}")


if __name__ == "__main__":
    main()
