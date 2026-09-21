# JAI Version: 0.7.0
"""JAI — interactive chat with persistent long-term memory."""

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


def build_chat_model():
    texts = [f"User: {user}\nJAI: {assistant}" for user, assistant in DIALOGUES]
    tokenizer = Tokenizer()
    tokenizer.build_vocabulary(texts)

    if CHECKPOINT.exists() and VOCABULARY.exists():
        loaded_tokenizer = Tokenizer.load(VOCABULARY)
        model = TransformerLanguageModel.load(CHECKPOINT)
        if len(loaded_tokenizer.tokens) == model.vocabulary_size:
            return model, loaded_tokenizer

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
    return model, tokenizer


def main() -> None:
    memory = MemoryManager(MEMORY_ROOT)
    model, tokenizer = build_chat_model()
    chat = ChatSession(
        model,
        tokenizer,
        memory=memory,
        max_new_tokens=32,
        temperature=0.35,
    )

    print("JAI 0.7.0 — Interactive Chat + Permanent Memory")
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
