# JAI Version: 0.1.0
"""JAI — learning foundations and the first tokenizer."""
from brain.network import NeuralNetwork
from tokenizer.tokenizer import Tokenizer
from training.trainer import train_network

def demonstrate_learning() -> None:
    jai = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, seed=7)
    samples = [([0.0, 0.0], [0.0]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [1.0])]
    initial_loss = sum((jai.forward(x)[0] - y[0]) ** 2 for x, y in samples) / len(samples)
    history = train_network(jai, samples, learning_rate=2.0, epochs=2000)
    print("JAI 0.1.0 — Learning Foundations")
    print(f"Network parameters: {jai.parameter_count()}")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {history[-1]:.6f}")

def demonstrate_tokenizer() -> None:
    tokenizer = Tokenizer()
    training_text = ["JAI learns one step at a time.", "JAI learns from examples.", "The machine reads tokens."]
    tokenizer.build_vocabulary(training_text)
    text = "JAI learns from examples."
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    print("\nTokenizer demonstration:")
    print(f"Vocabulary size: {len(tokenizer.tokens)}")
    print(f"Text:  {text}")
    print(f"IDs:   {ids}")
    print(f"Back:  {tokenizer.decode(ids, skip_special=True)}")

def main() -> None:
    demonstrate_learning()
    demonstrate_tokenizer()

if __name__ == "__main__":
    main()