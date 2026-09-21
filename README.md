# JAI — Jhave Artificial Intelligence

> Building an artificial intelligence from the ground up.

JAI (Jhave Artificial Intelligence) is an educational, experimental AI project built from first principles. The goal is to understand and implement the foundations of machine learning ourselves rather than starting with a pretrained language model.

## Current Version

**JAI 0.1.0 — The First Language Input**

JAI can now turn text into numerical token IDs. This is the bridge between human-readable language and the numerical data a future language model will learn from.

The tokenizer is deliberately built from scratch. It uses deterministic word, whitespace, and punctuation tokens, vocabulary building, special tokens, unknown-token handling, encode/decode operations, and JSON vocabulary persistence.

## What JAI Has Built

### 0.0.1 — The Seed
- Artificial neuron
- Sigmoid activation
- Weight and bias
- Loss calculation
- Gradient-based training

### 0.0.2 — The First Network
- Multiple neurons
- Hidden layer
- Output layer
- Fully connected forward pass
- Parameter counting
- Network tests

### 0.0.3 — Learning Through Backpropagation
- Output-layer gradients
- Hidden-layer gradients
- Weight and bias updates
- Epoch loss tracking
- Training tests

### 0.1.0 — The First Language Input
- Text splitting
- Vocabulary building
- Special tokens: <PAD>, <UNK>, <BOS>, <EOS>
- Text to token IDs
- Token IDs to text
- Unknown-token handling
- Vocabulary save/load
- Tokenizer tests

## Important Note

JAI is **not a language model yet**. The tokenizer gives JAI a numerical representation of text; the next major step is teaching a model to predict what token comes next.

## Roadmap

| Version | Goal |
|---|---|
| 0.0.1 | Artificial neuron |
| 0.0.2 | Multi-neuron network |
| 0.0.3 | Backpropagation |
| **0.1.0** | **Tokenizer** |
| 0.2 | Language model |
| 0.5 | External memory |
| 1.0 | Local JAI system |

## Principles
- Build from scratch where practical.
- Prefer understandable implementations over black boxes.
- Avoid pretrained models during the foundational stages.
- Keep dependencies minimal.
- Version every meaningful step.
- Test what JAI learns.
- Document how each part works.

## Running
Requires Python 3.10+.

    python main.py
    python -m unittest discover -s tests -v

## Project Structure
    brain/          Neural components
    tokenizer/      Text to token ID conversion
    training/       Learning components
    tests/          Verification
    main.py         JAI entry point