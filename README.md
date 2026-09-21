# JAI — Jhave Artificial Intelligence

> Building an artificial intelligence from the ground up.

JAI (Jhave Artificial Intelligence) is an educational, experimental AI project built from first principles. The goal is to understand and implement the foundations of machine learning ourselves rather than starting with a pretrained language model.

## Philosophy

JAI grows one layer at a time. The foundational implementations are intentionally small, transparent, and testable.

## Current Version

**JAI 0.0.2 — The First Network**

JAI now has a small feed-forward neural network:

```text
2 inputs → 3 hidden neurons → 1 output neuron
```

The network contains **13 trainable parameters**: 6 hidden-layer weights, 3 hidden biases, 3 output weights, and 1 output bias.

At this stage the network can perform a forward pass, but it is not yet being trained as a network. That is the next major learning step.

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

## Roadmap

| Version | Goal |
|---|---|
| 0.0.1 | Artificial neuron |
| **0.0.2** | **Multi-neuron network** |
| 0.0.3 | Train the network with backpropagation |
| 0.1 | Tokenizer |
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

Requires Python 3.

```bash
python main.py
python -m unittest discover -s tests -v
```

## Project Structure

```text
brain/          Neural components
training/       Learning components
tests/          Verification
main.py         JAI entry point
```
