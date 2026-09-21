# JAI — Jhave Artificial Intelligence

> Building an artificial intelligence from the ground up.

## Current Version

**JAI 0.3.0 — Attention / Transformer Foundations**

JAI now has a from-scratch implementation of single-head scaled dot-product self-attention and a minimal Transformer-style residual block. The 0.2.0 language model remains the trained language-model baseline while these Transformer mechanics are developed and tested separately.

This is intentionally tiny and educational. It is **not an LLM** and does not use pretrained weights or external machine-learning libraries.

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

### 0.0.3 — Learning Through Backpropagation
- Output-layer gradients
- Hidden-layer gradients
- Weight and bias updates
- Epoch loss tracking

### 0.1.0 — The First Language Input
- Vocabulary building
- Special tokens
- Text to token IDs
- Token IDs to text
- Vocabulary persistence

### 0.2.0 — The First Tiny Language Model
- Learned token embeddings
- Context window
- Next-token probabilities
- Softmax output
- Cross-entropy loss
- SGD training
- Next-token prediction

### 0.3.0 — Attention / Transformer Foundations
- Query, key, and value projections
- Scaled dot-product attention
- Attention probability matrix
- Attention-weighted value mixing
- Residual connection
- Inspectable Transformer-style block

## How JAI Learns Language

Text -> Tokenizer -> Token IDs -> Embeddings -> Attention -> Context-aware representations -> Next-token probabilities -> Loss -> Gradient descent

At 0.3.0, attention is implemented as a foundation and inspection tool. It is not yet the complete trained Transformer language model.

## Important Note

JAI is still extremely small. The current model uses averaged embeddings rather than a Transformer, so it cannot yet understand long-range relationships, attention, or rich language. Those are later engineering steps.

## Roadmap

| Version | Goal |
|---|---|
| 0.0.1 | Artificial neuron |
| 0.0.2 | Multi-neuron network |
| 0.0.3 | Backpropagation |
| 0.1.0 | Tokenizer |
| **0.2.0** | **Tiny language model** |
| **0.3.0** | **Attention / Transformer foundations** |
| 0.3 | Transformer/attention foundations |
| 0.5 | External memory |
| 1.0 | Local JAI system |

## Principles

- Build from scratch where practical.
- Prefer understandable implementations over black boxes.
- Avoid pretrained models during foundational stages.
- Keep dependencies minimal.
- Version every meaningful step.
- Test what JAI learns.
- Document how each part works.

## Running

Requires Python 3.10+.

    python main.py
    python -m unittest discover -s tests -v
