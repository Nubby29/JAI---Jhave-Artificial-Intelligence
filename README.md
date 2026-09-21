# JAI — Jhave Artificial Intelligence

> Building an artificial intelligence from the ground up.

## Current Version

**JAI 0.5.1 — Trainable Transformer + Checkpoints + Generation**

JAI now trains its decoder-style Transformer end-to-end using analytical backpropagation implemented with Python's standard library.

This remains a tiny educational model. It is **not an LLM**, and it does not use pretrained weights or external machine-learning libraries.

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

### 0.4.0 — Decoder-Style Transformer
- Token embeddings
- Sinusoidal positional encoding
- Causal self-attention
- Residual connections
- Layer normalization
- ReLU feed-forward network
- Next-token softmax output

### 0.4.1 — Transformer Backpropagation
- Cross-entropy gradient
- Output projection gradients
- Feed-forward gradients
- Layer-normalization gradients
- Attention softmax gradients
- Query/key/value gradients
- Embedding gradients
- End-to-end SGD
- Token generation

### 0.5.0 / 0.5.1 — Persistence and Generation
- JSON model checkpoints
- Model loading
- Temperature-based sampling
- Configurable generation length

## How JAI Learns

Text -> Tokenizer -> Token IDs -> Transformer -> Next-token probabilities -> Loss -> Backpropagation -> Gradient descent -> Updated parameters

At 0.4.1, JAI can actually change its Transformer parameters from examples instead of only running a forward pass.

## Important Note

JAI is intentionally tiny. A model this small and trained on a tiny dataset will not produce general human-level language. The purpose is to understand and build the machinery ourselves.

## Roadmap

| Version | Goal |
|---|---|
| 0.0.1 | Artificial neuron |
| 0.0.2 | Multi-neuron network |
| 0.0.3 | Backpropagation |
| 0.1.0 | Tokenizer |
| 0.2.0 | Tiny language model |
| 0.3.0 | Attention / Transformer foundations |
| 0.4.0 | Decoder-style Transformer forward pass |
| 0.4.1 | Transformer backpropagation + generation |
| **0.5.0 / 0.5.1** | **Checkpoints + improved generation** |
| 0.5.0 | Training data and checkpoint persistence |
| 0.5.1 | Text generation improvements |
| 0.6.0 | Interactive chat |
| 0.7.0 | Conversation memory |
| 1.0.0 | Local JAI system |

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
