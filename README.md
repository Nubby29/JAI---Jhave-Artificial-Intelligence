# JAI — Jhave Artificial Intelligence

> Building an artificial intelligence from the ground up.

## Current Version

**JAI 0.6.0 — Interactive Chat**

JAI can now train its own tiny decoder Transformer from a built-in dialogue dataset and use the learned parameters in an interactive terminal chat.

This is a small educational language model, **not an LLM**. It has no pretrained weights and no external machine-learning libraries.

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

### 0.4.0 — Decoder-Style Transformer
- Token embeddings
- Sinusoidal positional encoding
- Causal self-attention
- Residual connections
- Layer normalization
- ReLU feed-forward network
- Next-token softmax output

### 0.4.1 — Transformer Backpropagation
- Analytical gradients through the Transformer
- End-to-end SGD
- Next-token generation

### 0.5.0 / 0.5.1 — Persistence and Generation
- JSON model checkpoints
- Model loading
- Temperature-based generation
- Reusable generated-token interface

### 0.6.0 — Interactive Chat
- Built-in dialogue training set
- Automatic first-run training
- Saved model and vocabulary
- Interactive terminal conversation
- Basic conversation history
- User/JAI prompt formatting

## How JAI Works

Text -> Tokenizer -> Token IDs -> Embeddings -> Causal Attention -> Feed-Forward Network -> Next-token probabilities -> Loss -> Backpropagation -> Updated parameters -> Generation -> Chat response

The important part is that the chat responses come from the Transformer parameters that JAI trained from examples. There is no pretrained language model behind it.

## Running JAI

Requires Python 3.10+.

    python main.py
    python -m unittest discover -s tests -v

The first chat run trains a small model and creates:
- `jai_chat_model.json`
- `jai_chat_vocab.json`

Later runs reuse those files.

## Current Limitations

JAI is deliberately tiny. The chat model has a small vocabulary, a small dataset, one Transformer block, and a short context window. It can demonstrate learned conversational patterns, but it will not have the broad knowledge or fluency of a modern large language model.

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
| 0.5.0 / 0.5.1 | Checkpoints + generation |
| **0.6.0** | **Interactive chat** |
| 0.7.0 | Conversation memory |
| 0.8.0 | Larger training corpus |
| 0.9.0 | Better tokenizer and training pipeline |
| 1.0.0 | Local JAI system |

## Principles

- Build from scratch where practical.
- Prefer understandable implementations over black boxes.
- Avoid pretrained models during foundational stages.
- Keep dependencies minimal.
- Version every meaningful step.
- Test what JAI learns.
- Document how each part works.
