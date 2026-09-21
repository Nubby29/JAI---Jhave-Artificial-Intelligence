# JAI — Jhave Artificial Intelligence

> Building an artificial intelligence from the ground up.

## Current Version

**JAI 0.7.0 — Persistent Memory**

JAI now has a permanent external memory system. Every chat encounter can be recorded as an original raw memory and organized into a category for later recall.

This is still a small educational language model, **not an LLM**. It has no pretrained weights and no external machine-learning libraries.

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
- User/JAI prompt formatting

### 0.8.0 — Permanent Memory + Instant-Start Chat
- Permanent `memory/` root
- Raw encounter archive
- Organized memory categories
- Persistent memory index
- Keyword-based memory recall
- Recall statistics
- Conversation memory saved automatically
- Recalled memories supplied back to the chat session
- Original encounters are retained rather than overwritten

## Memory Architecture

JAI's memory is intentionally separate from its neural model.

```
memory/
├── raw/
│   └── YYYY/MM/
│       └── encounter.json
├── organized/
│   ├── conversations/
│   ├── knowledge/
│   ├── people/
│   ├── concepts/
│   ├── experiences/
│   ├── tasks/
│   ├── skills/
│   ├── procedures/
│   ├── errors/
│   └── relationships/
└── index.json
```

The design separates **what JAI encountered** from **how JAI organized it**.

For example:

```
Encounter
   ↓
Raw permanent record
   ↓
Category + keywords + metadata
   ↓
Persistent organized memory
   ↓
Future recall
   ↓
Context supplied to JAI
```

This is the foundation for future learning from experience.

## Long-Term Goal

The long-term goal is not merely to make JAI save conversations.

JAI should eventually be able to:

```
ENCOUNTER
    ↓
UNDERSTAND
    ↓
REMEMBER
    ↓
RECALL
    ↓
REASON
    ↓
ACT
    ↓
EVALUATE RESULT
    ↓
LEARN FROM RESULT
    └──────────────→ MEMORY
```

That requires additional systems for semantic understanding, relationships between memories, task planning, learned procedures, action execution, and experience evaluation. Version 0.7.0 establishes the persistent-memory foundation for those systems; it does not claim that JAI already performs those capabilities perfectly.

## How JAI Works

Text -> Tokenizer -> Token IDs -> Embeddings -> Causal Attention -> Feed-Forward Network -> Next-token probabilities -> Generation -> Chat

The memory layer now surrounds the conversation:

```
User message
    ↓
Memory recall
    ↓
Chat context
    ↓
JAI response
    ↓
Permanent memory
```

The neural model and the memory system have different jobs:
- **Transformer:** learns language patterns and generates responses.
- **Memory:** stores encounters and organized information outside the model.
- **Future reasoning system:** will learn to connect memories and use them to perform tasks.

## Running JAI

Requires Python 3.10+.

    python main.py
    python -m unittest discover -s tests -v

The first chat run trains a small model and creates:
- `jai_chat_model.json`
- `jai_chat_vocab.json`
- `memory/`

Later runs reuse the model, vocabulary, and persistent memory.

## Current Limitations

JAI's memory system is a foundation, not yet human-like understanding.

Currently:
- Memory organization uses transparent rule-based categorization.
- Recall uses keyword matching.
- The Transformer has a small vocabulary and dataset.
- The neural model does not automatically rewrite its weights from every memory.
- JAI does not yet autonomously plan and execute arbitrary tasks.
- A stored memory is not automatically guaranteed to be true.

These limitations are intentional. We are building the components ourselves so we can later replace simple mechanisms with more capable learned systems.

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
| 0.6.0 | Interactive chat |
| **0.7.0** | **Persistent memory** |
| 0.8.0 | Instant-start chat + permanent memory |
| 0.9.0 | Better tokenizer, retrieval, and training pipeline |
| 1.0.0 | Local JAI system |
| 1.x | Task planning, skills, experience learning, and autonomous memory use |

## Principles

- Build from scratch where practical.
- Prefer understandable implementations over black boxes.
- Avoid pretrained models during foundational stages.
- Keep dependencies minimal.
- Preserve original encounters.
- Let JAI organize memory through explicit, inspectable systems.
- Version every meaningful step.
- Test what JAI learns.
- Document how each part works.
