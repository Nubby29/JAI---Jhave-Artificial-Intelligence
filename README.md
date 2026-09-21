# JAI — Jhave Artificial Intelligence

> Building an artificial intelligence from the ground up.

JAI (Jhave Artificial Intelligence) is an educational, experimental AI project built from first principles. The goal is to understand and implement the foundations of machine learning ourselves rather than starting with a pretrained language model.

## Philosophy

JAI will grow one layer at a time:

1. Artificial neuron
2. Neural network
3. Training and optimization
4. Tokenization
5. Language modeling
6. External memory
7. Tools and interaction

The early versions are intentionally small and transparent. Every component should be understandable, inspectable, and testable.

## Current Version

**JAI 0.0.1 — The Seed**

The first version contains the foundation of a trainable artificial neuron. It learns a simple numerical relationship through gradient-based training.

This is **not yet a language model**. It is the first building block that will eventually lead toward one.

## Principles

- Build from scratch where practical.
- Prefer understandable implementations over black boxes.
- Avoid pretrained models during the foundational stages.
- Keep dependencies minimal.
- Version every meaningful step.
- Test what JAI learns.
- Document how each part works.

## Roadmap

| Version | Goal |
|---|---|
| 0.0.1 | Artificial neuron |
| 0.0.2 | Multi-neuron network |
| 0.0.3 | Learning system |
| 0.1 | Tokenizer |
| 0.2 | Language model |
| 0.5 | External memory |
| 1.0 | Local JAI system |

## Running

Requires Python 3.

```bash
python main.py
```

## Project Structure

```text
brain/          Neural components
training/       Learning components
tests/          Verification
main.py         JAI entry point
```

JAI is a learning project. The architecture will evolve as we discover what each layer requires.
