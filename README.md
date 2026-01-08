# Agentic RAG

**Agentic Retrieval-Augmented Generation on Constrained Hardware.**

`agentic-rag` is a local-first, systems-oriented research project focused on building reliable Agentic RAG pipelines that operate on consumer-grade hardware (specifically an RTX 3050 Laptop GPU with 6GB VRAM).

## Core Philosophy
- **Agent-driven reasoning**, not static prompt chaining.
- **Retrieval as a decision process**, not a lookup.
- **Explicit control** over memory, tools, and failure modes.
- **Hardware-aware design**: 6GB VRAM, Batch Size = 1.

## Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU (RTX 3050 or likely similar) with CUDA Toolkit installed.
- 6GB+ VRAM recommended.

### Installation
```bash
pip install -e .
```

### Model Setup
Models are stored in `models/`. The default recommended model is **Phi-3-mini-4k** (Quantized Q4).

## Architecture
- **Inference**: `llama-cpp-python` (Phi-3-mini-4k-instruct-q4_1.gguf)
- **Vector Store**: `chromadb` (Persistent)
- **Agent**: ReAct Loop with custom Tool Registry
- **Comparison**: See [Dify vs Agentic RAG](docs/dify_comparison.md) for architectural differences.
ieval and reasoning.
3.  **Agentic AI**: Explicit tooling and failure handling.
