# Agentic RAG: Enterprise-Grade AI on Consumer Hardware

![System Graph](assets/img.png)

## The Problem: "Silent Failure"

RAG systems often fail quietly. They return confident hallucinations because dense vector retrieval "smears" precise technical identifiers (like error codes) into generic semantic clusters.
**Engineers on constrained hardware are usually forced to choose: Privacy/Autonomy OR Performance.**

> *Read the full story on Medium:* [**Most RAG Systems Fail Quietly: How I Built a 98% Accurate Agent on a 6GB GPU**](https://medium.com/@stiflerxd/most-rag-systems-fail-quietly-heres-how-i-built-a-98-accurate-agent-on-a-6gb-gpu-4edc9d05116e)

## The Solution

**Agentic RAG** is a production-grade reasoning engine engineered for **100% recall** on consumer hardware (RTX 3050 / 6GB VRAM). It rejects "black box" abstractions for a controllable, graph-based architecture.

It turns your laptop into a **Self-Correcting Research Assistant**.

## Who Should Care?

* **Edge Engineers**: reliable RAG without cloud latency.
* **Privacy Devs**: Zero data egress to OpenAI/Anthropic.
* **Architects**: Explicit control flows (StateMachines) > Nondeterministic Prompts.

---

## ⚡ Leverage (Technical Highlights)

This system is not just a wrapper around a model. It is a **Reasoning Engine** built on four pillars of leverage:

### 0. Multi-Modal Intelligence (V2)

In V2, the system extracts and reasons over **Images and Diagrams** within technical documentation:

* **Ingestion**: Uses `PyMuPDF` to extract visual assets and anchor them to relevant text.
* **Embeddings**: Uses **CLIP** (`clip-ViT-B-32`) to search across both text and visuals in a single vector space.
* **Vision Reasoning**: The Agent calls a Vision-Language Model (Gemini fallback) via the `examine_image` tool to describe retrieved diagrams.
* **Recall**: 62.5% Accuracy on zero-shot visual reasoning tests.

### 1. Hybrid Hyper-Retrieval (100% Recall)

We beat dense-only retrieval by **100% on edge cases** using a 2-stage pipeline:

* **Recall**: Merges **BM25** (Keyword) and **ChromaDB** (Vector) to catch both specific IDs (`0xDEADBEEF`) and semantic concepts.
* **Precision**: Re-ranks candidates using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) on the GPU.
* **Benefit**: You get the broad knowledge of a large model with the precision of a search engine.

### 2. The Omni-Corpus Engine (Small-to-Big Retrieval)

We solved "Context Fragmentation" (where the LLM gets a snippet but lacks the story) using a structural ingestion strategy:

* **Strategy**:
    1. Split full text into **'Parent' chunks** (large, ~2000 chars).
    2. Split each Parent into **'Child' chunks** (small, ~400 chars).
    3. Assign **Parent ID** to Children.
    4. **Storage**: Children -> VectorDB (Search); Parents -> Key-Value Store (Context).
* **Result**: The Agent retrieves the *exact* needle, but reads the *entire* haystack paragraph to answer correctly.

### 3. Automated Self-Correction

Small models (3B params) often struggle with instruction following. We explicitly model "Self-Correction" as a graph node:

* **Reflection**: If the agent perceives a retrieval failure, it creates a new plan.
* **Query Rewriting**: Transforms vague user queries ("why failed?") into precise search terms ("QTP-X failure modes").

---

## 🚀 Key Metrics

| Metric | Value | Hardware Context |
| :--- | :--- | :--- |
| **Model** | Phi-3-mini-4k (3.8B) | 4-bit Quantized |
| **Accuracy (Text)** | **98.2%** | vs 0% Baseline (Hallucination) |
| **Accuracy (Vision)** | **62.5%** | Inaugural V2 Benchmarks |
| **Recall** | **100%** | On Technical Domain Data |
| **Latency** | **~5.8s** | Multi-modal Reasoning Loop |
| **Cost** | **$0.00** | 100% Local Inference (Text/CLIP) |

![Baseline vs RAG](docs/metrics_neon_line.png)

---

## 🛠️ Interface & Usage

We prioritize clean, single-purpose entry points.

### 1. Run the Agent

The main interactive loop. The agent will research its own codebase/docs to answer your questions.

```bash
python scripts/demo.py
```

### 2. Verify Performance

Run the full evaluation suite to generate the metrics and graphs shown above.

```bash
python scripts/evaluate_v2.py
```

### 3. Generate New Datasets

Create "Gold Standard" evaluation data from your own documents using our Synthetic Generator.

```bash
python scripts/generate_dataset.py
```

## 📂 Project Structure

```text
agentic-rag/
├── data/               # Vector Store & Raw Docs
├── models/             # Local GGUF Model Path
├── src/
│   └── agentic_rag/    # Core Logic
│       ├── agent.py    # LangGraph StateMachine
│       ├── ingestor.py # Omni-Corpus Ingestion
│       └── llm.py      # Inference Engine (Local + Fallback)
└── scripts/            # Executable Entry Points
```
