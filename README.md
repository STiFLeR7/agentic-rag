# Agentic RAG - Multi-Modal Reasoning Engine

<div align="center">

![Agentic RAG](https://img.shields.io/badge/Agentic--RAG-Vision%20%2B%20Reasoning-brightgreen?style=for-the-badge&logo=brain&logoColor=white)

**Perceive • Retrieve • Re-Rank • Reason • Audit**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-000000?style=flat-square&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-3fb1e3?style=flat-square&logo=google-cloud&logoColor=white)](https://trychroma.com)
[![CLIP](https://img.shields.io/badge/OpenAI--CLIP-Multi--Modal-6c5ce7?style=flat-square&logo=openai&logoColor=white)](https://openai.com/research/clip)
[![Phi-3](https://img.shields.io/badge/Phi--3-Local--SLM-00a4ef?style=flat-square&logo=microsoft&logoColor=white)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

</div>

---

## 🎯 Overview

**Agentic RAG** is a production-grade, multi-modal reasoning engine designed to run specifically on **constrained consumer hardware** (RTX 3050 / 6GB VRAM). Unlike traditional "Silent Failure" RAG systems, it uses a self-corrective StateGraph architecture to achieve **100% recall** on technical domain data.

In **V2**, the system transcends text, integrating **CLIP-based visual perception** to ingest, retrieve, and reason over diagrams and images within technical documentation.

---

## 🏗️ Architecture: The 4 Pillars of Leverage

### 1. Multi-Modal Perception (V2)

The system treats visual assets as first-class citizens:

* **Extraction**: Uses `PyMuPDF` to extract images from PDFs and anchors them to surrounding textual context.
* **Embeddings**: Employs **CLIP (ViT-B-32)** for a joint text-image semantic space.
* **Vision-Aware Agent**: Uses a Vision-Language Model (Gemini Flash) via an `examine_image` tool for nuanced analysis of retrieved diagrams.

### 2. Hybrid Hyper-Retrieval

We solve the "Semantic Smear" problem of dense vector search:

* **Recall**: Merges **BM25 (Keyword)** and **ChromaDB (Vector)** in a hybrid pipeline.
* **Re-Ranking**: Employs a local **TinyBERT Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) on GPU to surgically identify the most relevant context.

### 3. Omni-Corpus Management

Solves "Context Fragmentation" where the model gets small bits but lacks the full technical narrative:

* **Parent-Child Indexing**: Children (~400 chars) are used for high-precision search; upon a hit, the full **Parent context** (~2000 chars) is retrieved from a JSON store.

### 4. Graph-Based Self-Correction

Built on **LangGraph**, the agent iterates until it finds the "Gold" answer:

* **Query Rewriter**: Transforms vague user input into precise technical queries.
* **ReAct Loop**: Explicit Reasoning-Action-Observation loop that manages tool failure and hallucination.

---

## 🚀 Technical Specifications

| Feature | v1 (Baseline) | v2 (Multi-Modal) |
| :--- | :--- | :--- |
| **Main LLM** | Phi-3-mini-4k-instruct | Phi-3-mini-4k-instruct |
| **Fallback LLM** | Gemini 1.5 Pro (REST) | Gemini 2.0 Flash (REST) |
| **Embedding Model** | `all-MiniLM-L6-v2` | **CLIP (ViT-B-32)** |
| **Embedding Method** | Text-Only Semantic | **Joint Text-Image Semantic** |
| **Vector Space** | ChromaDB (Local) | ChromaDB (Local + Metadata) |
| **Context Window** | 4,096 Tokens | **128K - 1M (Cloud Fallback)** |
| **Tokens / Sec** | ~15 TPS | ~15 TPS (Local) |
| **Re-Ranker** | None | **TinyBERT (Cross-Encoder)** |
| **VRAM Usage** | ~4.2 GB | **~5.5 GB (on RTX 3050)** |

---

## 📊 Performance Benchmarks

### Project Orion (Technical Dataset)

Validation performed on 50+ fictional technical documentation pairs using the `Gold-Standard` dataset generator.

| Metric | local Phi-3 (Zero-Shot) | Agentic RAG (V2) |
| :--- | :--- | :--- |
| **Accuracy (Text)** | 0% (Hallucination) | **98.2%** |
| **Accuracy (Vision)**| N/A | **62.5%** |
| **Recall** | 12% | **100%** |
| **Mean Latency** | ~2.1s | **~5.8s** |

---

## 📂 Project Structure

```text
agentic-rag/
├── src/agentic_rag/
│   ├── agent.py        # LangGraph StateGraph & ReAct Loop
│   ├── ingestor.py     # Parent-Child Ingestion & Image Extraction
│   ├── retriever.py    # Hybrid (BM25 + Chroma) + Cross-Encoder
│   ├── embedding.py    # CLIP Multi-modal Embedding Logic
│   ├── llm.py          # Local Llama-cpp + Gemini REST Fallback
│   └── tools.py        # Discovery & Vision Analaysis Tools
├── scripts/
│   ├── demo_v2.py      # End-to-end Multi-modal Demo
│   ├── evaluate_v2.py  # Text-based RAG Benchmarks
│   └── evaluate_vision.py # Vision-based RAG Benchmarks
└── data/               # Vector Store & Vision Cache
```

---

## 🛠️ Usage

### ⚙️ Prerequisites

* Python 3.10+
* NVIDIA GPU (6GB+ VRAM recommended for local Re-ranking)
* Gemini API Key (for vision reasoning/fallback)

### 1. Ingest Documents

```bash
python scripts/test_v2_ingestion.py
```

### 2. Run Interactive Agent

```bash
python scripts/demo_v2.py
```

### 3. Run Benchmarks

```bash
python scripts/evaluate_vision.py
```

---

<div align="center">

**Built with ❤️ by [STiFLeR7](https://github.com/STiFLeR7)**

*Enterprise Intelligence without Cloud Reliance.*

</div>
