# Comparison: Agentic RAG vs Dify

This document maps the architectural concepts of **agentic-rag** to **Dify**, a popular open-source LLM app development platform. While Dify provides a high-level visual orchestration layer, `agentic-rag` implements these concepts from first principles in code, specifically optimized for constrained local hardware (Phi-3-mini / RTX 3050).

## 1. Core Architecture

| Concept | Dify (Visual/Cloud) | Agentic RAG (Code/Local) |
| :--- | :--- | :--- |
| **Orchestration** | **Workflow Engine**: A DAG (Directed Acyclic Graph) of distinct nodes (Start, LLM, Knowledge, Tool, End). | **ReAct Loop**: A cyclic `while` loop in `Agent.run()`. The "graph" is dynamic, emergent from the "Think/Action" loop rather than pre-defined edges. |
| **State Management** | **Conversation Variables**: Defined in UI, carried across nodes. | **Message History**: A list of `Dict[str, str]` (System, User, Assistant, Tool) managed explicitly in Python memory. |
| **Logic Control** | **Conditional Branches**: Visual IF/ELSE blocks. | **Python Logic**: Native `if/else` within Tools or the Agent's reasoning logic. |

## 2. Knowledge & Retrieval

| Feature | Dify "Knowledge" | Agentic RAG "Pillar II" |
| :--- | :--- | :--- |
| **Indexing** | Segmented cleaning, chunking (fixed/parent-child), and embedding (remote/local). | **Components**: `VectorStore` (ChromaDB) + `Retriever`. Uses `SentenceTransformer` for local embeddings. Supports simple semantic search with configurable Top-K. |
| **Retrieval Strategy** | Hybrid Search (Keyword + Vector), Re-ranking. | **Semantic Search**: Pure vector similarity (Cosine). Focus is on *transparency* of the score rather than complex re-ranking pipelines (due to compute constraints). |
| **Citation** | built-in "Citation" feature in responses. | **Explicit Observation**: The `search_knowledge_base` tool returns raw text which the LLM must synthesize. Citations are encouraged via system prompt. |

## 3. Agents & Tools

| Component | Dify "Agent" | Agentic RAG "Pillar III" |
| :--- | :--- | :--- |
| **Definition** | A specific node type or "Agent Mode" app. | **Class**: `Agent` in `src/agentic_rag/agent.py`. It wraps the LLM with a specific System Prompt structure (ReAct). |
| **Tools** | **Plugins**: First-party (Google Search, DALLE) or API-based. | **Python Protocol**: `Tool` protocol in `src/agentic_rag/tools.py`. Any Python function can be a tool (e.g., `ReadFileTool`). |
| **Routing** | Model decides internally or via specific Router Node. | **Prompt-Driven**: The LLM decides to call a tool by outputting `Action: <name>`. |

## 4. Why "agentic-rag"?

Dify represents the **Productization** of agents—hiding complexity to enable rapid app building.
`agentic-rag` represents the **Engineering** of agents—exposing complexity to enable control and understanding.

### Specific Advantages of this Implementation:
1.  **Hardware Alignment**: Dify's Docker stack is heavy (multiple containers, Postgres, Redis, Sandbox). `agentic-rag` runs in a single Python process (<200MB overhead + Model VRAM).
2.  **Transparency**: You can step-debug the actual "Think" loop. In Dify, the internal prompt engineering is often abstracted behind the "Agent" mode.
3.  **Local-First**: No dependence on OpenAI/Anthropic. Designed for the `Phi-3-mini` class of models.

## 5. Future Alignment
We aim to contribute to Dify by:
1.  Providing benchmarks for 3B-parameter models in agentic loops.
2.  Demonstrating "Hard" failure recovery patterns that can be adopted by Dify's workflow engine.
