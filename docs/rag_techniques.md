# Agentic RAG: Technical Deep Dive

This document explains the core technologies and strategies used in `agentic-rag`, along with industry-standard patterns for handling complex data (like PDFs with images).

## 1. Chunking Strategies
Chunking is the process of breaking text into smaller pieces for embedding.

### A. Fixed-Size Chunking (Used in agentic-rag v1)
- **Method**: Split text every `N` characters or tokens (e.g., 512 tokens).
- **Pros**: Fast, predictable, simple.
- **Cons**: Cuts sentences in half; loses context.

### B. Semantic Chunking (Recommended for Production)
- **Method**: Split text based on meaning. If the topic changes (detected via cosine similarity of sentence embeddings), create a new chunk.
- **Why we like it**: Keeps paragraphs together. The "context" remains intact for the LLM.

### C. Recursive Character Chunking (LangChain default)
- **Method**: Try splitting by `\n\n` (paragraphs). If too big, split by `\n` (lines). If still too big, split by space.
- **Verdict**: Best balance for general text.

## 2. Embeddings & Vector Space
Embeddings convert text into a list of numbers (vectors) such that similar meanings are mathematically close.

### The Model: `all-MiniLM-L6-v2`
- **Architecture**: BERT-based Transformer.
- **Dimensions**: 384 dimensions.
- **Size**: ~80MB (Tiny, perfect for local CPU).
- **Vector Space**: A 384-dimensional sphere. "King" and "Queen" are closer than "King" and "Apple".

### Retrieval (Cosine Similarity)
- We calculate the angle between the **Query Vector** and **Document Vectors**.
- Small angle = High similarity.

## 3. Image RAG Strategies (PDFs with Paste-ins)
Often, PDFs contain screenshots or diagrams that have no metadata. Standard text extractors (`pypdf`) ignore them or output garbage.

### How to Retrieve Info from Images?

#### Strategy A: Optical Character Recognition (OCR)
**Tools**: `tesseract`, `pdf2image`.
1.  **Detect**: Use `pdf2image` to convert PDF pages to JPGs.
2.  **Extract**: Run Tesseract OCR on the JPGs.
3.  **Embed**: Treat the OCR output text just like normal text.

#### Strategy B: Vision-Language Models (VLM) - The "Agentic" Way
**Tools**: `ColPali`, `Gemini Flash`, `GPT-4o`.
1.  **Screenshot**: Convert PDF page to image.
2.  **Captioning**: Pass the image to a VLM with the prompt: *"Describe this diagram in technical detail."*
3.  **Embedding**: Embed the **description**.
4.  **Retrieval**: When user asks about the diagram, the vector match finds the description, and the Agent reads it.

#### Strategy C: Multi-Modal RAG (ColPali)
- **Method**: The Embedding Model *itself* looks at the image patches (no text conversion).
- **Pros**: Extremely accurate for charts.
- **Cons**: Requires heavy GPU (VRAM > 24GB usually).

**Recommendation for `agentic-rag` (Consumer Hardware):**
Use **Strategy A (OCR)** for text-heavy images or **Strategy B (Gemini Flash as Describer)** for charts, utilizing the fallback key you just configured.
