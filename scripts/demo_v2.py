
import os
import sys
# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_rag.llm import InferenceEngine
from agentic_rag.vector_store import VectorStore
from agentic_rag.embedding import CLIPEmbeddingFunction
from agentic_rag.retriever import Retriever
from agentic_rag.tools import ToolRegistry, SearchKnowledgeBaseTool, ExamineImageTool, PythonCodeTool
from agentic_rag.agent import Agent

def main():
    print("--- Starting Agentic RAG V2 (Multi-Modal) ---")
    
    # 1. Initialize Large Components
    ef = CLIPEmbeddingFunction()
    vs = VectorStore(
        collection_name="v2_multimodal_clip_test", 
        persist_directory="d:/agentic-rag/data/chroma_v2_test", 
        embedding_function=ef
    )
    
    retriever = Retriever(vs, embedding_function=ef)
    
    # LLM Initialization (Phi-3 local, Gemini fallback/vision enabled)
    # For V2 Demo, we force Gemini to ensure multi-modal reasoning reliability
    model_path = "d:/agentic-rag/models/Phi-3-mini-4k-instruct-q4_1.gguf"
    llm = InferenceEngine(model_path=model_path, force_gemini=True)
    
    # 2. Tool Registration
    registry = ToolRegistry()
    registry.register(SearchKnowledgeBaseTool(retriever))
    registry.register(ExamineImageTool(llm))
    registry.register(PythonCodeTool())
    
    # 3. Agent Initialization
    agent = Agent(llm, registry)
    
    # 4. Multi-Modal Test Case
    # We know 'qatar_test_doc.pdf' was ingested. 
    # Let's ask a question that likely requires visual evidence.
    query = "Find any diagrams or images in the Qatar documentation and describe what is shown in them."
    
    print(f"\nUser Query: {query}\n")
    
    response = agent.run(query)
    
    print("\n--- Final Agent Response ---")
    print(response)

if __name__ == "__main__":
    main()
