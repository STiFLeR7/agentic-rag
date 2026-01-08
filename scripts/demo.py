
import os
import sys
# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_rag.llm import InferenceEngine
from agentic_rag.vector_store import VectorStore
from agentic_rag.retriever import Retriever
from agentic_rag.tools import ToolRegistry, SearchKnowledgeBaseTool, ReadFileTool
from agentic_rag.agent import Agent
from rich.console import Console

console = Console()

def main():
    console.print("[bold blue]Starting Agentic RAG Demo on RTX 3050...[/bold blue]")

    # 1. Load Model
    model_path = "D:/agentic-rag/models/Phi-3-mini-4k-instruct-q4_1.gguf"
    console.print(f"Loading Model: {model_path} ...")
    llm = InferenceEngine(model_path=model_path, n_gpu_layers=-1)
    
    # 2. Setup RAG
    console.print("Initializing Vector Store...")
    # Use distinct persist dir for demo
    vs = VectorStore(persist_directory="d:/agentic-rag/data/demo_chroma")
    
    # Add some dummy knowledge if empty
    if vs.count() == 0:
        console.print("Populating Vector Store with demo data...")
        vs.add_documents(
            documents=[
                "The project 'agentic-rag' runs on an RTX 3050 with 6GB VRAM.",
                "Phi-3-mini is a 3.8B parameter model suitable for local inference.",
                "Agents use tools to reason and act, unlike simple RAG pipelines."
            ],
            metadatas=[
                {"source": "manual"},
                {"source": "manual"},
                {"source": "manual"}
            ]
        )
    
    retriever = Retriever(vs)
    
    # 3. Setup Tools
    registry = ToolRegistry()
    registry.register(SearchKnowledgeBaseTool(retriever))
    registry.register(ReadFileTool())
    
    # 4. Run Agent
    agent = Agent(llm=llm, tools=registry)
    
    query = "What hardware does this project run on?"
    console.print(f"\n[bold green]User Query:[/bold green] {query}")
    
    response = agent.run(query)
    
    console.print(f"\n[bold yellow]Final Answer:[/bold yellow] {response}")

if __name__ == "__main__":
    main()
