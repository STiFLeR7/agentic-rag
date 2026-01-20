
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
from rich.prompt import Prompt

console = Console()

def main():
    console.print("[bold blue]Starting Agentic RAG: Interactive Mode[/bold blue]")
    console.print("Running on [green]RTX 3050 Compliant Settings[/green].")

    # 1. Load Model
    # Try to find model in standard paths
    model_path = "D:/agentic-rag/models/Phi-3-mini-4k-instruct-q4_1.gguf"
    if not os.path.exists(model_path):
        console.print(f"[red]Model not found at {model_path}![/red]")
        console.print("Please download Phi-3-mini-4k-instruct-q4_1.gguf to models/")
        return

    console.print(f"Loading Model: {os.path.basename(model_path)} ...")
    llm = InferenceEngine(model_path=model_path, n_gpu_layers=-1)
    
    # 2. Setup RAG
    console.print("Initializing Knowledge Base...")
    # Use Production Vector Store
    vs = VectorStore(persist_directory="d:/agentic-rag/data/chroma_db")
    
    if vs.count() == 0:
        console.print("[yellow]Warning: Vector Store is empty![/yellow]")
        console.print("Run [bold]python scripts/evaluate_ingestion.py[/bold] or check ingestion pipelines.")
        # Optional: Add dummy data if preferred, but for Asset mode, we want real connection.
    else:
        console.print(f"Connected to Vector Store with {vs.count()} chunks.")
    
    retriever = Retriever(vs)
    
    # 3. Setup Tools
    registry = ToolRegistry()
    registry.register(SearchKnowledgeBaseTool(retriever))
    registry.register(ReadFileTool())
    
    # 4. Run Agent
    agent = Agent(llm=llm, tools=registry)
    
    console.print("\n[bold green]Agent Ready! (Type 'exit' to quit)[/bold green]")
    
    while True:
        query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        if query.lower() in ("exit", "quit", "q"):
            break
            
        console.print("[dim]Thinking...[/dim]")
        try:
            response = agent.run(query)
            console.print(f"\n[bold yellow]Agent[/bold yellow]: {response}")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            
    console.print("[blue]Goodbye![/blue]")

if __name__ == "__main__":
    main()
