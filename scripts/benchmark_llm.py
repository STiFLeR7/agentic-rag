
import time
import psutil
import torch
import typer
from rich.console import Console
from rich.table import Table
from llama_cpp import Llama
import json
import os

console = Console()
app = typer.Typer()

def get_vram_usage():
    # Placeholder for VRAM usage. 
    # Accurate VRAM on Windows with minimal deps is tricky without pynvml.
    # We can try to rely on nvidia-smi if available or just skip.
    try:
        # Simple nvidia-smi query
        output = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").read()
        return int(output.strip())
    except:
        return 0

@app.command()
def run(
    model_path: str = "D:/agentic-rag/models/Phi-3-mini-4k-instruct-q4_1.gguf",
    n_gpu_layers: int = -1, # -1 = all
    n_ctx: int = 4096
):
    console.print(f"[bold green]Loading model:[/bold green] {model_path}")
    
    start_vram = get_vram_usage()
    console.print(f"Pre-load VRAM: {start_vram} MB")

    start_load = time.time()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False
    )
    load_time = time.time() - start_load
    
    post_load_vram = get_vram_usage()
    console.print(f"Post-load VRAM: {post_load_vram} MB (Delta: {post_load_vram - start_vram} MB)")
    console.print(f"Load time: {load_time:.2f}s")

    prompt = "<|user|>\nWrite a short poem about coding agents.<|end|>\n<|assistant|>"
    
    console.print("\n[bold]Generating...[/bold]")
    start_gen = time.time()
    output = llm(
        prompt,
        max_tokens=100,
        stop=["<|end|>"],
        echo=True
    )
    end_gen = time.time()
    
    usage = output['usage']
    total_tokens = usage['total_tokens']
    completion_tokens = usage['completion_tokens']
    duration = end_gen - start_gen
    tps = completion_tokens / duration

    console.print(output['choices'][0]['text'])
    
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Tokens Generated", str(completion_tokens))
    table.add_row("Duration", f"{duration:.2f}s")
    table.add_row("TPS", f"{tps:.2f}")
    table.add_row("VRAM Used (Approx)", f"{post_load_vram - start_vram} MB")
    
    console.print(table)

if __name__ == "__main__":
    app()
