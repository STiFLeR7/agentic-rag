
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from agentic_rag.llm import InferenceEngine
from agentic_rag.agent import Agent
from agentic_rag.vector_store import VectorStore
from agentic_rag.retriever import Retriever
from agentic_rag.tools import ToolRegistry, SearchKnowledgeBaseTool, ReadFileTool, PythonCodeTool

# --- Configuration ---
import os
MODEL_PATH = "d:/agentic-rag/models/Phi-3-mini-4k-instruct-q4_1.gguf"
DATA_FILE = "d:/agentic-rag/data/project_orion.txt"

# --- Fictional Knowledge (Project Orion) ---
PROJECT_ORION_CONTENT = """
Project Orion: Advanced Quantum Telemetry Protocol (QTP-X)
CONFIDENTIAL SPECIFICATIONS - 2026

1. Core Architecture
   The QTP-X system relies on a 'Hyper-State' entanglement bus running at 4.2 Tera-qubits per second.
   Unlike legacy QTP (v1.0), QTP-X introduces 'Chronos' latency compensation, which reduces packet loss by predicting state collapse 12ms into the future.

2. Hardware Requirements
   - Primary Processor: NeuroCore N-900 (requires liquid nitrogen cooling).
   - Memory: Minimum 128 Pentabytes of Holographic RAM.
   - Power Supply: Zero-Point Energy Module (ZPE-M) Type C.

3. Failure Modes
   - 'Cascade Resonance': Occurs if ZPE-M output fluctuation exceeds 0.003%.
   - Mitigation: Activate 'Aegis' dampeners immediately.
   - Error Code: 0xDEADBEEF relates to buffer overflow in the Flux Capacitor (Legacy).

4. Personnel
   - Chief Architect: Dr. Elena Voss.
   - Lead Engineer: Marcus Thorne.
   - Security Level: Crimson clearance required.
"""

# Questions
# We include specific Error Code questions which Dense retrieval often misses.
EVAL_SET = [
    {"q": "What is the primary processor for the QTP-X system?", "keywords": ["NeuroCore", "N-900"]},
    {"q": "What mitigates Cascade Resonance?", "keywords": ["Aegis", "dampeners"]},
    {"q": "Who is the Chief Architect of Project Orion?", "keywords": ["Elena", "Voss"]},
    {"q": "What is the bandwidth of the Hyper-State entanglement bus?", "keywords": ["4.2", "Tera-qubits"]},
    {"q": "What is the specific cooling requirement for the NeuroCore?", "keywords": ["liquid", "nitrogen"]},
    {"q": "What is the security clearance required?", "keywords": ["Crimson"]},
    {"q": "How does Chronos compensation reduce packet loss?", "keywords": ["predicting", "state", "collapse"]},
    {"q": "What triggers Cascade Resonance?", "keywords": ["ZPE-M", "fluctuation", "0.003"]},
    {"q": "Who is the Lead Engineer?", "keywords": ["Marcus", "Thorne"]},
    {"q": "What does Error Code 0xDEADBEEF mean?", "keywords": ["buffer", "overflow", "Flux"]}
]
EVAL_SET = EVAL_SET[:3] # Reduce to 3 for quick Smoke Test

def setup_system():
    with open(DATA_FILE, "w") as f:
        f.write(PROJECT_ORION_CONTENT)
    
    # EVAL_SET REDUCTION FOR SMOKE TEST
    # global EVAL_SET
    # EVAL_SET = EVAL_SET[:3] 
    
    llm = InferenceEngine(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=-1, force_gemini=False)
    
    # Store
    vector_store = VectorStore(collection_name="project_orion_eval_v2", persist_directory="d:/agentic-rag/data/chroma_eval_v2")
    vector_store.client.delete_collection("project_orion_eval_v2")
    vector_store = VectorStore(collection_name="project_orion_eval_v2", persist_directory="d:/agentic-rag/data/chroma_eval_v2")
    
    chunks = [c.strip() for c in PROJECT_ORION_CONTENT.split("\n\n") if c.strip()]
    ids = [f"id_{i}" for i in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=ids)
    
    # We create ONE retriever instance
    retriever = Retriever(vector_store) # Loads models
    
    return llm, retriever

def evaluate_answer(answer: str, keywords: list) -> int:
    hit_count = 0
    for k in keywords:
        if k.lower() in answer.lower():
            hit_count += 1
    return 1 if hit_count == len(keywords) else 0

def run_evaluation():
    print("Setting up V2 Environment...")
    llm, retriever = setup_system()
    
    results = []
    
    print("\n--- Starting Comparative Evaluation (v1 vs v2) ---")
    print("Mode: Full Agent Response Quality & Speed")
    
    # Warmup to settle CUDA/Tables
    print("Warming up models...")
    _ = retriever.retrieve("warmup", use_hybrid=True, use_rerank=True)
    
    memory_file = "d:/agentic-rag/agent_memory.json"
    
    for i, item in enumerate(EVAL_SET):
        q = item["q"]
        keywords = item["keywords"]
        print(f"\nQ{i+1}: {q}")
        
        # --- Run V1 (Dense Only) ---
        # Note: We use the Agent wrapper logic we defined earlier, or simplified for this script?
        # The script previously defined V1RetrieverWrapper/V2RetrieverWrapper inside run_evaluation
        # But I overwrote run_evaluation in the last step. I need to restore the Agent setup.
        
        # V1 Analysis
        if os.path.exists(memory_file): os.remove(memory_file)
        start_v1 = time.time()
        # Direct Agent Call with V1 Tool
        # We need to rebuild the agent or just swap the retriever behavior?
        # Swapping behavior is risky if parallel. 
        # Let's instantiate the wrappers properly here again.
        
        class V1Wrapper:
             def __init__(self, r): self.r = r
             def retrieve(self, q): return self.r.retrieve(q, top_k=3, use_hybrid=False, use_rerank=False)
        
        # We need a fresh registry for V1
        reg_v1 = ToolRegistry()
        reg_v1.register(SearchKnowledgeBaseTool(V1Wrapper(retriever)))
        agent_v1 = Agent(llm, reg_v1)
        
        out_v1 = agent_v1.run(q, max_steps=3) 
        time_v1 = time.time() - start_v1
        score_v1 = evaluate_answer(out_v1, keywords)
        
        # --- Run V2 (Hybrid Analysis) ---
        if os.path.exists(memory_file): os.remove(memory_file)
        
        class V2Wrapper:
             def __init__(self, r): self.r = r
             def retrieve(self, q): return self.r.retrieve(q, top_k=3, use_hybrid=True, use_rerank=True)
             
        reg_v2 = ToolRegistry()
        reg_v2.register(SearchKnowledgeBaseTool(V2Wrapper(retriever)))
        agent_v2 = Agent(llm, reg_v2)
        
        start_v2 = time.time()
        out_v2 = agent_v2.run(q, max_steps=3)
        time_v2 = time.time() - start_v2
        score_v2 = evaluate_answer(out_v2, keywords)
        
        print(f"  [v1 Dense ] Recall: {score_v1} | Time: {time_v1:.4f}s")
        print(f"  [v2 Hybrid] Recall: {score_v2} | Time: {time_v2:.4f}s")
        
        results.append({
            "Question": i + 1,
            "v1_Score": score_v1,
            "v1_Time": time_v1,
            "v2_Score": score_v2,
            "v2_Time": time_v2
        })

    # --- Visualization ---
    df = pd.DataFrame(results)
    
    print("\nResults Summary:")
    print(df)
    
    # Plot Accuracy
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Graph 1: Accuracy (Neon)
    color_v1 = '#FF0055' # Neon Pink
    color_v2 = '#00FFDD' # Neon Cyan
    
    ax1.plot(df['Question'], df['v1_Score'], marker='o', linestyle='--', color=color_v1, label='v1 (Dense)', linewidth=2)
    ax1.plot(df['Question'], df['v2_Score'], marker='o', linestyle='-', color=color_v2, label='v2 (Hybrid)', linewidth=3)
    
    # Glow
    for line, color in zip(ax1.lines, [color_v1, color_v2]):
        ax1.plot(df['Question'], line.get_ydata(), color=color, linewidth=10, alpha=0.2)
        
    ax1.set_title('Accuracy Comparison', color='white', fontsize=14)
    ax1.set_ylabel('Score (0/1)', color='white')
    ax1.legend()
    ax1.grid(color='#333333', linestyle=':')
    
    # Graph 2: Speed (Bar)
    width = 0.35
    x = df['Question']
    ax2.bar(x - width/2, df['v1_Time'], width, label='v1 Time', color=color_v1, alpha=0.8)
    ax2.bar(x + width/2, df['v2_Time'], width, label='v2 Time', color=color_v2, alpha=0.8)
    
    ax2.set_title('Speed Comparison (Latency)', color='white', fontsize=14)
    ax2.set_ylabel('Seconds', color='white')
    ax2.legend()
    
    plt.tight_layout()
    output_path = "d:/agentic-rag/docs/metrics_v2_compare.png"
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        import traceback
        traceback.print_exc()
