
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from agentic_rag.llm import InferenceEngine
from agentic_rag.agent import Agent
from agentic_rag.vector_store import VectorStore
from agentic_rag.retriever import Retriever
from agentic_rag.tools import ToolRegistry, SearchKnowledgeBaseTool, ReadFileTool, PythonCodeTool

# --- Configuration ---
MODEL_PATH = "d:/agentic-rag/models/Phi-3-mini-4k-instruct-q4_1.gguf"
DATA_FILE = "d:/agentic-rag/data/project_orion.txt"

# --- Fictional Knowledge (Project Orion) ---
# We write this to a file first so the agent can "read" or "ingest" it.
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

4. Personnel
   - Chief Architect: Dr. Elena Voss.
   - Lead Engineer: Marcus Thorne.
   - Security Level: Crimson clearance required.
"""

# Questions and Expected Keywords
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
    {"q": "what is ZPE-M?", "keywords": ["Zero-Point", "Energy", "Module"]}
]

def setup_system():
    # 1. Write Data
    with open(DATA_FILE, "w") as f:
        f.write(PROJECT_ORION_CONTENT)
    
    # 2. Init Components
    llm = InferenceEngine(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=-1)
    
    # 3. Ingest Data into Vector Store
    vector_store = VectorStore(collection_name="project_orion_eval", persist_directory="d:/agentic-rag/data/chroma_eval")
    vector_store.client.delete_collection("project_orion_eval") # Reset for fresh run
    vector_store = VectorStore(collection_name="project_orion_eval", persist_directory="d:/agentic-rag/data/chroma_eval")
    
    # Simple chunking by paragraph
    chunks = [c.strip() for c in PROJECT_ORION_CONTENT.split("\n\n") if c.strip()]
    ids = [f"id_{i}" for i in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=ids)
    
    retriever = Retriever(vector_store)
    
    # 4. Agent tools
    registry = ToolRegistry()
    registry.register(SearchKnowledgeBaseTool(retriever))
    registry.register(ReadFileTool())
    registry.register(PythonCodeTool())
    
    agent = Agent(llm, registry)
    
    return llm, agent

def evaluate_answer(answer: str, keywords: list) -> int:
    hit_count = 0
    for k in keywords:
        if k.lower() in answer.lower():
            hit_count += 1
    return 1 if hit_count == len(keywords) else 0

def run_evaluation():
    print("Setting up Project Orion environment...")
    llm, agent = setup_system()
    
    results = []
    
    print("\n--- Starting Evaluation (10 Questions) ---")
    for i, item in enumerate(EVAL_SET):
        q = item["q"]
        keywords = item["keywords"]
        print(f"\nQ{i+1}: {q}")
        
        # 1. Baseline (Zero-Shot LLM via Chat API for consistency)
        baseline_messages = [{"role": "user", "content": q}]
        # Use simple generation without RAG
        # We use chat() method but without tools injected in prompt
        baseline_out = llm.chat(baseline_messages)['choices'][0]['message']['content']
        s_base = evaluate_answer(baseline_out, keywords)
        print(f"[Baseline] Score: {s_base}")
        
        # 2. Agentic RAG
        agent_out = agent.run(q)
        s_agent = evaluate_answer(agent_out, keywords)
        print(f"[Agentic] Score: {s_agent}")
        
        results.append({
            "Question": i + 1,
            "Baseline": s_base,
            "Agentic RAG": s_agent
        })

    # --- Visualization (Neon Line Graph) ---
    df = pd.DataFrame(results)
    
    # Calculate Cumulative Accuracy for smooth line
    df['Baseline_Cum'] = df['Baseline'].cumsum()
    df['Agentic_Cum'] = df['Agentic RAG'].cumsum()
    
    print("\nResults Summary:")
    print(df)
    
    # Plotting
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Neon Colors
    color_base = '#FF0055' # Neon Pink
    color_agent = '#00FFDD' # Neon Cyan
    
    # Line Comparison
    # We plot the raw score (0 or 1) but with a line connecting them to show stability.
    
    ax.plot(df['Question'], df['Baseline'], marker='o', linestyle='--', color=color_base, label='Baseline (LLM Only)', linewidth=2, markersize=8)
    ax.plot(df['Question'], df['Agentic RAG'], marker='o', linestyle='-', color=color_agent, label='Agentic RAG', linewidth=3, markersize=8)
    
    # Add Glow
    for line, color in zip(ax.lines, [color_base, color_agent]):
        ax.plot(df['Question'], line.get_ydata(), color=color, linewidth=10, alpha=0.2)
    
    ax.set_xlabel('Question #', color='white', fontsize=12)
    ax.set_ylabel('Score (1=Correct, 0=Fail)', color='white', fontsize=12)
    ax.set_title('Quality Ablation: LLM vs Agentic RAG (Project Orion)', color='white', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Fail', 'Pass'])
    ax.set_xticks(df['Question'])
    
    ax.legend(facecolor='#1E1E1E', edgecolor='white')
    
    # Grid customization
    ax.grid(color='#333333', linestyle=':')
    
    plt.tight_layout()
    output_path = "d:/agentic-rag/docs/metrics_neon_line.png"
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    run_evaluation()
