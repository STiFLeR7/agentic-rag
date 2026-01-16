import os
import json
from agentic_rag.llm import InferenceEngine
from agentic_rag.dataset_generator import DatasetGenerator

def generate_orion_dataset():
    # 1. Setup
    # Ensure GEMINI_API_KEY is set in env or .env
    # Ensure GEMINI_API_KEY is set in env or .env
    if not os.environ.get("GEMINI_API_KEY"):
        # Manual load
        env_path = "d:/agentic-rag/.env"
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        key = line.strip().split("=", 1)[1]
                        os.environ["GEMINI_API_KEY"] = key
                        print("Loaded GEMINI_API_KEY from .env manually.")
                        break

                        os.environ["GEMINI_API_KEY"] = key
                        print("Loaded GEMINI_API_KEY from .env manually.")
                        break

    # Fallback to local model since API Key is invalid
    print("Switching to Local LLM (Phi-3) due to invalid API Key.")
    llm = InferenceEngine(
        model_path="d:/agentic-rag/models/Phi-3-mini-4k-instruct-q4_1.gguf", 
        force_gemini=False, 
        n_ctx=4096
    ) 
    generator = DatasetGenerator(llm)
    
    # 2. Load Data
    with open("d:/agentic-rag/data/project_orion.txt", "r", encoding="utf-8") as f:
        text = f.read()
        
    # 3. Generate
    print("Generating 'Technology' dataset for Project Orion...")
    dataset = generator.generate_synthetic_dataset(text, domain="Technology", num_questions=5)
    
    # 4. Save
    output_path = "d:/agentic-rag/data/project_orion_qa.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Saved {len(dataset)} QA pairs to {output_path}")

if __name__ == "__main__":
    generate_orion_dataset()
