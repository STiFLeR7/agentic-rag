
import os
import sys
import time
import json
from typing import List, Dict

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_rag.llm import InferenceEngine
from agentic_rag.vector_store import VectorStore
from agentic_rag.embedding import CLIPEmbeddingFunction
from agentic_rag.retriever import Retriever
from agentic_rag.tools import ToolRegistry, SearchKnowledgeBaseTool, ExamineImageTool
from agentic_rag.agent import Agent

# Define Vision Benchmarks
VISION_TEST_CASES = [
    {
        "query": "What are the six main machine learning models mentioned for Qatar's nowcasting?",
        "expected_substrings": ["LASSO", "Elastic Net", "Support Vector Machine", "Random Forest", "Gradient Boosting", "XGBoost"],
        "is_visual": True
    },
    {
        "query": "Describe Figure III. 3 in the Qatar documentation.",
        "expected_substrings": ["Machine Learning Process", "input", "output", "data"],
        "is_visual": True
    }
]

def evaluate_vision():
    print("--- Starting V2 Vision Evaluation ---")
    
    # 1. Setup V2 Engine
    ef = CLIPEmbeddingFunction()
    vs = VectorStore(
        collection_name="v2_multimodal_clip_test", 
        persist_directory="d:/agentic-rag/data/chroma_v2_test", 
        embedding_function=ef
    )
    retriever = Retriever(vs, embedding_function=ef)
    llm = InferenceEngine(model_path="d:/agentic-rag/models/Phi-3-mini-4k-instruct-q4_1.gguf", force_gemini=True)
    
    registry = ToolRegistry()
    registry.register(SearchKnowledgeBaseTool(retriever))
    registry.register(ExamineImageTool(llm))
    
    agent = Agent(llm, registry)
    
    results = []
    
    for i, test in enumerate(VISION_TEST_CASES, 1):
        print(f"\n[Test {i}] Query: {test['query']}")
        start_time = time.time()
        
        response = agent.run(test['query'])
        duration = time.time() - start_time
        
        # Simple string-based verification
        score = 0
        found = []
        for s in test['expected_substrings']:
            if s.lower() in response.lower():
                score += 1
                found.append(s)
        
        accuracy = score / len(test['expected_substrings'])
        
        print(f"Accuracy: {accuracy*100:.1f}% | Time: {duration:.2f}s")
        print(f"Found: {found}")
        
        results.append({
            "test_id": i,
            "query": test['query'],
            "accuracy": accuracy,
            "duration": duration,
            "response": response
        })

    # Save Results
    output_path = "d:/agentic-rag/data/vision_eval_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nEvaluation Complete. Results saved to {output_path}")

if __name__ == "__main__":
    evaluate_vision()
