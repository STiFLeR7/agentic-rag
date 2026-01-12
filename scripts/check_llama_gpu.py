
from llama_cpp import Llama
import os

model_path = "d:/agentic-rag/models/Phi-3-mini-4k-instruct-q4_1.gguf"

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
else:
    print("Loading Llama...")
    try:
        # Verbose=True prints build info to stderr
        llm = Llama(model_path=model_path, n_gpu_layers=-1, verbose=True)
        print("Model Loaded.")
        
        # We can't easily query the object for 'is_cuda', but the logs will show.
        # Check generated logs in the terminal output.
    except Exception as e:
        print(f"Error: {e}")
