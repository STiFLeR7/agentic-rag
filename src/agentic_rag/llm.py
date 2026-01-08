import os
from typing import Optional, List, Dict, Any
from llama_cpp import Llama
from pydantic import BaseModel

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    stop: Optional[List[str]] = None
    temperature: float = 0.7

class InferenceEngine:
    """
    Wrapper around llama-cpp-python to handle model loading and inference 
    with tracking for VRAM and latency managed externally or internally.
    """
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
            chat_format="chatml" # Phi-3 uses chatml-like or specific format, llama-cpp handles 'chatml' usually
        )
        print("Model loaded.")

    def generate(self, request: CompletionRequest) -> Dict[str, Any]:
        """
        Raw completion.
        """
        output = self.llm(
            request.prompt,
            max_tokens=request.max_tokens,
            stop=request.stop or ["<|end|>"],
            temperature=request.temperature,
            echo=False
        )
        return output

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Chat completion using llama-cpp's built-in chat handler.
        """
        # Default stop tokens if not provided
        stop = kwargs.pop("stop", ["<|end|>", "Observation:"])
        
        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=512, # Default cap
            stop=stop,
            **kwargs
        )
        return output
