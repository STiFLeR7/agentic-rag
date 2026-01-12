
import os
import requests
import json
from typing import Optional, List, Dict, Any
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv("d:/agentic-rag/.env")

class InferenceEngine:
    """
    Wrapper around llama-cpp-python to handle model loading and inference.
    Falls back to Gemini 2.0 Flash (REST) if local model fails or is forced.
    """
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1, force_gemini: bool = False):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        self.use_fallback = False
        self.gemini_key = None
        
        # Load API Key immediately
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_key:
             print("Warning: GEMINI_API_KEY not found in .env")

        if force_gemini:
            print("Forcing Gemini Usage (REST Mode).")
            self.use_fallback = True
            if not self.gemini_key:
                 raise ValueError("Force Gemini requested but no API Key available.")
        else:
            try:
                self._load_model()
            except Exception as e:
                print(f"CRITICAL: Failed to load local model ({e}). Switching to Gemini Fallback.")
                self.use_fallback = True

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        print(f"Loading model from {self.model_path}...")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
            chat_format="chatml"
        )
        print("Model loaded.")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Chat completion using local Llama or Gemini Fallback.
        """
        if self.use_fallback:
            return self._chat_gemini(messages, **kwargs)
            
        # Default stop tokens
        stop = kwargs.pop("stop", ["<|end|>", "Observation:"])
        
        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=1024,
                stop=stop,
                **kwargs
            )
            return output
        except Exception as e:
            print(f"Error during local inference: {e}. Switching to Fallback.")
            self.use_fallback = True
            return self._chat_gemini(messages, **kwargs)

    def _chat_gemini(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Uses Gemini via REST API (v1beta) targeting gemini-2.0-flash-exp.
        """
        if not self.gemini_key:
             return {"choices": [{"message": {"content": "Error: No Gemini API Key"}}]}
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.gemini_key}"
        
        # Convert messages to Gemini Content
        contents = []
        for m in messages:
            role = "model" if m['role'] == "assistant" else "user"
            # Merge system into user or prepend? Gemini 2.0 supports system instructions but REST payload is specific.
            # Simpler: Just map content.
            contents.append({
                "role": role,
                "parts": [{"text": m['content']}]
            })
            
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": 1024
            }
        }
        
        try:
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
            if response.status_code == 200:
                data = response.json()
                if "candidates" in data and data["candidates"]:
                    content = data["candidates"][0]["content"]["parts"][0]["text"]
                    return {"choices": [{"message": {"content": content}}]}
                else:
                    return {"choices": [{"message": {"content": f"Error: Empty Gemini Response: {data}"}}]}
            else:
                 # Fallback to 1.5-flash if 2.0 fails (e.g. 404)
                if response.status_code == 404:
                     print("Gemini 2.0 not found (404), trying 1.5-flash...")
                     url_15 = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
                     response = requests.post(url_15, headers={"Content-Type": "application/json"}, json=payload)
                     if response.status_code == 200:
                        data = response.json()
                        if "candidates" in data and data["candidates"]:
                            content = data["candidates"][0]["content"]["parts"][0]["text"]
                            return {"choices": [{"message": {"content": content}}]}

                return {"choices": [{"message": {"content": f"Error Gemini API {response.status_code}: {response.text}"}}]}
        except Exception as e:
            return {"choices": [{"message": {"content": f"Error calling Gemini REST: {e}"}}]}
