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

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class InferenceEngine:
    """
    Wrapper around llama-cpp-python to handle model loading and inference 
    with tracking for VRAM and latency managed externally or internally.
    Falls back to Gemini Flash if local model fails.
    """
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        self.use_fallback = False
        
        try:
             self._load_model()
        except Exception as e:
            print(f"CRITICAL: Failed to load local model ({e}). Switching to Gemini Fallback.")
            self._setup_fallback()

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

    def _setup_fallback(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY not found in .env. Fallback unusable.")
            raise RuntimeError("No Local Model and No Gemini Key.")
        
        genai.configure(api_key=api_key)
        self.fallback_model = genai.GenerativeModel('gemini-1.5-flash')
        self.use_fallback = True
        print("Gemini Fallback Active (gemini-1.5-flash).")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Chat completion using llama-cpp's built-in chat handler or Gemini Fallback.
        """
        if self.use_fallback:
            return self._chat_gemini(messages, **kwargs)
            
        # Default stop tokens if not provided
        stop = kwargs.pop("stop", ["<|end|>", "Observation:"])
        
        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=512, # Default cap
                stop=stop,
                **kwargs
            )
            return output
        except Exception as e:
            print(f"Error during local inference: {e}. Switching to Fallback.")
            self._setup_fallback()
            return self._chat_gemini(messages, **kwargs)

    def _chat_gemini(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        # Convert ChatML style messages to Gemini history
        # Gemini expects structure: contents=[{'role': 'user', 'parts': [...]}]
        # Local: [{'role': 'user', 'content': '...'}]
        
        gemini_history = []
        system_instruction = None
        
        for m in messages:
            role = m['role']
            content = m['content']
            
            if role == 'system':
                system_instruction = content
            elif role == 'user':
                gemini_history.append({'role': 'user', 'parts': [content]})
            elif role == 'assistant':
                gemini_history.append({'role': 'model', 'parts': [content]})
                
        # Handle current query (last user message)
        # Note: genai.ChatSession handles history, but here we are stateless "chat" wrapper.
        # We'll validly construct a chat with history excluding the last one, then send message.
        
        if not gemini_history:
             return {"choices": [{"message": {"content": "Error: No messages."}}]}
             
        last_msg = gemini_history[-1]
        if last_msg['role'] == 'user':
             prompt_parts = last_msg['parts']
             history = gemini_history[:-1]
        else:
             # Should not happen in standard flow (last is user)
             prompt_parts = ["Continue."]
             history = gemini_history

        try:
             model = self.fallback_model
             if system_instruction:
                 # Re-init with system instruction if needed? 
                 # Gemini 1.5 supports system_instruction argument in GenerativeModel constructor
                 # But we initialized it once. Let's send it as context in first message if needed.
                 # Or just re-instantiate.
                 model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction)
             
             chat = model.start_chat(history=history)
             response = chat.send_message(prompt_parts[0])
             
             return {
                 "choices": [
                     {
                         "message": {
                             "role": "assistant",
                             "content": response.text
                         }
                     }
                 ],
                 "usage": {"total_tokens": 0} # Dummy
             }
        except Exception as e:
             return {"choices": [{"message": {"content": f"Error in Gemini Fallback: {e}"}}]}

