
from sentence_transformers import SentenceTransformer
from PIL import Image
from typing import List, Union, Optional
import numpy as np
import torch

class CLIPEmbeddingFunction:
    """
    Custom Embedding Function for ChromaDB that handles both Text and Images.
    Uses sentence-transformers CLIP implementation.
    """
    def __init__(self, model_name: str = "d:/agentic-rag/models/clip-ViT-B-32", device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading CLIP model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def name(self) -> str:
        return "clip_multimodal"

    def encode_text(self, texts: List[str]) -> List[np.ndarray]:
        """Encodes text into the joint CLIP space."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def encode_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """Encodes images into the joint CLIP space."""
        images = [Image.open(p) for p in image_paths]
        return self.model.encode(images, convert_to_numpy=True).tolist()

    def __call__(self, input: Union[List[str], List[Image.Image]]) -> List[np.ndarray]:
        """
        Implementation of ChromaDB EmbeddingFunction protocol.
        Note: Chroma typically passed texts. We'll handle images via separate calls
        during ingestion to ensure precision.
        """
        # Default behavior for Chroma's query/add
        return self.model.encode(input, convert_to_numpy=True).tolist()

    def embed_query(self, input: str) -> List[float]:
        """ChromaDB v0.4+ compatibility."""
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input, convert_to_numpy=True).tolist()[0]

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """ChromaDB v0.4+ compatibility."""
        return self.model.encode(input, convert_to_numpy=True).tolist()
