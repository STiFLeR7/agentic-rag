import os
import json
import logging
import hashlib
import fitz  # PyMuPDF
from typing import List, Dict, Tuple
from pypdf import PdfReader
from PIL import Image
import io

class DocumentIngestor:
    """
    Handles ingestion of various document formats and 'Parent-Child' chunking.
    Support for Multi-Modal (Text + Image) ingestion.
    """
    def __init__(self, chunk_size: int = 400, parent_chunk_size: int = 2000, overlap: int = 50, vision_cache_dir: str = "d:/agentic-rag/data/vision_cache"):
        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be less than chunk_size ({chunk_size}).")
        self.chunk_size = chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.overlap = overlap
        self.vision_cache_dir = vision_cache_dir
        os.makedirs(self.vision_cache_dir, exist_ok=True)
        
    def load_file(self, file_path: str, enable_vision: bool = True) -> List[Dict]:
        """
        Loads a file and returns distinct chunks with parent linkage.
        Returns: List of dicts { 'text': str, 'metadata': dict, 'id': str, 'is_image': bool }
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            if enable_vision:
                return self._process_pdf_with_vision(file_path)
            else:
                text = self._load_pdf_pypdf(file_path)
                return self._create_parent_child_chunks(text, source=os.path.basename(file_path))
        elif ext in ['.txt', '.md', '.py', '.js']:
            text = self._load_text(file_path)
            return self._create_parent_child_chunks(text, source=os.path.basename(file_path))
        elif ext in ['.jpg', '.jpeg', '.png']:
            return self._process_raw_image(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _load_text(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _load_pdf_pypdf(self, path: str) -> str:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text

    def _process_pdf_with_vision(self, path: str) -> List[Dict]:
        """
        Extracts both text and images, anchoring images to their parent page text.
        """
        doc = fitz.open(path)
        source = os.path.basename(path)
        all_chunks = []
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            parent_id = hashlib.md5(f"{source}_p{page_num}".encode()).hexdigest()
            
            # 1. Create Parent Chunk for the page
            all_chunks.append({
                "id": parent_id,
                "text": page_text,
                "metadata": {"source": source, "type": "parent", "page": page_num},
                "is_parent": True,
                "is_image": False
            })
            
            # 2. Extract Images on this page
            images = page.get_images(full=True)
            for img_idx, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                img_filename = f"{source}_p{page_num}_img{img_idx}.{image_ext}"
                img_path = os.path.join(self.vision_cache_dir, img_filename)
                
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                
                # Image Chunk: Anchored to page text
                image_chunk_id = hashlib.md5(f"{source}_p{page_num}_img{img_idx}".encode()).hexdigest()
                all_chunks.append({
                    "id": image_chunk_id,
                    "text": f"Visual content in {source} on page {page_num}. Context: {page_text[:200]}...",
                    "metadata": {
                        "source": source,
                        "type": "image",
                        "image_path": img_path,
                        "parent_id": parent_id,
                        "page": page_num
                    },
                    "is_parent": False,
                    "is_image": True
                })
            
            # 3. Create Child Text Chunks from page text
            child_texts = self._split_text(page_text, self.chunk_size)
            for c_idx, c_text in enumerate(child_texts):
                child_id = hashlib.md5(f"{source}_p{page_num}_c{c_idx}".encode()).hexdigest()
                all_chunks.append({
                    "id": child_id,
                    "text": c_text,
                    "metadata": {
                        "source": source,
                        "type": "child",
                        "parent_id": parent_id,
                        "page": page_num
                    },
                    "is_parent": False,
                    "is_image": False
                })
        
        return all_chunks

    def _process_raw_image(self, path: str) -> List[Dict]:
        """Supports direct image ingestion (scans, photos)."""
        source = os.path.basename(path)
        parent_id = hashlib.md5(f"{source}_parent".encode()).hexdigest()
        image_id = hashlib.md5(f"{source}_img".encode()).hexdigest()
        
        # In a real scenario, we might use OCR here to get parent text.
        # For now, it's a standalone image asset.
        return [
            {
                "id": parent_id,
                "text": f"Raw image asset: {source}",
                "metadata": {"source": source, "type": "parent"},
                "is_parent": True,
                "is_image": False
            },
            {
                "id": image_id,
                "text": f"Image content from {source}",
                "metadata": {
                    "source": source,
                    "type": "image",
                    "image_path": path,
                    "parent_id": parent_id
                },
                "is_parent": False,
                "is_image": True
            }
        ]

    def _create_parent_child_chunks(self, full_text: str, source: str) -> List[Dict]:
        chunks = []
        parent_texts = self._split_text(full_text, self.parent_chunk_size)
        for p_idx, p_text in enumerate(parent_texts):
            parent_id = hashlib.md5(f"{source}_p{p_idx}".encode()).hexdigest()
            chunks.append({
                "id": parent_id,
                "text": p_text,
                "metadata": {"source": source, "type": "parent"},
                "is_parent": True,
                "is_image": False
            })
            child_texts = self._split_text(p_text, self.chunk_size)
            for c_idx, c_text in enumerate(child_texts):
                child_id = hashlib.md5(f"{source}_p{p_idx}_c{c_idx}".encode()).hexdigest()
                chunks.append({
                    "id": child_id,
                    "text": c_text,
                    "metadata": {
                        "source": source, 
                        "type": "child",
                        "parent_id": parent_id 
                    },
                    "is_parent": False,
                    "is_image": False
                })
        return chunks

    def _split_text(self, text: str, size: int) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start = end - self.overlap
        return chunks
