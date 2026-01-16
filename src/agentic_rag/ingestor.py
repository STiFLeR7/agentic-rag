import os
import json
import logging
import hashlib
from typing import List, Dict, Tuple
from pypdf import PdfReader

class DocumentIngestor:
    """
    Handles ingestion of various document formats and 'Parent-Child' chunking.
    """
    def __init__(self, chunk_size: int = 400, parent_chunk_size: int = 2000, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be less than chunk_size ({chunk_size}).")
        self.chunk_size = chunk_size # Small chunk for vector search
        self.parent_chunk_size = parent_chunk_size # Large chunk for context
        self.overlap = overlap
        
    def load_file(self, file_path: str) -> List[Dict]:
        """
        Loads a file and returns distinct chunks with parent linkage.
        Returns: List of dicts { 'text': str, 'metadata': dict, 'id': str }
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            text = self._load_pdf(file_path)
        elif ext in ['.txt', '.md', '.py', '.js']:
            text = self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        return self._create_parent_child_chunks(text, source=os.path.basename(file_path))

    def _load_text(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _load_pdf(self, path: str) -> str:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text

    def _create_parent_child_chunks(self, full_text: str, source: str) -> List[Dict]:
        """
        Strategy:
        1. Split full text into 'Parent' chunks (large).
        2. Split each Parent into 'Child' chunks (small).
        3. Assign Parent ID to Children.
        4. Return ONLY Children for vectorization, but with 'parent_content' in metadata?
           NO. Storing parent content in metadata Bloats ChromaDB.
           Better: Store Parent Content in a separate doc store (key-value),
           and just put 'parent_id' in Child Metadata.
           
           BUT for this simplified local setup, we can yield:
           - Children (for Vector Store)
           - Parents (for Doc Store)
        """
        chunks = []
        
        # 1. Create Parents
        parent_texts = self._split_text(full_text, self.parent_chunk_size)
        
        for p_idx, p_text in enumerate(parent_texts):
            parent_id = hashlib.md5(f"{source}_p{p_idx}".encode()).hexdigest()
            
            # Store Parent info (to be saved in DocStore later)
            # We tag this chunk as 'type': 'parent'
            chunks.append({
                "id": parent_id,
                "text": p_text,
                "metadata": {"source": source, "type": "parent"},
                "is_parent": True
            })
            
            # 2. Create Children from THIS Parent
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
                    "is_parent": False
                })
                
        return chunks

    def _split_text(self, text: str, size: int) -> List[str]:
        # Simple character splitter for now. 
        # Production would use RecursiveCharacterTextSplitter from langchain
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start = end - self.overlap # Overlap
        return chunks
