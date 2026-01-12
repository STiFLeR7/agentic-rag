
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os
import uuid

class VectorStore:
    def __init__(self, collection_name: str = "agentic_rag_docs", persist_directory: str = "d:/agentic-rag/data/chroma"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # Using default embedding function (all-MiniLM-L6-v2) for now.
            # It downloads automatically and runs locally.
        )
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """
        Add documents to the vector store.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
            
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} documents to collection '{self.collection.name}'")

    def query(self, query_text: str, n_results: int = 3) -> Dict:
        """
        Query the vector store.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """
        Dangerous: clears all data in collection.
        """
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(self.collection.name)

    def get_all_docs(self) -> List[str]:
        """
        Returns all document contents for building BM25 index.
        """
        # .get() returns dict with 'ids', 'documents', 'metadatas'
        result = self.collection.get()
        return result['documents'] if result and 'documents' in result else []

