
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os
import uuid
import json

class VectorStore:
    def __init__(self, collection_name: str = "agentic_rag_docs", persist_directory: str = "d:/agentic-rag/data/chroma"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Simple Key-Value Store for Parent Docs (JSON backed)
        self.doc_store_path = os.path.join(persist_directory, "doc_store.json")
        self.doc_store = self._load_doc_store()

    def _load_doc_store(self) -> Dict:
        if os.path.exists(self.doc_store_path):
            with open(self.doc_store_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_doc_store(self):
        with open(self.doc_store_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_store, f)

    def add_processed_chunks(self, chunks: List[Dict]):
        """
        Adds pre-processed chunks (from Ingestor) to appropriate stores.
        - Parents -> JSON Doc Store
        - Children -> ChromaDB
        """
        vector_docs = []
        vector_ids = []
        vector_metas = []
        
        new_parents = 0
        
        for chunk in chunks:
            if chunk.get("is_parent"):
                # Store Parent
                self.doc_store[chunk["id"]] = chunk["text"]
                new_parents += 1
            else:
                # Prepare Child for Vector
                vector_docs.append(chunk["text"])
                vector_ids.append(chunk["id"])
                vector_metas.append(chunk["metadata"])
                
        # Save Parents
        if new_parents > 0:
            self._save_doc_store()
            print(f"Stored {new_parents} Parent Documents in {self.doc_store_path}")
            
        # Save Children
        if vector_docs:
            self.collection.add(
                documents=vector_docs,
                metadatas=vector_metas,
                ids=vector_ids
            )
            print(f"Added {len(vector_docs)} Child Documents to ChromaDB")

    def get_parent_content(self, child_meta: Dict) -> Optional[str]:
        """
        Retrieves the parent content given a child's metadata.
        """
        parent_id = child_meta.get("parent_id")
        if parent_id:
            return self.doc_store.get(parent_id)
        return None

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """
        Legacy method for simple string addition.
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
        self.doc_store = {}
        if os.path.exists(self.doc_store_path):
            os.remove(self.doc_store_path)

    def get_all_docs(self) -> List[str]:
        """
        Returns all document contents for building BM25 index.
        """
        # .get() returns dict with 'ids', 'documents', 'metadatas'
        result = self.collection.get()
        return result['documents'] if result and 'documents' in result else []

