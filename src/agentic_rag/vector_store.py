import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os
import uuid
import json
from agentic_rag.embedding import CLIPEmbeddingFunction

class VectorStore:
    def __init__(self, collection_name: str = "agentic_rag_docs", persist_directory: str = "d:/agentic-rag/data/chroma", embedding_function: Optional[CLIPEmbeddingFunction] = None):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.ef = embedding_function
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.ef
        )
        
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
        - Children -> ChromaDB (with CLIP support)
        """
        vector_docs = []
        vector_ids = []
        vector_metas = []
        vector_embeddings = []
        
        new_parents = 0
        
        for chunk in chunks:
            if chunk.get("is_parent"):
                # Store Parent
                self.doc_store[chunk["id"]] = chunk["text"]
                new_parents += 1
            else:
                # Prepare Child for Vector
                # We normalize metadata to ensure keys like 'is_image' exist
                meta = chunk["metadata"]
                meta["is_image"] = chunk.get("is_image", False)
                
                # Handle Multi-Modal Embeddings
                if self.ef:
                    if meta["is_image"]:
                        # Embed the actual image
                        embedding = self.ef.encode_images([meta["image_path"]])[0]
                    else:
                        # Embed the text
                        embedding = self.ef.encode_text([chunk["text"]])[0]
                    vector_embeddings.append(embedding)
                
                vector_docs.append(chunk["text"])
                vector_ids.append(chunk["id"])
                vector_metas.append(meta)
                
        # Save Parents
        if new_parents > 0:
            self._save_doc_store()
            print(f"Stored {new_parents} Parent Documents in {self.doc_store_path}")
            
        # Save Children
        if vector_docs:
            add_kwargs = {
                "documents": vector_docs,
                "metadatas": vector_metas,
                "ids": vector_ids
            }
            if vector_embeddings:
                add_kwargs["embeddings"] = vector_embeddings
                
            self.collection.add(**add_kwargs)
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

    def query(self, query_text: Optional[str] = None, n_results: int = 3, query_embeddings: Optional[List[List[float]]] = None) -> Dict:
        """
        Query the vector store. Supports either text or pre-computed embeddings.
        """
        kwargs = {"n_results": n_results}
        if query_embeddings:
            kwargs["query_embeddings"] = query_embeddings
        elif query_text:
            kwargs["query_texts"] = [query_text]
        else:
            raise ValueError("Must provide either query_text or query_embeddings")

        results = self.collection.query(**kwargs)
        return results

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """
        Dangerous: clears all data in collection.
        """
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.ef
        )
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

