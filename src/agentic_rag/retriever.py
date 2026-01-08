
from typing import List, Dict
from agentic_rag.vector_store import VectorStore

class Retriever:
    """
    Orchestrates the retrieval process.
    Allows for expansion, filtering, and decision making (future).
    """
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Simple retrieval for now. 
        Returns a list of dicts with 'text', 'metadata', 'distance'.
        """
        raw_results = self.vector_store.query(query, n_results=top_k)
        
        # Parse ChromaDB result format into cleaner list
        parsed_results = []
        if raw_results['documents']:
             # result structure is list of lists (batched)
            docs = raw_results['documents'][0]
            metas = raw_results['metadatas'][0]
            distances = raw_results['distances'][0] if 'distances' in raw_results else [None]*len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                parsed_results.append({
                    "content": doc,
                    "metadata": meta,
                    "score": dist # lower is better for L2/cosine usually depends on chroma defaults
                })
        
        return parsed_results
