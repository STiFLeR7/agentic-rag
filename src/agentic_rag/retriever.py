
from typing import List, Dict
from agentic_rag.vector_store import VectorStore
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import torch

class Retriever:
    """
    Hybrid Retriever: BM25 + Vector Search + Cross-Encoder Re-ranking.
    """
    def __init__(self, vector_store: VectorStore, device: str = 'cuda'):
        self.vector_store = vector_store
        
        # 1. Initialize BM25 (In-Memory)
        # Fetch all docs from Vector Store (Assumption: Corpus fits in RAM)
        print("Initializing Hybrid Retriever...")
        self.docs = self.vector_store.get_all_docs()
        if self.docs:
            tokenized_corpus = [doc.split(" ") for doc in self.docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"BM25 Index built with {len(self.docs)} documents.")
        else:
            self.bm25 = None
            print("Warning: Vector Store empty. BM25 not initialized.")

        # 2. Initialize Cross-Encoder (Re-ranker)
        # We use a small, fast model. 
        # Using 'ms-marco-MiniLM-L-6-v2' (Standard for RAG).
        if torch.cuda.is_available() and device == 'cuda':
            print("Loading Re-ranker on GPU (CUDA)...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
        else:
             print("Loading Re-ranker on CPU...")
             self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')


    def retrieve(self, query: str, top_k: int = 3, use_hybrid: bool = True, use_rerank: bool = True) -> List[Dict]:
        """
        Hybrid Retrieval Process:
        1. Get Top-K from Vector Store (Semantic).
        2. Get Top-K from BM25 (Keyword) [Optional].
        3. Merge & Deduplicate.
        4. Re-rank with Cross-Encoder [Optional].
        5. Return Top-K.
        """
        # A. Vector Search
        raw_vector_results = self.vector_store.query(query, n_results=top_k * 2) # Fetch more for re-ranking
        vector_docs = []
        if raw_vector_results['documents']:
            vector_docs = raw_vector_results['documents'][0]

        # B. BM25 Search
        bm25_docs = []
        if use_hybrid and self.bm25:
            tokenized_query = query.split(" ")
            bm25_docs = self.bm25.get_top_n(tokenized_query, self.docs, n=top_k * 2)

        # C. Merge
        candidates = set(vector_docs + bm25_docs)
        if not use_hybrid:
             candidates = set(vector_docs) # Fallback to just vector
        
        if not candidates:
            return []

        candidate_list = list(candidates)
        
        # D. Re-Ranking
        if use_rerank:
            pairs = [[query, doc] for doc in candidate_list]
            scores = self.cross_encoder.predict(pairs)
            scored_results = sorted(zip(candidate_list, scores), key=lambda x: x[1], reverse=True)
            final_top_k = scored_results[:top_k]
        else:
            # If no rerank, we don't have good scores for the mixed set.
            # Just return top K from vector portion or random if mixed?
            # Ideally, without rerank, hybrid is hard to sort.
            # So if use_rerank=False, we assume Vector Only usually.
            # But if hybrid is true and rerank is false, we'll just take vector docs first then bm25.
            final_top_k = [(doc, 0.0) for doc in list(candidates)[:top_k]]
        
        # Format
        parsed_results = []
        for i, (content, score) in enumerate(final_top_k):
            # We don't have metadata for BM25 hits easily unless we map back.
            # For the demo, we construct a generic result.
            parsed_results.append({
                "content": content,
                "metadata": {"source": "hybrid"},
                "score": float(score) # numpy float to python float
            })
            
        return parsed_results
