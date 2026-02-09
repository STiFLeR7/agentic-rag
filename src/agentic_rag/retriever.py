from typing import List, Dict, Optional
from agentic_rag.vector_store import VectorStore
from agentic_rag.embedding import CLIPEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import torch

class Retriever:
    """
    Hybrid Multi-Modal Retriever: BM25 + Vector (CLIP) + Cross-Encoder.
    """
    def __init__(self, vector_store: VectorStore, embedding_function: Optional[CLIPEmbeddingFunction] = None, device: str = 'cuda'):
        self.vector_store = vector_store
        self.ef = embedding_function
        
        # 1. Initialize BM25 (In-Memory)
        print("Initializing Hybrid Multi-Modal Retriever...")
        self.docs = self.vector_store.get_all_docs()
        if self.docs:
            tokenized_corpus = [doc.split(" ") for doc in self.docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"BM25 Index built with {len(self.docs)} documents.")
        else:
            self.bm25 = None

        # 2. Initialize Cross-Encoder (Re-ranker)
        if torch.cuda.is_available() and device == 'cuda':
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
        else:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')


    def retrieve(self, query: str, top_k: int = 3, use_hybrid: bool = True, use_rerank: bool = True) -> List[Dict]:
        """
        Hybrid Multi-Modal Retrieval Process:
        1. Get Top-K from Vector Store (Semantic CLIP).
        2. Get Top-K from BM25 (Keyword).
        3. Merge & Deduplicate.
        4. Re-rank with Cross-Encoder.
        5. Return Top-K assets.
        """
        # A. Vector Search (CLIP)
        query_emb = None
        if self.ef:
            query_emb = self.ef.encode_text([query])
            
        raw_vector_results = self.vector_store.query(query_text=query, query_embeddings=query_emb, n_results=top_k * 2)
        
        vector_assets = []
        if raw_vector_results['documents']:
            for i in range(len(raw_vector_results['documents'][0])):
                vector_assets.append({
                    "content": raw_vector_results['documents'][0][i],
                    "metadata": raw_vector_results['metadatas'][0][i],
                    "id": raw_vector_results['ids'][0][i]
                })

        # B. BM25 Search (Keyword)
        bm25_assets = []
        if use_hybrid and self.bm25:
            tokenized_query = query.split(" ")
            # Note: BM25 only has access to document text, not full asset dicts.
            # We'll map back or just use the text.
            top_bm25_texts = self.bm25.get_top_n(tokenized_query, self.docs, n=top_k * 2)
            for text in top_bm25_texts:
                bm25_assets.append({"content": text, "metadata": {"source": "bm25"}, "id": "bm25"})

        # C. Merge & Deduplicate by content
        seen = set()
        candidates = []
        for asset in vector_assets + bm25_assets:
            if asset['content'] not in seen:
                candidates.append(asset)
                seen.add(asset['content'])
        
        if not candidates:
            return []

        # D. Re-Ranking (On Text Context)
        if use_rerank and candidates:
            pairs = [[query, asset['content']] for asset in candidates]
            scores = self.cross_encoder.predict(pairs)
            scored_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            final_top_k = scored_results[:top_k]
        else:
            final_top_k = [(asset, 0.0) for asset in candidates[:top_k]]
        
        # E. Format
        parsed_results = []
        for asset, score in final_top_k:
            result = {
                "content": asset['content'],
                "metadata": asset['metadata'],
                "score": float(score)
            }
            parsed_results.append(result)
            
        return parsed_results
