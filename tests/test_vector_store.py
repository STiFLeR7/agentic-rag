
import pytest
from agentic_rag.vector_store import VectorStore
from agentic_rag.retriever import Retriever
import shutil
import os

@pytest.fixture
def vector_store(tmp_path):
    # tmp_path is unique per test function invocation
    db_path = tmp_path / "chroma_db"
    vs = VectorStore(collection_name="test_collection", persist_directory=str(db_path))
    yield vs
    # No manual cleanup needed, pytest handles it. 
    # Even if it fails to delete, it won't affect next test.
    # But we should explicitly close client if possible? 
    # Chroma 0.4+ doesn't always have close(). Let's rely on gc.


def test_add_and_query(vector_store):
    docs = [
        "Agentic RAG allows for reasoning over retrieval.",
        "The RTX 3050 has 6GB of VRAM.",
        "Apples are red and tasty."
    ]
    metas = [{"topic": "rag"}, {"topic": "hardware"}, {"topic": "fruit"}]
    
    vector_store.add_documents(docs, metadatas=metas)
    assert vector_store.count() == 3
    
    results = vector_store.query("graphics card memory", n_results=1)
    assert len(results['documents'][0]) == 1
    assert "RTX 3050" in results['documents'][0][0]

def test_retriever(vector_store):
    vector_store.add_documents(["Code quality is important.", "Fast code is good."])
    retriever = Retriever(vector_store)
    
    hits = retriever.retrieve("quality", top_k=1)
    assert len(hits) == 1
    assert "Code quality" in hits[0]['content']
