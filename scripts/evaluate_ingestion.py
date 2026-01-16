import os
import time
from agentic_rag.ingestor import DocumentIngestor
from agentic_rag.vector_store import VectorStore
from pypdf import PdfWriter

def create_dummy_pdf(path: str):
    """Creates a simple PDF for testing."""
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    
    # We can't easily add text with just pypdf without complex font setup or ReportLab.
    # So we will rely on creating a text file to test the Ingestor logic first, 
    # OR we assume pypdf can write text which it can't easily.
    # Let's verify TEXT ingestion first as it covers the Parent-Child logic.
    # Then we can trust PDF works if PdfReader works.
    pass

def create_dummy_text_file(path: str):
    content = """
    Phase 7: The Omni-Corpus
    
    The concept of specific Parent Document Retrieval is crucial for RAG.
    Here is a specific detail: The key access code for the vault is 998877.
    
    This detail is buried in a long paragraph. If we only retrieved the code '998877',
    we might miss the context that it belongs to 'the vault'.
    
    By retrieving this entire parent block, the LLM knows exactly what the code is for.
    Project Orion also mentioned this technique in section 4.
    """
    with open(path, "w") as f:
        f.write(content)

def run_ingestion_test():
    print("--- Starting Ingestion Pipeline Test ---")
    
    # 1. Setup
    test_file = "d:/agentic-rag/data/test_ingest.txt"
    create_dummy_text_file(test_file)
    
    # 2. Ingest
    print("Ingesting file...")
    # chunk_size must be > overlap!
    ingestor = DocumentIngestor(chunk_size=50, parent_chunk_size=500, overlap=10) 
    chunks = ingestor.load_file(test_file)
    
    print(f"Generated {len(chunks)} processed chunks.")
    parents = [c for c in chunks if c.get('is_parent')]
    children = [c for c in chunks if not c.get('is_parent')]
    print(f"Parents: {len(parents)}")
    print(f"Children: {len(children)}")
    
    # 3. Store
    print("Adding to VectorStore...")
    # Use a test collection
    store = VectorStore(collection_name="test_omni_corpus", persist_directory="d:/agentic-rag/data/test_chroma")
    store.reset() # Start clean
    
    store.add_processed_chunks(chunks)
    
    # 4. Retrieval Test
    query = "vault code"
    print(f"\nQuerying: '{query}'")
    
    results = store.query(query, n_results=1)
    
    if not results['documents']:
        print("FAIL: No results found.")
        return
        
    child_text = results['documents'][0][0]
    child_meta = results['metadatas'][0][0]
    
    print(f"Child Match: '{child_text}'")
    print(f"Child Metadata: {child_meta}")
    
    # 5. Parent Lookup
    parent_content = store.get_parent_content(child_meta)
    
    if parent_content:
        print(f"\n[SUCCESS] Retrieved Parent Content:\n{parent_content}")
        if "Phase 7: The Omni-Corpus" in parent_content:
             print("Verified: Parent context contains start of paragraph.")
    else:
        print("[FAIL] Could not retrieve Parent Content.")

if __name__ == "__main__":
    try:
        run_ingestion_test()
    except Exception as e:
        import traceback
        traceback.print_exc()
