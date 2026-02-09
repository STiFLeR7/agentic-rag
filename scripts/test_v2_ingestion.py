
import os
import sys
# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_rag.ingestor import DocumentIngestor
from agentic_rag.vector_store import VectorStore
from agentic_rag.embedding import CLIPEmbeddingFunction

def main():
    print("Testing V2 Multi-Modal Ingestion with CLIP...")
    ingestor = DocumentIngestor()
    # Using one of the provided PDFs
    pdf_path = "d:/agentic-rag/data/qatar_test_doc.pdf"
    
    if not os.path.exists(pdf_path):
        # Fallback to another one found in list_dir
        pdf_path = "d:/agentic-rag/data/Leo-Huang.pdf"

    print(f"Ingesting: {pdf_path}")
    
    # Clean fresh directory to avoid embedding conflicts
    persist_dir = "d:/agentic-rag/data/chroma_v2_test"
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
        print(f"Cleared stale directory: {persist_dir}")

    # Initialize CLIP
    ef = CLIPEmbeddingFunction()
    
    # Initialize Vector Store (new collection for v2)
    vs = VectorStore(collection_name="v2_multimodal_clip_test", persist_directory=persist_dir, embedding_function=ef)
    vs.reset() # Start fresh
    
    # Load and process
    chunks = ingestor.load_file(pdf_path)
    
    print(f"Created {len(chunks)} total chunks.")
    
    images = [c for c in chunks if c.get("is_image")]
    print(f"Found {len(images)} images in PDF.")
    
    # Add to vector store
    vs.add_processed_chunks(chunks)
    
    # Verify vision cache
    cache_files = os.listdir("d:/agentic-rag/data/vision_cache")
    print(f"Vision Cache contains {len(cache_files)} files.")
    for f in cache_files[:5]:
        print(f" - {f}")

    print("\nVerifying Retrieval of an image...")
    # Query for something visual if we can guess context
    query_text = "visual content"
    query_emb = ef.encode_text([query_text])
    results = vs.query(query_embeddings=query_emb, n_results=5)
    
    found_image = False
    for i in range(len(results['documents'][0])):
        meta = results['metadatas'][0][i]
        if meta.get("is_image"):
            print(f"HIT! Found image at: {meta.get('image_path')}")
            found_image = True
    
    if not found_image:
        print("No image found in top-5 results for 'visual content'.")

if __name__ == "__main__":
    main()
