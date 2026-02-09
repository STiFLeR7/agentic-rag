
from agentic_rag.embedding import CLIPEmbeddingFunction
import numpy as np

def smoke_test():
    print("Smoking testing CLIPEmbeddingFunction...")
    try:
        ef = CLIPEmbeddingFunction()
        
        test_text = ["Hello world"]
        emb_text = ef(test_text)
        print(f"Text embedding shape: {len(emb_text)} x {len(emb_text[0])}")
        print(f"Type of first element: {type(emb_text[0][0])}")
        
        # Test individual methods
        emb_query = ef.embed_query("test query")
        print(f"Query embedding length: {len(emb_query)}")
        
        print("Smoke test PASSED.")
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    smoke_test()
