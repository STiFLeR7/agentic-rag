from typing import List, Dict
from agentic_rag.llm import InferenceEngine

class DomainRouter:
    """
    Classifies queries to route them to the appropriate domain/collection.
    """
    def __init__(self, llm: InferenceEngine):
        self.llm = llm
        
    def route_query(self, query: str) -> str:
        """
        Returns 'technical' or 'general'.
        """
        # Simple Keyword Heuristic for speed (LLM fallback if needed)
        keywords_tech = ['code', 'python', 'error', 'bug', 'deploy', 'api', 'class', 'function', 'import']
        if any(k in query.lower() for k in keywords_tech):
            return "technical"
            
        # LLM Classifier (if keywords fail)
        # prompt = f"Classify this query into 'technical' or 'general': {query}"
        # For now, stay fast.
        return "general"
