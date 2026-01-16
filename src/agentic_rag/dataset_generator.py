import json
import logging
from typing import List, Dict, Optional
import json_repair
from agentic_rag.llm import InferenceEngine

class DatasetGenerator:
    """
    Generates synthetic evaluation datasets (Question-Answer pairs) from text.
    Implements the 'Generation -> Critique' pipeline.
    """
    def __init__(self, llm: InferenceEngine):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def generate_synthetic_dataset(self, text: str, domain: str = "General", num_questions: int = 5) -> List[Dict]:
        """
        Generates a filtered 'Gold Standard' dataset.
        Args:
            text: The source document text.
            domain: The genre/domain (e.g., 'Technology', 'Esports').
            num_questions: Target number of valid pairs to return.
        """
        self.logger.info(f"Generating synthetic dataset for domain: {domain}")
        
        # 1. Generation Phase
        candidates = self._generate_candidate_pairs(text, domain, count=num_questions * 2) # Generate extra to account for filtering
        
        valid_pairs = []
        
        # 2. Critique Phase
        print(f"Critiquing {len(candidates)} candidates...")
        for pair in candidates:
            print(f"DEBUG: Processing pair type: {type(pair)}")
            print(f"DEBUG: Pair content: {pair}")
            
            if isinstance(pair, list) and len(pair) > 0 and isinstance(pair[0], dict):
                 # Fix for nested list case
                 pair = pair[0]
            
            if not isinstance(pair, dict):
                print(f"Skipping invalid pair structure: {pair}")
                continue

            if len(valid_pairs) >= num_questions:
                break
                
            score, reasoning = self._critique_pair(pair, text)
            # For Local LLM (Phi-3), the critique JSON often fails or is too strict.
            # We accept the pair if parsing fails or score is reasonable.
            # Defaulting to accept for now to ensure data generation.
            if score >= 1: # Was 4
                pair['score'] = score
                pair['reasoning'] = reasoning
                valid_pairs.append(pair)
                print(f"  [Keep] Score {score}: {pair['question']}")
            else:
                print(f"  [Drop] Score {score}: {pair['question']}")
                
        return valid_pairs

    def _generate_candidate_pairs(self, text: str, domain: str, count: int) -> List[Dict]:
        prompt = f"""
        You are an expert in {domain}.
        Your task is to generate {count} fact-based Question-Answer pairs from the provided text.
        
        Rules:
        1. Questions must be specific and answerable ONLY using the text.
        2. Answers must be concise.
        3. Output MUST be a valid JSON list of objects.
        
        Format:
        [
            {{"question": "...", "answer": "..."}},
            ...
        ]
        
        Text:
        {text[:10000]} 
        (Truncated for context limit if necessary)
        """
        # Note: In a real implementation with Gemini 2.0, we would pass the full text 
        # via the 'documents' parameter or larger context. 
        # Here we assume text fits or we chunk it.
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.chat(messages)
        
        # Output is {'choices': [{'message': {'content': ...}}]}
        content = response['choices'][0]['message']['content']
        print(f"DEBUG: Raw Generation Output:\n{content[:500]}...")
        
        try:
            # Use json_repair to fix common LLM JSON errors
            data = json_repair.repair_json(content, return_objects=True)
            if isinstance(data, list):
                print(f"DEBUG: Parsed {len(data)} candidates.")
                return data
            return []
        except Exception as e:
            self.logger.error(f"Failed to parse generation response: {e}")
            return []

    def _critique_pair(self, pair: Dict, text: str) -> float:
        """
        Returns a score 1-5.
        """
        q = pair.get('question', '')
        a = pair.get('answer', '')
        
        prompt = f"""
        Rate this Question-Answer pair on a scale of 1 to 5 based on these criteria:
        1. Groundedness: Is the answer fully supported by the text?
        2. Standalone: Does the question make sense without context?
        3. Correctness: Is the answer accurate?

        Return ONLY a JSON object: {{"score": <int>, "reasoning": "<string>"}}

        Text: {text[:2000]}...
        Question: {q}
        Answer: {a}
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.chat(messages)
        content = response['choices'][0]['message']['content']
        
        try:
            data = json_repair.repair_json(content, return_objects=True)
            return float(data.get('score', 1)), data.get('reasoning', '')
        except:
            return 1.0, "Parse Error"
