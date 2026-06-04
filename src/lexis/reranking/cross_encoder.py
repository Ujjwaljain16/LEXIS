from typing import List
from lexis.retrieval.interfaces import Query, Candidate
from lexis.reranking.interfaces import Reranker
from sentence_transformers import CrossEncoder

class BAAICrossEncoder(Reranker):
    """
    Reranks candidates by jointly scoring the query and candidate chunk content.
    Uses BAAI/bge-reranker-v2-m3. Highly accurate but computationally expensive.
    Should be applied before Sentence Window Expansion.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        # Using a smaller cross-encoder for local development speed if needed, 
        # but defaulting to bge-reranker-v2-m3 as per plan.
        self.model = CrossEncoder(model_name)
        
    async def transform(self, query: Query, candidates: List[Candidate]) -> List[Candidate]:
        if not candidates:
            return []
            
        # Format for cross-encoder: list of (query, document) pairs
        pairs = [(query.text, c.content) for c in candidates]
        scores = self.model.predict(pairs)
        
        # Update scores
        for c, score in zip(candidates, scores):
            c.score = float(score)
            
        # Sort by new cross-encoder score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
