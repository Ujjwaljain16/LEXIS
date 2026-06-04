from typing import List
from lexis.retrieval.interfaces import Query, Candidate
from lexis.reranking.interfaces import Reranker

class LostInMiddleReordering(Reranker):
    """
    Combats the 'Lost in the Middle' phenomenon observed in LLMs where they 
    over-index on the beginning and end of a context window.
    
    Reorders candidates so that the highest scoring are at the edges:
    e.g., [1, 3, 5, 4, 2]
    """
    async def transform(self, query: Query, candidates: List[Candidate]) -> List[Candidate]:
        if not candidates:
            return []
            
        # Ensure candidates are sorted descending by score first
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        
        reordered = []
        for i, candidate in enumerate(sorted_candidates):
            # Alternate placing at beginning and end
            if i % 2 == 0:
                reordered.insert(0, candidate)
            else:
                reordered.append(candidate)
                
        return reordered
