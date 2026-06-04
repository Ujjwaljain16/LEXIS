from typing import Protocol, List
from lexis.retrieval.interfaces import Candidate, Query

class Reranker(Protocol):
    """
    Standard interface for all reranking and context-expansion transformations.
    Transformations take a list of candidates and a query, and return a re-ordered
    or expanded list of candidates.
    """
    async def transform(self, query: Query, candidates: List[Candidate]) -> List[Candidate]:
        ...
