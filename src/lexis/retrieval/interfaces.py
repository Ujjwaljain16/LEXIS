from typing import Protocol, Dict, Any, List
from pydantic import BaseModel, Field

class Query(BaseModel):
    """Core domain query object for retrieval."""
    text: str
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 20

class Candidate(BaseModel):
    """A standardized chunk candidate returned by any retrieval path."""
    chunk_id: str
    score: float
    source_path: str  # e.g., 'path_b_global', 'path_d_bm25'
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content: str = ""

class RetrievalPath(Protocol):
    """
    Standard interface for all retrieval paths in Lexis.
    Enforces that every path returns the exact same Candidate object.
    """
    async def retrieve(self, query: Query) -> List[Candidate]:
        ...
