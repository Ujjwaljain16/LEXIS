from typing import List
from lexis.retrieval.interfaces import RetrievalPath, Query, Candidate
from lexis.indexing.bm25_index import LexisBM25Index
from lexis.config import settings
import logging

logger = logging.getLogger(__name__)

class LocalBM25Retrieval(RetrievalPath):
    """
    Path D: BM25 Lexical Retrieval.
    Replaces Elasticsearch with a local fast implementation (bm25s)
    as authorized by the architect to reduce operational overhead for <10M
    chunk scale (see docs/ADR.md ADR-003).
    """
    def __init__(self, index_dir: str = None):
        self.index = LexisBM25Index(index_dir=index_dir or settings.bm25_index_dir)

    async def retrieve(self, query: Query) -> List[Candidate]:
        import asyncio
        hits = await asyncio.to_thread(self.index.search, query.text, query.top_k)

        candidates = []
        for hit in hits:
            candidates.append(Candidate(
                chunk_id=hit.get("chunk_id", ""),
                score=hit.get("score", 0.0),
                source_path="path_d_bm25",
                metadata=hit.get("payload", {}),
                content=hit.get("payload", {}).get("content", ""),
            ))
        return candidates
