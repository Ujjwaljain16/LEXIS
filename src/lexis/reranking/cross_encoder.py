import asyncio
from typing import List, Optional
from lexis.retrieval.interfaces import Query, Candidate
from lexis.reranking.interfaces import Reranker
from sentence_transformers import CrossEncoder

class BAAICrossEncoder(Reranker):
    """
    Reranks candidates by jointly scoring the query and candidate chunk content.
    Uses BAAI/bge-reranker-v2-m3. Highly accurate but computationally expensive.
    Should be applied before Sentence Window Expansion.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", model: Optional[CrossEncoder] = None):
        # model= lets a test inject a fake without downloading/loading a real ~600MB+ model.
        # Using a smaller cross-encoder for local development speed if needed,
        # but defaulting to bge-reranker-v2-m3 as per plan.
        self.model = model if model is not None else CrossEncoder(model_name)

    async def transform(self, query: Query, candidates: List[Candidate]) -> List[Candidate]:
        if not candidates:
            return []

        # Format for cross-encoder: list of (query, document) pairs
        pairs = [(query.text, c.content) for c in candidates]
        # CrossEncoder.predict() is synchronous and CPU-bound -- run it in a worker thread so it
        # doesn't block the event loop under concurrent requests, matching how
        # hybrid_retriever.py's BM25 path (also sync/CPU-bound) already does this.
        scores = await asyncio.to_thread(self.model.predict, pairs)

        # Builds NEW Candidate objects rather than mutating the input's score in place -- the
        # input candidates may be a slice sharing object references with a caller's own list
        # (e.g. hybrid_retriever.py's RetrievalTrace.fused_candidates, the pre-rerank RRF stage
        # kept for diagnostics), and mutating those objects would silently corrupt that
        # already-returned/recorded state. Matches fusion.py::apply_rrf's own non-mutating style.
        rescored = [c.model_copy(update={"score": float(score)}) for c, score in zip(candidates, scores)]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored
