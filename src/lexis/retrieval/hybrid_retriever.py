import asyncio
import logging
from typing import List, Dict, Any, Optional

from lexis.config import settings
from lexis.indexing.qdrant_client import LexisQdrantClient
from lexis.indexing.es_client import LexisElasticsearchClient
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.retrieval.fusion import apply_rrf
from lexis.retrieval.interfaces import Candidate

logger = logging.getLogger(__name__)

class RetrievalEngine:
    """
    Week 1-2 Foundation Retriever.
    Executes Path B (Global Dense) and Path D (BM25) concurrently, fusing with RRF.
    """
    def __init__(self):
        self.es = LexisElasticsearchClient()
        self.qdrant = LexisQdrantClient()
        self.embedder = BGEM3Embedder()
        self.timeout_sec = 2.0  # 2 second max per path

    async def _safe_execute(self, task, path_name: str) -> List[Any]:
        """Wraps a retrieval task in a timeout to guarantee UI latency targets."""
        try:
            return await asyncio.wait_for(task, timeout=self.timeout_sec)
        except asyncio.TimeoutError:
            logger.warning(f"Retrieval Path [{path_name}] timed out after {self.timeout_sec}s")
            return []
        except Exception as e:
            logger.error(f"Retrieval Path [{path_name}] failed: {str(e)}")
            return []

    async def _path_b_global(self, query_emb: List[float], top_k: int) -> List[Candidate]:
        """Searches Global Dense Embeddings."""
        results = await self.qdrant.search(settings.qdrant_collection_primary, query_emb, top_k=top_k)
        candidates = []
        for r in results:
            candidates.append(Candidate(
                chunk_id=r.payload.get("chunk_id", ""),
                score=r.score,
                source_path="path_b_global",
                metadata=r.payload,
                content=r.payload.get("content", "")
            ))
        return candidates

    async def _path_d_bm25(self, query_text: str, top_k: int) -> List[Candidate]:
        """Searches Elasticsearch with BM25."""
        try:
            hits = await self.es.search(query_text, size=top_k)
            candidates = []
            for h in hits:
                candidates.append(Candidate(
                    chunk_id=h["chunk_id"],
                    score=h["score"],
                    source_path="path_d_bm25",
                    metadata=h["payload"],
                    content=h["payload"].get("content", "")
                ))
            return candidates
        except Exception as e:
            logger.error(f"ES search failed: {e}")
            return []

    async def retrieve(self, query: str, top_k_per_path: int = 15, top_n_rrf: int = 15) -> List[dict]:
        """
        Executes foundation multi-path retrieval.
        Returns dict-based chunks matching downstream expectations.
        """
        top_k_per_path = top_k_per_path or settings.retrieval_top_k_per_path
        
        # Embed Query
        query_emb = self.embedder.embed_text(query).tolist()
        
        # Concurrently execute paths
        task_b = self._safe_execute(self._path_b_global(query_emb, top_k_per_path), "Path B (Global)")
        task_d = self._safe_execute(self._path_d_bm25(query, top_k_per_path), "Path D (BM25)")
        
        res_b, res_d = await asyncio.gather(task_b, task_d)
        
        # Fuse with RRF
        candidate_lists = []
        if res_b: candidate_lists.append(res_b)
        if res_d: candidate_lists.append(res_d)
            
        fused = apply_rrf(candidate_lists, k=settings.rrf_k)
        
        # Format for downstream
        final_chunks = []
        for c in fused[:top_n_rrf]:
            final_chunks.append({
                "id": c.chunk_id,
                "score": c.score,
                "rrf_score": c.score,
                "source_path": c.source_path,
                "payload": c.metadata,
                "text": c.content
            })
            
        return final_chunks
