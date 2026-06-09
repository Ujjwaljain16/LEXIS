"""
Hybrid Retrieval Engine for LEXIS.

Rationale: Executes the 4-Path Retrieval architectural specification.
Source Inspiration: plan.md (Path A, B, C, D) and Haystack concurrent retrieval patterns.
Deviations from Source Repos: We implement strict timeouts per path to prevent slow DBs from stalling the query.
Expected Impact on Metrics: High recall due to multi-modal search surfaces; bounded latency p99.
"""
import asyncio
from typing import List, Dict, Any, Set
from lexis.indexing.es_client import LexisElasticsearchClient
from lexis.indexing.qdrant_client import LexisQdrantClient
from lexis.ingestion.embedder import BGEM3Embedder
import logging

logger = logging.getLogger(__name__)

class RetrievalEngine:
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

    async def _path_a_raptor(self, query_emb: List[float], top_k: int) -> List[Any]:
        """Searches RAPTOR Clusters."""
        return await self.qdrant.search("clusters_v2", query_emb, top_k=top_k)

    async def _path_b_hype(self, query_emb: List[float], top_k: int) -> List[Any]:
        """Searches Hypothetical Document Embeddings."""
        return await self.qdrant.search("hype_v2", query_emb, top_k=top_k)

    async def _path_c_propositions(self, query_emb: List[float], top_k: int) -> List[Any]:
        """Searches the Proposition Graph."""
        return await self.qdrant.search("propositions_v2", query_emb, top_k=top_k)

    async def _path_d_bm25(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Searches Elasticsearch with BM25."""
        try:
            return await self.es.search(query_text, size=top_k)
        except Exception as e:
            logger.warning(f"Elasticsearch is unavailable, falling back to Vector-Only Mode. Error: {e}")
            return []
        
    async def _path_primary_vector(self, query_emb: List[float], top_k: int) -> List[Any]:
        """Fallback base vector search on primary chunks."""
        return await self.qdrant.search("primary_v2", query_emb, top_k=top_k)

    async def retrieve(self, query: str, top_k_per_path: int = 5, top_n_rrf: int = None) -> List[Dict[str, Any]]:
        """
        Executes concurrent retrieval across all configured paths.
        De-duplicates by chunk_id.
        """
        query_emb = self.embedder.embed_text(query).tolist()
        
        # Build tasks
        task_a = self._path_a_raptor(query_emb, top_k_per_path)
        task_b = self._path_b_hype(query_emb, top_k_per_path)
        task_c = self._path_c_propositions(query_emb, top_k_per_path)
        task_d = self._path_d_bm25(query, top_k_per_path)
        task_primary = self._path_primary_vector(query_emb, top_k_per_path)

        # Fire concurrently with safety bounds
        results = await asyncio.gather(
            self._safe_execute(task_a, "A_RAPTOR"),
            self._safe_execute(task_b, "B_HYPE"),
            self._safe_execute(task_c, "C_PROPOSITIONS"),
            self._safe_execute(task_d, "D_BM25"),
            self._safe_execute(task_primary, "PRIMARY_VECTOR")
        )

        res_a, res_b, res_c, res_d, res_primary = results
        
        # Reciprocal Rank Fusion (RRF)
        # RRF_Score = sum(1 / (61 + rank))
        rrf_scores: Dict[str, float] = {}
        chunk_payloads: Dict[str, Dict[str, Any]] = {}

        def _apply_rrf(hit_list: List[Any], source_path: str):
            for rank, hit in enumerate(hit_list):
                payload = hit if isinstance(hit, dict) else hit.payload
                c_id = payload.get("chunk_id")
                if not c_id:
                    continue
                
                if c_id not in chunk_payloads:
                    payload["_source_path"] = source_path
                    chunk_payloads[c_id] = payload
                    rrf_scores[c_id] = 0.0
                
                # k=61 is the industry standard constant for RRF
                rrf_scores[c_id] += 1.0 / (61 + rank)

        _apply_rrf(res_d, "D_BM25")
        _apply_rrf(res_primary, "PRIMARY_VECTOR")
        _apply_rrf(res_b, "B_HYPE")
        _apply_rrf(res_c, "C_PROPOSITIONS")
        _apply_rrf(res_a, "A_RAPTOR")

        # Sort by RRF score descending
        sorted_c_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        final_chunks = [chunk_payloads[cid] for cid in sorted_c_ids]

        if top_n_rrf is not None:
            final_chunks = final_chunks[:top_n_rrf]

        return final_chunks
