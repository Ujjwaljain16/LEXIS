from typing import List, Callable, Awaitable
from lexis.retrieval.interfaces import Query, Candidate
from lexis.reranking.interfaces import Reranker
import logging

logger = logging.getLogger(__name__)

class SentenceWindowExpansion(Reranker):
    """
    Expands high-scoring candidates with their adjacent chunks (e.g., chunk_idx-1, chunk_idx+1)
    to provide the LLM with surrounding context. 
    Crucially, this is applied AFTER the CrossEncoder to minimize token processing costs.
    """
    def __init__(self, fetch_chunk_fn: Callable[[str], Awaitable[Candidate]], window_size: int = 1):
        """
        Args:
            fetch_chunk_fn: An async callable that takes a chunk_id and returns the Candidate.
            window_size: Number of adjacent chunks to fetch on each side.
        """
        self.fetch_chunk_fn = fetch_chunk_fn
        self.window_size = window_size

    async def transform(self, query: Query, candidates: List[Candidate]) -> List[Candidate]:
        expanded_candidates = []
        seen_ids = set()
        
        for candidate in candidates:
            if candidate.chunk_id in seen_ids:
                continue
                
            # Attempt to parse chunk sequence index if it follows a pattern like doc_123_chunk_5
            # For this MVP, we assume metadata contains 'doc_id' and 'chunk_index'
            doc_id = candidate.metadata.get("doc_id")
            chunk_index = candidate.metadata.get("chunk_index")
            
            if doc_id is None or chunk_index is None:
                if candidate.chunk_id not in seen_ids:
                    expanded_candidates.append(candidate)
                    seen_ids.add(candidate.chunk_id)
                continue
                
            # Fetch surrounding chunks
            context_chunks = []
            for offset in range(-self.window_size, self.window_size + 1):
                target_idx = int(chunk_index) + offset
                if target_idx < 0:
                    continue
                    
                target_chunk_id = f"{doc_id}_chunk_{target_idx}"
                if target_chunk_id not in seen_ids:
                    try:
                        adj_chunk = await self.fetch_chunk_fn(target_chunk_id)
                        if adj_chunk:
                            context_chunks.append(adj_chunk)
                            seen_ids.add(target_chunk_id)
                    except Exception as e:
                        logger.warning(f"Failed to fetch adjacent chunk {target_chunk_id}: {e}")
            
            # Combine the content logically or just append the adjacent chunks as new candidates.
            # Usually, window expansion merges them into the original candidate to keep it a single continuous text block.
            # We'll merge them for efficiency:
            if context_chunks:
                # Sort by original chunk index to maintain reading order
                context_chunks.sort(key=lambda c: int(c.metadata.get("chunk_index", 0)))
                merged_content = "\n\n".join([c.content for c in context_chunks])
                
                expanded_candidate = Candidate(
                    chunk_id=candidate.chunk_id, # keep original anchor ID
                    score=candidate.score,
                    source_path=candidate.source_path + ",window_expanded",
                    metadata=candidate.metadata,
                    content=merged_content
                )
                expanded_candidates.append(expanded_candidate)
            else:
                expanded_candidates.append(candidate)
                
        return expanded_candidates
