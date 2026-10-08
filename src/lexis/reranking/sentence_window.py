from typing import List, Callable, Awaitable, Optional
from lexis.retrieval.interfaces import Query, Candidate
from lexis.reranking.interfaces import Reranker
import logging

logger = logging.getLogger(__name__)

class SentenceWindowExpansion(Reranker):
    """
    Expands high-scoring candidates with their adjacent chunks (by document-relative
    position, not any assumed document structure) to provide the LLM with surrounding
    context.
    Crucially, this is applied AFTER the CrossEncoder to minimize token processing costs.
    """
    def __init__(self, fetch_chunk_fn: Callable[[str, int], Awaitable[Optional[Candidate]]], window_size: int = 1):
        """
        Args:
            fetch_chunk_fn: An async callable that takes (doc_id, chunk_index) and returns
                the Candidate at that position, or None if it doesn't exist. Chunk ids are
                content-derived hashes (see Chunk.create) and cannot be constructed from
                doc_id/index alone, so lookup must happen by position, not by a guessed id.
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
            seen_ids.add(candidate.chunk_id)

            # chunk_index is the chunk's position within its document's chunk sequence
            # (see Chunk.split_idx), populated by the ingestion pipeline. It carries no
            # assumption about document type or internal structure.
            doc_id = candidate.metadata.get("doc_id")
            chunk_index = candidate.metadata.get("chunk_index")

            if doc_id is None or chunk_index is None:
                expanded_candidates.append(candidate)
                continue

            chunk_index = int(chunk_index)

            # Fetch surrounding chunks by (doc_id, chunk_index); start with the anchor itself.
            context_chunks = [candidate]
            for offset in range(-self.window_size, self.window_size + 1):
                if offset == 0:
                    continue
                target_idx = chunk_index + offset
                if target_idx < 0:
                    continue

                try:
                    adj_chunk = await self.fetch_chunk_fn(doc_id, target_idx)
                except Exception as e:
                    logger.warning(f"Failed to fetch adjacent chunk for doc_id={doc_id} chunk_index={target_idx}: {e}")
                    continue

                if adj_chunk and adj_chunk.chunk_id not in seen_ids:
                    context_chunks.append(adj_chunk)
                    seen_ids.add(adj_chunk.chunk_id)

            # Combine the content logically or just append the adjacent chunks as new candidates.
            # Usually, window expansion merges them into the original candidate to keep it a single continuous text block.
            # We'll merge them for efficiency:
            if len(context_chunks) > 1:
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
