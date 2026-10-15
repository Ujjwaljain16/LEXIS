"""
R7: document-preamble prior (document-scoped retrieval only).

Questions about a document's identity -- its title, parties, agreement/effective date -- are answered by
its opening text, not by whichever clause is semantically closest to the question. On the dev split,
82-91% of the Document Name / Parties / Agreement Date answers start within the first 2,000 characters
(about one chunk) and 69% of Effective Date answers do, versus 5-15% for ordinary clause categories.
Dense and lexical similarity rank those opening chunks poorly (Document Name MRR 0.08, Agreement Date
0.14 on the frozen baseline), so this adds the document's first `preamble_prior_chunks` chunks as one more
ranked list in RRF -- the same fusion the other paths use, no special-casing of the question.

Only used when retrieval is already restricted to known documents: across a whole corpus the "first
chunks" of every document would flood the fused list. Off by default (0): an ablation flag, evaluated on
dev before any default changes.

"First" is by chunk_index (document order). chunk_index is not contiguous (sub-split chunks are
base*100+sub), so the first n are found by sorting each document's chunk_index values, never by a range
filter.
"""
from typing import Dict, List

from qdrant_client.http import models as qmodels

from lexis.config import settings
from lexis.retrieval.interfaces import Candidate

SOURCE_PATH = "path_preamble"


class PreambleIndex:
    """Per-document cache of opening chunks. A document's chunks are immutable once ingested, so the
    cache never needs invalidation within a process; re-ingesting a document needs a new process
    (or `clear()`)."""

    def __init__(self, qdrant_async_client, collection: str):
        self._client = qdrant_async_client
        self._collection = collection
        self._cache: Dict[str, List[Candidate]] = {}

    def clear(self) -> None:
        self._cache.clear()

    async def _load_document(self, doc_id: str) -> List[Candidate]:
        flt = qmodels.Filter(must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))])
        records, offset = [], None
        while True:
            batch, offset = await self._client.scroll(
                collection_name=self._collection, scroll_filter=flt, limit=settings.preamble_scroll_batch,
                offset=offset, with_payload=True, with_vectors=False)
            records.extend(batch)
            if offset is None:
                break
        records.sort(key=lambda r: (r.payload.get("chunk_index") is None, r.payload.get("chunk_index", 0)))
        return [Candidate(chunk_id=r.payload.get("chunk_id", ""), score=0.0, source_path=SOURCE_PATH,
                          metadata=r.payload, content=r.payload.get("content", "")) for r in records]

    async def first_chunks(self, doc_id: str, n: int) -> List[Candidate]:
        if doc_id not in self._cache:
            self._cache[doc_id] = await self._load_document(doc_id)
        # Copies: fusion mutates source_path on the candidate objects it is given.
        return [c.model_copy() for c in self._cache[doc_id][:n]]

    async def for_scope(self, doc_ids: List[str], n: int) -> List[Candidate]:
        """The opening chunks of every scoped document, documents in the order given."""
        out: List[Candidate] = []
        for doc_id in doc_ids:
            out.extend(await self.first_chunks(doc_id, n))
        return out
