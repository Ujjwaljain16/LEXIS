"""
Regression test for ingestion/pipeline.py::_upsert_to_databases.

Bug fixed: the Qdrant payload built for every chunk never included its
chunk_index (document-relative position), so sentence-window expansion had
nothing to key neighbour lookups on. The value must come from the chunk's
own split_idx (set during parsing/chunking), not be hardcoded or assumed
sequential from zero -- this test uses non-contiguous, arbitrary indices on
a generic (non-legal) document to prove that.

Only the payload-construction path is exercised here; qdrant/es/pg are
replaced with in-memory fakes (dependency-injection style already used
elsewhere in this codebase, e.g. test_dependency_injection.py) so this test
needs no live infrastructure and never loads a real embedding model.
"""
import numpy as np
import pytest

from lexis.ingestion.interfaces import BaseEmbedder
from lexis.ingestion.pipeline import IngestionPipeline
from lexis.indexing.schema import Chunk, ChunkMetadata
from lexis.config import settings


class FakeEmbedder(BaseEmbedder):
    def embed_text(self, text: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3])

    def embed_batch(self, texts):
        return [np.array([0.1, 0.2, 0.3]) for _ in texts]


class FakeQdrant:
    def __init__(self):
        self.upserted = []  # list of (collection_name, points)

    async def upsert_chunks(self, collection_name, points):
        self.upserted.append((collection_name, points))


class FakeBM25:
    def __init__(self):
        self.indexed = []

    def add_documents(self, docs):
        self.indexed.append(docs)


class FakePG:
    def __init__(self):
        self.schema_initialized = False
        self.citations = []

    async def initialize_schema(self):
        self.schema_initialized = True

    async def insert_citation(self, citation):
        self.citations.append(citation)


def make_pipeline() -> IngestionPipeline:
    pipeline = IngestionPipeline(embedder=FakeEmbedder())
    pipeline.qdrant = FakeQdrant()
    pipeline.bm25 = FakeBM25()
    pipeline.pg = FakePG()
    return pipeline


@pytest.mark.asyncio
async def test_chunk_index_in_qdrant_payload_matches_split_idx_for_arbitrary_positions():
    # Arbitrary, non-contiguous split indices on a generic document (e.g. a
    # policy or report, not a contract) -- proves chunk_index is read from
    # the real per-chunk value, not a hardcoded 0/1/2 sequence.
    meta = ChunkMetadata(source_file="quarterly_report.txt", page_num=1, document_type="unknown")
    chunk_a = Chunk.create(doc_id="generic-doc-1", split_idx=0, raw_content="Opening passage.", metadata=meta)
    chunk_b = Chunk.create(doc_id="generic-doc-1", split_idx=17, raw_content="A much later passage.", metadata=meta)

    pipeline = make_pipeline()
    await pipeline._upsert_to_databases([chunk_a, chunk_b])

    assert len(pipeline.qdrant.upserted) == 1
    collection_name, points = pipeline.qdrant.upserted[0]
    assert collection_name == settings.qdrant_collection_primary

    payload_by_chunk_id = {p.payload["chunk_id"]: p.payload for p in points}
    assert payload_by_chunk_id[chunk_a.chunk_id]["chunk_index"] == 0
    assert payload_by_chunk_id[chunk_b.chunk_id]["chunk_index"] == 17


@pytest.mark.asyncio
async def test_chunk_index_present_for_a_second_unrelated_document_with_different_indices():
    meta = ChunkMetadata(source_file="policy.txt", page_num=2, document_type="unknown")
    chunks = [
        Chunk.create(doc_id="policy-doc-9", split_idx=idx, raw_content=f"Passage {idx}.", metadata=meta)
        for idx in (2, 5, 9)
    ]

    pipeline = make_pipeline()
    await pipeline._upsert_to_databases(chunks)

    _, points = pipeline.qdrant.upserted[0]
    indices = sorted(p.payload["chunk_index"] for p in points)
    assert indices == [2, 5, 9]
