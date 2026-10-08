"""
Proves ingestion and the LIVE retrieval path agree on which Qdrant collection
holds primary chunks -- not by coincidence of matching string literals, but
because both read the same settings field, and changing that field moves
both sides together.

The live retrieval engine is retrieval/hybrid_retriever.py::RetrievalEngine
(constructed by serving/routes/query.py and evaluation/run_eval.py); the
standalone retrieval/path_b_global.py is a separate, currently orphaned
duplicate (see test_retrieval_path_collection_config.py) and is not what's
checked here.
"""
import numpy as np
import pytest

from lexis.config import settings
from lexis.ingestion.interfaces import BaseEmbedder
from lexis.ingestion.pipeline import IngestionPipeline
from lexis.indexing.schema import Chunk, ChunkMetadata
from lexis.retrieval.hybrid_retriever import RetrievalEngine


class FakeEmbedder(BaseEmbedder):
    def embed_text(self, text: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3])

    def embed_batch(self, texts):
        return [np.array([0.1, 0.2, 0.3]) for _ in texts]


class FakeQdrantWriter:
    def __init__(self):
        self.upserted = []

    async def upsert_chunks(self, collection_name, points):
        self.upserted.append((collection_name, points))


class FakeBM25:
    def __init__(self):
        self.indexed = []

    def add_documents(self, docs):
        self.indexed.append(docs)


class FakePG:
    async def initialize_schema(self):
        pass

    async def insert_citation(self, citation):
        pass


@pytest.mark.asyncio
async def test_ingestion_write_and_live_retrieval_read_use_the_same_configured_collection(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_collection_primary", "sentinel_shared_collection")

    # --- Ingestion side: where does a chunk actually get written? ---
    pipeline = IngestionPipeline(embedder=FakeEmbedder())
    fake_qdrant_writer = FakeQdrantWriter()
    pipeline.qdrant = fake_qdrant_writer
    pipeline.bm25 = FakeBM25()
    pipeline.pg = FakePG()

    meta = ChunkMetadata(source_file="generic_document.txt", page_num=1, document_type="unknown")
    chunk = Chunk.create(doc_id="doc-agreement-1", split_idx=0, raw_content="Arbitrary passage.", metadata=meta)
    await pipeline._upsert_to_databases([chunk])

    assert len(fake_qdrant_writer.upserted) == 1
    written_collection, _ = fake_qdrant_writer.upserted[0]

    # --- Retrieval side: where does the live engine actually search? ---
    engine = RetrievalEngine()
    read_collections = []

    async def fake_search(collection_name, query_vector, top_k=10):
        read_collections.append(collection_name)
        return []

    engine.qdrant.search = fake_search
    await engine._path_b_global(query_emb=[0.1, 0.2, 0.3], top_k=5)

    assert len(read_collections) == 1
    read_collection = read_collections[0]

    # Both sides resolved to the sentinel we injected, and therefore to each other.
    assert written_collection == "sentinel_shared_collection"
    assert read_collection == "sentinel_shared_collection"
    assert written_collection == read_collection


@pytest.mark.asyncio
async def test_changing_config_moves_both_ingestion_and_retrieval_together(monkeypatch):
    """A second, differently-valued sentinel to rule out a coincidental match
    against a single hardcoded default somewhere in the chain."""
    monkeypatch.setattr(settings, "qdrant_collection_primary", "another_sentinel_value_999")

    pipeline = IngestionPipeline(embedder=FakeEmbedder())
    fake_qdrant_writer = FakeQdrantWriter()
    pipeline.qdrant = fake_qdrant_writer
    pipeline.bm25 = FakeBM25()
    pipeline.pg = FakePG()

    meta = ChunkMetadata(source_file="another_document.txt", page_num=1, document_type="unknown")
    chunk = Chunk.create(doc_id="doc-agreement-2", split_idx=3, raw_content="Another passage.", metadata=meta)
    await pipeline._upsert_to_databases([chunk])
    written_collection, _ = fake_qdrant_writer.upserted[0]

    engine = RetrievalEngine()
    read_collections = []

    async def fake_search(collection_name, query_vector, top_k=10):
        read_collections.append(collection_name)
        return []

    engine.qdrant.search = fake_search
    await engine._path_b_global(query_emb=[0.1, 0.2, 0.3], top_k=5)

    assert written_collection == "another_sentinel_value_999"
    assert read_collections[0] == "another_sentinel_value_999"
