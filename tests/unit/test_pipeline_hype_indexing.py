"""
Regression tests for ingestion/pipeline.py's R2 HyPE wiring
(IngestionPipeline._build_hype_points / _upsert_to_databases).

settings.hype_enabled defaults to False, so ingestion behavior for every
existing caller is unchanged unless a test/config explicitly opts in.
"""
import numpy as np
import pytest

from lexis.config import settings
from lexis.ingestion.interfaces import BaseEmbedder
from lexis.ingestion.pipeline import IngestionPipeline
from lexis.indexing.schema import Chunk, ChunkMetadata


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
    def add_documents(self, docs):
        pass


class FakePG:
    async def initialize_schema(self):
        pass

    async def insert_citation(self, citation):
        pass


class FakeHyPEGenerator:
    def __init__(self, questions_by_text=None, default=("Q1?", "Q2?")):
        self.questions_by_text = questions_by_text or {}
        self.default = list(default)
        self.calls = []

    async def generate_questions(self, chunk_text):
        self.calls.append(chunk_text)
        return list(self.questions_by_text.get(chunk_text, self.default))


def make_pipeline(hype_generator=None) -> IngestionPipeline:
    pipeline = IngestionPipeline(embedder=FakeEmbedder())
    pipeline.qdrant = FakeQdrant()
    pipeline.bm25 = FakeBM25()
    pipeline.pg = FakePG()
    if hype_generator is not None:
        pipeline.hype_generator = hype_generator
    return pipeline


def make_chunk(doc_id="doc-1", split_idx=0, text="Some clause text."):
    meta = ChunkMetadata(source_file="f.txt", page_num=1, document_type="contract")
    return Chunk.create(doc_id=doc_id, split_idx=split_idx, raw_content=text, metadata=meta)


@pytest.mark.asyncio
async def test_hype_disabled_by_default_never_calls_the_generator_or_upserts(monkeypatch):
    monkeypatch.setattr(settings, "hype_enabled", False)
    fake_gen = FakeHyPEGenerator()
    pipeline = make_pipeline(hype_generator=fake_gen)

    await pipeline._upsert_to_databases([make_chunk()])

    assert fake_gen.calls == []
    assert [c for c, _ in pipeline.qdrant.upserted] == [settings.qdrant_collection_primary]


@pytest.mark.asyncio
async def test_hype_enabled_generates_questions_and_upserts_to_the_hype_collection(monkeypatch):
    monkeypatch.setattr(settings, "hype_enabled", True)
    fake_gen = FakeHyPEGenerator(default=("What is the term?", "Who are the parties?"))
    pipeline = make_pipeline(hype_generator=fake_gen)
    chunk = make_chunk(text="The term of this agreement is five years.")

    await pipeline._upsert_to_databases([chunk])

    assert fake_gen.calls == ["The term of this agreement is five years."]
    collections_written = [c for c, _ in pipeline.qdrant.upserted]
    assert settings.qdrant_collection_hype in collections_written

    _, hype_points = next((c, pts) for c, pts in pipeline.qdrant.upserted if c == settings.qdrant_collection_hype)
    assert len(hype_points) == 2
    questions = {p.payload["question"] for p in hype_points}
    assert questions == {"What is the term?", "Who are the parties?"}
    for p in hype_points:
        # Denormalized: points carry the ORIGINATING chunk's real content/doc_id, not just a
        # reference -- so a retrieval hit can build a Candidate without a second lookup.
        assert p.payload["chunk_id"] == chunk.chunk_id
        assert p.payload["doc_id"] == chunk.doc_id
        assert p.payload["content"] == chunk.raw_content


@pytest.mark.asyncio
async def test_hype_enabled_but_generator_returns_nothing_upserts_no_hype_points(monkeypatch):
    monkeypatch.setattr(settings, "hype_enabled", True)
    fake_gen = FakeHyPEGenerator(default=())
    pipeline = make_pipeline(hype_generator=fake_gen)

    await pipeline._upsert_to_databases([make_chunk()])

    collections_written = [c for c, _ in pipeline.qdrant.upserted]
    assert settings.qdrant_collection_hype not in collections_written


@pytest.mark.asyncio
async def test_hype_points_use_distinct_ids_per_question_on_the_same_chunk(monkeypatch):
    monkeypatch.setattr(settings, "hype_enabled", True)
    fake_gen = FakeHyPEGenerator(default=("Question A?", "Question B?"))
    pipeline = make_pipeline(hype_generator=fake_gen)

    await pipeline._upsert_to_databases([make_chunk()])

    _, hype_points = next((c, pts) for c, pts in pipeline.qdrant.upserted if c == settings.qdrant_collection_hype)
    ids = [p.id for p in hype_points]
    assert len(ids) == len(set(ids))
