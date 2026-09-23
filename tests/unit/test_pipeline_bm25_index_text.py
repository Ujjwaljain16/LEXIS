"""
Regression test for ingestion/pipeline.py::_upsert_to_databases's BM25 doc
construction (R1 of the retrieval-improvement ladder in LEXIS_FINAL_PLAN.md
section 4: the chunker already prepends a deterministic contextual header --
doc title/type/section -- before embedding (chunker.py::_build_cch_header),
but BM25 previously indexed raw_content only, so keyword search never
benefited from that context the way dense retrieval already did).

Bug risk this guards against: LexisBM25Index returns its corpus dict
verbatim as each hit's payload, and retrieval/path_d_bm25.py reads
payload["content"] as the candidate's actual displayed/cited text. Simply
pointing "content" at the CCH-prefixed text (the naive fix) would leak the
header into every BM25 candidate's content. The correct fix -- proven here --
is a separate "index_text" field: BM25 tokenizes index_text (the CCH-prefixed
text) but "content" stays the clean raw_content.
"""
import numpy as np
import pytest

from lexis.ingestion.interfaces import BaseEmbedder
from lexis.ingestion.pipeline import IngestionPipeline
from lexis.indexing.schema import Chunk, ChunkMetadata


class FakeEmbedder(BaseEmbedder):
    def embed_text(self, text: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3])

    def embed_batch(self, texts):
        return [np.array([0.1, 0.2, 0.3]) for _ in texts]


class FakeQdrant:
    async def upsert_chunks(self, collection_name, points):
        pass


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


def make_pipeline() -> IngestionPipeline:
    pipeline = IngestionPipeline(embedder=FakeEmbedder())
    pipeline.qdrant = FakeQdrant()
    pipeline.bm25 = FakeBM25()
    pipeline.pg = FakePG()
    return pipeline


@pytest.mark.asyncio
async def test_bm25_doc_carries_cch_prefixed_index_text_separate_from_clean_content():
    meta = ChunkMetadata(source_file="master_supply.txt", page_num=1, document_type="contract")
    chunk = Chunk.create(
        doc_id="doc-cch-1", split_idx=0,
        raw_content="The term of this agreement is five years.",
        metadata=meta,
    )
    chunk.expanded_content = "Document: Master Supply Agreement\nType: contract\nSection: Term\n\n" + chunk.raw_content

    pipeline = make_pipeline()
    await pipeline._upsert_to_databases([chunk])

    assert len(pipeline.bm25.indexed) == 1
    doc = pipeline.bm25.indexed[0][0]

    assert doc["content"] == chunk.raw_content
    assert "Document:" not in doc["content"]
    assert doc["index_text"] == chunk.expanded_content
    assert doc["index_text"].startswith("Document: Master Supply Agreement")


@pytest.mark.asyncio
async def test_bm25_doc_index_text_equals_content_when_no_expansion_is_set():
    """A chunk with no expanded_content (content falls back to raw_content)
    should still populate index_text -- it's just identical to content, not
    a broken/missing key."""
    meta = ChunkMetadata(source_file="plain.txt", page_num=1, document_type="unknown")
    chunk = Chunk.create(doc_id="doc-plain-1", split_idx=0, raw_content="A plain passage.", metadata=meta)

    pipeline = make_pipeline()
    await pipeline._upsert_to_databases([chunk])

    doc = pipeline.bm25.indexed[0][0]
    assert doc["content"] == "A plain passage."
    assert doc["index_text"] == "A plain passage."
