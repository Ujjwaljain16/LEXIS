"""
Regression test for serving/routes/query.py::fetch_chunk.

Bug fixed: fetch_chunk looked points up in a hardcoded collection name
("primary_v2") that nothing in the ingestion pipeline ever writes to (the
real, configured collection is settings.qdrant_collection_primary), and it
fetched by a constructed chunk id that can never match a real point id.
fetch_chunk now takes (doc_id, chunk_index) and looks the neighbour up via
the configured collection.
"""
import pytest

from lexis.serving.routes import query as query_module
from lexis.config import settings


class FakeQdrantClient:
    def __init__(self, records):
        self._records = records
        self.calls = []

    async def find_by_payload(self, collection_name, must_match, limit=1):
        self.calls.append((collection_name, must_match, limit))
        return self._records


@pytest.mark.asyncio
async def test_fetch_chunk_uses_configured_collection_not_hardcoded_name(monkeypatch):
    fake_record = type("Rec", (), {"payload": {
        "chunk_id": "pqac-neighbor",
        "content": "neighbour text",
        "source_file": "arbitrary_document.pdf",
    }})()
    fake_qdrant = FakeQdrantClient([fake_record])
    monkeypatch.setattr(query_module, "get_qdrant_client", lambda: fake_qdrant)

    result = await query_module.fetch_chunk("doc-123", 9)

    assert len(fake_qdrant.calls) == 1
    collection_name, must_match, limit = fake_qdrant.calls[0]
    assert collection_name == settings.qdrant_collection_primary
    assert collection_name != "primary_v2"
    assert must_match == {"doc_id": "doc-123", "chunk_index": 9}
    assert limit == 1

    assert result is not None
    assert result.chunk_id == "pqac-neighbor"
    assert result.content == "neighbour text"


@pytest.mark.asyncio
async def test_fetch_chunk_returns_none_when_no_match(monkeypatch):
    fake_qdrant = FakeQdrantClient([])
    monkeypatch.setattr(query_module, "get_qdrant_client", lambda: fake_qdrant)

    result = await query_module.fetch_chunk("doc-unknown", 0)

    assert result is None
