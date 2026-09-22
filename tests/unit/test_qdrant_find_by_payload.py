"""
Tests for LexisQdrantClient.find_by_payload -- the lookup added so
sentence-window expansion can find a neighbouring chunk by its (doc_id,
chunk_index) position instead of by id. Point ids in Qdrant are
content-derived hashes (see Chunk.create) and cannot be constructed from
doc_id/index alone, so this filter-based scroll is required for correctness.
"""
import pytest
from unittest.mock import AsyncMock

from qdrant_client.http import models
from lexis.indexing.qdrant_client import LexisQdrantClient


class FakeRecord:
    def __init__(self, id_, payload):
        self.id = id_
        self.payload = payload


@pytest.mark.asyncio
async def test_find_by_payload_filters_on_all_given_fields_and_forwards_limit():
    client = LexisQdrantClient()
    fake_record = FakeRecord("uuid-1", {"doc_id": "doc-1", "chunk_index": 4, "content": "neighbour text"})
    client.client.scroll = AsyncMock(return_value=([fake_record], None))

    result = await client.find_by_payload(
        "chunks_primary",
        {"doc_id": "doc-1", "chunk_index": 4},
        limit=1,
    )

    assert result == [fake_record]

    _, kwargs = client.client.scroll.call_args
    assert kwargs["collection_name"] == "chunks_primary"
    assert kwargs["limit"] == 1
    assert kwargs["with_payload"] is True

    scroll_filter = kwargs["scroll_filter"]
    assert isinstance(scroll_filter, models.Filter)
    matched = {(c.key, c.match.value) for c in scroll_filter.must}
    assert matched == {("doc_id", "doc-1"), ("chunk_index", 4)}


@pytest.mark.asyncio
async def test_find_by_payload_returns_empty_list_when_nothing_matches():
    client = LexisQdrantClient()
    client.client.scroll = AsyncMock(return_value=([], None))

    result = await client.find_by_payload("chunks_primary", {"doc_id": "no-such-doc", "chunk_index": 999})

    assert result == []
