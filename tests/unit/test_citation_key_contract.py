"""
Regression tests for the citation-key contract (plan defect D3 follow-up):
ingestion/pipeline.py reads Chunk.pqac_key when writing citation rows, and
map_reduce_filter reads a key from the Qdrant payload. Both previously
referenced a key that did not exist, so PDF ingestion (which has bounding
boxes) would raise AttributeError and deep-mode citations were "unknown".
"""
import re

from lexis.indexing.schema import Chunk, ChunkMetadata

# Same pattern the streaming synthesizer intercepts (generation/synthesizer.py).
KEY_RE = re.compile(r"^pqac-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
META = ChunkMetadata(source_file="f.txt", page_num=1, document_type="unknown")


def make_chunk(idx=0, text="some generic passage"):
    return Chunk.create(doc_id="doc-1", split_idx=idx, raw_content=text, metadata=META)


def test_chunk_has_pqac_key_equal_to_chunk_id():
    c = make_chunk()
    assert c.pqac_key == c.chunk_id


def test_pqac_key_is_a_full_uuid_key_the_synthesizer_regex_accepts():
    assert KEY_RE.match(make_chunk().pqac_key)


def test_pqac_key_is_deterministic_and_content_sensitive():
    assert make_chunk(0, "a").pqac_key == make_chunk(0, "a").pqac_key
    assert make_chunk(0, "a").pqac_key != make_chunk(0, "b").pqac_key
    assert make_chunk(0, "a").pqac_key != make_chunk(1, "a").pqac_key


def test_map_reduce_falls_back_to_chunk_id_when_payload_has_no_pqac_key():
    import asyncio
    from unittest.mock import AsyncMock, patch

    from lexis.reranking import map_reduce_filter as mrf

    cached = type("Cached", (), {"reason": "evidence text", "score": 0.9})()
    fake_cache = type("FakeCache", (), {"get": AsyncMock(return_value=cached)})()
    chunk = {"payload": {"chunk_id": "pqac-real-id", "content": "text"}}

    with patch.object(mrf, "get_cache", return_value=fake_cache):
        node = asyncio.run(mrf._map_single_chunk("q", "h", chunk, "m"))

    assert node.citations == ["pqac-real-id"]
