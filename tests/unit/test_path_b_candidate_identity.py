"""
Regression tests for retrieval/path_b_global.py::GlobalDenseRetrieval.retrieve.

Bug fixed: Candidate.chunk_id was built from Qdrant's internal point id
(str(point.id)) instead of the application-level chunk_id stored in the
point's payload (written by IngestionPipeline). Qdrant's point id is a
deterministic UUID derived from the real chunk_id purely so Qdrant will
accept it as a point key -- it is not the citation-bearing identity anything
downstream (citations, sentence-window lookups, judge_dep, etc.) should see.

Test IDs are deliberately unrelated in shape/length/content to each other
(not a truncation, not a prefix, not derivable from doc_id) so an
implementation that reconstructs or derives the id some other way cannot
accidentally pass.
"""
import numpy as np
import pytest

from lexis.retrieval.interfaces import Query
from lexis.retrieval.path_b_global import GlobalDenseRetrieval


class FakeEmbedder:
    """Avoids loading the real bge-m3 model in a unit test."""
    def embed_text(self, text: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3])


class FakePoint:
    def __init__(self, id_, payload, score=0.9):
        self.id = id_
        self.payload = payload
        self.score = score


def make_path() -> GlobalDenseRetrieval:
    path = GlobalDenseRetrieval()
    path.embedder = FakeEmbedder()
    return path


@pytest.mark.asyncio
async def test_candidate_chunk_id_is_the_payload_identity_not_the_qdrant_point_id():
    qdrant_internal_id = "11111111-2222-3333-4444-555555555555"
    application_chunk_id = "pqac-deadbeefcafefeed"  # deliberately different length/shape

    fake_point = FakePoint(
        id_=qdrant_internal_id,
        payload={
            "chunk_id": application_chunk_id,
            "doc_id": "generic-report-7",
            "content": "An arbitrary passage from a generic document.",
        },
    )

    path = make_path()

    async def fake_search(collection_name, query_vector, top_k):
        return [fake_point]

    path.qdrant.search = fake_search

    candidates = await path.retrieve(Query(text="irrelevant query text", top_k=1))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.chunk_id == application_chunk_id
    assert candidate.chunk_id != qdrant_internal_id
    assert candidate.content == "An arbitrary passage from a generic document."
    assert candidate.metadata["doc_id"] == "generic-report-7"


@pytest.mark.asyncio
async def test_multiple_points_each_keep_their_own_application_identity():
    points = [
        FakePoint(id_="qid-alpha-000", payload={"chunk_id": "pqac-1111aaaa", "content": "First."}),
        FakePoint(id_="qid-beta-999", payload={"chunk_id": "app-id-completely-unrelated-shape", "content": "Second."}),
    ]

    path = make_path()

    async def fake_search(collection_name, query_vector, top_k):
        return points

    path.qdrant.search = fake_search

    candidates = await path.retrieve(Query(text="q", top_k=2))

    returned_ids = {c.chunk_id for c in candidates}
    assert returned_ids == {"pqac-1111aaaa", "app-id-completely-unrelated-shape"}
    assert "qid-alpha-000" not in returned_ids
    assert "qid-beta-999" not in returned_ids


@pytest.mark.asyncio
async def test_falls_back_to_qdrant_point_id_only_when_payload_has_no_chunk_id():
    """Defensive fallback for malformed/legacy points, not the normal path --
    proves the fallback exists without masking the primary bug."""
    qdrant_internal_id = "fallback-only-id-000"
    fake_point = FakePoint(id_=qdrant_internal_id, payload={"doc_id": "doc-x", "content": "text"})

    path = make_path()

    async def fake_search(collection_name, query_vector, top_k):
        return [fake_point]

    path.qdrant.search = fake_search

    candidates = await path.retrieve(Query(text="q", top_k=1))

    assert candidates[0].chunk_id == qdrant_internal_id
