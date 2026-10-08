"""
Collection-name reconciliation regression tests.

Bug fixed: path_a_cluster.py, path_b_global.py, and path_c_concept.py each
hardcoded a stale Qdrant collection name ("clusters_v2", "primary_v2",
"propositions_v2") that does not match anything setup_collections.py
actually creates or ingestion/pipeline.py actually writes to (the real
names live in config.py: qdrant_collection_clusters="clusters",
qdrant_collection_primary="chunks_primary",
qdrant_collection_propositions="propositions"). Each class now reads its
collection name from `settings` instead.

These tests intentionally assert against dynamically-injected sentinel
values (never the production default) so a test cannot pass merely because
someone reintroduces a hardcoded string that happens to match today's
config default -- the point is to prove config actually flows into the
object, not to freeze a particular string.

path_a_cluster.py, path_b_global.py, and path_c_concept.py are NOT wired
into the live query pipeline (hybrid_retriever.py implements its own
inline Path B + Path D and is the only retrieval engine actually
constructed by serving/routes/query.py and evaluation/run_eval.py). These
three modules remain orphaned after this change -- only their collection
name is corrected, nothing here claims they are production-integrated.
"""
import numpy as np
import pytest

from lexis.config import settings
from lexis.retrieval.interfaces import Query
from lexis.retrieval.path_a_cluster import ClusterHierarchicalRetrieval
from lexis.retrieval.path_b_global import GlobalDenseRetrieval
from lexis.retrieval.path_c_concept import ConceptRoutingRetrieval


class FakeEmbedder:
    """Avoids loading the real bge-m3 model (~GBs) in a unit test."""
    def embed_text(self, text: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3])

    def embed_batch(self, texts):
        return [np.array([0.1, 0.2, 0.3]) for _ in texts]


# --- Configured collection name flows into each orphaned path's constructor ---

def test_path_a_reads_clusters_collection_from_config(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_collection_clusters", "sentinel_clusters_abc123")
    path = ClusterHierarchicalRetrieval()
    assert path.collection_name == "sentinel_clusters_abc123"
    assert path.collection_name not in ("clusters_v2", "primary_v2", "propositions_v2")


def test_path_b_reads_primary_collection_from_config(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_collection_primary", "sentinel_primary_xyz456")
    path = GlobalDenseRetrieval()
    assert path.collection_name == "sentinel_primary_xyz456"
    assert path.collection_name not in ("clusters_v2", "primary_v2", "propositions_v2")


def test_path_c_reads_propositions_collection_from_config(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_collection_propositions", "sentinel_props_789")
    path = ConceptRoutingRetrieval()
    assert path.collection_name == "sentinel_props_789"
    assert path.collection_name not in ("clusters_v2", "primary_v2", "propositions_v2")


def test_each_path_reads_its_own_dedicated_setting_not_a_shared_one(monkeypatch):
    """Changing one collection setting must not affect the others -- proves
    each path is wired to its own config field, not an aliased/shared one."""
    monkeypatch.setattr(settings, "qdrant_collection_clusters", "only_clusters_changed")

    path_a = ClusterHierarchicalRetrieval()
    path_b = GlobalDenseRetrieval()
    path_c = ConceptRoutingRetrieval()

    assert path_a.collection_name == "only_clusters_changed"
    assert path_b.collection_name == settings.qdrant_collection_primary
    assert path_c.collection_name == settings.qdrant_collection_propositions
    assert path_b.collection_name != "only_clusters_changed"
    assert path_c.collection_name != "only_clusters_changed"


# --- The collection name actually reaches the Qdrant call, not just __init__ ---

@pytest.mark.asyncio
async def test_path_a_retrieve_searches_the_configured_collection(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_collection_clusters", "sentinel_clusters_call")
    path = ClusterHierarchicalRetrieval()

    captured = {}

    async def fake_search(collection_name, query_vector, top_k=10):
        captured["collection_name"] = collection_name
        return ["fake-result"]

    path.qdrant.search = fake_search

    result = await path.retrieve([0.1, 0.2, 0.3], top_k=3)

    assert captured["collection_name"] == "sentinel_clusters_call"
    assert result == ["fake-result"]


@pytest.mark.asyncio
async def test_path_b_retrieve_searches_the_configured_collection(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_collection_primary", "sentinel_primary_call")
    path = GlobalDenseRetrieval()
    path.embedder = FakeEmbedder()  # avoid loading the real embedding model

    captured = {}

    async def fake_search(collection_name, query_vector, top_k):
        captured["collection_name"] = collection_name
        return []

    path.qdrant.search = fake_search

    result = await path.retrieve(Query(text="irrelevant text", top_k=5))

    assert captured["collection_name"] == "sentinel_primary_call"
    assert result == []


# Note: path_c_concept.py's retrieve() calls acompletion(model=settings.llm_model, ...)
# BEFORE ever reaching Qdrant, and `llm_model` is not a field defined anywhere on
# Settings (config.py only defines gemini_model_synthesis/feature/vision) -- calling
# retrieve() would raise AttributeError unrelated to collection naming. That is a
# separate, pre-existing integration bug (reported, not fixed, in this step), so no
# retrieve()-level test is added for path_c here -- only the constructor-level
# collection-name test above, which is unaffected by that bug.
