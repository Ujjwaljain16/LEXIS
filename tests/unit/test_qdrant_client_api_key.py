"""
Tests that LexisQdrantClient forwards settings.qdrant_api_key to the
underlying AsyncQdrantClient (needed for Qdrant Cloud auth), and that an
empty key (local/unauthenticated Qdrant) is passed as None rather than an
empty string.
"""
from lexis.config import settings
from lexis.indexing import qdrant_client as qdrant_client_module
from lexis.indexing.qdrant_client import LexisQdrantClient


def test_forwards_configured_api_key(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_api_key", "sentinel-cloud-api-key")
    captured = {}

    class FakeAsyncQdrantClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(qdrant_client_module, "AsyncQdrantClient", FakeAsyncQdrantClient)

    LexisQdrantClient()

    assert captured["api_key"] == "sentinel-cloud-api-key"


def test_empty_api_key_is_passed_as_none_not_empty_string(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_api_key", "")
    captured = {}

    class FakeAsyncQdrantClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(qdrant_client_module, "AsyncQdrantClient", FakeAsyncQdrantClient)

    LexisQdrantClient()

    assert captured["api_key"] is None
