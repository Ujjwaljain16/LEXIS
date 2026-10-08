"""
Regression test for ingestion/embedder.py::BGEM3Embedder.embed_batch.

Bug context: batch_size was hardcoded to 32, giving no way to shrink peak
GPU memory per encode() call on memory-constrained hardware (a live Colab
T4 run OOM'd mid-ingest on a batch of ordinary contract chunks). Now driven
by settings.embedding_batch_size so it can be lowered via config/env without
a code change, without touching the encode() call's other behaviour.
"""
import pytest

from lexis.config import settings
from lexis.ingestion import embedder as embedder_module
from lexis.ingestion.embedder import BGEM3Embedder


class FakeModel:
    def __init__(self, captured):
        self._captured = captured

    def encode(self, texts, **kwargs):
        self._captured.update(kwargs)
        self._captured["texts"] = texts
        return [[0.0] * 4 for _ in texts]


def test_embed_batch_uses_configured_batch_size(monkeypatch):
    captured = {}
    monkeypatch.setattr(embedder_module, "get_embedder", lambda: FakeModel(captured))

    BGEM3Embedder().embed_batch(["a", "b", "c"])

    assert captured["batch_size"] == settings.embedding_batch_size


def test_changing_configured_batch_size_changes_the_encode_call(monkeypatch):
    monkeypatch.setattr(settings, "embedding_batch_size", 4)
    captured = {}
    monkeypatch.setattr(embedder_module, "get_embedder", lambda: FakeModel(captured))

    BGEM3Embedder().embed_batch(["a", "b", "c"])

    assert captured["batch_size"] == 4


def test_no_hardcoded_batch_size_survives_a_config_change(monkeypatch):
    monkeypatch.setattr(settings, "embedding_batch_size", 7)
    captured = {}
    monkeypatch.setattr(embedder_module, "get_embedder", lambda: FakeModel(captured))

    BGEM3Embedder().embed_batch(["a"])

    assert captured["batch_size"] == 7
    assert captured["batch_size"] != 32
