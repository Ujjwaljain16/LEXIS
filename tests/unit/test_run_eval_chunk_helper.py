"""
Tests for evaluation/run_eval.py::chunk_document_text -- the helper that
runs LEXIS's real parser+chunker over benchmark document text (so ground
truth is matched against LEXIS's actual chunk boundaries). Uses fake
parser/chunker objects (matching the same duck-typed interface
IngestionPipeline itself accepts) so no real model or file-format library
is needed.
"""
import os

from lexis.evaluation.run_eval import chunk_document_text


class FakeParser:
    def __init__(self):
        self.calls = []

    def parse(self, file_path, doc_id):
        self.calls.append((file_path, doc_id))
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        return [{"content": content, "doc_id": doc_id, "split_idx": 0}]


class FakeChunker:
    def __init__(self):
        self.calls = []

    def chunk(self, elements):
        self.calls.append(elements)
        return [f"fake-chunk-for:{e['content']}" for e in elements]


def test_writes_text_to_a_temp_file_parses_and_chunks_it():
    parser = FakeParser()
    chunker = FakeChunker()

    result = chunk_document_text(parser, chunker, "Some generic document text.", "doc-generic-1")

    assert result == ["fake-chunk-for:Some generic document text."]
    assert len(parser.calls) == 1
    tmp_path, doc_id = parser.calls[0]
    assert doc_id == "doc-generic-1"
    assert tmp_path.endswith(".txt")


def test_temp_file_is_cleaned_up_afterwards():
    parser = FakeParser()
    chunker = FakeChunker()
    captured_path = {}

    class CapturingParser(FakeParser):
        def parse(self, file_path, doc_id):
            captured_path["path"] = file_path
            return super().parse(file_path, doc_id)

    chunk_document_text(CapturingParser(), chunker, "text", "doc-x")

    assert "path" in captured_path
    assert not os.path.exists(captured_path["path"])


def test_temp_file_is_cleaned_up_even_if_chunking_raises():
    parser = FakeParser()

    class FailingChunker(FakeChunker):
        def chunk(self, elements):
            raise RuntimeError("simulated chunking failure")

    captured_path = {}

    class CapturingParser(FakeParser):
        def parse(self, file_path, doc_id):
            captured_path["path"] = file_path
            return super().parse(file_path, doc_id)

    try:
        chunk_document_text(CapturingParser(), FailingChunker(), "text", "doc-x")
    except RuntimeError:
        pass

    assert "path" in captured_path
    assert not os.path.exists(captured_path["path"])
