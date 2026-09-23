"""
Local BM25 lexical index (Path D keyword search) using bm25s.

Rationale: see docs/ADR.md ADR-003. Replaces Elasticsearch as the keyword-
search backend, per the architecture direction already noted in
retrieval/path_d_bm25.py ("Replaces Elasticsearch with a local fast
implementation (e.g. BM25S) ... to reduce operational overhead for <10M
chunk scale"). bm25s is pure Python (numpy/scipy), runs in-process, and
needs no server, JVM, or network connection -- eliminating Elasticsearch's
operational and memory footprint entirely.

bm25s builds its sparse retrieval matrix over a full corpus in one pass; it
has no notion of incremental per-document indexing the way Elasticsearch
does. To keep newly-ingested documents searchable immediately without a
separate reindex step, this class persists the full corpus as a JSONL file
and rebuilds the in-memory index whenever documents are added. At the
<10M chunk scale this is designed for, a full rebuild is seconds, not
minutes -- an explicit, accepted tradeoff, not an oversight.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import bm25s

logger = logging.getLogger(__name__)


class LexisBM25Index:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.corpus_path = os.path.join(index_dir, "corpus.jsonl")
        self._retriever: Optional[bm25s.BM25] = None
        self._loaded_from_disk = False

    def _load_corpus(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.corpus_path):
            return []
        docs = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return docs

    def _rebuild(self, corpus: List[Dict[str, Any]]) -> None:
        if not corpus:
            self._retriever = None
            return
        # index_text (when present) is what gets tokenized for matching -- it may carry extra
        # context (e.g. a contextual chunk header) that "content" deliberately excludes, since
        # "content" is returned verbatim as each hit's payload (see search()) and callers treat
        # it as the chunk's actual displayed/cited text.
        texts = [doc.get("index_text", doc.get("content", "")) for doc in corpus]
        tokenized = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(tokenized, show_progress=False)
        self._retriever = retriever

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Merges documents into the persisted corpus (by chunk_id -- a
        document with the same chunk_id as an existing entry replaces it,
        matching Qdrant's upsert-by-id semantics rather than blindly
        appending a duplicate) and rebuilds the index so they are
        searchable immediately. Each doc must have a 'chunk_id' and
        'content' key; an optional 'index_text' key is tokenized for
        matching instead of 'content' when present. Other keys are carried
        through into search results but are otherwise opaque to this
        class."""
        if not docs:
            return
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)

        merged = {doc["chunk_id"]: doc for doc in self._load_corpus()}
        for doc in docs:
            merged[doc["chunk_id"]] = doc

        with open(self.corpus_path, "w", encoding="utf-8") as f:
            for doc in merged.values():
                f.write(json.dumps(doc) + "\n")

        self._rebuild(list(merged.values()))
        logger.info(f"BM25 index rebuilt with {len(merged)} total documents.")

    def _ensure_loaded(self) -> None:
        if self._retriever is None and not self._loaded_from_disk:
            self._rebuild(self._load_corpus())
            self._loaded_from_disk = True

    def corpus_size(self) -> int:
        """Number of documents currently in the index (loads from disk if
        not already loaded). Lets a caller request a full-corpus ranking
        from search() without guessing a top_k large enough -- used by
        evaluation/run_hard_miss_diagnostics.py, not by production search."""
        self._ensure_loaded()
        return len(self._retriever.corpus) if self._retriever is not None else 0

    def contains(self, chunk_id: str) -> bool:
        """Whether chunk_id exists in the persisted corpus at all,
        independent of any particular query's score/rank. Used by
        diagnostics to distinguish "never indexed" from "indexed but ranked
        beyond what was scanned"."""
        self._ensure_loaded()
        if self._retriever is None:
            return False
        return any(doc.get("chunk_id") == chunk_id for doc in self._retriever.corpus)

    def search(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Synchronous, CPU-only search -- callers on an event loop should
        run this via asyncio.to_thread to avoid blocking concurrent
        requests. Returns results shaped like the previous Elasticsearch
        client's search() output: [{"chunk_id", "score", "payload"}, ...]."""
        self._ensure_loaded()
        if self._retriever is None or not query_text.strip():
            return []

        corpus_size = len(self._retriever.corpus)
        k = min(top_k, corpus_size)
        if k <= 0:
            return []

        query_tokens = bm25s.tokenize(query_text, stopwords="en", show_progress=False)
        results, scores = self._retriever.retrieve(query_tokens, k=k, show_progress=False)

        hits = []
        for doc, score in zip(results[0], scores[0]):
            hits.append({
                "chunk_id": doc.get("chunk_id"),
                "score": float(score),
                "payload": doc,
            })
        return hits
