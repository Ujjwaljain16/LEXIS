"""
Document-scoped hybrid retrieval (plan R6).

Mirrors RetrievalEngine.retrieve_with_trace() -- same embedder, same
collection, same dense/BM25 path shapes, the same unmodified apply_rrf at
the same settings.rrf_k -- but restricts BOTH paths to a set of documents:

  dense  a real Qdrant payload filter on doc_id (the field is indexed)
  bm25   the FULL corpus ranking under global IDF (exactly what a filter on
         an inverted index does), then filtered to the scope. It is not a
         separate per-document index, whose IDF would differ.

With the scope set to every indexed document this must reproduce the
unscoped trace exactly; run_doc_scoping.py enforces that as a control.

The single implementation used by two callers: the ad hoc
run_doc_scoping.py experiment (which sizes routing/label-strictness
questions with bootstrap CIs and per-case blocker analysis) and
harness.py's "doc_scoped" protocol (the general, repeatable production
evaluation path -- see run_eval.py's --protocol flag). Both call this
function rather than each re-implementing scoped retrieval. This lives in
the evaluation layer and never modifies RetrievalEngine.
"""
import asyncio
from typing import Sequence

from qdrant_client.http import models as qmodels

from lexis.config import settings
from lexis.retrieval.fusion import apply_rrf
from lexis.retrieval.hybrid_retriever import RetrievalEngine, RetrievalTrace
from lexis.retrieval.interfaces import Candidate, Query


async def retrieve_scoped(engine: RetrievalEngine, query: str, doc_ids: Sequence[str],
                          top_k_per_path: int, top_n_rrf: int) -> RetrievalTrace:
    scope = list(doc_ids)
    scope_set = set(scope)
    query_emb = engine.embedder.embed_text(query).tolist()

    response = await engine.qdrant.client.query_points(
        collection_name=settings.qdrant_collection_primary,
        query=query_emb,
        limit=top_k_per_path,
        query_filter=qmodels.Filter(must=[
            qmodels.FieldCondition(key="doc_id", match=qmodels.MatchAny(any=scope))
        ]),
    )
    dense = [
        Candidate(chunk_id=r.payload.get("chunk_id", ""), score=r.score, source_path="path_b_global",
                  metadata=r.payload, content=r.payload.get("content", ""))
        for r in response.points
    ]

    corpus_size = engine.bm25.corpus_size()
    hits = await asyncio.to_thread(engine.bm25.search, query, corpus_size)
    scoped_hits = [h for h in hits if (h["payload"] or {}).get("doc_id") in scope_set][:top_k_per_path]
    bm25 = [
        Candidate(chunk_id=h["chunk_id"], score=h["score"], source_path="path_d_bm25",
                  metadata=h["payload"], content=h["payload"].get("content", ""))
        for h in scoped_hits
    ]

    hype: list = []
    if settings.hype_enabled:
        hype_response = await engine.qdrant.client.query_points(
            collection_name=settings.qdrant_collection_hype,
            query=query_emb,
            limit=top_k_per_path,
            query_filter=qmodels.Filter(must=[
                qmodels.FieldCondition(key="doc_id", match=qmodels.MatchAny(any=scope))
            ]),
        )
        hype = [
            Candidate(chunk_id=r.payload.get("chunk_id", ""), score=r.score, source_path="path_hype",
                      metadata=r.payload, content=r.payload.get("content", ""))
            for r in hype_response.points
        ]

    lists = [lst for lst in (dense, bm25, hype) if lst]
    fused = apply_rrf(lists, k=settings.rrf_k)

    # R4: same top-rerank_top_k-then-append-the-rest policy as
    # RetrievalEngine._maybe_rerank -- duplicated rather than called (getattr, not a hard
    # dependency on RetrievalEngine's exact interface) so the lightweight `engine`-like fakes
    # this module's own tests use only need a `reranker` attribute, not every RetrievalEngine
    # method.
    reranker = getattr(engine, "reranker", None)
    if reranker is not None and fused:
        head, tail = fused[:settings.rerank_top_k], fused[settings.rerank_top_k:]
        ranked = (await reranker.transform(Query(text=query), head)) + tail
    else:
        ranked = fused

    final_chunks = [
        {"id": c.chunk_id, "score": c.score, "rrf_score": c.score, "source_path": c.source_path,
         "payload": c.metadata, "text": c.content}
        for c in ranked[:top_n_rrf]
    ]
    return RetrievalTrace(query=query, dense_candidates=dense, bm25_candidates=bm25,
                          fused_candidates=fused, final_chunks=final_chunks, top_n_rrf=top_n_rrf,
                          hype_candidates=hype)
