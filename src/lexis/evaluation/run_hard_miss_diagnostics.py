"""
CLI: hard-miss diagnostic analysis (Step 10).

Step 9 identified a "hard miss" set: relevant chunks absent from BOTH the
dense and BM25 retrieval paths even at the largest diagnostic depth tested
(--max-depth, default 200 -- matching Step 9's ceiling, but independently
configurable here). This script independently rebuilds that same set (it
does not read Step 9's JSON artifact, so it has no coupling to that file's
schema or to a stale prior run) and gathers generic, document-agnostic
evidence about each one: lexical query/chunk overlap, chunk/answer-span
characteristics, and -- when it can be obtained safely -- direct dense
vector and BM25 evidence for the exact (query, relevant_chunk) pair.

This is measurement only. It does not modify RetrievalEngine.retrieve(),
retrieve_with_trace(), RRF, per-path top-k, the embedding model, chunking,
or the ground-truth mapping algorithm, and it draws no conclusion about why
a chunk is a hard miss -- see evaluation/hard_miss_diagnostics.py's
docstring and this run's printed summary for what the numbers do and do
not show.

Dense/BM25 "exact rank" requires scanning the full indexed population/
corpus. At the scale this project runs at (a few hundred chunks for the
10-contract CUAD baseline) that is cheap and exact, not approximate; if a
population/corpus exceeds --max-population-for-exact-rank /
--max-corpus-for-exact-rank, the scan is skipped and the limitation is
recorded explicitly (rank_unavailable_reason) rather than estimated.

Example:
    python -m lexis.evaluation.run_hard_miss_diagnostics --benchmark cuad \\
        --cuad-path data/cuad_raw/CUAD_v1/CUAD_v1.json \\
        --num-contracts 10 --max-questions 50 --max-depth 200
"""
import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from qdrant_client.http import models as qdrant_models

from lexis.config import settings
from lexis.evaluation.dataset.cuad_loader import CUADAdapter, CUADLoader, select_contracts
from lexis.evaluation.dataset.mapping import deterministic_document_id, map_answer_texts_to_chunk_ids
from lexis.evaluation.depth_sensitivity import IndexIntegrityStatus, check_index_integrity
from lexis.evaluation.diagnostics import rank_of
from lexis.evaluation.hard_miss_diagnostics import (
    BM25Evidence,
    BM25TextStatus,
    DenseEvidence,
    VectorStatus,
    build_hard_miss_record,
    build_hard_miss_report,
    format_hard_miss_summary,
)
from lexis.evaluation.run_eval import chunk_document_text
from lexis.indexing.bm25_index import LexisBM25Index
from lexis.indexing.schema import Chunk
from lexis.ingestion.chunker import SemanticChunker
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.ingestion.parser import LexisParser
from lexis.retrieval.hybrid_retriever import RetrievalEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _fetch_target_vector_by_filter(retriever: RetrievalEngine, chunk_id: str):
    """Fallback single-point lookup, used ONLY when the full population was
    too large to fetch entirely (see _fetch_full_population). Requires a
    keyword payload index on the "chunk_id" field; the primary collection
    currently only indexes "doc_id" (see
    indexing/qdrant_client.py::initialize_collections), so this raises
    UnexpectedResponse (400) against the live collection. Creating a new
    index is a schema change this diagnostic does not make -- the caller
    must treat any exception here as "lookup unavailable", not crash.
    Returns (vector_or_None, error_or_None)."""
    try:
        points, _ = await retriever.qdrant.client.scroll(
            collection_name=settings.qdrant_collection_primary,
            scroll_filter=qdrant_models.Filter(
                must=[qdrant_models.FieldCondition(key="chunk_id", match=qdrant_models.MatchValue(value=chunk_id))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=True,
        )
        return (points[0].vector if points else None), None
    except Exception as e:
        return None, str(e)


async def _fetch_full_population(retriever: RetrievalEngine, max_population: int):
    """Returns (vectors_by_chunk_id or None, population_size, limitation or None).
    vectors_by_chunk_id is None only when population_size exceeds
    max_population -- the scan is then skipped entirely rather than
    approximated."""
    count_result = await retriever.qdrant.client.count(
        collection_name=settings.qdrant_collection_primary, exact=True
    )
    population_size = count_result.count
    if population_size > max_population:
        return None, population_size, (
            f"population size {population_size} exceeds configured max "
            f"{max_population} for a full exact-rank scan"
        )

    vectors_by_chunk_id: Dict[str, List[float]] = {}
    offset = None
    while True:
        points, next_offset = await retriever.qdrant.client.scroll(
            collection_name=settings.qdrant_collection_primary,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in points:
            cid = (p.payload or {}).get("chunk_id")
            if cid and p.vector is not None:
                vectors_by_chunk_id[cid] = p.vector
        if next_offset is None:
            break
        offset = next_offset
    return vectors_by_chunk_id, population_size, None


async def _dense_evidence_for(
    retriever: RetrievalEngine,
    chunk_id: str,
    query_vector: np.ndarray,
    vectors_by_chunk_id: Optional[Dict[str, List[float]]],
    population_size: int,
    population_rank_limitation: Optional[str],
) -> DenseEvidence:
    """When the full population was fetched (the common case at this
    project's scale), the target's own vector is looked up directly from
    that already-fetched dict -- no second, per-chunk query against
    Qdrant is needed, so this never depends on a payload index existing
    for "chunk_id". Only when the population was too large to fetch in
    full does this fall back to a filtered single-point lookup, which may
    itself be unavailable (see _fetch_target_vector_by_filter)."""
    if vectors_by_chunk_id is not None:
        target_vector = vectors_by_chunk_id.get(chunk_id)
        if target_vector is None:
            return DenseEvidence(
                vector_status=VectorStatus.MISSING_FROM_VECTOR_STORE.value,
                query_chunk_similarity=None,
                population_size=population_size,
                exact_rank=None,
                rank_unavailable_reason="target chunk has no stored vector in the collection",
            )

        similarity = float(np.dot(query_vector, np.asarray(target_vector)))
        scored = sorted(
            ((cid, float(np.dot(query_vector, np.asarray(vec)))) for cid, vec in vectors_by_chunk_id.items()),
            key=lambda pair: pair[1],
            reverse=True,
        )
        exact_rank = next((i + 1 for i, (cid, _) in enumerate(scored) if cid == chunk_id), None)
        return DenseEvidence(
            vector_status=VectorStatus.FOUND.value,
            query_chunk_similarity=similarity,
            population_size=population_size,
            exact_rank=exact_rank,
            rank_unavailable_reason=None if exact_rank is not None else (
                "target chunk was not found during the full-population scan (unexpected)"
            ),
        )

    target_vector, lookup_error = await _fetch_target_vector_by_filter(retriever, chunk_id)
    if lookup_error is not None:
        return DenseEvidence(
            vector_status=VectorStatus.LOOKUP_UNAVAILABLE.value,
            query_chunk_similarity=None,
            population_size=population_size,
            exact_rank=None,
            rank_unavailable_reason=f"per-chunk vector lookup failed: {lookup_error}",
        )
    if target_vector is None:
        return DenseEvidence(
            vector_status=VectorStatus.MISSING_FROM_VECTOR_STORE.value,
            query_chunk_similarity=None,
            population_size=population_size,
            exact_rank=None,
            rank_unavailable_reason="target chunk has no stored vector in the collection",
        )
    similarity = float(np.dot(query_vector, np.asarray(target_vector)))
    return DenseEvidence(
        vector_status=VectorStatus.FOUND.value,
        query_chunk_similarity=similarity,
        population_size=population_size,
        exact_rank=None,
        rank_unavailable_reason=population_rank_limitation,
    )


def _bm25_evidence_for(bm25_index: LexisBM25Index, query_text: str, chunk_id: str, max_corpus: int) -> BM25Evidence:
    corpus_size = bm25_index.corpus_size()

    if not bm25_index.contains(chunk_id):
        return BM25Evidence(
            text_status=BM25TextStatus.MISSING_FROM_BM25_CORPUS.value,
            score=None,
            corpus_size=corpus_size,
            exact_rank=None,
            rank_unavailable_reason="chunk_id not present in the BM25 corpus",
        )

    if corpus_size > max_corpus:
        return BM25Evidence(
            text_status=BM25TextStatus.FOUND.value,
            score=None,
            corpus_size=corpus_size,
            exact_rank=None,
            rank_unavailable_reason=(
                f"corpus size {corpus_size} exceeds configured max {max_corpus} for a full exact-rank/score scan"
            ),
        )

    hits = bm25_index.search(query_text, top_k=corpus_size)
    for i, hit in enumerate(hits):
        if hit["chunk_id"] == chunk_id:
            return BM25Evidence(
                text_status=BM25TextStatus.FOUND.value,
                score=hit["score"],
                corpus_size=corpus_size,
                exact_rank=i + 1,
                rank_unavailable_reason=None,
            )

    return BM25Evidence(
        text_status=BM25TextStatus.FOUND.value,
        score=None,
        corpus_size=corpus_size,
        exact_rank=None,
        rank_unavailable_reason="chunk present in corpus but absent from a full-corpus search result (unexpected)",
    )


async def run_cuad_hard_miss_diagnostics(
    cuad_path: str,
    num_contracts: int,
    max_questions: int,
    max_depth: int,
    max_population_for_exact_rank: int,
    max_corpus_for_exact_rank: int,
    output_path: str,
):
    logger.info(f"Loading CUAD dataset from {cuad_path}")
    raw = CUADLoader().load(cuad_path)
    contracts = select_contracts(raw, num_contracts)
    logger.info(f"Selected {len(contracts)} contracts (deterministic, sorted by title).")

    parser = LexisParser()
    embedder = BGEM3Embedder()
    chunker = SemanticChunker(embedder=embedder)

    chunks_by_doc_id: Dict[str, List[Chunk]] = {}
    chunk_by_id: Dict[str, Chunk] = {}
    for contract in contracts:
        title = contract["title"]
        doc_id = deterministic_document_id(title, prefix="cuad")
        context = contract["paragraphs"][0]["context"]
        logger.info(f"Chunking contract '{title}' (doc_id={doc_id})...")
        chunks = chunk_document_text(parser, chunker, context, doc_id)
        chunks_by_doc_id[doc_id] = chunks
        for c in chunks:
            chunk_by_id[c.chunk_id] = c
        logger.info(f"  -> {len(chunks)} chunks")

    cases, unmapped = CUADAdapter().build_cases(raw, chunks_by_doc_id, num_contracts, max_questions)
    logger.info(f"Built {len(cases)} scoreable benchmark cases; {len(unmapped)} unmapped.")

    if not cases:
        logger.error("No scoreable benchmark cases were produced. Cannot run the hard-miss diagnostic.")
        return None

    retriever = RetrievalEngine()

    settings_snapshot_before = (
        settings.retrieval_top_k_per_path, settings.retrieval_fusion_top_k, settings.rrf_k,
        settings.qdrant_collection_primary, settings.embedding_model,
    )

    scored_cases = [c for c in cases if c.has_chunk_level_ground_truth()]

    # --- Identify the hard-miss set: (case, chunk_id) pairs absent from
    # BOTH raw paths at max_depth. Independently rebuilt here (not read from
    # Step 9's artifact) via the same retrieve_with_trace() diagnostic
    # method, one call per case.
    hard_miss_pairs = []  # List[Tuple[BenchmarkCase, str]]
    for case in scored_cases:
        trace = await retriever.retrieve_with_trace(case.query, top_k_per_path=max_depth, top_n_rrf=max_depth)
        for chunk_id in case.relevant_chunk_ids:
            dense_rank = rank_of(chunk_id, trace.dense_candidates)
            bm25_rank = rank_of(chunk_id, trace.bm25_candidates)
            if dense_rank is None and bm25_rank is None:
                hard_miss_pairs.append((case, chunk_id))

    logger.info(f"Hard-miss set: {len(hard_miss_pairs)} (case, relevant_chunk) pairs at max_depth={max_depth}.")

    # --- Gather evidence. Dense population is fetched ONCE (it does not
    # depend on any query); BM25 corpus size/membership likewise come from
    # the single shared index. Per-query embeddings are cached since a case
    # can contribute more than one hard-miss chunk.
    vectors_by_chunk_id, population_size, population_rank_limitation = await _fetch_full_population(
        retriever, max_population_for_exact_rank
    )
    if population_rank_limitation:
        logger.warning(population_rank_limitation)

    query_vector_cache: Dict[str, np.ndarray] = {}
    records = []
    for case, chunk_id in hard_miss_pairs:
        if case.query not in query_vector_cache:
            query_vector_cache[case.query] = np.asarray(embedder.embed_text(case.query))
        query_vector = query_vector_cache[case.query]

        doc_id = next(iter(case.relevant_doc_ids), None)
        chunk = chunk_by_id.get(chunk_id)
        chunk_content = chunk.raw_content if chunk is not None else ""

        answer_texts = case.metadata.get("answer_texts", [])
        matched_answer_texts = [
            a for a in answer_texts
            if chunk is not None and map_answer_texts_to_chunk_ids([a], [chunk])
        ]

        index_integrity = check_index_integrity(chunk_id, doc_id, chunks_by_doc_id).value

        dense_evidence = await _dense_evidence_for(
            retriever, chunk_id, query_vector, vectors_by_chunk_id, population_size, population_rank_limitation
        )
        bm25_evidence = _bm25_evidence_for(retriever.bm25, case.query, chunk_id, max_corpus_for_exact_rank)

        records.append(build_hard_miss_record(
            case_id=case.case_id,
            chunk_id=chunk_id,
            doc_id=doc_id,
            query=case.query,
            chunk_content=chunk_content,
            matched_answer_texts=matched_answer_texts,
            index_integrity=index_integrity,
            dense_evidence=dense_evidence,
            bm25_evidence=bm25_evidence,
        ))

    settings_snapshot_after = (
        settings.retrieval_top_k_per_path, settings.retrieval_fusion_top_k, settings.rrf_k,
        settings.qdrant_collection_primary, settings.embedding_model,
    )
    if settings_snapshot_before != settings_snapshot_after:
        raise RuntimeError(
            "Production Settings changed during the hard-miss diagnostic -- "
            "this must never happen; refusing to write a report."
        )

    run_config = {
        "num_contracts": num_contracts,
        "max_questions": max_questions,
        "max_depth": max_depth,
        "max_population_for_exact_rank": max_population_for_exact_rank,
        "max_corpus_for_exact_rank": max_corpus_for_exact_rank,
        "embedding_model": settings.embedding_model,
        "rrf_k": settings.rrf_k,
    }
    report = build_hard_miss_report(records, benchmark="cuad", run_config=run_config)
    summary = format_hard_miss_summary(report)
    logger.info("\n" + summary)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": report.benchmark,
        "cuad_source_path": cuad_path,
        "run_config": report.run_config,
        "num_cases": len(scored_cases),
        "num_hard_misses": report.num_hard_misses,
        "dense_population_size": population_size,
        "dense_population_rank_limitation": population_rank_limitation,
        "bm25_corpus_size": retriever.bm25.corpus_size(),
        "lexical_overlap_ratio_distribution": report.lexical_overlap_ratio_distribution,
        "chunk_char_length_distribution": report.chunk_char_length_distribution,
        "dense_similarity_distribution": report.dense_similarity_distribution,
        "dense_rank_distribution": report.dense_rank_distribution,
        "bm25_score_distribution": report.bm25_score_distribution,
        "bm25_rank_distribution": report.bm25_rank_distribution,
        "num_with_nonzero_lexical_overlap": report.num_with_nonzero_lexical_overlap,
        "num_with_valid_dense_vector": report.num_with_valid_dense_vector,
        "num_dense_evidence_available": report.num_dense_evidence_available,
        "num_with_nonzero_bm25_score": report.num_with_nonzero_bm25_score,
        "num_bm25_evidence_available": report.num_bm25_evidence_available,
        "num_index_integrity_anomalies": report.num_index_integrity_anomalies,
        "summary_text": summary,
        "records": [asdict(r) for r in records],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Hard-miss diagnostic report written to {output_path}")
    return output


def main():
    arg_parser = argparse.ArgumentParser(description="Diagnose the hard-miss set (relevant chunks absent from both retrieval paths at max depth) -- measurement only.")
    arg_parser.add_argument("--benchmark", choices=["cuad"], default="cuad")
    arg_parser.add_argument("--cuad-path", default="data/cuad_raw/CUAD_v1/CUAD_v1.json")
    arg_parser.add_argument("--num-contracts", type=int, default=10)
    arg_parser.add_argument("--max-questions", type=int, default=50)
    arg_parser.add_argument(
        "--max-depth", type=int, default=200,
        help="Per-path/RRF depth used to (re)identify the hard-miss set (diagnostic checkpoint, not a production setting).",
    )
    arg_parser.add_argument(
        "--max-population-for-exact-rank", type=int, default=20000,
        help="Skip the full-collection dense exact-rank scan if the collection exceeds this many points.",
    )
    arg_parser.add_argument(
        "--max-corpus-for-exact-rank", type=int, default=20000,
        help="Skip the full-corpus BM25 exact-rank/score scan if the corpus exceeds this many documents.",
    )
    arg_parser.add_argument("--output", default="evaluation/reports/cuad_hard_miss_diagnostics.json")
    args = arg_parser.parse_args()

    if args.benchmark == "cuad":
        asyncio.run(run_cuad_hard_miss_diagnostics(
            args.cuad_path, args.num_contracts, args.max_questions, args.max_depth,
            args.max_population_for_exact_rank, args.max_corpus_for_exact_rank, args.output,
        ))
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
