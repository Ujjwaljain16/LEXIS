"""
CLI: retrieval-depth sensitivity experiment (Step 9).

Answers: for relevant chunks absent from both retrieval paths at the
production per-path depth, are they genuinely unretrievable, or merely
ranked beyond that depth? Measurement only -- does not modify
RetrievalEngine.retrieve(), production top-k/RRF-k/model/chunking
settings, or the ground-truth mapping algorithm. Uses
RetrievalEngine.retrieve_with_trace() (added in Step 8, itself proven
identical to retrieve() for the same parameters) at each configured depth.

Reuses the exact same CUAD adapter/chunking/case-building path as
evaluation/run_eval.py so the diagnosed question set is identical to the
production baseline's.

Example:
    python -m lexis.evaluation.run_depth_sensitivity --benchmark cuad \\
        --cuad-path data/cuad_raw/CUAD_v1/CUAD_v1.json \\
        --num-contracts 10 --max-questions 50 --depths 10,30,60,100,200

Methodology note on efficiency: for each case, one retrieve_with_trace()
call is made per configured depth (so that each depth's RRF/Recall/MRR
figures come from retrieval genuinely run at that depth -- see the
docstring in evaluation/depth_sensitivity.py for why a single deep call
truncated afterward would NOT be equivalent for the fused ranking, even
though it IS equivalent for the raw per-path dense/BM25 rankings, which are
depth-invariant). The largest configured depth's call additionally supplies
the dense/BM25 rank data used for coverage at every smaller depth.
"""
import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from lexis.config import settings
from lexis.evaluation.dataset.cuad_loader import CUADAdapter, CUADLoader, select_contracts
from lexis.evaluation.dataset.mapping import deterministic_document_id
from lexis.evaluation.depth_sensitivity import (
    build_chunk_profile,
    build_rank_distribution,
    compute_depth_coverage,
    DEFINITIONS,
    format_depth_table,
    set_rrf_hit_depths,
)
from lexis.evaluation.metrics import reciprocal_rank, recall_at_k
from lexis.evaluation.run_eval import chunk_document_text
from lexis.indexing.schema import Chunk
from lexis.ingestion.chunker import SemanticChunker
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.ingestion.parser import LexisParser
from lexis.retrieval.hybrid_retriever import RetrievalEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_cuad_depth_sensitivity(
    cuad_path: str,
    num_contracts: int,
    max_questions: int,
    depths: List[int],
    output_path: str,
):
    depths = sorted(set(depths))
    max_depth = depths[-1]

    logger.info(f"Loading CUAD dataset from {cuad_path}")
    raw = CUADLoader().load(cuad_path)
    contracts = select_contracts(raw, num_contracts)
    logger.info(f"Selected {len(contracts)} contracts (deterministic, sorted by title).")

    parser = LexisParser()
    embedder = BGEM3Embedder()
    chunker = SemanticChunker(embedder=embedder)

    chunks_by_doc_id: Dict[str, List[Chunk]] = {}
    for contract in contracts:
        title = contract["title"]
        doc_id = deterministic_document_id(title, prefix="cuad")
        context = contract["paragraphs"][0]["context"]
        logger.info(f"Chunking contract '{title}' (doc_id={doc_id})...")
        chunks = chunk_document_text(parser, chunker, context, doc_id)
        chunks_by_doc_id[doc_id] = chunks
        logger.info(f"  -> {len(chunks)} chunks")

    cases, unmapped = CUADAdapter().build_cases(raw, chunks_by_doc_id, contracts, max_questions)
    logger.info(f"Built {len(cases)} scoreable benchmark cases; {len(unmapped)} unmapped.")

    if not cases:
        logger.error("No scoreable benchmark cases were produced. Cannot run the depth-sensitivity experiment.")
        return None

    retriever = RetrievalEngine()

    # settings snapshot taken before any retrieval calls, and compared again
    # at the end, as a live guard that this experiment never mutates
    # production configuration (see also the static source-inspection test).
    settings_snapshot_before = (
        settings.retrieval_top_k_per_path, settings.retrieval_fusion_top_k, settings.rrf_k,
        settings.qdrant_collection_primary, settings.embedding_model,
    )

    # traces_by_depth[depth][case_id] = RetrievalTrace
    traces_by_depth: Dict[int, Dict[str, object]] = {d: {} for d in depths}

    for case in cases:
        if not case.has_chunk_level_ground_truth():
            continue
        for depth in depths:
            trace = await retriever.retrieve_with_trace(case.query, top_k_per_path=depth, top_n_rrf=depth)
            traces_by_depth[depth][case.case_id] = trace

    scored_cases = [c for c in cases if c.has_chunk_level_ground_truth()]

    # Chunk profiles use the largest depth's trace for per-path rank data
    # (depth-invariant ranking -- see module docstring), and index-integrity
    # from the already-built chunk map.
    chunk_profiles = []
    for case in scored_cases:
        max_trace = traces_by_depth[max_depth][case.case_id]
        for chunk_id in case.relevant_chunk_ids:
            chunk_profiles.append(build_chunk_profile(case, chunk_id, max_trace, depths, chunks_by_doc_id))

    # RRF hit sets per depth, from that depth's OWN properly-scoped trace.
    # Each set holds (case_id, chunk_id) pairs -- see the comment below.
    rrf_hits_by_depth: Dict[int, set] = {}
    per_case_metrics_by_depth: Dict[int, List[tuple]] = {}
    for depth in depths:
        hit_pairs = set()
        per_case_metrics = []
        for case in scored_cases:
            trace = traces_by_depth[depth][case.case_id]
            retrieved_ids = [c["id"] for c in trace.final_chunks]
            relevant_ids = list(case.relevant_chunk_ids)
            for cid in relevant_ids:
                if cid in retrieved_ids:
                    # keyed by (case_id, chunk_id): the same chunk_id can be
                    # relevant to more than one case, and each case's own
                    # query can retrieve it independently -- a flat
                    # chunk_id set would falsely mark it "hit" for every
                    # case referencing it as soon as any one of them found it.
                    hit_pairs.add((case.case_id, cid))
            per_case_metrics.append((
                recall_at_k(relevant_ids, retrieved_ids, k=depth),
                reciprocal_rank(relevant_ids, retrieved_ids),
            ))
        rrf_hits_by_depth[depth] = hit_pairs
        per_case_metrics_by_depth[depth] = per_case_metrics

    set_rrf_hit_depths(chunk_profiles, rrf_hits_by_depth)

    coverages = [
        compute_depth_coverage(depth, chunk_profiles, rrf_hits_by_depth[depth], per_case_metrics_by_depth[depth])
        for depth in depths
    ]
    rank_distribution = build_rank_distribution(chunk_profiles, depths)

    settings_snapshot_after = (
        settings.retrieval_top_k_per_path, settings.retrieval_fusion_top_k, settings.rrf_k,
        settings.qdrant_collection_primary, settings.embedding_model,
    )
    if settings_snapshot_before != settings_snapshot_after:
        raise RuntimeError(
            "Production Settings changed during the depth-sensitivity experiment -- "
            "this must never happen; refusing to write a report."
        )

    summary_table = format_depth_table(coverages)
    logger.info("\n=== Retrieval Depth Sensitivity ===\n" + summary_table)

    run_config = {
        "num_contracts": num_contracts,
        "max_questions": max_questions,
        "depths": depths,
        "embedding_model": settings.embedding_model,
        "rrf_k": settings.rrf_k,
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "cuad",
        "cuad_source_path": cuad_path,
        "run_config": run_config,
        "num_cases": len(scored_cases),
        "num_relevant_chunks": len(chunk_profiles),
        "definitions": DEFINITIONS,
        "per_depth_coverage": [
            {
                "depth": c.depth,
                "num_relevant_chunks": c.num_relevant_chunks,
                "dense_coverage": c.dense_coverage,
                "bm25_coverage": c.bm25_coverage,
                "either_coverage": c.either_coverage,
                "both_coverage": c.both_coverage,
                "neither_coverage": c.neither_coverage,
                "rrf_coverage": c.rrf_coverage,
                "num_cases": c.num_cases,
                "mean_recall_at_depth": c.mean_recall_at_depth,
                "mean_reciprocal_rank_at_depth": c.mean_reciprocal_rank_at_depth,
            }
            for c in coverages
        ],
        "rank_distribution": rank_distribution,
        "summary_table_markdown": summary_table,
        "chunk_profiles": [
            {
                "case_id": p.case_id,
                "chunk_id": p.chunk_id,
                "doc_id": p.doc_id,
                "dense_rank_at_max_depth": p.dense_rank_at_max_depth,
                "bm25_rank_at_max_depth": p.bm25_rank_at_max_depth,
                "first_depth_dense_hit": p.first_depth_dense_hit,
                "first_depth_bm25_hit": p.first_depth_bm25_hit,
                "first_depth_rrf_hit": p.first_depth_rrf_hit,
                "index_integrity": p.index_integrity.value,
            }
            for p in chunk_profiles
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Depth-sensitivity report written to {output_path}")
    return output


def main():
    arg_parser = argparse.ArgumentParser(description="Measure retrieval coverage/Recall/MRR across multiple retrieval depths (diagnostic only).")
    arg_parser.add_argument("--benchmark", choices=["cuad"], default="cuad")
    arg_parser.add_argument("--cuad-path", default="data/cuad_raw/CUAD_v1/CUAD_v1.json")
    arg_parser.add_argument("--num-contracts", type=int, default=10)
    arg_parser.add_argument("--max-questions", type=int, default=50)
    arg_parser.add_argument(
        "--depths", default="10,30,60,100,200",
        help="Comma-separated diagnostic retrieval depths (checkpoints only, not production settings).",
    )
    arg_parser.add_argument("--output", default="evaluation/reports/cuad_depth_sensitivity.json")
    args = arg_parser.parse_args()

    depths = [int(d.strip()) for d in args.depths.split(",") if d.strip()]

    if args.benchmark == "cuad":
        asyncio.run(run_cuad_depth_sensitivity(args.cuad_path, args.num_contracts, args.max_questions, depths, args.output))
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
