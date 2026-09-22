"""
CLI entry point for running a retrieval benchmark against the live LEXIS
retrieval stack (lexis.retrieval.hybrid_retriever.RetrievalEngine).

This is the only place benchmark selection happens. It composes:
    benchmark adapter (e.g. CUADLoader/CUADAdapter)
        -> real LEXIS parsing/chunking (to compute chunk-level ground truth
           against LEXIS's own actual chunks, not an assumption about them)
        -> real ingestion (so the evaluated corpus is genuinely indexed the
           same way production data is)
        -> the generic EvalHarness against the real RetrievalEngine

No CUAD-specific knowledge exists outside the `--benchmark cuad` branch;
adding a second benchmark means adding another adapter and another branch
here, not touching harness.py or metrics.py.

Supports two evaluation protocols via --protocol (see harness.py): 'pooled'
(default -- the frozen continuity baseline, retrieval searches the whole
corpus) and 'doc_scoped' (retrieval restricted to each case's own
relevant_doc_ids -- the intended within-document retrieval task; a P0
experiment showed CUAD queries do not reliably identify their own document,
so this uses the ground-truth document as a known task-context scope, not a
system-chosen one). Report both, labelled, side by side -- never average or
compare them as if one were an improvement on the other.

Example:
    python -m lexis.evaluation.run_eval --benchmark cuad \\
        --cuad-path data/cuad_raw/CUAD_v1/CUAD_v1.json \\
        --num-contracts 10 --max-questions 50 --top-k 30 --protocol pooled

All dataset/subset/output parameters are CLI flags with defaults matching
the plan's Week 1-2 gate (10 contracts, 50 questions, Recall@30) -- none of
these numbers are embedded inside the harness or metrics code.
"""
import argparse
import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from lexis.config import settings
from lexis.evaluation.dataset.cuad_loader import CUADAdapter, CUADLoader, select_contracts
from lexis.evaluation.dataset.mapping import deterministic_document_id
from lexis.evaluation.harness import EvalHarness
from lexis.evaluation.scoped_retrieval import retrieve_scoped
from lexis.indexing.schema import Chunk
from lexis.ingestion.chunker import SemanticChunker
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.ingestion.parser import LexisParser
from lexis.ingestion.pipeline import IngestionPipeline
from lexis.retrieval.hybrid_retriever import RetrievalEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_document_text(parser: LexisParser, chunker: SemanticChunker, text: str, doc_id: str) -> List[Chunk]:
    """Runs LEXIS's real parser+chunker over a piece of already-extracted
    document text, the same way IngestionPipeline would for a .txt source
    file. Used so benchmark ground truth is matched against LEXIS's actual
    chunk boundaries instead of an assumption about them."""
    fd, tmp_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        elements = parser.parse(tmp_path, doc_id)
        return chunker.chunk(elements)
    finally:
        os.unlink(tmp_path)


async def run_cuad_benchmark(
    cuad_path: str,
    num_contracts: int,
    max_questions: int,
    top_k: int,
    output_path: str,
    skip_ingest: bool,
    diagnostics_output: str = None,
    skip_diagnostics: bool = False,
    protocol: str = "pooled",
):
    logger.info(f"Loading CUAD dataset from {cuad_path}")
    raw = CUADLoader().load(cuad_path)
    contracts = select_contracts(raw, num_contracts)
    logger.info(f"Selected {len(contracts)} contracts (deterministic, sorted by title).")

    parser = LexisParser()
    embedder = BGEM3Embedder()
    chunker = SemanticChunker(embedder=embedder)
    pipeline = IngestionPipeline(parser=parser, embedder=embedder, chunker=chunker)

    chunks_by_doc_id: Dict[str, List[Chunk]] = {}
    for contract in contracts:
        title = contract["title"]
        doc_id = deterministic_document_id(title, prefix="cuad")
        context = contract["paragraphs"][0]["context"]

        logger.info(f"Chunking contract '{title}' (doc_id={doc_id})...")
        chunks = chunk_document_text(parser, chunker, context, doc_id)
        chunks_by_doc_id[doc_id] = chunks
        logger.info(f"  -> {len(chunks)} chunks")

        if not skip_ingest:
            logger.info(f"  Ingesting into Qdrant/Elasticsearch/Postgres...")
            await pipeline._upsert_to_databases(chunks)

    cases, unmapped = CUADAdapter().build_cases(raw, chunks_by_doc_id, num_contracts, max_questions)
    logger.info(f"Built {len(cases)} scoreable benchmark cases; {len(unmapped)} could not be mapped to any produced chunk.")
    for u in unmapped:
        logger.warning(f"UNMAPPED ({u['reason']}): {u['case_id']}")

    if not cases:
        logger.error("No scoreable benchmark cases were produced. Cannot run a retrieval evaluation.")
        return None

    retriever = RetrievalEngine()

    async def retrieve_fn(query: str, k: int) -> List[str]:
        results = await retriever.retrieve(query, top_k_per_path=k, top_n_rrf=k)
        return [r["id"] for r in results]

    async def scoped_retrieve_fn(query: str, k: int, doc_ids) -> List[str]:
        trace = await retrieve_scoped(retriever, query, list(doc_ids), k, k)
        return [c["id"] for c in trace.final_chunks]

    harness = EvalHarness(
        retrieve_fn=retrieve_fn if protocol == "pooled" else None,
        scoped_retrieve_fn=scoped_retrieve_fn if protocol == "doc_scoped" else None,
        top_k=top_k, protocol=protocol,
    )
    report = await harness.run(cases, benchmark_name="cuad")

    logger.info(f"========== CUAD RETRIEVAL EVALUATION (protocol={protocol}) ==========")
    logger.info(f"Cases scored:    {report.num_cases_scored}")
    logger.info(f"Cases excluded (no ground truth / no document scope): {report.num_cases_excluded}")
    logger.info(f"Cases unmapped (adapter could not locate answer in any chunk): {len(unmapped)}")
    logger.info(f"Recall@{top_k}: {report.mean_recall_at_k:.4f}")
    logger.info(f"MRR:       {report.mean_reciprocal_rank:.4f}")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "cuad",
        "cuad_source_path": cuad_path,
        "protocol": protocol,
        "run_config": {
            "num_contracts": num_contracts,
            "max_questions": max_questions,
            "top_k": top_k,
            "protocol": protocol,
            "embedding_model": settings.embedding_model,
            "rrf_k": settings.rrf_k,
        },
        "num_cases_scored": report.num_cases_scored,
        "num_cases_excluded_no_ground_truth": report.num_cases_excluded,
        "num_cases_unmapped": len(unmapped),
        "unmapped_cases": unmapped,
        "mean_recall_at_k": report.mean_recall_at_k,
        "mean_reciprocal_rank": report.mean_reciprocal_rank,
        "per_case": [
            {
                "case_id": c.case_id,
                "recall_at_k": c.recall_at_k,
                "reciprocal_rank": c.reciprocal_rank,
                "retrieved_count": c.retrieved_count,
                "relevant_count": c.relevant_count,
            }
            for c in report.per_case
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Report written to {output_path}")

    # --- Retrieval error-analysis diagnostics (Step 8) ---
    # A deliberately separate pass, using RetrievalEngine.retrieve_with_trace
    # (not retrieve()) and evaluation/metrics.py directly. Everything above
    # this point -- including report.mean_recall_at_k/mean_reciprocal_rank
    # and the file just written -- is already final and untouched by what
    # follows.
    if protocol == "doc_scoped" and not skip_diagnostics:
        logger.info("Skipping retrieval error-analysis diagnostics: diagnostics.py's ChunkOutcome taxonomy "
                    "(dense_rank/bm25_rank against the pooled candidate lists) is defined for the pooled "
                    "protocol only; running it against doc_scoped ranks would silently mix the two protocols.")
        skip_diagnostics = True

    if not skip_diagnostics:
        from lexis.evaluation.diagnostics import build_diagnostic_report, diagnose_case, format_summary
        from lexis.evaluation.metrics import reciprocal_rank, recall_at_k

        case_diagnostics = []
        for case in cases:
            if not case.has_chunk_level_ground_truth():
                continue
            trace = await retriever.retrieve_with_trace(case.query, top_k_per_path=top_k, top_n_rrf=top_k)
            retrieved_ids = [c["id"] for c in trace.final_chunks]
            relevant_ids = list(case.relevant_chunk_ids)
            case_recall = recall_at_k(relevant_ids, retrieved_ids, k=top_k)
            case_rr = reciprocal_rank(relevant_ids, retrieved_ids)
            case_diagnostics.append(diagnose_case(case, trace, case_recall, case_rr, top_n_rrf=top_k))

        diag_report = build_diagnostic_report(case_diagnostics, benchmark="cuad", run_config=output["run_config"])
        logger.info("\n" + format_summary(diag_report))

        diag_path = diagnostics_output or str(Path(output_path).with_name(Path(output_path).stem + "_diagnostics.json"))
        diag_output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": diag_report.benchmark,
            "run_config": diag_report.run_config,
            "num_cases": diag_report.num_cases,
            "aggregate_chunk_outcomes": diag_report.aggregate_chunk_outcomes,
            "aggregate_document_outcomes": diag_report.aggregate_document_outcomes,
            "category_definitions": diag_report.category_definitions,
            "per_case": [
                {
                    "case_id": cd.case_id,
                    "query": cd.query,
                    "relevant_chunk_ids": cd.relevant_chunk_ids,
                    "recall_at_k": cd.recall_at_k,
                    "reciprocal_rank": cd.reciprocal_rank,
                    "final_topk_doc_ids": cd.final_topk_doc_ids,
                    "document_outcome": cd.document_outcome.value if cd.document_outcome else None,
                    "chunk_diagnostics": [
                        {
                            "chunk_id": chd.chunk_id,
                            "dense_rank": chd.dense_rank,
                            "bm25_rank": chd.bm25_rank,
                            "rrf_rank": chd.rrf_rank,
                            "in_final_topk": chd.in_final_topk,
                            "outcome": chd.outcome.value,
                        }
                        for chd in cd.chunk_diagnostics
                    ],
                }
                for cd in diag_report.per_case
            ],
        }

        Path(diag_path).parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag_output, f, indent=2)
        logger.info(f"Diagnostic report written to {diag_path}")

    return report


def main():
    arg_parser = argparse.ArgumentParser(description="Run a retrieval benchmark against the live LEXIS retrieval stack.")
    arg_parser.add_argument("--benchmark", choices=["cuad"], default="cuad")
    arg_parser.add_argument(
        "--cuad-path", default="data/cuad_raw/CUAD_v1/CUAD_v1.json",
        help="Path to the real CUAD_v1.json (theatticusproject/cuad on the Hugging Face Hub).",
    )
    arg_parser.add_argument("--num-contracts", type=int, default=10, help="Number of contracts to include (deterministic subset).")
    arg_parser.add_argument("--max-questions", type=int, default=50, help="Maximum total benchmark questions.")
    arg_parser.add_argument("--top-k", type=int, default=30, help="Retrieval depth for Recall@k.")
    arg_parser.add_argument(
        "--protocol", choices=["pooled", "doc_scoped"], default="pooled",
        help="'pooled' (default, frozen continuity baseline): search the whole corpus. "
             "'doc_scoped': restrict retrieval to each case's own relevant_doc_ids "
             "(the intended within-document retrieval task).",
    )
    arg_parser.add_argument("--output", default="evaluation/reports/cuad_report.json")
    arg_parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip writing chunks to Qdrant/Elasticsearch/Postgres; assumes the corpus is already indexed.",
    )
    arg_parser.add_argument(
        "--diagnostics-output", default=None,
        help="Path for the retrieval error-analysis diagnostic report (default: <output>_diagnostics.json).",
    )
    arg_parser.add_argument(
        "--skip-diagnostics", action="store_true",
        help="Skip the retrieval error-analysis pass; only compute the standard Recall@k/MRR report.",
    )
    args = arg_parser.parse_args()

    if args.benchmark == "cuad":
        asyncio.run(run_cuad_benchmark(
            args.cuad_path, args.num_contracts, args.max_questions, args.top_k, args.output, args.skip_ingest,
            diagnostics_output=args.diagnostics_output, skip_diagnostics=args.skip_diagnostics,
            protocol=args.protocol,
        ))
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
