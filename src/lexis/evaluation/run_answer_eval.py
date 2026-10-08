"""
CLI: end-to-end answer-quality evaluation on a CUAD split (see evaluation/answer_eval.py for definitions).

    python -m lexis.evaluation.run_answer_eval --cuad-path CUAD_v1.json \
        --split-manifest evaluation/splits/cuad_split_v1.json --split-name dev --max-contracts 60 \
        --n-answerable 150 --n-unanswerable 150 --output answer_eval_dev.json

Requires the split's contracts to already be indexed (same collection / BM25 dir as a retrieval run -- see
run_eval.py's --ingest-checkpoint) and a working LLM key. Every question is answered through the same
AnswerService the API serves, document-scoped, so this measures the shipped path, not a re-implementation.
Develop and tune prompts on dev; report the held-out test split once, with the prompt frozen.
"""
import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from lexis.config import settings
from lexis.evaluation.answer_eval import build_answer_cases, records_to_json, run_answer_eval, summarize
from lexis.evaluation.dataset.cuad_loader import CUADLoader
from lexis.evaluation.dataset.cuad_split import cap_split_contracts, contracts_for_split
from lexis.evaluation.dataset.mapping import deterministic_document_id
from lexis.evaluation.provenance import build_provenance
from lexis.evaluation.run_eval import chunk_document_text, device_info
from lexis.generation.synthesizer import LexisSynthesizer
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.ingestion.parser import LexisParser
from lexis.ingestion.rate_limiter import AsyncRateLimiter
from lexis.ingestion.chunker import SemanticChunker
from lexis.retrieval.hybrid_retriever import RetrievalEngine
from lexis.serving.service import AnswerService, EngineRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main(args) -> int:
    raw = CUADLoader().load(args.cuad_path)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    contracts = cap_split_contracts(contracts_for_split(raw, manifest, args.split_name), args.max_contracts, args.split_name)
    logger.info("Evaluating answers on %d contracts of split %r", len(contracts), args.split_name)

    parser = LexisParser()
    embedder = BGEM3Embedder()
    chunker = SemanticChunker(embedder=embedder)
    chunks_by_doc_id = {}
    for contract in contracts:
        doc_id = deterministic_document_id(contract["title"], prefix="cuad")
        chunks_by_doc_id[doc_id] = chunk_document_text(parser, chunker, contract["paragraphs"][0]["context"], doc_id)

    cases, unmapped = build_answer_cases(raw, chunks_by_doc_id, contracts, args.n_answerable, args.n_unanswerable, args.seed)
    n_ans = sum(c.answerable for c in cases)
    logger.info("%d answerable + %d unanswerable questions (%d answerable unmapped and excluded)",
                n_ans, len(cases) - n_ans, len(unmapped))

    verifier = None
    if args.verify:
        from lexis.evaluation.nli_checker import NLIChecker
        verifier = NLIChecker()
    service = AnswerService(EngineRetriever(RetrievalEngine()), LexisSynthesizer(), verifier)
    limiter = AsyncRateLimiter(settings.answer_eval_requests_per_minute)

    def progress(rec):
        logger.info("[%s] %-9s %s", rec.case_id[-40:], rec.outcome, rec.error or rec.abstain_reason or "")

    records = await run_answer_eval(service, cases, limiter.acquire, on_record=progress)
    summary = summarize(records, seed=args.seed)

    gateway = urlparse(settings.llm_api_base).netloc if settings.llm_api_base else None
    run_config = {
        "split_manifest": args.split_manifest, "split_name": args.split_name, "max_contracts": args.max_contracts,
        "n_answerable": args.n_answerable, "n_unanswerable": args.n_unanswerable, "sample_seed": args.seed,
        "generator_model": settings.gemini_model_synthesis, "llm_gateway_host": gateway,
        "answer_context_chunks": settings.answer_context_chunks, "temperature": settings.answer_temperature,
        "verify_claims": args.verify, "nli_entailment_threshold": settings.nli_entailment_threshold,
        "rerank_enabled": settings.rerank_enabled, "preamble_prior_chunks": settings.preamble_prior_chunks,
        "embedding_model": settings.embedding_model, "rrf_k": settings.rrf_k,
        "qdrant_collection_primary": settings.qdrant_collection_primary,
        "contract_titles_used": [c["title"] for c in contracts],
    }
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kind": "answer_quality_eval", "benchmark": "cuad", "protocol": "doc_scoped",
        "run_config": run_config,
        "provenance": build_provenance(
            config=run_config, data_paths=[args.cuad_path, args.split_manifest],
            packages=["sentence-transformers", "bm25s", "qdrant-client", "torch", "numpy", "litellm"],
            extra={"device": device_info()}),
        "num_unmapped_answerable_excluded": len(unmapped),
        "summary": summary,
        "per_case": records_to_json(records),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Report written to %s", args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuad-path", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--split-name", choices=["dev", "test"], required=True)
    ap.add_argument("--max-contracts", type=int, default=None, help="dev only; refused for test")
    ap.add_argument("--n-answerable", type=int, default=None, help="sample size (default: all)")
    ap.add_argument("--n-unanswerable", type=int, default=None, help="sample size (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="also run the per-claim NLI check")
    ap.add_argument("--output", required=True)
    raise SystemExit(asyncio.run(main(ap.parse_args())))
