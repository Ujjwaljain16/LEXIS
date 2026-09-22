"""
Load the frozen baseline's scope (contracts, cases, chunks) once, for
diagnostic runners. Scope and reference artifact come from
config/defaults.yaml (frozen_baseline), never from literals here.
"""
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from lexis.evaluation.dataset.cuad_loader import CUADAdapter, CUADLoader, select_contracts
from lexis.evaluation.dataset.mapping import deterministic_document_id
from lexis.evaluation.run_eval import chunk_document_text
from lexis.evaluation.types import BenchmarkCase
from lexis.indexing.schema import Chunk
from lexis.ingestion.chunker import SemanticChunker
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.ingestion.parser import LexisParser


def load_frozen_scope(frozen_cfg: Mapping[str, Any], cuad_path: str
                      ) -> Tuple[List[BenchmarkCase], Dict[str, List[Chunk]], Dict[str, Chunk]]:
    raw = CUADLoader().load(cuad_path)
    contracts = select_contracts(raw, frozen_cfg["contracts"])
    parser, embedder = LexisParser(), BGEM3Embedder()
    chunker = SemanticChunker(embedder=embedder)

    chunks_by_doc: Dict[str, List[Chunk]] = {}
    chunk_by_id: Dict[str, Chunk] = {}
    for contract in contracts:
        doc_id = deterministic_document_id(contract["title"], prefix="cuad")
        chunks = chunk_document_text(parser, chunker, contract["paragraphs"][0]["context"], doc_id)
        chunks_by_doc[doc_id] = chunks
        chunk_by_id.update({c.chunk_id: c for c in chunks})

    cases, _ = CUADAdapter().build_cases(raw, chunks_by_doc, frozen_cfg["contracts"], frozen_cfg["questions"])
    return [c for c in cases if c.has_chunk_level_ground_truth()], chunks_by_doc, chunk_by_id
