"""
Generic evaluation harness for LEXIS.

Executes a list of BenchmarkCase objects (see lexis.evaluation.types)
against any retrieval function matching the RetrieveFn/ScopedRetrieveFn
signature below, and computes deterministic retrieval metrics (Recall@k,
MRR) via lexis.evaluation.metrics. This module contains no knowledge of any
particular benchmark (CUAD or otherwise) -- that lives entirely in each
benchmark's adapter (e.g. lexis.evaluation.dataset.cuad_loader).

Two protocols, both first-class:

  pooled      retrieval searches the WHOLE indexed corpus. This is the
              frozen continuity baseline: it measures retrieval when the
              system must find the right document as well as the right
              chunk within it.

  doc_scoped  retrieval is restricted to each case's OWN
              case.relevant_doc_ids -- the document(s) the task/benchmark
              context already identifies as relevant (e.g. "the user has
              this contract open"), never a document chosen by the
              retrieval system itself. This isolates within-document
              chunk-ranking quality from corpus-wide document selection,
              which a P0 experiment showed cannot reliably be inferred from
              CUAD query text alone (see evaluation/doc_routing.py and
              evaluation/run_doc_scoping.py). A case with chunk-level
              ground truth but no relevant_doc_ids has no known scope under
              this protocol and is excluded, not silently scored.

pooled and doc_scoped answer different questions over the same cases and
must never be averaged together or compared as if one were an improvement
on the other -- report them side by side, both labelled.

This step intentionally implements only deterministic retrieval metrics.
RAGAS, LLM-as-judge, NLI/faithfulness, and citation-quality scoring are out
of scope here and must not be added to this module.
"""
import logging
from typing import Awaitable, Callable, FrozenSet, List, Literal, Optional

from lexis.evaluation.metrics import reciprocal_rank, recall_at_k
from lexis.evaluation.types import BenchmarkCase, CaseResult, EvaluationReport

logger = logging.getLogger(__name__)

Protocol = Literal["pooled", "doc_scoped"]

# pooled: given a query string and a top_k depth, return the ranked list of
# retrieved identifiers (chunk ids), searching the whole corpus.
RetrieveFn = Callable[[str, int], Awaitable[List[str]]]
# doc_scoped: as above, but additionally given the set of document ids
# retrieval must be restricted to.
ScopedRetrieveFn = Callable[[str, int, FrozenSet[str]], Awaitable[List[str]]]


class EvalHarness:
    """Main evaluation runner for LEXIS: executes benchmark cases against a
    retrieval function and computes generic retrieval metrics, under either
    the pooled or doc_scoped protocol."""

    def __init__(
        self,
        retrieve_fn: Optional[RetrieveFn] = None,
        scoped_retrieve_fn: Optional[ScopedRetrieveFn] = None,
        top_k: int = 30,
        protocol: Protocol = "pooled",
    ):
        if protocol not in ("pooled", "doc_scoped"):
            raise ValueError(f"unknown protocol {protocol!r}; must be 'pooled' or 'doc_scoped'")
        if protocol == "pooled" and retrieve_fn is None:
            raise ValueError("protocol='pooled' requires retrieve_fn")
        if protocol == "doc_scoped" and scoped_retrieve_fn is None:
            raise ValueError("protocol='doc_scoped' requires scoped_retrieve_fn")
        self.retrieve_fn = retrieve_fn
        self.scoped_retrieve_fn = scoped_retrieve_fn
        self.top_k = top_k
        self.protocol = protocol

    async def run(self, cases: List[BenchmarkCase], benchmark_name: str = "unknown") -> EvaluationReport:
        """
        For each case with chunk-level ground truth (and, under doc_scoped,
        a known document scope), retrieves the top_k results and scores
        Recall@top_k and MRR against case.relevant_chunk_ids. Cases that
        cannot be scored are excluded (never scored as 0, never silently
        dropped -- their case_ids are reported in excluded_case_ids).
        """
        scored: List[CaseResult] = []
        excluded_case_ids: List[str] = []

        for case in cases:
            if not case.has_chunk_level_ground_truth():
                excluded_case_ids.append(case.case_id)
                logger.warning(f"Excluding case {case.case_id!r} from scoring: no chunk-level ground truth.")
                continue
            if self.protocol == "doc_scoped" and not case.relevant_doc_ids:
                excluded_case_ids.append(case.case_id)
                logger.warning(f"Excluding case {case.case_id!r} from doc_scoped scoring: no known document scope.")
                continue

            if self.protocol == "doc_scoped":
                retrieved_ids = await self.scoped_retrieve_fn(case.query, self.top_k, case.relevant_doc_ids)
            else:
                retrieved_ids = await self.retrieve_fn(case.query, self.top_k)
            relevant_ids = list(case.relevant_chunk_ids)

            case_recall = recall_at_k(relevant_ids, retrieved_ids, k=self.top_k)
            case_rr = reciprocal_rank(relevant_ids, retrieved_ids)

            scored.append(CaseResult(
                case_id=case.case_id,
                recall_at_k=case_recall,
                reciprocal_rank=case_rr,
                retrieved_count=len(retrieved_ids),
                relevant_count=len(relevant_ids),
            ))

        mean_recall = sum(c.recall_at_k for c in scored) / len(scored) if scored else 0.0
        mean_rr = sum(c.reciprocal_rank for c in scored) / len(scored) if scored else 0.0

        return EvaluationReport(
            benchmark=benchmark_name,
            top_k=self.top_k,
            num_cases_scored=len(scored),
            num_cases_excluded=len(excluded_case_ids),
            mean_recall_at_k=mean_recall,
            mean_reciprocal_rank=mean_rr,
            per_case=scored,
            excluded_case_ids=excluded_case_ids,
            protocol=self.protocol,
        )
