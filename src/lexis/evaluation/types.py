"""
Generic evaluation data contracts for LEXIS.

Rationale: Retrieval benchmarks (CUAD today, others later) must feed the
evaluation harness through one normalized shape so the harness and the
metrics it uses never need to know which benchmark produced a case. All
benchmark-specific parsing (question templates, category names, answer
formats, dataset file layout) stays inside that benchmark's own adapter
module (e.g. evaluation/dataset/cuad_loader.py) and is never represented
here.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet


@dataclass(frozen=True)
class BenchmarkCase:
    """One evaluable query from a benchmark, normalized to what the generic
    harness needs and nothing more.

    relevant_chunk_ids: the ground-truth chunk identifiers a retrieval run
        should surface, if the source benchmark's ground truth could be
        mapped to actual indexed chunks (see the benchmark's adapter for how
        that mapping is derived). Empty if no chunk-level mapping exists.
    relevant_doc_ids: the ground-truth document identifiers, when the
        benchmark's ground truth is only known at document granularity (or
        in addition to chunk granularity).
    source_benchmark: a plain label (e.g. "cuad") for reporting/grouping.
        The harness must never branch on this value -- if it needs to, that
        logic belongs in the adapter, not here.
    metadata: adapter-specific extras (e.g. CUAD's category name, contract
        title, is_impossible flag). Opaque to the harness; carried through
        purely for reporting/debugging.
    """
    case_id: str
    query: str
    relevant_chunk_ids: FrozenSet[str] = field(default_factory=frozenset)
    relevant_doc_ids: FrozenSet[str] = field(default_factory=frozenset)
    source_benchmark: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_chunk_level_ground_truth(self) -> bool:
        return len(self.relevant_chunk_ids) > 0


@dataclass(frozen=True)
class CaseResult:
    """Per-case scoring output. Kept separate from BenchmarkCase so a case
    can be scored more than once (e.g. different top_k) without mutation."""
    case_id: str
    recall_at_k: float
    reciprocal_rank: float
    retrieved_count: int
    relevant_count: int


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregated result of running a set of BenchmarkCases through a
    retrieval function. Contains only generic retrieval metrics -- no
    RAGAS/NLI/faithfulness/LLM-judge fields belong here (out of scope for
    this step).

    protocol: "pooled" (retrieval searches the whole indexed corpus -- the
        frozen continuity baseline) or "doc_scoped" (retrieval is restricted
        to each case's own relevant_doc_ids, i.e. the document(s) the task
        context already identifies as relevant -- the intended
        within-document retrieval task). The two protocols answer different
        questions and must never be averaged or compared without saying
        which is which; see evaluation/harness.py."""
    benchmark: str
    top_k: int
    num_cases_scored: int
    num_cases_excluded: int
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    per_case: list  # List[CaseResult]
    excluded_case_ids: list  # cases with no chunk-level ground truth (or, under doc_scoped, no known document scope)
    protocol: str = "pooled"
