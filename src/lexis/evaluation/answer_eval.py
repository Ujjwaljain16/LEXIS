"""
End-to-end answer-quality evaluation: does the cited-answer path (retrieval -> numbered sources -> LLM ->
validated citations -> abstention) behave correctly?

What is measured is objective and needs no LLM judge:

  Answerable questions (CUAD gold answer spans, mapped to chunks):
    answer_rate                 share the system answered rather than declined
    gold_in_context_rate        share where a gold chunk was actually in the prompt
    citation_hit_rate           among answered: some cited chunk is a gold chunk
    citation_precision          among answered with citations: share of cited chunks that are gold
    uncited_answer_rate         among answered: answers that cite nothing
    answers_with_invalid_citation_rate   among answered: the model cited a source number that did not exist
    *_given_gold_in_context     the same rates restricted to questions where the evidence WAS available
  Unanswerable questions (CUAD marks the clause category as not present in that contract):
    correct_abstention_rate     share the system declined instead of answering
  Combined:
    abstention_balanced_accuracy = mean(correct_abstention_rate, answer_rate_given_gold_in_context)
  Optional (a verifier is configured): claim-level NLI support rate. The NLI model's accuracy on legal
    text is itself unmeasured, so this is reported as "NLI-judged support", never as ground truth.
  Cost/latency: time-to-first-token and total time, P50/P95, measured from the machine that ran it.

What is NOT measured: free-text answer correctness. CUAD answers are extracted clause spans; a generated
paraphrase cannot be graded against them without an LLM judge or human review, and neither is claimed here.

Failures (provider errors, timeouts) are counted and reported, never silently dropped or scored as abstentions.
"""
import hashlib
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from lexis.evaluation.dataset.cuad_loader import CUADAdapter
from lexis.evaluation.dataset.mapping import deterministic_document_id
from lexis.evaluation.stats import bootstrap_ci
from lexis.indexing.schema import Chunk
from lexis.serving.service import AnswerService


@dataclass(frozen=True)
class AnswerCase:
    case_id: str
    query: str
    doc_id: str
    answerable: bool
    gold_chunk_ids: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def category(self) -> str:
        return self.case_id.split("__")[-1]


@dataclass
class AnswerRecord:
    case_id: str
    category: str
    answerable: bool
    outcome: str                                   # "answered" | "abstained" | "failed"
    abstain_reason: Optional[str] = None
    source_chunk_ids: List[str] = field(default_factory=list)
    cited_chunk_ids: List[str] = field(default_factory=list)
    gold_chunk_ids: List[str] = field(default_factory=list)
    n_invalid_citations: int = 0
    answer_text: str = ""
    n_claims: int = 0
    n_claims_supported: int = 0
    ttft_s: Optional[float] = None
    total_s: Optional[float] = None
    error: Optional[str] = None


# ---- case construction ----------------------------------------------------------------------------------

def _rank_key(seed: int, case_id: str) -> str:
    return hashlib.sha256(f"{seed}|{case_id}".encode("utf-8")).hexdigest()


def _sample(cases: List[AnswerCase], n: Optional[int], seed: int) -> List[AnswerCase]:
    ordered = sorted(cases, key=lambda c: _rank_key(seed, c.case_id))
    return ordered if n is None else ordered[:n]


def build_answer_cases(raw: Dict[str, Any], chunks_by_doc_id: Dict[str, List[Chunk]], contracts: List[Dict[str, Any]],
                       n_answerable: Optional[int], n_unanswerable: Optional[int],
                       seed: int = 0) -> Tuple[List[AnswerCase], List[Dict[str, Any]]]:
    """Deterministic, seeded samples of answerable (gold mapped to chunks) and unanswerable questions.
    Returns (cases, unmapped_answerable). Sampling is by hash of the case id, so it is stable under
    reordering and independent of contract order."""
    scored, unmapped = CUADAdapter().build_cases(raw, chunks_by_doc_id, contracts, sys.maxsize)
    answerable = [AnswerCase(c.case_id, c.query, next(iter(c.relevant_doc_ids)), True, c.relevant_chunk_ids)
                  for c in scored]

    unanswerable: List[AnswerCase] = []
    for contract in contracts:
        doc_id = deterministic_document_id(contract["title"], prefix="cuad")
        for qa in contract["paragraphs"][0]["qas"]:
            if qa.get("is_impossible") and not qa.get("answers"):
                unanswerable.append(AnswerCase(qa["id"], qa["question"], doc_id, False))

    return _sample(answerable, n_answerable, seed) + _sample(unanswerable, n_unanswerable, seed), unmapped


# ---- running --------------------------------------------------------------------------------------------

async def answer_one(service: AnswerService, case: AnswerCase,
                     clock: Callable[[], float] = time.perf_counter) -> AnswerRecord:
    rec = AnswerRecord(case.case_id, case.category, case.answerable, outcome="failed",
                       gold_chunk_ids=sorted(case.gold_chunk_ids))
    start = clock()
    async for event in service.stream(case.query, [case.doc_id]):
        if event.type == "context":
            rec.source_chunk_ids = list(event.data["chunk_ids"])
        elif event.type == "token" and rec.ttft_s is None:
            rec.ttft_s = clock() - start
        elif event.type == "answer":
            rec.answer_text = event.data["text"]
            rec.n_invalid_citations = len(event.data["invalid_citation_indices"])
            rec.outcome = "answered"
        elif event.type == "citations":
            rec.cited_chunk_ids = [c["chunk_id"] for c in event.data["citations"]]
        elif event.type == "abstained":
            rec.outcome, rec.abstain_reason = "abstained", event.data["reason"]
        elif event.type == "verification":
            claims = event.data["claims"]
            rec.n_claims = len(claims)
            rec.n_claims_supported = sum(c["status"] == "supported" for c in claims)
        elif event.type == "failed":
            rec.outcome, rec.error = "failed", event.data["error"]
            break
    rec.total_s = clock() - start
    if rec.outcome == "failed" and rec.error is None:
        rec.error = "stream ended without answer, abstention, or failure"
    return rec


async def run_answer_eval(service: AnswerService, cases: Sequence[AnswerCase], acquire: Callable[[], Awaitable[None]],
                          on_record: Optional[Callable[[AnswerRecord], None]] = None) -> List[AnswerRecord]:
    """Sequential on purpose: one question at a time keeps the latency numbers meaningful and the provider
    rate limit simple. `acquire` blocks until a request slot is free (an AsyncRateLimiter.acquire)."""
    records: List[AnswerRecord] = []
    for case in cases:
        await acquire()
        rec = await answer_one(service, case)
        records.append(rec)
        if on_record:
            on_record(rec)
    return records


# ---- summarising ----------------------------------------------------------------------------------------

def _rate(values: Sequence[float], n_resamples: Optional[int], seed: int) -> Dict[str, Any]:
    if not values:
        return {"value": None, "lo": None, "hi": None, "n": 0}
    kwargs = {} if n_resamples is None else {"n_resamples": n_resamples}   # None -> the stats module's default
    ci = bootstrap_ci(list(values), seed=seed, **kwargs)
    return {"value": ci.mean, "lo": ci.lo, "hi": ci.hi, "n": len(values)}


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(records: Sequence[AnswerRecord], n_resamples: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    ok = [r for r in records if r.outcome != "failed"]
    answerable = [r for r in ok if r.answerable]
    unanswerable = [r for r in ok if not r.answerable]
    answered = [r for r in answerable if r.outcome == "answered"]

    def gold_in_context(r: AnswerRecord) -> bool:
        return bool(set(r.source_chunk_ids) & set(r.gold_chunk_ids))

    def cites_gold(r: AnswerRecord) -> bool:
        return bool(set(r.cited_chunk_ids) & set(r.gold_chunk_ids))

    def rate(values):
        return _rate(values, n_resamples, seed)

    with_gold = [r for r in answerable if gold_in_context(r)]
    answered_with_gold = [r for r in with_gold if r.outcome == "answered"]
    cited = [r for r in answered if r.cited_chunk_ids]

    out: Dict[str, Any] = {
        "counts": {"total": len(records), "failed": len(records) - len(ok), "answerable": len(answerable),
                   "unanswerable": len(unanswerable), "answered": len(answered)},
        "failure_rate": rate([float(r.outcome == "failed") for r in records]),
        "answerable": {
            "answer_rate": rate([float(r.outcome == "answered") for r in answerable]),
            "gold_in_context_rate": rate([float(gold_in_context(r)) for r in answerable]),
            "answer_rate_given_gold_in_context": rate([float(r.outcome == "answered") for r in with_gold]),
            "citation_hit_rate": rate([float(cites_gold(r)) for r in answered]),
            "citation_hit_rate_given_gold_in_context": rate([float(cites_gold(r)) for r in answered_with_gold]),
            "citation_precision": rate([len(set(r.cited_chunk_ids) & set(r.gold_chunk_ids)) / len(set(r.cited_chunk_ids))
                                        for r in cited]),
            "uncited_answer_rate": rate([float(not r.cited_chunk_ids) for r in answered]),
            "answers_with_invalid_citation_rate": rate([float(r.n_invalid_citations > 0) for r in answered]),
        },
        "unanswerable": {
            "correct_abstention_rate": rate([float(r.outcome == "abstained") for r in unanswerable]),
        },
    }

    abst = out["unanswerable"]["correct_abstention_rate"]["value"]
    ans = out["answerable"]["answer_rate_given_gold_in_context"]["value"]
    out["abstention_balanced_accuracy"] = None if abst is None or ans is None else (abst + ans) / 2

    verified = [r for r in answered if r.n_claims]
    out["nli_judged_support"] = {
        "claim_support_rate": (sum(r.n_claims_supported for r in verified) / sum(r.n_claims for r in verified))
        if verified else None,
        "answers_verified": len(verified),
    }
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    total = [r.total_s for r in ok if r.total_s is not None]
    out["latency_s"] = {
        "ttft_p50": _percentile(ttft, 0.5), "ttft_p95": _percentile(ttft, 0.95),
        "total_p50": _percentile(total, 0.5), "total_p95": _percentile(total, 0.95),
        "total_mean": statistics.fmean(total) if total else None,
    }
    out["by_category_answer_rate"] = {
        cat: statistics.fmean(float(r.outcome == "answered") for r in answerable if r.category == cat)
        for cat in sorted({r.category for r in answerable})
    }
    return out


def records_to_json(records: Sequence[AnswerRecord]) -> List[Dict[str, Any]]:
    return [asdict(r) for r in records]
