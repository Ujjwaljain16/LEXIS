"""evaluation/answer_eval.py: case construction, per-case recording from the real event stream, and the
hand-checkable arithmetic of every reported metric."""
import asyncio

import pytest

from lexis.evaluation.answer_eval import (
    AnswerCase, AnswerRecord, answer_one, build_answer_cases, run_answer_eval, summarize,
)
from lexis.evaluation.dataset.mapping import deterministic_document_id
from lexis.indexing.schema import Chunk, ChunkMetadata
from lexis.serving.service import AnswerEvent

META = ChunkMetadata(source_file="f.txt", page_num=1, document_type="contract")


# ---- case construction ----------------------------------------------------------------------------------

def contract(title, qas):
    return {"title": title, "paragraphs": [{"context": "irrelevant here", "qas": qas}]}


def qa(cat, title, impossible=False, answer=None):
    return {"id": f"{title}__{cat}", "question": f"Highlight {cat}", "is_impossible": impossible,
            "answers": [] if impossible else [{"text": answer, "answer_start": 0}]}


def test_build_cases_maps_gold_chunks_and_samples_unanswerable_deterministically():
    title = "ACME_Agreement"
    doc_id = deterministic_document_id(title, prefix="cuad")
    chunks = [Chunk.create(doc_id, 0, "This Agreement is governed by Delaware law.", META),
              Chunk.create(doc_id, 1, "Payment is due in thirty days.", META)]
    qas = [qa("Governing Law", title, answer="governed by Delaware law"),
           qa("Exclusivity", title, impossible=True), qa("Non-Compete", title, impossible=True),
           qa("Cap On Liability", title, answer="text found in no chunk")]
    raw = {"data": [contract(title, qas)]}

    cases, unmapped = build_answer_cases(raw, {doc_id: chunks}, raw["data"], n_answerable=None, n_unanswerable=1)

    answerable = [c for c in cases if c.answerable]
    unanswerable = [c for c in cases if not c.answerable]
    assert [c.category for c in answerable] == ["Governing Law"]
    assert answerable[0].gold_chunk_ids == frozenset({chunks[0].chunk_id})
    assert answerable[0].doc_id == doc_id
    assert len(unanswerable) == 1 and unanswerable[0].gold_chunk_ids == frozenset()
    assert [u["case_id"] for u in unmapped] == [f"{title}__Cap On Liability"]   # reported, not dropped or scored

    again, _ = build_answer_cases(raw, {doc_id: chunks}, raw["data"], None, 1)
    assert [c.case_id for c in again] == [c.case_id for c in cases]               # deterministic


# ---- recording from the event stream ---------------------------------------------------------------------

class ScriptedService:
    def __init__(self, events):
        self.events, self.calls = events, []

    async def stream(self, query, document_ids):
        self.calls.append((query, document_ids))
        for e in self.events:
            yield e


def ev(t, **data):
    return AnswerEvent(t, data)


def run(coro):
    return asyncio.run(coro)


CASE = AnswerCase("T__Governing Law", "q?", "doc-1", True, frozenset({"g1"}))


def test_answered_case_is_recorded_with_context_citations_and_scoped_retrieval():
    svc = ScriptedService([
        ev("status", stage="RETRIEVAL"), ev("context", chunk_ids=["g1", "x2"]), ev("token", text="Delaware"),
        ev("answer", text="Delaware [1].", invalid_citation_indices=[9]),
        ev("citations", citations=[{"chunk_id": "g1"}]),
        ev("verification", support_rate=1.0, claims=[{"status": "supported"}, {"status": "unsupported"}]),
        ev("completed"),
    ])
    clock = iter([0.0, 0.4, 1.0]).__next__          # start, first token, end
    rec = run(answer_one(svc, CASE, clock=clock))
    assert svc.calls == [("q?", ["doc-1"])]         # always document-scoped
    assert (rec.outcome, rec.source_chunk_ids, rec.cited_chunk_ids) == ("answered", ["g1", "x2"], ["g1"])
    assert rec.n_invalid_citations == 1 and rec.n_claims == 2 and rec.n_claims_supported == 1
    assert rec.ttft_s == pytest.approx(0.4) and rec.total_s == pytest.approx(1.0)
    assert rec.gold_chunk_ids == ["g1"] and rec.category == "Governing Law"


def test_abstention_and_failure_are_distinct_outcomes():
    clock = iter([0.0, 1.0]).__next__
    rec = run(answer_one(ScriptedService([ev("abstained", reason="model_declined"), ev("completed")]), CASE, clock=clock))
    assert (rec.outcome, rec.abstain_reason) == ("abstained", "model_declined")

    clock = iter([0.0, 1.0]).__next__
    rec = run(answer_one(ScriptedService([ev("failed", error="RateLimitError")]), CASE, clock=clock))
    assert (rec.outcome, rec.error) == ("failed", "RateLimitError")


def test_empty_stream_is_a_failure_not_a_silent_abstention():
    clock = iter([0.0, 1.0]).__next__
    rec = run(answer_one(ScriptedService([]), CASE, clock=clock))
    assert rec.outcome == "failed" and rec.error


def test_runner_acquires_a_rate_limit_slot_per_question_and_reports_progress():
    svc = ScriptedService([ev("abstained", reason="no_context"), ev("completed")])
    slots, seen = [], []

    async def acquire():
        slots.append(1)

    cases = [AnswerCase(f"c{i}", "q", "d", False) for i in range(3)]
    records = run(run_answer_eval(svc, cases, acquire, on_record=seen.append))
    assert len(records) == 3 == len(slots) == len(seen)


# ---- metric arithmetic -----------------------------------------------------------------------------------

def rec(case_id, answerable, outcome, src=(), cited=(), gold=(), invalid=0, claims=(0, 0), ttft=None, total=None):
    return AnswerRecord(case_id, case_id.split("__")[-1], answerable, outcome, source_chunk_ids=list(src),
                        cited_chunk_ids=list(cited), gold_chunk_ids=list(gold), n_invalid_citations=invalid,
                        n_claims=claims[0], n_claims_supported=claims[1], ttft_s=ttft, total_s=total)


RECORDS = [
    # answerable: gold in context, answered, cites gold only
    rec("a__X", True, "answered", src=["g1", "n1"], cited=["g1"], gold=["g1"], claims=(2, 2), ttft=0.5, total=2.0),
    # answerable: gold in context, answered, cites one gold + one non-gold, with an invalid marker
    rec("b__X", True, "answered", src=["g2", "n2"], cited=["g2", "n2"], gold=["g2"], invalid=1, claims=(2, 1), ttft=1.0, total=3.0),
    # answerable: gold in context but declined (false abstention)
    rec("c__Y", True, "abstained", src=["g3"], gold=["g3"], total=1.0),
    # answerable: gold NOT in context, answered with an uncited answer
    rec("d__Y", True, "answered", src=["n4"], cited=[], gold=["g4"], ttft=0.5, total=1.5),
    # unanswerable: abstained (correct) / answered (wrong)
    rec("e__Z", False, "abstained", total=1.0),
    rec("f__Z", False, "answered", src=["n6"], cited=["n6"], ttft=0.5, total=1.5),
    # provider failure: must be counted, excluded from every rate
    rec("g__Z", True, "failed", gold=["g7"]),
]


def test_summary_arithmetic():
    s = summarize(RECORDS, n_resamples=200)
    assert s["counts"] == {"total": 7, "failed": 1, "answerable": 4, "unanswerable": 2, "answered": 3}
    assert s["failure_rate"]["value"] == pytest.approx(1 / 7)

    a = s["answerable"]
    assert a["answer_rate"]["value"] == pytest.approx(3 / 4)
    assert a["gold_in_context_rate"]["value"] == pytest.approx(3 / 4)
    assert a["answer_rate_given_gold_in_context"]["value"] == pytest.approx(2 / 3)       # a, b answered; c declined
    assert a["citation_hit_rate"]["value"] == pytest.approx(2 / 3)                       # a, b cite gold; d cites nothing
    assert a["citation_hit_rate_given_gold_in_context"]["value"] == pytest.approx(1.0)
    assert a["citation_precision"]["value"] == pytest.approx((1.0 + 0.5) / 2)            # over answers that cite something
    assert a["uncited_answer_rate"]["value"] == pytest.approx(1 / 3)
    assert a["answers_with_invalid_citation_rate"]["value"] == pytest.approx(1 / 3)

    assert s["unanswerable"]["correct_abstention_rate"]["value"] == pytest.approx(1 / 2)
    assert s["abstention_balanced_accuracy"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert s["nli_judged_support"] == {"claim_support_rate": 3 / 4, "answers_verified": 2}


def test_latency_percentiles_ignore_failures():
    s = summarize(RECORDS, n_resamples=200)["latency_s"]
    assert s["ttft_p50"] == pytest.approx(0.5)          # ttft values: 0.5, 1.0, 0.5, 0.5
    assert s["total_p50"] == pytest.approx(1.5)         # totals: 2, 3, 1, 1.5, 1, 1.5
    assert s["total_p95"] <= 3.0 and s["total_p95"] > s["total_p50"]


def test_intervals_are_bootstrap_cis_that_contain_the_estimate():
    r = summarize(RECORDS, n_resamples=500)["answerable"]["answer_rate"]
    assert r["lo"] <= r["value"] <= r["hi"] and r["n"] == 4


def test_summary_of_nothing_does_not_crash():
    s = summarize([])
    assert s["answerable"]["answer_rate"]["value"] is None and s["abstention_balanced_accuracy"] is None
