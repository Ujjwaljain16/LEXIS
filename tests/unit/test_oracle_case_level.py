"""
Tests for the case-level oracle analysis, using generic invented text and
synthetic ranked lists (no retrieval, no CUAD content).
"""
import pytest

from lexis.evaluation.oracle_case_level import (
    DIFFERENT_DOCUMENT,
    HIGH_OVERLAP,
    LOW_OVERLAP,
    analyze_ranked_case,
    answer_token_coverage,
    best_of,
    categorize_blocker,
    summarize_blockers,
)

ANSWER = ["The supplier shall deliver goods within thirty days"]
THRESH = 0.8


def item(cid, doc="d1", text="unrelated filler words entirely"):
    return {"id": cid, "doc_id": doc, "text": text}


def analyze(ranked, relevant=("rel",), docs=("d1",), k=30):
    return analyze_ranked_case(ranked, relevant, docs, ANSWER, THRESH, k)


# --- answer_token_coverage ---

def test_coverage_full_partial_none():
    assert answer_token_coverage(ANSWER, "supplier shall deliver goods within thirty days") == 1.0
    assert 0 < answer_token_coverage(ANSWER, "supplier deliver goods") < 1
    assert answer_token_coverage(ANSWER, "completely different topic") == 0.0


def test_coverage_edge_cases():
    assert answer_token_coverage([], "anything") == 0.0
    assert answer_token_coverage([""], "anything") == 0.0
    assert answer_token_coverage(ANSWER, "") == 0.0


def test_coverage_unions_multiple_answer_spans():
    both = answer_token_coverage(["alpha beta", "gamma delta"], "alpha beta gamma delta")
    assert both == 1.0


def test_coverage_is_case_and_punctuation_insensitive():
    assert answer_token_coverage(["Deliver GOODS!"], "goods, deliver.") == 1.0


# --- categorize_blocker ---

def test_blocker_categories():
    assert categorize_blocker("other", ["d1"], 0.99, THRESH) == DIFFERENT_DOCUMENT
    assert categorize_blocker("d1", ["d1"], 0.9, THRESH) == HIGH_OVERLAP
    assert categorize_blocker("d1", ["d1"], 0.79, THRESH) == LOW_OVERLAP


def test_threshold_boundary_is_inclusive():
    assert categorize_blocker("d1", ["d1"], THRESH, THRESH) == HIGH_OVERLAP


def test_missing_top_doc_counts_as_different_document():
    assert categorize_blocker(None, ["d1"], 1.0, THRESH) == DIFFERENT_DOCUMENT


# --- analyze_ranked_case ---

def test_relevant_first_gives_perfect_rr_and_no_blocker():
    a = analyze([item("rel"), item("x")])
    assert a.reciprocal_rank == 1.0 and a.top1_is_relevant and a.blocker is None and a.first_relevant_rank == 1


def test_relevant_second_gives_half_and_a_blocker():
    a = analyze([item("x", text="filler"), item("rel")])
    assert a.reciprocal_rank == 0.5 and a.first_relevant_rank == 2
    assert not a.top1_is_relevant and a.blocker == LOW_OVERLAP


def test_blocker_from_other_document():
    a = analyze([item("x", doc="other"), item("rel")])
    assert a.blocker == DIFFERENT_DOCUMENT


def test_blocker_high_overlap_same_doc_signals_label_strictness():
    near = item("x", text="the supplier shall deliver goods within thirty days of order")
    a = analyze([near, item("rel")])
    assert a.blocker == HIGH_OVERLAP and a.top1_answer_coverage == 1.0


def test_no_relevant_in_list():
    a = analyze([item("x"), item("y")])
    assert a.reciprocal_rank == 0.0 and a.first_relevant_rank is None and a.recall == 0.0


def test_empty_ranking_is_safe():
    a = analyze([])
    assert a.reciprocal_rank == 0.0 and a.top1_chunk_id is None and a.blocker is None


def test_recall_respects_k_and_multiple_relevant():
    ranked = [item("rel"), item("x"), item("rel2")]
    assert analyze(ranked, relevant=("rel", "rel2"), k=1).recall == 0.5
    assert analyze(ranked, relevant=("rel", "rel2"), k=3).recall == 1.0


def test_uses_project_metric_functions_not_a_reimplementation():
    import inspect
    import lexis.evaluation.oracle_case_level as m
    src = inspect.getsource(m)
    assert "def recall_at_k" not in src and "def reciprocal_rank" not in src


# --- summarize / best_of ---

def test_summarize_counts_each_category_and_totals():
    analyses = [
        analyze([item("rel")]),
        analyze([item("x", doc="o"), item("rel")]),
        analyze([item("x", text="supplier shall deliver goods within thirty days"), item("rel")]),
        analyze([item("x", text="filler"), item("rel")]),
        analyze([item("x")]),
    ]
    s = summarize_blockers(analyses)
    assert s["cases"] == 5 and s["top1_relevant"] == 1 and s["top1_not_relevant"] == 4
    assert (s[DIFFERENT_DOCUMENT], s[HIGH_OVERLAP], s[LOW_OVERLAP]) == (1, 1, 2)
    assert s["no_relevant_in_list"] == 1


def test_best_of_picks_highest_rr_then_recall():
    weak = analyze([item("x"), item("rel")])
    strong = analyze([item("rel")])
    assert best_of([weak, strong]) is strong
    with pytest.raises(ValueError):
        best_of([])


def test_analysis_is_deterministic():
    ranked = [item("x", text="supplier deliver"), item("rel")]
    assert analyze(ranked) == analyze(ranked)


# --- call_until_healthy (measurement integrity against silent path timeouts) ---

import asyncio

from lexis.evaluation.oracle_case_level import call_until_healthy


def _flaky(results):
    it = iter(results)

    async def call():
        return next(it)
    return call


def test_healthy_first_try_uses_one_attempt():
    out = asyncio.run(call_until_healthy(_flaky(["ok"]), lambda r: r == "ok", max_retries=3))
    assert (out.result, out.attempts, out.healthy) == ("ok", 1, True)


def test_retries_until_healthy_and_counts_attempts():
    out = asyncio.run(call_until_healthy(_flaky(["bad", "bad", "ok"]), lambda r: r == "ok", max_retries=3))
    assert (out.attempts, out.healthy) == (3, True)


def test_gives_up_after_max_retries_and_flags_unhealthy():
    out = asyncio.run(call_until_healthy(_flaky(["bad"] * 10), lambda r: r == "ok", max_retries=2))
    assert (out.attempts, out.healthy, out.result) == (3, False, "bad")


def test_zero_retries_means_single_attempt():
    out = asyncio.run(call_until_healthy(_flaky(["bad", "ok"]), lambda r: r == "ok", max_retries=0))
    assert (out.attempts, out.healthy) == (1, False)
