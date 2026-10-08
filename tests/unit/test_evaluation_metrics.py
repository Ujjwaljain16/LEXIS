"""
Tests for the existing generic retrieval metrics in evaluation/metrics.py
(reciprocal_rank, recall_at_k) against the exact behaviour specified for
the Week 1-2 evaluation gate. These functions were already implemented and
unused/untested before this step -- this file is regression coverage for
functions the new harness now depends on, not a rewrite of their logic.
"""
import pytest

from lexis.evaluation.metrics import reciprocal_rank, recall_at_k


# --- Recall@30 ---

def test_recall_relevant_at_rank_1_is_one():
    retrieved = ["relevant-1"] + [f"noise-{i}" for i in range(29)]
    assert recall_at_k(["relevant-1"], retrieved, k=30) == 1.0


def test_recall_relevant_at_rank_30_is_one():
    retrieved = [f"noise-{i}" for i in range(29)] + ["relevant-1"]
    assert len(retrieved) == 30
    assert recall_at_k(["relevant-1"], retrieved, k=30) == 1.0


def test_recall_relevant_at_rank_31_is_zero():
    retrieved = [f"noise-{i}" for i in range(30)] + ["relevant-1"]
    assert len(retrieved) == 31
    assert recall_at_k(["relevant-1"], retrieved, k=30) == 0.0


def test_recall_no_relevant_result_is_zero():
    retrieved = [f"noise-{i}" for i in range(30)]
    assert recall_at_k(["relevant-1"], retrieved, k=30) == 0.0


def test_recall_with_multiple_relevant_items_is_proportional():
    """Explicit, tested definition for the multi-relevant-item case: the
    standard IR recall -- fraction of ground-truth relevant items that
    appear anywhere in the top k -- not a binary hit/miss."""
    relevant = ["a", "b", "c"]
    retrieved = ["a", "x", "b", "y", "z"]  # 2 of 3 relevant items present
    assert recall_at_k(relevant, retrieved, k=30) == pytest.approx(2 / 3)


def test_recall_all_relevant_items_present_is_one():
    relevant = ["a", "b"]
    retrieved = ["a", "b", "z"]
    assert recall_at_k(relevant, retrieved, k=30) == 1.0


def test_recall_empty_relevant_set_is_zero_not_a_crash():
    assert recall_at_k([], ["a", "b"], k=30) == 0.0


# --- MRR ---

def test_mrr_relevant_at_rank_1_is_one():
    assert reciprocal_rank(["relevant-1"], ["relevant-1", "x", "y"]) == 1.0


def test_mrr_relevant_at_rank_2_is_half():
    assert reciprocal_rank(["relevant-1"], ["x", "relevant-1", "y"]) == pytest.approx(0.5)


def test_mrr_relevant_at_rank_5_is_one_fifth():
    assert reciprocal_rank(["relevant-1"], ["a", "b", "c", "d", "relevant-1"]) == pytest.approx(0.2)


def test_mrr_no_relevant_result_is_zero():
    assert reciprocal_rank(["relevant-1"], ["a", "b", "c"]) == 0.0


def test_mrr_with_multiple_relevant_items_uses_first_relevant_rank_only():
    """Two of the three retrieved candidates are relevant (rank 2 and rank
    4) -- MRR must reflect only the FIRST one (rank 2 -> 0.5), not the best
    achievable or an average."""
    relevant = ["b", "d"]
    retrieved = ["a", "b", "c", "d"]
    assert reciprocal_rank(relevant, retrieved) == pytest.approx(0.5)
