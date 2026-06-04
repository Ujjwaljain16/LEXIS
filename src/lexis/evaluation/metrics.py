"""
WHAT: Quantitative metrics for Lexis system performance.
"""
from typing import List
import numpy as np

def reciprocal_rank(relevant_docs: List[str], retrieved_docs: List[str]) -> float:
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0

def average_precision(relevant_docs: List[str], retrieved_docs: List[str], k: int = 10) -> float:
    hits = 0
    sum_precisions = 0.0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            hits += 1
            sum_precisions += hits / (i + 1.0)
    if not relevant_docs:
        return 0.0
    return sum_precisions / min(len(relevant_docs), k)

def recall_at_k(relevant_docs: List[str], retrieved_docs: List[str], k: int = 10) -> float:
    if not relevant_docs:
        return 0.0
    hits = sum(1 for doc in retrieved_docs[:k] if doc in relevant_docs)
    return hits / len(relevant_docs)

def citation_precision(supported_claims: int, total_claims: int) -> float:
    """Moat 2 Core Metric: What % of claims are supported by a valid citation."""
    if total_claims == 0:
        return 1.0
    return supported_claims / total_claims

def grounded_accuracy(is_answer_correct: bool, citation_precision_score: float, threshold: float = 0.95) -> bool:
    """Strict boolean: Is the answer factually correct AND safely grounded?"""
    return is_answer_correct and (citation_precision_score >= threshold)
