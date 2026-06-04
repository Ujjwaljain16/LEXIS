"""
WHAT: Analyzes and buckets system failures for actionable evaluation reports.
"""
from typing import Dict, List
from enum import Enum

class FailureCategory(str, Enum):
    RETRIEVAL_FAILURE = "retrieval_failure"       # Ground truth not in top-K
    CITATION_FAILURE = "citation_failure"         # Found truth, but cited wrong chunk or hallucinated
    REASONING_FAILURE = "reasoning_failure"       # Found truth, cited truth, but concluded incorrectly
    GENERATION_FAILURE = "generation_failure"     # Malformed output, truncation, budget exceeded
    CRAG_FAILURE = "crag_failure"                 # Fallback triggered incorrectly or hallucinated from web
    SUCCESS = "success"

def categorize_failure(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    is_answer_correct: bool,
    citation_precision: float,
    crag_triggered: bool,
    is_malformed: bool
) -> FailureCategory:
    """Categorizes a single eval run into the primary point of failure."""
    
    if is_malformed:
        return FailureCategory.GENERATION_FAILURE
        
    has_relevant = any(doc in relevant_docs for doc in retrieved_docs)
    
    if not has_relevant:
        if crag_triggered:
            return FailureCategory.CRAG_FAILURE
        return FailureCategory.RETRIEVAL_FAILURE
        
    if not is_answer_correct:
        return FailureCategory.REASONING_FAILURE
        
    if citation_precision < 0.95:
        return FailureCategory.CITATION_FAILURE
        
    return FailureCategory.SUCCESS

def generate_report(failures: List[FailureCategory]) -> Dict[str, float]:
    total = len(failures)
    if total == 0:
        return {}
        
    report = {
        cat.value: sum(1 for f in failures if f == cat) / total
        for cat in FailureCategory
    }
    return report
