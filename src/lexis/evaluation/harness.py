"""
WHAT: The CI Evaluation Gate for Lexis.
WHY: Prevents regressions. Enforces Hard Launch Thresholds.
"""
import logging
from typing import List, Dict
from pydantic import BaseModel
from .failure_analyzer import FailureCategory, generate_report

logger = logging.getLogger(__name__)

class BenchmarkVersion(BaseModel):
    version: str
    grounded_accuracy_target: float
    citation_precision_target: float
    recall_at_10_target: float

# Lexis V1 Launch Criteria
LEXIS_V1_BENCHMARK = BenchmarkVersion(
    version="v1.0",
    grounded_accuracy_target=0.80,
    citation_precision_target=0.95,
    recall_at_10_target=0.90
)

def assert_baseline(
    actual_grounded_acc: float,
    actual_citation_prec: float,
    actual_recall_10: float,
    benchmark: BenchmarkVersion = LEXIS_V1_BENCHMARK
) -> bool:
    """
    Hard Launch Threshold gate.
    If any of these fail, CI must fail and deployment stops.
    """
    failures = []
    if actual_grounded_acc < benchmark.grounded_accuracy_target:
        failures.append(f"Grounded Accuracy {actual_grounded_acc:.2f} < {benchmark.grounded_accuracy_target}")
        
    if actual_citation_prec < benchmark.citation_precision_target:
        failures.append(f"Citation Precision {actual_citation_prec:.2f} < {benchmark.citation_precision_target}")
        
    if actual_recall_10 < benchmark.recall_at_10_target:
        failures.append(f"Recall@10 {actual_recall_10:.2f} < {benchmark.recall_at_10_target}")
        
    if failures:
        msg = "EVALUATION REGRESSION:\n" + "\n".join(failures)
        logger.error(msg)
        raise AssertionError(msg)
        
    logger.info(f"Passed baseline version {benchmark.version}!")
    return True

def summarize_run(failures: List[FailureCategory]):
    """Outputs the actionable failure bucket report."""
    report = generate_report(failures)
    
    print("\n=== LEXIS EVALUATION REPORT ===")
    for category, pct in report.items():
        print(f"{category.ljust(25)} : {pct*100:.1f}%")
    print("===============================\n")
