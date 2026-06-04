from pydantic import BaseModel, Field
from typing import Optional
import time

class ResearchBudget(BaseModel):
    max_cost_usd: float = 0.50  # Hard stop at 50 cents per query
    max_tokens: int = 100_000

class QueryCost(BaseModel):
    """Tracks token consumption and latency for a single query workflow."""
    query_id: str
    
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    verification_ms: float = 0.0
    generation_ms: float = 0.0
    
    retrieval_tokens: int = 0
    verification_tokens: int = 0
    generation_tokens: int = 0
    
    @property
    def estimated_cost_usd(self) -> float:
        """
        Calculates an estimated cost assuming GPT-4o pricing:
        $5.00 / 1M input tokens
        $15.00 / 1M output tokens
        (Simplified estimation for tracking Deep Mode budget)
        """
        input_tokens = self.retrieval_tokens + self.verification_tokens
        output_tokens = self.generation_tokens
        return (input_tokens / 1_000_000 * 5.0) + (output_tokens / 1_000_000 * 15.0)

class CostLedger:
    """Singleton/Instance ledger for tracking execution costs during a request."""
    def __init__(self, query_id: str, budget: Optional[ResearchBudget] = None):
        self.cost = QueryCost(query_id=query_id)
        self.budget = budget or ResearchBudget()
        self._timers = {}

    def is_budget_exceeded(self) -> bool:
        """Checks if the current cost exceeds the defined ResearchBudget."""
        if self.cost.estimated_cost_usd > self.budget.max_cost_usd:
            return True
        total_tokens = self.cost.retrieval_tokens + self.cost.verification_tokens + self.cost.generation_tokens
        if total_tokens > self.budget.max_tokens:
            return True
        return False

    def start_timer(self, phase: str):
        self._timers[phase] = time.perf_counter()
        
    def stop_timer(self, phase: str):
        if phase in self._timers:
            elapsed_ms = (time.perf_counter() - self._timers[phase]) * 1000
            setattr(self.cost, f"{phase}_ms", getattr(self.cost, f"{phase}_ms", 0.0) + elapsed_ms)
            del self._timers[phase]

    def add_tokens(self, phase: str, tokens: int):
        field_name = f"{phase}_tokens"
        if hasattr(self.cost, field_name):
            setattr(self.cost, field_name, getattr(self.cost, field_name) + tokens)

    def get_receipt(self) -> QueryCost:
        return self.cost
