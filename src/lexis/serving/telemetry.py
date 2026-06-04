import contextvars
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode

tracer = trace.get_tracer(__name__)
# Context variable to hold the current trace_id and request_id for logging
current_trace_id = contextvars.ContextVar("current_trace_id", default=None)

def get_trace_id() -> str:
    span = trace.get_current_span()
    if span.is_recording():
        return format(span.get_span_context().trace_id, "032x")
    return current_trace_id.get() or "unknown_trace"

class LexisTracer:
    """
    Helper class to standardize span creation and telemetry emission across Lexis.
    """
    @staticmethod
    def start_span(name: str):
        return tracer.start_as_current_span(name)
        
    @staticmethod
    def record_cost(span, cost_ledger):
        """
        Extracts cost and token usage from CostLedger and adds them as span attributes
        so Jaeger/Grafana can quickly identify expensive queries.
        """
        receipt = cost_ledger.get_receipt()
        span.set_attribute("lexis.cost.retrieval_ms", receipt.retrieval_ms)
        span.set_attribute("lexis.cost.rerank_ms", receipt.rerank_ms)
        span.set_attribute("lexis.cost.verification_ms", receipt.verification_ms)
        span.set_attribute("lexis.cost.generation_ms", receipt.generation_ms)
        
        span.set_attribute("lexis.tokens.retrieval", receipt.retrieval_tokens)
        span.set_attribute("lexis.tokens.verification", receipt.verification_tokens)
        span.set_attribute("lexis.tokens.generation", receipt.generation_tokens)
        span.set_attribute("lexis.cost.estimated_usd", receipt.estimated_cost_usd)

    @staticmethod
    def record_error(span, error: Exception):
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error)))
