"""Re-export: document-scoped retrieval now lives in lexis.retrieval.scoped so the serving layer can
use it without importing the evaluation package. Kept so existing evaluation imports keep working."""
from lexis.retrieval.scoped import retrieve_scoped  # noqa: F401

__all__ = ["retrieve_scoped"]
