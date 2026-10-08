from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # LLM
    gemini_api_key: str = ""
    gemini_model_synthesis: str = "gemini/gemini-2.5-flash"
    gemini_model_feature: str = "gemini/gemini-2.5-flash"
    gemini_model_vision: str = "gemini/gemini-2.5-pro"
    # Optional OpenAI-compatible gateway (OpenRouter-style). With it, set GEMINI_MODEL_SYNTHESIS to the
    # gateway's model id (e.g. "openai/<model>") and GEMINI_API_KEY to that gateway's key; the field names
    # predate multi-provider support. None = LiteLLM's default routing for the model name.
    llm_api_base: Optional[str] = None

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32  # lower on memory-constrained GPUs (e.g. Colab free-tier T4)

    # Vector Store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""  # required for Qdrant Cloud; empty for a local/unauthenticated instance
    qdrant_timeout_s: int = 60  # a remote/free-tier cluster needs more than the client's short default
    qdrant_collection_primary: str = "chunks_primary"
    qdrant_collection_hype: str = "hype_questions"
    qdrant_collection_propositions: str = "propositions"
    qdrant_collection_clusters: str = "clusters"

    # Search (legacy Elasticsearch client -- see docs/ADR.md ADR-003; no longer
    # used by the live retrieval/ingestion path, kept only so es_client.py
    # remains importable for anyone still using it directly)
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "chunks_bm25"

    # BM25 keyword search (ADR-003: replaces Elasticsearch for Path D)
    bm25_index_dir: str = "data/bm25_index"

    # Redis (for future worker queues, kept for compatibility)
    redis_url: str = "redis://localhost:6379/0"

    # Postgres (citation storage). pg_client.py also checks the real
    # POSTGRES_URL process environment variable first, for deployments that
    # set it directly rather than via .env.
    postgres_url: str = "postgresql://postgres:postgres@localhost:5432/postgres"

    # Retrieval
    rrf_k: int = 61
    retrieval_top_k_per_path: int = 15
    retrieval_fusion_top_k: int = 50

    # R2: HyPE (Hypothetical Document Embeddings) question index. Default off -- an ablation
    # flag, not a silent behavior change; enabling it makes ingestion generate 3 LLM questions
    # per chunk (indexed into qdrant_collection_hype) and adds a third retrieval path.
    hype_enabled: bool = False
    hype_questions_per_chunk: int = 3
    hype_generation_timeout_s: int = 30
    # Gemini's free tier caps gemini-2.5-flash at 20 requests/minute; a document with more
    # chunks than that firing every call at once (the naive asyncio.gather approach) makes
    # nearly all of them fail with RateLimitError instantly. Bounds how many
    # HyPEGenerator.generate_questions() calls run concurrently.
    hype_max_concurrent_requests: int = 5
    # A concurrency bound alone does not bound RATE (fast calls still exceed the per-minute cap
    # even with few in flight at once) -- confirmed live. hype_requests_per_minute paces actual
    # call starts via a sliding-window limiter. 5 matches the free-tier gemini-2.5-flash quota
    # actually observed live (GenerateRequestsPerMinutePerProjectPerModel-FreeTier = 5, not the
    # 20 originally assumed) -- override upward for a paid tier or a key with a higher quota.
    # Some free-tier keys also cap at as few as 20 requests/DAY, which no amount of per-minute
    # pacing can work around; see hype_generator.py's daily-quota-exhaustion short-circuit.
    hype_requests_per_minute: int = 5
    hype_max_retries: int = 3
    hype_retry_backoff_s: float = 5.0

    # R4: cross-encoder rerank. Off by default -- an ablation flag. No external API dependency
    # (local model), so unlike R2/HyPE this is not rate-limited; the cost is query-time latency.
    rerank_enabled: bool = False
    rerank_top_k: int = 50
    # None = let sentence-transformers pick (CUDA if present). Set "cpu" to keep the ~2.3 GB reranker
    # off a GPU that is already mostly full (e.g. a shared/partly-occupied Colab T4).
    rerank_device: Optional[str] = None

    # R7: document-preamble prior for document-scoped retrieval (retrieval/preamble.py). 0 = off. An
    # ablation flag: the number of opening chunks per document added as an extra RRF list.
    preamble_prior_chunks: int = 0
    preamble_scroll_batch: int = 256
    # Weight of the preamble list in RRF relative to the dense/BM25 lists (1.0 = equal vote). Below 1.0 it
    # acts as a tie-breaker so it lifts opening chunks that already have some retrieval support without
    # pushing irrelevant opening chunks above correct clause hits.
    preamble_prior_weight: float = 1.0

    # NLI/faithfulness checking (plan section 5, "Real NLI"). 0.5 ("more likely entailed than
    # not") is a deliberately conservative starting point, not a measured value -- the plan
    # calls for hand-labeling ~100 legal claims and reporting measured accuracy before treating
    # any specific threshold as validated (see config/models.yaml's per-NLI-model accuracy notes).
    nli_entailment_threshold: float = 0.5

    # Answer path (serving/service.py). Retrieval stays hybrid + RRF exactly as evaluated; these
    # only size how much of the fused list reaches the generator.
    answer_top_k_per_path: int = 15
    answer_top_n_fused: int = 15
    answer_context_chunks: int = 5
    # Claim-level support check on the finished answer (NLI). Off by default: it loads a second
    # model, and its accuracy on legal text is not yet measured (see nli_entailment_threshold).
    answer_verify_claims: bool = False
    answer_temperature: float = 0.1
    api_max_query_chars: int = 4000
    api_max_document_ids: int = 50

    # API security. LEXIS_API_KEYS="key1:tenantA,key2:tenantB". Empty = fail closed (503) unless
    # auth_disabled is set explicitly, which is for local development only.
    lexis_api_keys: str = ""
    auth_disabled: bool = False
    # Ingest accepts server-side paths; they must resolve inside this directory (no traversal).
    ingest_root_dir: str = "data/uploads"
    rate_limit_window_s: float = 60.0
    rate_limit_fast_per_window: int = 30
    rate_limit_deep_per_window: int = 5

    # Frozen-artifact verification tolerances (evaluation/verify_frozen.py). Means are exact sums, so the
    # tolerance is float noise; CI bounds come from a seeded bootstrap whose RNG stream can differ across
    # numpy major versions.
    verify_mean_tolerance: float = 1e-9
    verify_ci_tolerance: float = 5e-4

    # Answer-quality evaluation (evaluation/run_answer_eval.py): LLM calls per minute, kept under the
    # provider's limit. Conservative default; raise it for a paid key.
    answer_eval_requests_per_minute: int = 10

    # Chunking
    semantic_chunking_threshold: float = 0.4
    chunk_target_tokens: int = 500
    chunk_max_tokens: int = 750

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
