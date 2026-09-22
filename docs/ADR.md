# Architecture Decision Record (ADR) - LEXIS

## ADR-001: Composite UUID Generation for Chunks
- **Date**: 2026-05-30
- **Status**: Accepted
- **Context**: `plan.md` derived `UUID5(Hash(content))` from `paperqa`. PaperQA uses this intentionally to deduplicate identical texts across distributed runs. However, combining this with RAGFlow's physical bounding boxes means identical boilerplate across two different documents will collide in Qdrant, overwriting the bounding boxes and source document metadata of the first.
- **Decision**: Use a composite strategy: `UUID5(Hash(doc_id | split_idx | content))`.
- **Rationale**: We must preserve physical bounding boxes per instance of text while retaining determinism.
- **Impact**: Citation bounding boxes will correctly map to the precise document and page they were extracted from, even for standard boilerplate clauses.

## ADR-002: Replace OpenAI/Anthropic with Gemini and Groq
- **Date**: 2026-05-30
- **Status**: Accepted
- **Context**: The user requested changing the generative foundation models to Gemini and LLaMA (via Groq) due to personal ecosystem preferences and cost/latency benefits.
- **Decision**: Update dependencies and configurations to support google-generativeai and groq clients instead of openai and anthropic.
- **Rationale**: Groq provides ultra-low TTFT for Fast Mode. Gemini 1.5 Pro offers an enormous context window for heavy aggregation.
- **Impact**: Fast mode latency should drop below 1000ms. We must ensure rigorous testing of structured JSON outputs during Task 4 since LLaMA-3 can occasionally fail schema adherence compared to GPT-4o.

## ADR-003: Replace Elasticsearch with bm25s for Path D keyword search
- **Date**: 2026-09-15
- **Status**: Accepted
- **Context**: `retrieval/path_d_bm25.py` already carried a note that this replacement was "authorized by the architect to reduce operational overhead for <10M chunk scale," but the decision had never been formally recorded, and the live retrieval path (`hybrid_retriever.py`) and ingestion path (`ingestion/pipeline.py`) still called a real Elasticsearch instance directly. Running Elasticsearch (JVM, ~1GB+ real footprint even with a capped heap) alongside Qdrant, the embedding model, and everything else was also empirically confirmed to exceed available memory on the primary 16GB development machine, and Elasticsearch has no realistic always-free managed cloud tier the way Qdrant (Qdrant Cloud) and Postgres (Neon) do.
- **Decision**: Path D keyword search is served by `bm25s` (pure Python/numpy, in-process, no server) via `indexing/bm25_index.py::LexisBM25Index`, wired into both `ingestion/pipeline.py` (writes) and `retrieval/hybrid_retriever.py` (reads). `indexing/es_client.py` and the `elasticsearch` service in `docker-compose.yml` are retired from the live path; the ES client code is left in place but unused, in case it's needed for reference.
- **Rationale**: bm25s eliminates a JVM process, a network hop, and the only infrastructure dependency in the stack without a viable always-free managed hosting option, while implementing the same BM25 ranking function. It builds its index over the full corpus in one pass rather than indexing incrementally per document; `LexisBM25Index` accepts this by persisting the corpus to a JSONL file and rebuilding the in-memory index on every write, which is fast (seconds, not minutes) at the "<10M chunk scale" this decision is explicitly scoped to.
- **Impact**: No Elasticsearch server, credentials, or network dependency anywhere in ingestion or retrieval. Query-time keyword scoring is now CPU-bound inside the serving process rather than delegated to a separate service, so `hybrid_retriever.py`/`path_d_bm25.py` run it via `asyncio.to_thread` to avoid blocking the event loop under concurrent requests. Elasticsearch's legal-synonym analyzer (indemnification/indemnity/hold harmless, etc.) is not reproduced by bm25s's default tokenizer -- if that synonym expansion is later found to matter for retrieval quality, it would need to be re-implemented as a query/corpus preprocessing step, not inside bm25s itself.
