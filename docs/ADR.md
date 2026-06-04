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
