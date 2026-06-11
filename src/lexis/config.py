from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # LLM
    gemini_api_key: str = ""
    gemini_model_synthesis: str = "gemini/gemini-2.5-flash"
    gemini_model_feature: str = "gemini/gemini-2.5-flash"
    gemini_model_vision: str = "gemini/gemini-2.5-pro"

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # Vector Store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_primary: str = "chunks_primary"
    qdrant_collection_hype: str = "hype_questions"
    qdrant_collection_propositions: str = "propositions"
    qdrant_collection_clusters: str = "clusters"

    # Search
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "chunks_bm25"

    # Redis (for future worker queues, kept for compatibility)
    redis_url: str = "redis://localhost:6379/0"

    # Retrieval
    rrf_k: int = 61
    retrieval_top_k_per_path: int = 15
    retrieval_fusion_top_k: int = 50

    # Chunking
    semantic_chunking_threshold: float = 0.4
    chunk_target_tokens: int = 500
    chunk_max_tokens: int = 750

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
