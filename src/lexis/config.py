"""
Configuration module for LEXIS.

Rationale: Centralize all environment variables and configuration to avoid magic strings.
Source Inspiration: Standard 12-factor app configuration using pydantic-settings.
Deviations from Source: N/A.
Expected Impact on Metrics: N/A (Infrastructure level).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    groq_api_key: str = ""
    gemini_api_key: str = ""
    llm_model: str = "groq/llama-3.3-70b-versatile"  # Change to "gemini/gemini-2.5-flash" in .env
    
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    es_host: str = "http://localhost:9200"
    
    redis_url: str = "redis://localhost:6379/0"
    
    environment: str = "development"
    log_level: str = "INFO"
    
    # Deep Mode Configuration
    deep_mode_rrf_candidates: int = 8
    deep_mode_top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
