"""
Централізована конфігурація через env (12-Factor §III).
Усі змінні з .env.example мапляться сюди як типізовані атрибути.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    openrouter_api_key: str = ""

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_chunks_collection: str = "chunks"
    qdrant_cache_collection: str = "cache"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Cost tracking
    database_url: str = "sqlite+aiosqlite:///./data/cost.db"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # App tuning
    llm_concurrency_limit: int = 20
    llm_request_timeout_s: int = 15
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 3600
    max_input_chars: int = 4000
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


settings = Settings()
