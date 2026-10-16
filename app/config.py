from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings are read from environment variables and .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG Knowledge Assistant"
    database_url: str = "postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_assistant"
    ollama_base_url: str = "http://host.docker.internal:11434"
    chat_model: str = "qwen2.5:3b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_dimensions: int = 384
    chunk_size: int = 900
    chunk_overlap: int = 150
    semantic_top_k: int = 8
    bm25_top_k: int = 8
    final_top_k: int = 5
    max_cosine_distance: float = 0.70
    enable_llm_reranker: bool = False
    max_upload_bytes: int = 20 * 1024 * 1024


settings = Settings()
