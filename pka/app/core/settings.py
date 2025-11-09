from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Centralised application configuration loaded from environment or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    app_env: Literal["development", "production", "test"] = Field(default="development")
    app_name: str = Field(default="NestAi")
    log_level: str = Field(default="INFO")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_chat_model: str = Field(default="llama3.1:8b")
    ollama_embed_model: str = Field(default="nomic-embed-text")
    ollama_timeout_seconds: int = Field(default=60)
    ollama_num_predict: int | None = Field(default=None)
    ollama_num_ctx: int | None = Field(default=None)
    ollama_keep_alive: str | None = Field(default="30m")

    llm_temperature: float = Field(default=0.0)
    llm_seed: int = Field(default=42)

    knowledge_notes_dir: Path = Field(default=Path.home() / "KnowledgeBase" / "notes")
    knowledge_pdfs_dir: Path = Field(default=Path.home() / "KnowledgeBase" / "pdfs")
    knowledge_emails_dir: Path = Field(default=Path.home() / "KnowledgeBase" / "exports" / "emails")

    database_url: str = Field(default="postgresql+psycopg://pka:pka@localhost:5432/pka")
    vector_dim: int = Field(default=768)
    bm25_index_path: Path = Field(default=Path(".indices") / "bm25")

    enable_reranker: bool = Field(default=False)
    strict_mode: bool = Field(default=False)
    skip_health_checks: bool = Field(default=False, validation_alias="skip_health_checks")

    @validator(
        "knowledge_notes_dir",
        "knowledge_pdfs_dir",
        "knowledge_emails_dir",
        "bm25_index_path",
        pre=True,
    )
    def _expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()


@lru_cache
def get_settings() -> AppSettings:
    """Return cached settings instance."""

    return AppSettings()


settings = get_settings()
