from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PGKG_",
        extra="ignore",
    )

    database_url: str = "postgresql://postgres:postgres@localhost:5432/pgkg"
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    embed_dim: int = 1024
    # Pinned model IDs — dated suffixes ensure reproducible benchmark comparisons.
    llm_model: str = "gpt-4o-mini-2024-07-18"
    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    # When set, overrides llm_model for extraction only.
    # Useful for "extract with one model, answer with another" Mem0-style setups.
    extractor_model: str | None = None
    # Pinned judge model — matches LongMemEval/LoCoMo published evaluation setups.
    judge_model: str = "gpt-4o-2024-08-06"
    judge_provider: str = "openai"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    # Point at OpenRouter (https://openrouter.ai/api/v1) or Groq, etc.
    openai_base_url: str | None = None
    default_namespace: str = "default"
    offline_extract: str = "0"
    # Informational: the prompt version used for extraction (source of truth is
    # the PROMPT_VERSION constant in ml.py; this field is logged into BenchReport).
    prompt_version: str = "v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Alias for external use
MemoryConfig = Settings
