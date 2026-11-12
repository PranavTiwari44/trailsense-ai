"""App settings, loaded from environment variables and the .env file.

Think of this like Spring Boot's @ConfigurationProperties: one typed class,
fields filled in from the environment, with defaults and validation for free.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Tells pydantic-settings to also read a .env file in the working directory.
    # Real environment variables win over .env values.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Literal = only these exact strings are allowed (like a Java enum).
    # LLM_PROVIDER=gpt in .env would fail fast at startup with a clear error.
    llm_provider: Literal["ollama", "claude"] = "ollama"

    ollama_model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"

    # `str | None` means "a string, or nothing". Optional until Milestone 9.
    anthropic_api_key: str | None = None
    anthropic_model: str | None = None


# @lru_cache makes this a lazy singleton: the first call builds Settings,
# every later call returns the same object (like a Spring @Bean).
@lru_cache
def get_settings() -> Settings:
    return Settings()
