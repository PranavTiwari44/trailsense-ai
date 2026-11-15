"""App settings, read from environment variables and the .env file."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All the app's settings in one place, each with a default.

    A value in .env (or a real environment variable) replaces the default.
    For example, OLLAMA_MODEL=llama3.1:8b in .env changes `ollama_model`.
    """

    # Also read the .env file. Real environment variables win over .env values.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Which AI to use. Only these two values are allowed; anything else stops the app at startup.
    llm_provider: Literal["ollama", "claude"] = "ollama"

    ollama_model: str = "qwen3.5:9b"
    ollama_host: str = "http://localhost:11434"
    # "Thinking" makes the model reason privately before answering. In our test it
    # took 149 s instead of 3 s for a one-sentence answer, so it's off by default.
    ollama_think: bool = False
    # How long to wait for the AI to answer before giving up.
    llm_timeout_s: float = 300.0

    # Outside data websites (all free, no key needed).
    open_meteo_url: str = "https://api.open-meteo.com/v1"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    http_timeout_s: float = 10.0
    # Overpass is often slow. After this many seconds we give up and use the saved trails file.
    overpass_timeout_s: float = 15.0

    # Only needed later, if we try Claude.
    anthropic_api_key: str | None = None
    anthropic_model: str | None = None


@lru_cache  # build the settings once, then hand back the same object every time
def get_settings() -> Settings:
    """Return the app's settings."""
    return Settings()
