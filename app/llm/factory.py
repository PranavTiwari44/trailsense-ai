"""Picks which AI client to use, based on LLM_PROVIDER in the settings."""

from app.config import get_settings
from app.llm.base import LLMClient
from app.llm.ollama_client import OllamaClient


def get_llm_client() -> LLMClient:
    """Return the AI client named by LLM_PROVIDER ("ollama" or "claude")."""
    settings = get_settings()
    if settings.llm_provider == "ollama":
        return OllamaClient(
            host=settings.ollama_host,
            model=settings.ollama_model,
            think=settings.ollama_think,
            timeout_s=settings.llm_timeout_s,
        )
    if settings.llm_provider == "claude":
        raise NotImplementedError("The Claude client comes in Milestone 9. Use LLM_PROVIDER=ollama for now.")
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
