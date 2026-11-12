"""Builds the HTTP client every service uses, so they all share the same settings."""

import httpx

from app.config import get_settings

# Overpass refuses requests that don't say who they're from, so we send a name.
USER_AGENT = "trailsense-ai/0.1 (learning project)"


def make_client(timeout_s: float | None = None) -> httpx.AsyncClient:
    """Make an HTTP client with our User-Agent and a timeout.

    Uses the timeout from settings unless a different one is passed in.
    """
    settings = get_settings()
    if timeout_s is None:
        timeout_s = settings.http_timeout_s
    return httpx.AsyncClient(timeout=timeout_s, headers={"User-Agent": USER_AGENT})
