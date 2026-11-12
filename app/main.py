"""The web app: creates FastAPI and lists the URLs it answers."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.config import get_settings

app = FastAPI(
    title="TrailSense AI Explainer",
    description="Ask about a hiking trail; tools gather data, a rules engine decides difficulty. "
    "Check official park conditions before hiking.",
    version="0.1.0",
)


@app.get("/", include_in_schema=False)  # include_in_schema=False hides it from /docs
def root() -> RedirectResponse:
    """Send visitors of the home page to the API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    """Quick check that the app is running, and which AI it is set to use."""
    settings = get_settings()
    return {"status": "ok", "llm_provider": settings.llm_provider}
