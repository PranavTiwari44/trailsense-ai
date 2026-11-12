"""FastAPI entry point. Routes live here (like a Spring @RestController)."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.config import get_settings

app = FastAPI(
    title="TrailSense AI Explainer",
    description="Ask about a hiking trail; tools gather data, a rules engine decides difficulty. "
    "Check official park conditions before hiking.",
    version="0.1.0",
)


# include_in_schema=False hides this route from /docs, since it's only a convenience.
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    # FastAPI turns the returned dict into JSON automatically.
    settings = get_settings()
    return {"status": "ok", "llm_provider": settings.llm_provider}
