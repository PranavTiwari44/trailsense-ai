"""The web app: creates FastAPI and lists the URLs it answers."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from app.config import get_settings
from app.models import DifficultyResult
from app.services.errors import TrailNotFoundError
from app.services.trail_difficulty import get_trail_difficulty

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


@app.get("/difficulty")
async def difficulty(
    trail_id: str = Query(
        description="Trail id. Saved trails: wolf-trail, king-mountain-trail, luskville-falls-trail, pink-lake-trail",
        examples=["wolf-trail"],
    ),
) -> DifficultyResult:
    """How hard a trail is right now, from the rules engine only (no AI).

    Uses live weather and elevation. Answers 404 if the trail id is unknown.
    """
    try:
        return await get_trail_difficulty(trail_id)
    except TrailNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
