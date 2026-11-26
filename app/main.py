"""The web app: creates FastAPI and lists the URLs it answers."""

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from app.config import get_settings
from app.llm.agent import AnswerError, explain
from app.llm.base import LLMError
from app.llm.factory import get_llm_client
from app.models import DifficultyResult, ExplainRequest, ExplainResponse
from app.services.errors import TrailNotFoundError
from app.services.trail_difficulty import get_trail_difficulty

# Show our INFO logs (each tool the AI runs, with timing) in the server's terminal.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide a log line for every web request

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


@app.post("/explain")
async def explain_trail(request: ExplainRequest) -> ExplainResponse:
    """Ask about a trail in plain English. The AI gathers data with tools and explains it;
    the rules engine decides the level and score.

    Takes 5 to 30 seconds (the AI runs locally). Answers 503 if the AI can't be
    reached, and 502 if it couldn't produce a valid answer.
    """
    try:
        return await explain(request.question, get_llm_client())
    except LLMError as error:
        raise HTTPException(status_code=503, detail=f"The AI is not available right now: {error}")
    except AnswerError as error:
        raise HTTPException(status_code=502, detail=str(error))
