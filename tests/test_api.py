"""Tests for the web URLs. Outside websites are faked with respx, so no internet is used."""

import httpx
import respx
from fastapi.testclient import TestClient

from app import main
from app.config import get_settings
from app.llm.agent import AnswerError
from app.llm.base import LLMError
from app.main import app
from app.models import SAFETY_NOTE, ExplainResponse

client = TestClient(app)
settings = get_settings()


def test_home_page_is_the_web_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "TrailSense" in response.text
    assert "text/html" in response.headers["content-type"]


def test_web_page_files_are_served():
    assert client.get("/styles.css").status_code == 200
    assert client.get("/logo.svg").status_code == 200
    script = client.get("/app.js")
    assert script.status_code == 200
    assert "/explain" in script.text  # the compiled script calls our API


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_difficulty_unknown_trail_is_404():
    response = client.get("/difficulty", params={"trail_id": "no-such-trail"})
    assert response.status_code == 404


def test_difficulty_needs_a_trail_id():
    response = client.get("/difficulty")
    assert response.status_code == 422  # FastAPI rejects the request: trail_id is required


@respx.mock
def test_difficulty_is_unknown_when_weather_and_elevation_are_down():
    respx.get(url__startswith=settings.open_meteo_url).mock(return_value=httpx.Response(500))

    response = client.get("/difficulty", params={"trail_id": "wolf-trail"})

    assert response.status_code == 200
    body = response.json()
    assert body["level"] == "Unknown"
    assert body["score"] is None


# ---------- POST /explain (the AI part is faked, so no model is needed) ----------

def fake_answer() -> ExplainResponse:
    """A finished answer, as if the whole agent had run."""
    return ExplainResponse(trail_name="Wolf Trail", level="Moderate", score=35,
                           summary="Moderate: 8.9 km with 360 m of climb.", reasons=["8.9 km long"],
                           warnings=[], gear_tips=["Water"], data_time=None)


def test_explain_returns_the_answer_as_json(monkeypatch):
    async def fake_explain(question, llm_client):
        return fake_answer()

    monkeypatch.setattr(main, "explain", fake_explain)

    response = client.post("/explain", json={"question": "How hard is Wolf Trail?"})

    assert response.status_code == 200
    body = response.json()
    assert body["level"] == "Moderate"
    assert body["note"] == SAFETY_NOTE  # every answer carries the safety note


def test_explain_says_503_when_the_ai_is_down(monkeypatch):
    async def broken_explain(question, llm_client):
        raise LLMError("Failed to connect to Ollama")

    monkeypatch.setattr(main, "explain", broken_explain)

    response = client.post("/explain", json={"question": "How hard is Wolf Trail?"})

    assert response.status_code == 503


def test_explain_says_502_when_the_ai_gives_no_valid_answer(monkeypatch):
    async def confused_explain(question, llm_client):
        raise AnswerError("No valid answer after 2 tries")

    monkeypatch.setattr(main, "explain", confused_explain)

    response = client.post("/explain", json={"question": "How hard is Wolf Trail?"})

    assert response.status_code == 502


def test_explain_rejects_a_too_short_question():
    response = client.post("/explain", json={"question": "hi"})

    assert response.status_code == 422  # min_length=3 on ExplainRequest
