"""Tests for the web URLs. Outside websites are faked with respx, so no internet is used."""

import httpx
import respx
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)
settings = get_settings()


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
