"""Tests for the AI's tools. No model and no internet: the weather service is replaced with a fake."""

import json
from datetime import datetime, timedelta, timezone

from app.llm import tools
from app.llm.base import ToolCall
from app.models import DifficultyResult, ElevationProfile, TrailInfo, Weather, WeatherHour
from app.services.errors import DataUnavailableError, TrailNotFoundError

NOW = datetime(2026, 11, 14, 10, 0, tzinfo=timezone(timedelta(hours=-5)))


def make_weather() -> Weather:
    """A small, mild test forecast: just the current hour and one hour ahead."""
    hour = WeatherHour(time=NOW, temp_c=4.0, precip_mm=1.2, rain_mm=1.2, snowfall_cm=0.0,
                       wind_kmh=15.0, weather_code=61)
    return Weather(lat=45.5, lon=-75.9, current=hour, recent_hours=[], next_hours=[hour])


def test_weather_tool_tells_the_ai_its_inputs():
    schema = tools.WEATHER_TOOL.parameters

    assert schema["required"] == ["lat", "lon"]
    assert schema["properties"]["lat"]["type"] == "number"


async def test_weather_tool_returns_the_weather_as_json(monkeypatch):
    async def fake_get_weather(lat, lon):
        return make_weather()

    monkeypatch.setattr(tools, "get_weather", fake_get_weather)

    result = await tools.run_tool(ToolCall(name="get_weather", arguments={"lat": 45.5, "lon": -75.9}))

    data = json.loads(result)
    assert data["current"]["temp_c"] == 4.0
    assert data["current"]["time"] == "2026-11-14T10:00:00-05:00"  # dates become text


async def test_numbers_sent_as_text_are_accepted(monkeypatch):
    # Small models sometimes send "45.5" instead of 45.5. Pydantic converts it for us.
    async def fake_get_weather(lat, lon):
        assert lat == 45.5
        return make_weather()

    monkeypatch.setattr(tools, "get_weather", fake_get_weather)

    result = await tools.run_tool(ToolCall(name="get_weather", arguments={"lat": "45.5", "lon": "-75.9"}))

    assert "error" not in json.loads(result)


async def test_unknown_tool_gives_an_error_message():
    result = await tools.run_tool(ToolCall(name="get_lunch", arguments={}))

    assert "Unknown tool 'get_lunch'" in json.loads(result)["error"]


async def test_bad_inputs_give_an_error_message():
    result = await tools.run_tool(ToolCall(name="get_weather", arguments={"lat": 200}))

    error = json.loads(result)["error"]
    assert "Bad inputs for get_weather" in error
    assert "lat" in error  # says what was wrong: lat too big...
    assert "lon" in error  # ...and lon missing


async def test_website_down_gives_an_error_message(monkeypatch):
    async def broken_get_weather(lat, lon):
        raise DataUnavailableError("Weather service is down")

    monkeypatch.setattr(tools, "get_weather", broken_get_weather)

    result = await tools.run_tool(ToolCall(name="get_weather", arguments={"lat": 45.5, "lon": -75.9}))

    assert json.loads(result) == {"error": "Weather service is down"}


# ---------- All four tools ----------

def test_every_tool_shown_to_the_ai_can_be_run():
    # If we add a tool to the menu but forget its function (or the other way round), this fails.
    names = []
    for spec in tools.TOOL_SPECS:
        names.append(spec.name)

    assert sorted(names) == sorted(tools.TOOL_FUNCTIONS)
    assert len(names) == 4


async def test_find_trail_returns_details_without_the_map_points(monkeypatch):
    async def fake_find_trail(name, near=None):
        return TrailInfo(id="wolf-trail", name="Wolf Trail", length_km=8.9, sac_scale="mountain_hiking",
                         start_lat=45.5, start_lon=-75.9, source="fallback",
                         segments=[[(45.5, -75.9), (45.6, -75.8)]])

    monkeypatch.setattr(tools, "find_trail", fake_find_trail)

    result = await tools.run_tool(ToolCall(name="find_trail", arguments={"name": "Wolf Trail"}))

    data = json.loads(result)
    assert data["id"] == "wolf-trail"
    assert data["start_lat"] == 45.5
    assert "segments" not in data  # hundreds of points the AI doesn't need


async def test_find_trail_not_found_gives_an_error_message(monkeypatch):
    async def fake_find_trail(name, near=None):
        raise TrailNotFoundError(f"No trail found matching {name!r}")

    monkeypatch.setattr(tools, "find_trail", fake_find_trail)

    result = await tools.run_tool(ToolCall(name="find_trail", arguments={"name": "Narnia Loop"}))

    assert json.loads(result) == {"error": "No trail found matching 'Narnia Loop'"}


async def test_find_trail_passes_the_place_through(monkeypatch):
    places_asked = []

    async def fake_find_trail(name, near=None):
        places_asked.append(near)
        return TrailInfo(id="kay-gardiner-beltline-trail", name="Kay Gardiner Beltline Trail", length_km=4.5,
                         start_lat=43.7, start_lon=-79.4, source="overpass")

    monkeypatch.setattr(tools, "find_trail", fake_find_trail)

    await tools.run_tool(ToolCall(name="find_trail", arguments={"name": "Beltline", "near": "Toronto"}))
    await tools.run_tool(ToolCall(name="find_trail", arguments={"name": "Wolf Trail"}))

    assert places_asked == ["Toronto", None]  # near is optional


async def test_find_trail_with_an_empty_name_gives_an_error_message():
    result = await tools.run_tool(ToolCall(name="find_trail", arguments={"name": ""}))

    assert "Bad inputs for find_trail" in json.loads(result)["error"]


async def test_elevation_gain_for_a_saved_trail(monkeypatch):
    # "wolf-trail" is in our saved file, so get_trail finds it with no internet.
    async def fake_profile(segments):
        assert len(segments) > 0  # we were given the trail's real map points
        return ElevationProfile(gain_m=310, min_m=180, max_m=340, samples=90)

    monkeypatch.setattr(tools, "get_elevation_profile", fake_profile)

    result = await tools.run_tool(ToolCall(name="get_elevation_gain", arguments={"trail_id": "wolf-trail"}))

    assert json.loads(result)["gain_m"] == 310


async def test_unknown_trail_id_tells_the_ai_to_find_the_trail_first():
    result = await tools.run_tool(ToolCall(name="get_elevation_gain", arguments={"trail_id": "made-up-trail"}))

    assert "Call find_trail first" in json.loads(result)["error"]


async def test_difficulty_score_returns_the_rules_engine_result(monkeypatch):
    async def fake_difficulty(trail_id):
        return DifficultyResult(score=35, level="Moderate", reasons=["8 km long (+8)"], data_time=NOW)

    monkeypatch.setattr(tools, "get_trail_difficulty", fake_difficulty)

    result = await tools.run_tool(ToolCall(name="get_difficulty_score", arguments={"trail_id": "wolf-trail"}))

    data = json.loads(result)
    assert data["level"] == "Moderate"
    assert data["score"] == 35
    assert data["reasons"] == ["8 km long (+8)"]
