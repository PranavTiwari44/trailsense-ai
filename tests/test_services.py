"""Tests for the data code (trails, weather, elevation).

None of these tests use the internet. The respx library catches every web
request and answers with a fake response that we set up in the test.
"""

from datetime import timedelta

import httpx
import pytest
import respx

from app.config import get_settings
from app.services import trails
from app.services.elevation import distance_m, get_elevation_profile
from app.services.errors import DataUnavailableError, TrailNotFoundError
from app.services.places import DEFAULT_AREA, area_around, find_place, is_inside, search_area_for
from app.services.trails import find_trail, get_trail, join_pieces, make_id, search_words
from app.services.weather import get_weather

settings = get_settings()
FORECAST_URL = f"{settings.open_meteo_url}/forecast"
ELEVATION_URL = f"{settings.open_meteo_url}/elevation"


@pytest.fixture(autouse=True)  # autouse=True: runs around every test in this file
def forget_found_trails():
    """Clear the list of found trails before and after each test, so tests don't affect each other."""
    trails._found_trails.clear()
    yield  # the test runs here
    trails._found_trails.clear()


# ---------- Weather ----------

def fake_open_meteo_answer() -> dict:
    """A short copy of a real Open-Meteo answer: 6 past hours (rainy) and 6 upcoming hours (thunderstorm)."""
    times = [
        "2026-11-11T19:00", "2026-11-11T20:00", "2026-11-11T21:00",
        "2026-11-11T22:00", "2026-11-11T23:00", "2026-11-12T00:00",
        "2026-11-12T01:00", "2026-11-12T02:00", "2026-11-12T03:00",
        "2026-11-12T04:00", "2026-11-12T05:00", "2026-11-12T06:00",
    ]
    return {
        "utc_offset_seconds": -18000,  # 5 hours behind UTC
        "current": {
            "time": "2026-11-12T01:15", "temperature_2m": 0.1, "precipitation": 0.0, "rain": 0.0,
            "snowfall": 0.0, "wind_speed_10m": 13.7, "weather_code": 3,
        },
        "hourly": {
            "time": times,
            "temperature_2m": [2.0] * 12,  # [2.0] * 12 = a list of twelve 2.0s
            "precipitation": [1.5] * 6 + [0.0] * 6,
            "rain": [1.5] * 6 + [0.0] * 6,
            "snowfall": [0.0] * 12,
            "wind_speed_10m": [10.0] * 12,
            "weather_code": [61] * 6 + [95] * 6,
        },
    }


@respx.mock
async def test_weather_splits_past_and_upcoming_hours():
    respx.get(url__startswith=FORECAST_URL).mock(return_value=httpx.Response(200, json=fake_open_meteo_answer()))

    weather = await get_weather(45.5, -75.9)

    assert weather.current.temp_c == 0.1
    assert weather.current.time.utcoffset() == timedelta(hours=-5)  # the timezone was attached
    assert len(weather.recent_hours) == 6
    assert weather.recent_hours[0].rain_mm == 1.5
    assert len(weather.next_hours) == 6
    assert weather.next_hours[0].time.hour == 1  # the hour we're in (01:15) counts as upcoming
    assert weather.next_hours[0].weather_code == 95


@respx.mock
async def test_weather_server_error_becomes_data_unavailable():
    respx.get(url__startswith=FORECAST_URL).mock(return_value=httpx.Response(500))

    with pytest.raises(DataUnavailableError):  # the test passes only if this error is raised
        await get_weather(45.5, -75.9)


@respx.mock
async def test_weather_with_missing_value_becomes_data_unavailable():
    answer = fake_open_meteo_answer()
    answer["current"]["temperature_2m"] = None  # Open-Meteo sometimes sends null
    respx.get(url__startswith=FORECAST_URL).mock(return_value=httpx.Response(200, json=answer))

    with pytest.raises(DataUnavailableError):
        await get_weather(45.5, -75.9)


@respx.mock
async def test_weather_timeout_becomes_data_unavailable():
    respx.get(url__startswith=FORECAST_URL).mock(side_effect=httpx.ReadTimeout("too slow"))

    with pytest.raises(DataUnavailableError):
        await get_weather(45.5, -75.9)


# ---------- Elevation ----------

def fake_elevation_api(height_for_lat):
    """Set up a fake elevation website that answers each point with height_for_lat(lat)."""
    def answer(request: httpx.Request) -> httpx.Response:
        lats = request.url.params["latitude"].split(",")
        heights = [height_for_lat(float(lat)) for lat in lats]
        return httpx.Response(200, json={"elevation": heights})
    return respx.get(url__startswith=ELEVATION_URL).mock(side_effect=answer)


def test_one_degree_of_latitude_is_about_111_km():
    assert distance_m((45.0, -75.0), (46.0, -75.0)) == pytest.approx(111_195, rel=0.001)


@respx.mock
async def test_climb_is_half_of_all_ups_and_downs():
    # Up 50, down 30, down 20: total change 100, so the climb is 50.
    heights = {45.0: 100, 45.01: 150, 45.02: 120, 45.03: 100}
    fake_elevation_api(lambda lat: heights[round(lat, 2)])
    piece = [(45.0, -75.0), (45.01, -75.0), (45.02, -75.0), (45.03, -75.0)]

    profile = await get_elevation_profile([piece])

    assert profile.gain_m == 50
    assert profile.min_m == 100
    assert profile.max_m == 150


@respx.mock
async def test_elevation_asks_in_groups_of_100_points():
    route = fake_elevation_api(lambda lat: 200)  # completely flat
    piece = [(45.0 + i * 0.002, -75.0) for i in range(250)]  # points ~220 m apart, so none are dropped

    profile = await get_elevation_profile([piece])

    assert route.call_count == 3  # 100 + 100 + 50 points
    assert profile.samples == 250
    assert profile.gain_m == 0


async def test_elevation_without_points_is_data_unavailable():
    with pytest.raises(DataUnavailableError):
        await get_elevation_profile([])


# ---------- Trails: small helpers ----------

def test_search_words_drops_filler_words():
    assert search_words("The Wolf Trail") == ["wolf"]
    assert search_words("Sentier du Lac-Pink") == ["lac", "pink"]
    assert search_words("Trail") == ["trail"]  # nothing useful left, so keep everything


def test_make_id():
    assert make_id("Sentier du Lac-Pink") == "sentier-du-lac-pink"
    assert make_id("Rivière Trail") == "riviere-trail"


def osm_piece(piece_id: int, english_name: str, difficulty: str | None, lats: list[float]) -> dict:
    """Build one fake OpenStreetMap piece, shaped like what Overpass sends back."""
    tags = {"highway": "path", "name": f"Sentier {piece_id}", "name:en": english_name}
    if difficulty:
        tags["sac_scale"] = difficulty
    points = [{"lat": lat, "lon": -75.9} for lat in lats]
    return {"type": "way", "id": piece_id, "tags": tags, "geometry": points}


OVERPASS_PIECES = [
    osm_piece(1, "Wolf Trail", "mountain_hiking", [45.50, 45.51]),  # ~1.1 km
    osm_piece(2, "Wolf Trail", "demanding_mountain_hiking", [45.51, 45.52]),  # ~1.1 km
    osm_piece(3, "Wolf Creek Path", None, [45.60, 45.601]),  # a different, short trail
]


def test_join_pieces_makes_one_trail_with_the_hardest_difficulty():
    trail = join_pieces(OVERPASS_PIECES, ["wolf"])

    assert trail.name == "Wolf Trail"  # the longer of the two matching trails
    assert trail.id == "wolf-trail"
    assert len(trail.segments) == 2
    assert trail.length_km == pytest.approx(2.22, abs=0.01)
    assert trail.sac_scale == "demanding_mountain_hiking"
    assert trail.start_lat == 45.50


def test_overpass_query_only_searches_walking_paths():
    query = trails.build_overpass_query(["mer", "bleue"], 10, DEFAULT_AREA)
    assert "path|footway" in query
    assert "cycleway" in query  # shared walking/cycling paths like Toronto's Beltline
    assert '["highway"]' not in query  # a plain "highway" filter would match roads too


def test_overpass_query_searches_the_given_area():
    query = trails.build_overpass_query(["beltline"], 10, (43.4, -79.8, 44.0, -79.0))
    assert "(43.4,-79.8,44.0,-79.0)" in query


# ---------- Places: searching near a city or town ----------

GEOCODING_URL = f"{settings.geocoding_url}/search"
TORONTO = {"results": [{"name": "Toronto", "latitude": 43.70643, "longitude": -79.39864}]}


def test_area_around_reaches_about_30_km_each_way():
    south, west, north, east = area_around(43.7, -79.4)

    assert distance_m((south, -79.4), (north, -79.4)) == pytest.approx(60_000, rel=0.02)
    assert distance_m((43.7, west), (43.7, east)) == pytest.approx(60_000, rel=0.02)
    assert is_inside((south, west, north, east), 43.7, -79.4)
    assert not is_inside((south, west, north, east), 45.4, -75.7)  # Ottawa is far outside


@respx.mock
async def test_find_place_uses_only_the_town_name():
    route = respx.get(GEOCODING_URL).mock(return_value=httpx.Response(200, json=TORONTO))

    place = await find_place("Toronto, Ontario")

    assert place == (43.70643, -79.39864)
    assert route.calls.last.request.url.params["name"] == "Toronto"  # ", Ontario" dropped


@respx.mock
async def test_unknown_place_searches_ottawa_gatineau():
    respx.get(GEOCODING_URL).mock(return_value=httpx.Response(200, json={}))  # no "results" at all

    assert await search_area_for("Gatineau Park") == DEFAULT_AREA  # parks aren't towns


@respx.mock
async def test_geocoder_down_searches_ottawa_gatineau():
    respx.get(GEOCODING_URL).mock(return_value=httpx.Response(503))

    assert await search_area_for("Toronto") == DEFAULT_AREA


async def test_no_place_searches_ottawa_gatineau():
    assert await search_area_for(None) == DEFAULT_AREA


# ---------- Trails: find_trail ----------

@respx.mock
async def test_find_trail_uses_overpass_and_remembers_the_trail():
    respx.post(settings.overpass_url).mock(return_value=httpx.Response(200, json={"elements": OVERPASS_PIECES}))

    trail = await find_trail("Wolf Trail")

    assert trail.source == "overpass"
    assert get_trail("wolf-trail") is trail  # other code can now get it by id


@respx.mock
async def test_find_trail_uses_saved_file_when_overpass_is_down():
    respx.post(settings.overpass_url).mock(return_value=httpx.Response(502))

    trail = await find_trail("Wolf Trail")

    assert trail.source == "fallback"
    assert trail.name == "Wolf Trail"


@respx.mock
async def test_find_trail_by_french_name_in_saved_file():
    respx.post(settings.overpass_url).mock(side_effect=httpx.ReadTimeout("too slow"))

    trail = await find_trail("Sentier des Loups")

    assert trail.id == "wolf-trail"


@respx.mock
async def test_find_trail_with_unknown_name_raises():
    respx.post(settings.overpass_url).mock(return_value=httpx.Response(200, json={"elements": []}))

    with pytest.raises(TrailNotFoundError) as error:
        await find_trail("Definitely Not A Real Trail")

    # The message tells the AI (and so the person) how to search outside Ottawa–Gatineau.
    assert "Ottawa–Gatineau" in str(error.value)
    assert "city" in str(error.value)


@respx.mock
async def test_find_trail_near_a_city_searches_around_it():
    respx.get(GEOCODING_URL).mock(return_value=httpx.Response(200, json=TORONTO))
    overpass = respx.post(settings.overpass_url).mock(return_value=httpx.Response(200, json={"elements": []}))

    with pytest.raises(TrailNotFoundError, match="near Toronto"):
        await find_trail("Beltline", near="Toronto")

    query = overpass.calls.last.request.content.decode()
    assert "43.4" in query and "-79." in query  # the Toronto box, not Ottawa's


@respx.mock
async def test_saved_ottawa_trails_are_not_used_for_another_city():
    respx.get(GEOCODING_URL).mock(return_value=httpx.Response(200, json=TORONTO))
    respx.post(settings.overpass_url).mock(return_value=httpx.Response(502))  # Overpass down

    # The saved file has Wolf Trail, but it's in Gatineau, not Toronto.
    with pytest.raises(TrailNotFoundError):
        await find_trail("Wolf Trail", near="Toronto")


def test_get_trail_with_unknown_id_raises():
    with pytest.raises(TrailNotFoundError):
        get_trail("nope")
