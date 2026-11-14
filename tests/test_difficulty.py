"""Tests for the difficulty rules engine. No internet and no AI: just data in, answer out."""

from datetime import datetime, timedelta, timezone

import pytest

from app.engine.difficulty import level_for_score, score_difficulty
from app.models import TrailInfo, Weather, WeatherHour

NOW = datetime(2026, 11, 14, 10, 0, tzinfo=timezone(timedelta(hours=-5)))


# ---------- Helpers to build test data ----------

def make_trail(length_km: float = 2.0, sac_scale: str | None = "hiking") -> TrailInfo:
    """A simple test trail. Change only the parts a test cares about."""
    return TrailInfo(
        id="test-trail", name="Test Trail", length_km=length_km, sac_scale=sac_scale,
        start_lat=45.5, start_lon=-75.9, source="fallback",
    )


def make_hour(hours_from_now: int, temp_c: float, rain_mm: float, snowfall_cm: float,
              wind_kmh: float, weather_code: int) -> WeatherHour:
    """One hour of test weather.

    precip_mm is rain + snow (as water), like Open-Meteo. We use the snow cm as a rough stand-in for its water.
    """
    return WeatherHour(
        time=NOW + timedelta(hours=hours_from_now), temp_c=temp_c, precip_mm=rain_mm + snowfall_cm,
        rain_mm=rain_mm, snowfall_cm=snowfall_cm, wind_kmh=wind_kmh, weather_code=weather_code,
    )


def make_weather(temp_c: float = 15.0, rain_mm: float = 0.0, snowfall_cm: float = 0.0,
                 wind_kmh: float = 10.0, weather_code: int = 0, recent_rain_mm: float = 0.0) -> Weather:
    """Test weather: now and the next 6 hours all have the same conditions.

    The defaults are a mild, dry, calm day. `recent_rain_mm` is the rain in each of the last 6 hours.
    """
    current = make_hour(0, temp_c, rain_mm, snowfall_cm, wind_kmh, weather_code)
    next_hours = []
    for h in range(6):
        next_hours.append(make_hour(h, temp_c, rain_mm, snowfall_cm, wind_kmh, weather_code))
    recent_hours = []
    for h in range(-6, 0):
        recent_hours.append(make_hour(h, 5.0, recent_rain_mm, 0.0, 10.0, 0))
    return Weather(lat=45.5, lon=-75.9, current=current, recent_hours=recent_hours, next_hours=next_hours)


def reasons_text(result) -> str:
    """All the reasons joined into one lowercase string, so tests can look for a word."""
    return " ".join(result.reasons).lower()


# ---------- The 4 cases from the spec ----------

def test_short_flat_trail_sunny_is_easy():
    result = score_difficulty(make_trail(length_km=2.0), elevation_gain_m=20, weather=make_weather())

    assert result.level == "Easy"
    assert result.score == 2  # 2 km = 2 points, 20 m climb = 0, "hiking" = 0, nice weather = 0
    assert result.data_time == NOW
    assert len(result.reasons) > 0  # always explain


def test_long_steep_trail_with_heavy_rain_is_hard():
    trail = make_trail(length_km=20.0, sac_scale="mountain_hiking")

    result = score_difficulty(trail, elevation_gain_m=1000, weather=make_weather(rain_mm=5.0))

    assert result.score == 60  # 20 (length) + 20 (climb) + 10 (mountain_hiking) + 10 (rain)
    assert result.level == "Hard"
    assert "rain" in reasons_text(result)


def test_thunderstorm_is_dangerous_even_on_an_easy_trail():
    result = score_difficulty(make_trail(length_km=2.0), elevation_gain_m=20, weather=make_weather(weather_code=95))

    assert result.level == "Dangerous"
    assert result.score == 2  # the score stays the same; only the level changes
    assert "thunderstorm" in reasons_text(result)


def test_missing_weather_is_unknown():
    result = score_difficulty(make_trail(), elevation_gain_m=20, weather=None)

    assert result.level == "Unknown"
    assert result.score is None
    assert "weather" in reasons_text(result)


# ---------- More cases ----------

def test_missing_trail_is_unknown():
    result = score_difficulty(None, elevation_gain_m=20, weather=make_weather())

    assert result.level == "Unknown"
    assert result.score is None


def test_missing_climb_is_unknown():
    result = score_difficulty(make_trail(), elevation_gain_m=None, weather=make_weather())

    assert result.level == "Unknown"


def test_part_km_and_part_climb_steps_dont_count():
    result = score_difficulty(make_trail(length_km=8.9), elevation_gain_m=99, weather=make_weather())

    assert result.score == 9  # 8 whole km + 1 whole 50 m step


def test_length_and_climb_are_capped_at_25_each():
    result = score_difficulty(make_trail(length_km=100.0), elevation_gain_m=5000, weather=make_weather())

    assert result.score == 50  # 25 + 25, not 100 + 100
    assert result.level == "Hard"


def test_rain_of_exactly_2mm_does_not_count():
    result = score_difficulty(make_trail(length_km=2.0), elevation_gain_m=0, weather=make_weather(rain_mm=2.0))

    assert result.score == 2  # the rule is MORE than 2 mm


def test_freezing_after_recent_rain_is_ice():
    weather = make_weather(temp_c=-2.0, recent_rain_mm=1.0)

    result = score_difficulty(make_trail(length_km=2.0), elevation_gain_m=0, weather=weather)

    assert result.score == 17  # 2 + 15 (ice)
    assert "ice" in reasons_text(result)


def test_freezing_without_recent_rain_is_not_ice():
    result = score_difficulty(make_trail(length_km=2.0), elevation_gain_m=0, weather=make_weather(temp_c=-2.0))

    assert result.score == 2


def test_snow_is_ice():
    result = score_difficulty(make_trail(length_km=2.0), elevation_gain_m=0, weather=make_weather(snowfall_cm=0.5))

    assert result.score == 17


def test_heavy_snow_is_ice_not_heavy_rain():
    # 3 cm of snow means lots of precipitation, but none of it is rain.
    result = score_difficulty(make_trail(length_km=2.0), elevation_gain_m=0, weather=make_weather(snowfall_cm=3.0))

    assert result.score == 17  # 2 + 15 (ice), no rain points
    assert "rain" not in reasons_text(result)


def test_ice_points_count_only_once():
    # Snowing AND freezing after rain: still only +15, not +30.
    weather = make_weather(temp_c=-2.0, snowfall_cm=0.5, recent_rain_mm=1.0)

    result = score_difficulty(make_trail(length_km=2.0), elevation_gain_m=0, weather=weather)

    assert result.score == 17


def test_strong_wind_and_heat():
    weather = make_weather(temp_c=32.0, wind_kmh=60.0)

    result = score_difficulty(make_trail(length_km=2.0), elevation_gain_m=0, weather=weather)

    assert result.score == 22  # 2 + 10 (wind) + 10 (heat)


def test_score_never_goes_above_100():
    trail = make_trail(length_km=100.0, sac_scale="alpine_hiking")
    weather = make_weather(rain_mm=5.0, snowfall_cm=1.0, wind_kmh=80.0)

    result = score_difficulty(trail, elevation_gain_m=5000, weather=weather)

    assert result.score == 100  # 25 + 25 + 30 + 10 + 15 + 10 = 115, capped
    assert result.level == "Dangerous"


# parametrize runs the same test once for each (score, level) pair below.
@pytest.mark.parametrize(
    ("score", "expected_level"),
    [(0, "Easy"), (24, "Easy"), (25, "Moderate"), (49, "Moderate"),
     (50, "Hard"), (74, "Hard"), (75, "Dangerous"), (100, "Dangerous")],
)
def test_level_boundaries(score, expected_level):
    assert level_for_score(score) == expected_level
