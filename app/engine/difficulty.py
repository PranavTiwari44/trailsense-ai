"""The difficulty rules engine.

Takes trail, elevation and weather data and works out how hard the hike is.
Plain functions only: no web calls, no AI. Same input always gives the same answer,
which is what makes it easy to test.
"""

from app.models import DifficultyResult, Level, TrailInfo, Weather

# ---------- The rules (from CLAUDE.md section 4) ----------

MAX_LENGTH_POINTS = 25  # +1 per whole km, at most 25
METRES_PER_CLIMB_POINT = 50  # +1 per 50 m of climb...
MAX_CLIMB_POINTS = 25  # ...at most 25

# Points for each OpenStreetMap difficulty tag. A missing or unknown tag counts as 0.
SAC_POINTS = {
    "hiking": 0,
    "mountain_hiking": 10,
    "demanding_mountain_hiking": 20,
    "alpine_hiking": 30,
    "demanding_alpine_hiking": 30,
    "difficult_alpine_hiking": 30,
}

HEAVY_RAIN_MM = 2.0  # more than this in an hour: +10 (slippery)
RAIN_POINTS = 10
ICE_POINTS = 15  # any snow, OR below 0 °C after recent rain: +15 (counted once)
STRONG_WIND_KMH = 50.0  # more than this: +10
WIND_POINTS = 10
HOT_TEMP_C = 30.0  # more than this: +10
HEAT_POINTS = 10
THUNDERSTORM_CODE = 95  # weather codes 95 and up are thunderstorms -> always "Dangerous"

MAX_SCORE = 100


# ---------- Your code goes below ----------
# Write each function body. Run `uv run pytest tests/test_difficulty.py -v` as you go.
# Tip: start with level_for_score (the easiest), then trail_points, then weather_points,
# then has_thunderstorm, and score_difficulty last (it uses all the others).


def level_for_score(score: int) -> Level:
    """Turn a score into a level: 0-24 Easy, 25-49 Moderate, 50-74 Hard, 75+ Dangerous."""
    if score <= 24:
        return "Easy"
    elif score <= 49:
        return "Moderate"
    elif score <= 74:
        return "Hard"
    else:
        return "Dangerous"

def trail_points(trail: TrailInfo, elevation_gain_m: float) -> tuple[int, list[str]]:
    """Points for the trail itself: length, climb and the OpenStreetMap difficulty tag.

    Returns (points, reasons). Each rule that adds points should add one reason,
    e.g. "8 km long (+8)". Count whole km and whole 50 m steps only (8.9 km -> 8 points).
    """
    points = 0
    reasons = []
    points += min(MAX_LENGTH_POINTS, int(trail.length_km))
    reasons.append(f"{trail.length_km} km long (+{min(MAX_LENGTH_POINTS, int(trail.length_km))})")

    points += min(MAX_CLIMB_POINTS, int(elevation_gain_m / METRES_PER_CLIMB_POINT))
    reasons.append(f"{elevation_gain_m} m of climb (+{min(MAX_CLIMB_POINTS, int(elevation_gain_m / METRES_PER_CLIMB_POINT))})")

    if trail.sac_scale is not None:
        sac_points = SAC_POINTS.get(trail.sac_scale, 0)
        points += sac_points
        if sac_points > 0:
            reasons.append(f"OSM difficulty tag '{trail.sac_scale}' (+{sac_points})")

    return points, reasons


def weather_points(weather: Weather) -> tuple[int, list[str]]:
    """Points for the weather: rain, ice, wind and heat.

    Look at the current reading AND the next hours. A rule counts if ANY of those
    hours breaks it, and each rule adds its points only once.
    Ice = any snow in those hours, OR below 0 °C in those hours after any rain
    in `weather.recent_hours`.
    Returns (points, reasons), one reason per rule that added points.
    """
    reasons = []
    points = 0

    hours_to_check = [weather.current] + weather.next_hours

    rained = False
    hotTemp = False
    strongWind = False
    heavyRain = False
    ice = False

    for hour in weather.recent_hours:
        if hour.rain_mm > 0:
            rained = True
            break

    for hour in hours_to_check:
        if hour.temp_c > HOT_TEMP_C:
            hotTemp = True
        if hour.wind_kmh > STRONG_WIND_KMH:
            strongWind = True
        if hour.rain_mm > HEAVY_RAIN_MM:
            heavyRain = True
        if hour.snowfall_cm > 0 or (rained and hour.temp_c < 0):
            ice = True

    if heavyRain:
        points += RAIN_POINTS
        reasons.append(f"Heavy rain, more than {HEAVY_RAIN_MM} mm/h (+{RAIN_POINTS})")
    if hotTemp:
        points += HEAT_POINTS
        reasons.append(f"Hot temperature (+{HEAT_POINTS})")
    if strongWind:
        points += WIND_POINTS
        reasons.append(f"Strong wind, more than {STRONG_WIND_KMH} km/h (+{WIND_POINTS})")
    if ice:
        points += ICE_POINTS
        reasons.append(f"Ice conditions (+{ICE_POINTS})")

    return points, reasons

def has_thunderstorm(weather: Weather) -> bool:
    """True if the current reading or any of the next hours has a thunderstorm code."""

    hours_to_check = [weather.current] + weather.next_hours
    for hour in hours_to_check:
        if hour.weather_code >= THUNDERSTORM_CODE:
            return True
    return False


def score_difficulty(
    trail: TrailInfo | None,
    elevation_gain_m: float | None,
    weather: Weather | None,
) -> DifficultyResult:
    """Work out the difficulty of a hike. This is the function the rest of the app calls.

    - If the trail, the climb or the weather is missing: level "Unknown", score None,
      and a reason saying what's missing. Never guess "Easy".
    - Otherwise: add trail points and weather points, cap the score at 100,
      and pick the level from the score.
    - A thunderstorm makes the level "Dangerous" whatever the score (keep the score as it is,
      and add a reason about the thunderstorm).
    - data_time is the time of the current weather reading.
    """

    reasons = []
    score = 0
    data_time = None

    if trail is None:
        reasons.append("Trail data missing")
    if elevation_gain_m is None:
        reasons.append("Elevation gain data missing")
    if weather is None:
        reasons.append("Weather data missing")

    if reasons:
        return DifficultyResult(score=None, level="Unknown", reasons=reasons, data_time=data_time)

    trail_score, trail_reasons = trail_points(trail, elevation_gain_m)
    weather_score, weather_reasons = weather_points(weather)

    score = min(MAX_SCORE, trail_score + weather_score)
    reasons.extend(trail_reasons)
    reasons.extend(weather_reasons)

    if has_thunderstorm(weather):
        level = "Dangerous"
        reasons.append("Thunderstorm detected")
    else:
        level = level_for_score(score)

    data_time = weather.current.time

    return DifficultyResult(score=score, level=level, reasons=reasons, data_time=data_time)
    
