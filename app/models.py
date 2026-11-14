"""The data shapes used across the app.

Each class lists its fields and their types. Pydantic checks the values when an
object is created and can turn the object into JSON.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# A point on the map: (latitude, longitude).
LatLon = tuple[float, float]


class TrailInfo(BaseModel):
    """One hiking trail: its name, length, difficulty tag and where it starts."""

    id: str  # short id like "wolf-trail"
    name: str
    length_km: float
    sac_scale: str | None = None  # OpenStreetMap difficulty tag, e.g. "mountain_hiking"
    start_lat: float
    start_lon: float
    source: Literal["overpass", "fallback"]  # where we got it: live OpenStreetMap or our saved file
    # The trail's points, as a list of pieces. Each piece is a list of points in order.
    # repr=False stops these hundreds of points from showing up when we print a trail.
    segments: list[list[LatLon]] = Field(default_factory=list, repr=False)


class WeatherHour(BaseModel):
    """The weather for one hour (also used for "right now")."""

    time: datetime
    temp_c: float
    precip_mm: float  # rain + melted snow in that hour
    rain_mm: float
    snowfall_cm: float
    wind_kmh: float
    weather_code: int  # standard weather code; 95 to 99 means thunderstorm


class Weather(BaseModel):
    """The weather at one spot: right now, the last 6 hours and the next 6 hours."""

    lat: float
    lon: float
    current: WeatherHour
    recent_hours: list[WeatherHour]  # used to spot ice: rain earlier, freezing now
    next_hours: list[WeatherHour]


# The difficulty levels. "Unknown" is used when data is missing (never guess "Easy").
Level = Literal["Easy", "Moderate", "Hard", "Dangerous", "Unknown"]


class DifficultyResult(BaseModel):
    """What the rules engine decided: a score, a level, and the reasons for it."""

    score: int | None  # 0 to 100; None when the level is "Unknown"
    level: Level
    reasons: list[str]  # plain sentences, e.g. "8 km long (+8)"
    data_time: datetime | None  # time of the weather reading used; None if there was no weather


class ElevationProfile(BaseModel):
    """How much climbing a trail has, plus its lowest and highest points."""

    gain_m: float
    min_m: float
    max_m: float
    samples: int  # how many points we looked up
