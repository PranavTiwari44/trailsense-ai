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


SAFETY_NOTE = "Check official park conditions before hiking. This is information, not a safety guarantee."


class AIAnswer(BaseModel):
    """What the AI writes at the end: the words only.

    We show this shape to the AI as a JSON Schema and it must reply in exactly
    this shape. The numbers (score, data time) are added by our code later.
    """

    trail_name: str | None = Field(description="The trail's name from find_trail, or null if no trail was found")
    level: Level = Field(description="Copy the level from get_difficulty_score. Unknown if it is missing")
    summary: str = Field(description="One or two sentences explaining how hard the trail is right now, and why")
    reasons: list[str] = Field(description="Short facts from the tool results, e.g. '8.9 km long'")
    warnings: list[str] = Field(description="Dangers to watch for. Empty list if none")
    gear_tips: list[str] = Field(description="What to bring for these conditions")


class ExplainResponse(BaseModel):
    """The final answer POST /explain sends back: the AI's words plus the rules engine's numbers."""

    trail_name: str | None
    level: Level  # always the rules engine's level, never the AI's
    score: int | None  # from the rules engine; None when the level is "Unknown"
    summary: str
    reasons: list[str]
    warnings: list[str]
    gear_tips: list[str]
    data_time: datetime | None  # time of the weather reading the engine used
    note: str = SAFETY_NOTE


class ExplainRequest(BaseModel):
    """What someone sends to POST /explain: their question."""

    question: str = Field(min_length=3, max_length=500,
                          examples=["How hard is the Wolf Trail in Gatineau Park right now?"])


class ElevationProfile(BaseModel):
    """How much climbing a trail has, plus its lowest and highest points."""

    gain_m: float
    min_m: float
    max_m: float
    samples: int  # how many points we looked up
