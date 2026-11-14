"""Gets everything the rules engine needs for one trail, then runs it.

Used by GET /difficulty now, and by the AI's get_difficulty_score tool later.
"""

import asyncio
import logging

from app.engine.difficulty import score_difficulty
from app.models import DifficultyResult, TrailInfo, Weather
from app.services.elevation import get_elevation_profile
from app.services.errors import DataUnavailableError
from app.services.trails import get_trail
from app.services.weather import get_weather

logger = logging.getLogger(__name__)


async def weather_or_none(trail: TrailInfo) -> Weather | None:
    """Get the weather at the trail's start, or None if the weather website failed."""
    try:
        return await get_weather(trail.start_lat, trail.start_lon)
    except DataUnavailableError as error:
        logger.warning("%s", error)
        return None


async def climb_or_none(trail: TrailInfo) -> float | None:
    """Get the trail's climb in metres, or None if the elevation website failed."""
    try:
        profile = await get_elevation_profile(trail.segments)
        return profile.gain_m
    except DataUnavailableError as error:
        logger.warning("%s", error)
        return None


async def get_trail_difficulty(trail_id: str) -> DifficultyResult:
    """Look up a trail, fetch its weather and climb, and score it with the rules engine.

    If the weather or climb can't be fetched, the engine gets None and answers "Unknown".
    Raises TrailNotFoundError if the trail id is unknown.
    """
    trail = get_trail(trail_id)
    # Weather and climb don't depend on each other, so we fetch both at the same time.
    weather, climb_m = await asyncio.gather(weather_or_none(trail), climb_or_none(trail))
    return score_difficulty(trail, climb_m, weather)
