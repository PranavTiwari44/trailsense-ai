"""Gets the weather for a spot from Open-Meteo (free, no key needed)."""

from datetime import datetime, timedelta, timezone

import httpx
from pydantic import ValidationError

from app.config import get_settings
from app.models import Weather, WeatherHour
from app.services.errors import DataUnavailableError
from app.services.http import make_client

# The weather values we ask Open-Meteo for.
OPEN_METEO_FIELDS = "temperature_2m,precipitation,rain,snowfall,wind_speed_10m,weather_code"


async def get_weather(lat: float, lon: float) -> Weather:
    """Get the weather right now, for the last 6 hours and for the next 6 hours.

    Raises DataUnavailableError if Open-Meteo fails or sends back bad data.
    """
    settings = get_settings()
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": OPEN_METEO_FIELDS,
        "hourly": OPEN_METEO_FIELDS,
        "past_hours": 6,
        "forecast_hours": 6,
        "timezone": "auto",  # give us times in the trail's local time
    }
    try:
        async with make_client() as client:
            response = await client.get(f"{settings.open_meteo_url}/forecast", params=params)
            response.raise_for_status()  # turn 4xx/5xx answers into an error
        return parse_weather(response.json(), lat, lon)
    except (httpx.HTTPError, ValueError, KeyError, ValidationError) as error:
        raise DataUnavailableError(f"Weather unavailable for ({lat}, {lon}): {error}") from error


def parse_weather(data: dict, lat: float, lon: float) -> Weather:
    """Turn Open-Meteo's JSON answer into our Weather object."""
    # Open-Meteo sends times like "2026-11-12T01:00" with no timezone,
    # plus a separate number: how many seconds the local time is from UTC.
    local_tz = timezone(timedelta(seconds=data["utc_offset_seconds"]))

    current = read_current(data["current"], local_tz)

    # Sort each hour into "recent" (before this hour) or "next" (this hour and later).
    this_hour = current.time.replace(minute=0, second=0, microsecond=0)
    hourly = data["hourly"]
    recent_hours = []
    next_hours = []
    for i in range(len(hourly["time"])):
        hour = read_hour(hourly, i, local_tz)
        if hour.time < this_hour:
            recent_hours.append(hour)
        else:
            next_hours.append(hour)

    return Weather(lat=lat, lon=lon, current=current, recent_hours=recent_hours, next_hours=next_hours)


def read_time(text: str, local_tz: timezone) -> datetime:
    """Read a time like "2026-11-12T01:00" and attach the local timezone to it."""
    return datetime.fromisoformat(text).replace(tzinfo=local_tz)


def read_current(current: dict, local_tz: timezone) -> WeatherHour:
    """Build a WeatherHour from the "current" part of Open-Meteo's answer."""
    return WeatherHour(
        time=read_time(current["time"], local_tz),
        temp_c=current["temperature_2m"],
        precip_mm=current["precipitation"],
        rain_mm=current["rain"],
        snowfall_cm=current["snowfall"],
        wind_kmh=current["wind_speed_10m"],
        weather_code=current["weather_code"],
    )


def read_hour(hourly: dict, i: int, local_tz: timezone) -> WeatherHour:
    """Build a WeatherHour for hour number `i` from the "hourly" part of the answer.

    Open-Meteo sends hourly data as one list per value: all the times, all the
    temperatures, and so on. Item `i` of every list belongs to the same hour.
    """
    return WeatherHour(
        time=read_time(hourly["time"][i], local_tz),
        temp_c=hourly["temperature_2m"][i],
        precip_mm=hourly["precipitation"][i],
        rain_mm=hourly["rain"][i],
        snowfall_cm=hourly["snowfall"][i],
        wind_kmh=hourly["wind_speed_10m"][i],
        weather_code=hourly["weather_code"][i],
    )
