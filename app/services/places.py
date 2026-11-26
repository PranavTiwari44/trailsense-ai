"""Turns a place name like "Toronto" into an area of the map to search for trails.

Uses Open-Meteo's free geocoding (same provider as the weather, no key needed).
"""

import logging
import math

import httpx

from app.config import get_settings
from app.services.errors import DataUnavailableError
from app.services.http import make_client

logger = logging.getLogger(__name__)

# An area of the map: (south, west, north, east). This is the order Overpass wants.
Area = tuple[float, float, float, float]

# The area we search when no place is given: Ottawa–Gatineau.
DEFAULT_AREA: Area = (45.20, -76.20, 45.75, -75.40)

# How far around a place we look for trails.
SEARCH_RADIUS_KM = 30


def area_around(lat: float, lon: float, radius_km: float = SEARCH_RADIUS_KM) -> Area:
    """A box around a point, reaching about radius_km north, south, east and west."""
    lat_step = radius_km / 111.0  # one degree of latitude is about 111 km everywhere
    # Degrees of longitude get shorter as you move away from the equator, so we need more of them.
    lon_step = radius_km / (111.0 * math.cos(math.radians(lat)))
    return (
        round(lat - lat_step, 4),
        round(lon - lon_step, 4),
        round(lat + lat_step, 4),
        round(lon + lon_step, 4),
    )


def is_inside(area: Area, lat: float, lon: float) -> bool:
    """True if the point is inside the area."""
    south, west, north, east = area
    return south <= lat <= north and west <= lon <= east


async def find_place(name: str) -> tuple[float, float] | None:
    """Look up the centre (lat, lon) of a city or town. Returns None if no place has that name.

    Raises DataUnavailableError if the geocoding website failed.
    """
    # "Toronto, Ontario" -> "Toronto": the search matches place names only.
    place_name = name.split(",")[0].strip()
    settings = get_settings()
    params = {"name": place_name, "count": 1, "language": "en", "format": "json"}
    try:
        async with make_client() as client:
            response = await client.get(f"{settings.geocoding_url}/search", params=params)
            response.raise_for_status()
        results = response.json().get("results") or []  # no "results" at all when nothing matched
    except (httpx.HTTPError, ValueError) as error:
        raise DataUnavailableError(f"Place search failed: {error!r}") from error

    if not results:
        return None
    return results[0]["latitude"], results[0]["longitude"]


async def search_area_for(near: str | None) -> Area:
    """The area to search for trails: around `near` if it's a known place, otherwise Ottawa–Gatineau.

    Parks and other non-towns (like "Gatineau Park") aren't in the geocoder, so they also
    fall back to the default area, which covers Gatineau Park.
    """
    if not near:
        return DEFAULT_AREA
    try:
        place = await find_place(near)
    except DataUnavailableError as error:
        logger.warning("%s; searching Ottawa–Gatineau instead", error)
        return DEFAULT_AREA
    if place is None:
        logger.info("No town called %r; searching Ottawa–Gatineau instead", near)
        return DEFAULT_AREA
    return area_around(place[0], place[1])
