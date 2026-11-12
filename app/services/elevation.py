"""Works out how much climbing a trail has, using Open-Meteo's elevation API."""

import asyncio
import math

import httpx

from app.config import get_settings
from app.models import ElevationProfile, LatLon
from app.services.errors import DataUnavailableError
from app.services.http import make_client

POINTS_PER_REQUEST = 100  # Open-Meteo answers at most 100 points per request
MIN_GAP_M = 100  # their height data is on a ~90 m grid, so points closer than this add nothing
EARTH_RADIUS_M = 6_371_000


def distance_m(a: LatLon, b: LatLon) -> float:
    """Distance in metres between two (lat, lon) points.

    Uses the haversine formula, the usual way to measure distance on a round Earth.
    """
    lat1 = math.radians(a[0])
    lon1 = math.radians(a[1])
    lat2 = math.radians(b[0])
    lon2 = math.radians(b[1])
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def thin_points(points: list[LatLon], min_gap_m: float) -> list[LatLon]:
    """Drop points that are closer than `min_gap_m` to the previous kept point.

    Always keeps the first and last point, so the piece keeps its full length.
    """
    if len(points) <= 2:
        return points
    kept = [points[0]]
    for point in points[1:-1]:  # every point except the first and last
        if distance_m(kept[-1], point) >= min_gap_m:
            kept.append(point)
    kept.append(points[-1])
    return kept


async def fetch_heights(client: httpx.AsyncClient, points: list[LatLon]) -> list[float]:
    """Ask Open-Meteo for the height in metres of each point (at most 100 points)."""
    settings = get_settings()
    lats = ",".join(f"{lat:.5f}" for lat, lon in points)
    lons = ",".join(f"{lon:.5f}" for lat, lon in points)
    response = await client.get(f"{settings.open_meteo_url}/elevation", params={"latitude": lats, "longitude": lons})
    response.raise_for_status()
    heights = response.json()["elevation"]
    if len(heights) != len(points):
        raise ValueError(f"asked for {len(points)} heights, got {len(heights)}")
    return heights


async def get_elevation_profile(segments: list[list[LatLon]]) -> ElevationProfile:
    """Work out the total climb of a trail made of one or more pieces.

    Raises DataUnavailableError if there are no points or Open-Meteo fails.
    """
    # 1. Drop points that are too close together, so we ask for fewer heights.
    thinned = [thin_points(piece, MIN_GAP_M) for piece in segments]
    all_points = []
    for piece in thinned:
        all_points.extend(piece)
    if len(all_points) < 2:
        raise DataUnavailableError("Trail has no points to measure elevation on")

    # 2. Split the points into groups of 100 and ask for all groups at the same time.
    groups = []
    for start in range(0, len(all_points), POINTS_PER_REQUEST):
        groups.append(all_points[start : start + POINTS_PER_REQUEST])
    try:
        async with make_client() as client:
            requests = [fetch_heights(client, group) for group in groups]
            results = await asyncio.gather(*requests)  # waits until every group has answered
    except (httpx.HTTPError, ValueError, KeyError) as error:
        raise DataUnavailableError(f"Elevation unavailable: {error}") from error

    heights = []
    for group_heights in results:
        heights.extend(group_heights)

    # 3. Add up every step up AND every step down, one piece at a time.
    #    (We don't join the end of one piece to the start of the next.)
    total_change = 0.0
    position = 0
    for piece in thinned:
        piece_heights = heights[position : position + len(piece)]
        position += len(piece)
        for i in range(1, len(piece_heights)):
            total_change += abs(piece_heights[i] - piece_heights[i - 1])

    # 4. OpenStreetMap pieces can point either way, so we can't tell up from down.
    #    On a loop you climb exactly as much as you descend, so climb = total change / 2.
    #    On a one-way trail this is the average of walking it in either direction.
    return ElevationProfile(
        gain_m=round(total_change / 2),
        min_m=min(heights),
        max_m=max(heights),
        samples=len(heights),
    )
