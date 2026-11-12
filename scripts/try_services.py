"""Tries the data code against the REAL websites and prints what comes back.

Run from the project root:  uv run python -m scripts.try_services "Wolf Trail"
"""

import asyncio
import logging
import sys
import time

from app.services.elevation import get_elevation_profile
from app.services.trails import find_trail
from app.services.weather import get_weather


async def main(name: str) -> None:
    """Find a trail, then get its weather and climb, printing each result and how long it took."""
    started = time.perf_counter()
    trail = await find_trail(name)
    print(f"[{time.perf_counter() - started:.1f}s] TRAIL   {trail!r}")

    started = time.perf_counter()
    # Weather and elevation only need the trail, so we ask for both at the same time.
    weather, elevation = await asyncio.gather(
        get_weather(trail.start_lat, trail.start_lon),
        get_elevation_profile(trail.segments),
    )
    print(f"[{time.perf_counter() - started:.1f}s] ELEV    {elevation!r}")
    now = weather.current
    print(f"          WEATHER now {now.time:%H:%M}: {now.temp_c} C, wind {now.wind_kmh} km/h, code {now.weather_code}")
    print(f"          next {len(weather.next_hours)} h precip: {[h.precip_mm for h in weather.next_hours]}")
    print(f"          past {len(weather.recent_hours)} h precip: {[h.precip_mm for h in weather.recent_hours]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) > 1:
        trail_name = sys.argv[1]
    else:
        trail_name = "Wolf Trail"
    asyncio.run(main(trail_name))
