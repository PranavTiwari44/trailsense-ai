"""Rebuilds app/data/trails_fallback.json from live OpenStreetMap data.

Run from the project root:  uv run python -m scripts.build_fallback

The saved file is a copy of real OpenStreetMap data, so the app still works
when Overpass is slow or down, and nobody has to type trail numbers by hand.
"""

import asyncio
import json
from datetime import date

from app.models import TrailInfo
from app.services.elevation import thin_points
from app.services.places import DEFAULT_AREA
from app.services.trails import FALLBACK_FILE, search_overpass

# Each entry: (name to search for, other names the trail can be found by).
# Mer Bleue isn't here: its boardwalk isn't a named walking path in OpenStreetMap.
TRAILS = [
    ("Wolf Trail", ["Sentier des Loups"]),
    ("King Mountain", ["Sentier King Mountain"]),
    ("Luskville Falls", ["Chutes de Luskville"]),
    ("Pink Lake", ["Lac Pink", "Lac-Pink"]),
]
ATTEMPTS = 3


async def fetch(name: str) -> TrailInfo | None:
    """Search Overpass for a trail, trying up to 3 times because Overpass is often busy."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return await search_overpass(name, DEFAULT_AREA)
        except Exception as error:
            print(f"  attempt {attempt} failed: {error}")
            await asyncio.sleep(5)
    return None


async def main() -> None:
    """Fetch every trail in TRAILS and save them all to the fallback file."""
    saved = []
    for name, aliases in TRAILS:
        print(f"Searching {name!r}...")
        trail = await fetch(name)
        if trail is None:
            print("  -> not found, skipped")
            continue
        # Keep the length worked out from the full data, but store fewer points so the file stays small.
        smaller_pieces = [thin_points(piece, 30) for piece in trail.segments]
        trail = trail.model_copy(update={"source": "fallback", "segments": smaller_pieces})
        print(f"  -> {trail.name}: {trail.length_km} km, sac={trail.sac_scale}, {len(trail.segments)} pieces")
        saved.append({"aliases": aliases, "trail": trail.model_dump()})

    if not saved:
        print("No trails fetched (Overpass busy?). Kept the existing file. Try again later.")
        return

    data = {
        "generated_at": date.today().isoformat(),
        "note": "Copy of OpenStreetMap data (c) OpenStreetMap contributors, ODbL. Rebuild with scripts/build_fallback.py.",
        "trails": saved,
    }
    FALLBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    FALLBACK_FILE.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(saved)} trails to {FALLBACK_FILE}")


if __name__ == "__main__":  # only runs when this file is run directly, not when imported
    asyncio.run(main())
