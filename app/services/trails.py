"""Finds hiking trails by name.

First we ask OpenStreetMap (through the Overpass API). If that fails or is too
slow, we look in a small saved file of trails instead.

In OpenStreetMap one trail is usually stored as many short pieces that share
the same name. We join those pieces back into one trail.
"""

import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import httpx
from pydantic import BaseModel

from app.config import get_settings
from app.models import LatLon, TrailInfo
from app.services.elevation import distance_m
from app.services.errors import DataUnavailableError, TrailNotFoundError
from app.services.http import make_client

logger = logging.getLogger(__name__)

FALLBACK_FILE = Path(__file__).parent.parent / "data" / "trails_fallback.json"

# The area we search: Ottawa–Gatineau, as (south, west, north, east).
SEARCH_AREA = (45.20, -76.20, 45.75, -75.40)

# OpenStreetMap path types you can walk on. Roads are also tagged "highway"
# in OpenStreetMap, so without this filter "Mer Bleue" matched a road.
WALKING_PATH_TYPES = "path|footway|track|steps|bridleway"

# OpenStreetMap difficulty tags, from easiest to hardest.
DIFFICULTY_ORDER = [
    "hiking",
    "mountain_hiking",
    "demanding_mountain_hiking",
    "alpine_hiking",
    "demanding_alpine_hiking",
    "difficult_alpine_hiking",
]

# Words we ignore when searching, so "the Wolf Trail" searches for "wolf".
IGNORED_WORDS = {"the", "trail", "trails", "sentier", "path", "loop", "hike", "hiking", "de", "des", "du", "la", "le", "of"}

# Trails we've found so far, by id, so other parts of the app can look them up later.
_found_trails: dict[str, TrailInfo] = {}


# ---------- Small helpers ----------

def make_id(name: str) -> str:
    """Turn a trail name into a simple id: "Sentier du Lac-Pink" -> "sentier-du-lac-pink"."""
    # Remove accents first (é -> e), then replace anything that isn't a letter or digit with "-".
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def search_words(query: str) -> list[str]:
    """Pick the useful words out of a search: "The Wolf Trail" -> ["wolf"]."""
    words = re.findall(r"\w+", query.lower())  # split into words, dropping punctuation
    useful = [word for word in words if word not in IGNORED_WORDS]
    if useful:
        return useful
    return words  # the search was only ignored words (like "trail"), so keep them all


def name_matches(words: list[str], names: list[str | None]) -> bool:
    """True if every search word appears in at least one of the names."""
    all_names = " ".join(name.lower() for name in names if name)
    for word in words:
        if word not in all_names:
            return False
    return True


def trail_length_km(pieces: list[list[LatLon]]) -> float:
    """Add up the length of every piece of a trail, in km."""
    total_m = 0.0
    for piece in pieces:
        for i in range(1, len(piece)):
            total_m += distance_m(piece[i - 1], piece[i])
    return round(total_m / 1000, 2)


def hardest_difficulty(tags: list[str | None]) -> str | None:
    """Return the hardest OpenStreetMap difficulty tag in the list (None if there are none)."""
    hardest = None
    for tag in tags:
        if tag not in DIFFICULTY_ORDER:
            continue  # missing or unknown tag
        if hardest is None or DIFFICULTY_ORDER.index(tag) > DIFFICULTY_ORDER.index(hardest):
            hardest = tag
    return hardest


def points_of(piece: dict) -> list[LatLon]:
    """Get the (lat, lon) points of one OpenStreetMap piece, in order."""
    return [(p["lat"], p["lon"]) for p in piece["geometry"]]


# ---------- Searching OpenStreetMap ----------

def build_overpass_query(words: list[str], timeout_s: int) -> str:
    """Write the Overpass search: walking paths in our area whose name contains the words."""
    name_pattern = ".*".join(words)  # ["lac", "pink"] -> "lac.*pink"
    area = ",".join(str(value) for value in SEARCH_AREA)
    path_filter = f'["highway"~"^({WALKING_PATH_TYPES})$"]'
    # We check both the local name and the English name ("name:en").
    # `out tags geom` asks for each piece's tags and its points.
    return f"""[out:json][timeout:{timeout_s}];
(
  way{path_filter}["name"~"{name_pattern}",i]({area});
  way{path_filter}["name:en"~"{name_pattern}",i]({area});
);
out tags geom;"""


def join_pieces(elements: list[dict], words: list[str]) -> TrailInfo | None:
    """Join the OpenStreetMap pieces that share a name into one trail.

    If pieces from several different trails matched, we pick the longest trail.
    Returns None if nothing matched.
    """
    # Group the pieces by trail name.
    pieces_by_name: dict[str, list[dict]] = {}
    for element in elements:
        tags = element.get("tags", {})
        name = tags.get("name:en") or tags.get("name")  # prefer the English name
        if element.get("type") != "way" or not name or not element.get("geometry"):
            continue
        if not name_matches(words, [tags.get("name"), tags.get("name:en")]):
            continue
        if name not in pieces_by_name:
            pieces_by_name[name] = []
        pieces_by_name[name].append(element)

    if not pieces_by_name:
        return None

    # Pick the longest trail.
    best_name = None
    best_pieces = []
    best_length = -1.0
    for name, pieces in pieces_by_name.items():
        length = trail_length_km([points_of(piece) for piece in pieces])
        if length > best_length:
            best_name = name
            best_pieces = pieces
            best_length = length

    segments = [points_of(piece) for piece in best_pieces]
    start_lat, start_lon = segments[0][0]
    return TrailInfo(
        id=make_id(best_name),
        name=best_name,
        length_km=best_length,
        sac_scale=hardest_difficulty([piece["tags"].get("sac_scale") for piece in best_pieces]),
        start_lat=start_lat,
        start_lon=start_lon,
        source="overpass",
        segments=segments,
    )


async def search_overpass(name: str) -> TrailInfo | None:
    """Search OpenStreetMap for a trail by name.

    Returns None if nothing matched. Raises DataUnavailableError if Overpass failed or was too slow.
    """
    settings = get_settings()
    words = search_words(name)
    query = build_overpass_query(words, timeout_s=int(settings.overpass_timeout_s))
    try:
        # Give the HTTP request a few seconds more than Overpass's own time limit.
        async with make_client(timeout_s=settings.overpass_timeout_s + 5) as client:
            response = await client.post(settings.overpass_url, data={"data": query})
            response.raise_for_status()
        elements = response.json()["elements"]
    except (httpx.HTTPError, ValueError, KeyError) as error:
        raise DataUnavailableError(f"Overpass failed: {error!r}") from error
    return join_pieces(elements, words)


# ---------- Searching the saved file ----------

class FallbackTrail(BaseModel):
    """One trail in the saved file, plus other names it can be found by (like its French name)."""

    aliases: list[str]
    trail: TrailInfo


@lru_cache  # read the file only once, then reuse what we read
def load_fallback_trails() -> list[FallbackTrail]:
    """Read the saved trails file. Returns an empty list if the file is missing."""
    if not FALLBACK_FILE.exists():
        logger.warning("Fallback file missing: %s", FALLBACK_FILE)
        return []
    data = json.loads(FALLBACK_FILE.read_text(encoding="utf-8"))
    return [FallbackTrail.model_validate(item) for item in data["trails"]]


def search_fallback(name: str) -> TrailInfo | None:
    """Look for a trail by name (or other name) in the saved file. Returns None if not found."""
    words = search_words(name)
    for saved in load_fallback_trails():
        if name_matches(words, [saved.trail.name] + saved.aliases):
            return saved.trail
    return None


# ---------- What the rest of the app calls ----------

async def find_trail(name: str) -> TrailInfo:
    """Find a trail by name: OpenStreetMap first, then the saved file.

    Raises TrailNotFoundError if neither has it.
    """
    trail = None
    try:
        trail = await search_overpass(name)
        if trail is None:
            logger.info("Overpass found no trail matching %r; trying the saved file", name)
    except DataUnavailableError as error:
        logger.warning("%s; using the saved file", error)

    if trail is None:
        trail = search_fallback(name)
    if trail is None:
        raise TrailNotFoundError(f"No trail found matching {name!r}")

    _found_trails[trail.id] = trail
    return trail


def get_trail(trail_id: str) -> TrailInfo:
    """Get a trail by id: one we found earlier, or one from the saved file.

    Raises TrailNotFoundError if we don't know that id.
    """
    if trail_id in _found_trails:
        return _found_trails[trail_id]
    for saved in load_fallback_trails():
        if saved.trail.id == trail_id:
            return saved.trail
    raise TrailNotFoundError(f"Unknown trail id {trail_id!r}. Call find_trail first.")
