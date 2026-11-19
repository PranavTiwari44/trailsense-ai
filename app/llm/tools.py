"""The tools the AI can ask us to run, and the code that runs them.

Each tool has two parts:
  - an input model (a Pydantic class): checks the AI's inputs, and also gives
    us the JSON Schema we show the AI, so the two can never disagree
  - a run_... function: calls our real service and returns plain data

run_tool() is the only thing the agent calls. It never crashes on a bad
tool call: it returns an error message the AI can read and react to.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.llm.base import ToolCall, ToolSpec
from app.services.elevation import get_elevation_profile
from app.services.errors import DataUnavailableError, TrailNotFoundError
from app.services.trail_difficulty import get_trail_difficulty
from app.services.trails import find_trail, get_trail
from app.services.weather import get_weather

logger = logging.getLogger(__name__)


# ---------- find_trail ----------

class FindTrailInput(BaseModel):
    """The inputs for find_trail: the trail's name."""

    name: str = Field(min_length=1, description="The trail's name, e.g. 'Wolf Trail'")


FIND_TRAIL_TOOL = ToolSpec(
    name="find_trail",
    description=(
        "Find a hiking trail by name. Returns its id (needed by the other trail tools), "
        "length in km, OpenStreetMap difficulty tag (sac_scale) and the lat/lon of its start."
    ),
    parameters=FindTrailInput.model_json_schema(),
)


async def run_find_trail(arguments: dict[str, Any]) -> dict[str, Any]:
    """Check the inputs, look the trail up, and return its details (without the map points)."""
    inputs = FindTrailInput.model_validate(arguments)
    trail = await find_trail(inputs.name)
    # The map points are hundreds of numbers the AI doesn't need, so we leave them out.
    return trail.model_dump(mode="json", exclude={"segments"})


# ---------- get_elevation_gain and get_difficulty_score (both take a trail id) ----------

class TrailIdInput(BaseModel):
    """The inputs for tools that work on a trail we already found: its id."""

    trail_id: str = Field(min_length=1, description="The trail id from find_trail, e.g. 'wolf-trail'")


ELEVATION_TOOL = ToolSpec(
    name="get_elevation_gain",
    description="Get how much climbing a trail has, in metres, plus its lowest and highest points.",
    parameters=TrailIdInput.model_json_schema(),
)


async def run_get_elevation_gain(arguments: dict[str, Any]) -> dict[str, Any]:
    """Check the inputs, measure the trail's climb, and return it."""
    inputs = TrailIdInput.model_validate(arguments)
    trail = get_trail(inputs.trail_id)
    profile = await get_elevation_profile(trail.segments)
    return profile.model_dump(mode="json")


DIFFICULTY_TOOL = ToolSpec(
    name="get_difficulty_score",
    description=(
        "Get the official difficulty of a trail right now: a score from 0 to 100, a level "
        "(Easy, Moderate, Hard, Dangerous or Unknown) and the reasons. "
        "Always use this for the difficulty level; never guess it."
    ),
    parameters=TrailIdInput.model_json_schema(),
)


async def run_get_difficulty_score(arguments: dict[str, Any]) -> dict[str, Any]:
    """Check the inputs, run our rules engine on the trail with live data, and return its decision."""
    inputs = TrailIdInput.model_validate(arguments)
    result = await get_trail_difficulty(inputs.trail_id)
    return result.model_dump(mode="json")


# ---------- get_weather ----------

class GetWeatherInput(BaseModel):
    """The inputs for get_weather: a spot on the map."""

    lat: float = Field(ge=-90, le=90, description="Latitude of the spot, e.g. 45.5")
    lon: float = Field(ge=-180, le=180, description="Longitude of the spot, e.g. -75.9")


WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description=(
        "Get the weather at a spot: right now, the last 6 hours and the next 6 hours "
        "(temperature, rain, snow, wind and weather code)."
    ),
    parameters=GetWeatherInput.model_json_schema(),
)


async def run_get_weather(arguments: dict[str, Any]) -> dict[str, Any]:
    """Check the inputs, fetch the weather, and return it as plain JSON-ready data."""
    inputs = GetWeatherInput.model_validate(arguments)
    weather = await get_weather(inputs.lat, inputs.lon)
    return weather.model_dump(mode="json")


# ---------- The list of tools, and running them ----------

# What we show the AI.
TOOL_SPECS: list[ToolSpec] = [FIND_TRAIL_TOOL, WEATHER_TOOL, ELEVATION_TOOL, DIFFICULTY_TOOL]

# Which function runs each tool, looked up by the tool's name.
TOOL_FUNCTIONS = {
    "find_trail": run_find_trail,
    "get_weather": run_get_weather,
    "get_elevation_gain": run_get_elevation_gain,
    "get_difficulty_score": run_get_difficulty_score,
}


def error_result(message: str) -> str:
    """A tool result that tells the AI something went wrong, as JSON text."""
    return json.dumps({"error": message})


async def run_tool(call: ToolCall) -> str:
    """Run the tool the AI asked for and return its result as JSON text.

    Problems (unknown tool, bad inputs, a website down) come back as
    {"error": "..."} instead of crashing, so the AI can try again or explain.
    """
    tool_function = TOOL_FUNCTIONS.get(call.name)
    if tool_function is None:
        return error_result(f"Unknown tool '{call.name}'. Available tools: {list(TOOL_FUNCTIONS)}")

    try:
        result = await tool_function(call.arguments)
    except ValidationError as error:
        return error_result(f"Bad inputs for {call.name}: {error}")
    except (DataUnavailableError, TrailNotFoundError) as error:
        return error_result(str(error))

    return json.dumps(result)
