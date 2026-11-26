"""Runs every saved scenario through the REAL AI and scores its answers.

The tools return each scenario's fixed data instead of calling the internet, so every
run sees exactly the same facts. Only the AI's behaviour can change between runs.
get_difficulty_score still runs our real rules engine on that fixed data.

Run from the project root (Ollama must be running):
    uv run python -m evals.run_evals                     # all scenarios
    uv run python -m evals.run_evals hot-day fresh-snow  # only these ones
"""

import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.config import get_settings
from app.engine.difficulty import score_difficulty
from app.llm import tools
from app.llm.agent import (
    AnswerError,
    add_weather_if_missing,
    ask_for_answer,
    build_response,
    get_engine_result,
    run_agent,
)
from app.llm.base import LLMClient, LLMError, Message
from app.llm.factory import get_llm_client
from app.llm.tools import TOOL_SPECS, FindTrailInput, GetWeatherInput, TrailIdInput
from app.models import Level, TrailInfo, Weather, WeatherHour
from app.services.errors import DataUnavailableError, TrailNotFoundError

SCENARIOS_FILE = Path(__file__).parent / "scenarios.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Every scenario happens at the same time and place, so runs can be compared.
NOW = datetime(2026, 11, 25, 10, 0, tzinfo=timezone(timedelta(hours=-5)))
START_LAT = 45.5
START_LON = -75.9


# ---------- The scenario file ----------

class ScenarioTrail(BaseModel):
    """The trail in a scenario: just the facts the AI and the engine need."""

    id: str
    name: str
    length_km: float
    sac_scale: str | None = None


class ScenarioWeather(BaseModel):
    """The weather in a scenario. It is the same for now and the next 6 hours."""

    temp_c: float
    rain_mm: float = 0.0
    snowfall_cm: float = 0.0
    wind_kmh: float = 10.0
    weather_code: int = 0
    recent_rain_mm: float = 0.0  # rain in each of the last 6 hours (for the ice rule)


class Scenario(BaseModel):
    """One saved situation, plus what a good answer must look like."""

    id: str
    question: str
    trail: ScenarioTrail | None  # None = the trail can't be found
    climb_m: float | None  # None = the elevation service is down
    weather: ScenarioWeather | None  # None = the weather service is down
    expected_level: Level
    must_mention: list[str]  # "a|b" means either word is fine


def load_scenarios() -> list[Scenario]:
    """Read every scenario from scenarios.json."""
    data = json.loads(SCENARIOS_FILE.read_text(encoding="utf-8"))
    scenarios = []
    for item in data["scenarios"]:
        scenarios.append(Scenario.model_validate(item))
    return scenarios


def make_trail(scenario: Scenario) -> TrailInfo | None:
    """Turn the scenario's trail into the TrailInfo our tools and engine use."""
    if scenario.trail is None:
        return None
    return TrailInfo(
        id=scenario.trail.id,
        name=scenario.trail.name,
        length_km=scenario.trail.length_km,
        sac_scale=scenario.trail.sac_scale,
        start_lat=START_LAT,
        start_lon=START_LON,
        source="fallback",
    )


def make_hour(hours_from_now: int, temp_c: float, rain_mm: float, snowfall_cm: float,
              wind_kmh: float, weather_code: int) -> WeatherHour:
    """One hour of scenario weather."""
    return WeatherHour(
        time=NOW + timedelta(hours=hours_from_now),
        temp_c=temp_c,
        precip_mm=rain_mm + snowfall_cm,
        rain_mm=rain_mm,
        snowfall_cm=snowfall_cm,
        wind_kmh=wind_kmh,
        weather_code=weather_code,
    )


def make_weather(scenario: Scenario) -> Weather | None:
    """Turn the scenario's weather into a full Weather: now, the last 6 hours and the next 6."""
    if scenario.weather is None:
        return None
    w = scenario.weather
    current = make_hour(0, w.temp_c, w.rain_mm, w.snowfall_cm, w.wind_kmh, w.weather_code)
    next_hours = []
    for hour in range(1, 7):
        next_hours.append(make_hour(hour, w.temp_c, w.rain_mm, w.snowfall_cm, w.wind_kmh, w.weather_code))
    recent_hours = []
    for hour in range(-6, 0):
        recent_hours.append(make_hour(hour, w.temp_c, w.recent_rain_mm, 0.0, w.wind_kmh, 3))
    return Weather(lat=START_LAT, lon=START_LON, current=current, recent_hours=recent_hours, next_hours=next_hours)


# ---------- Tools that return the scenario's fixed data ----------

def fixed_tools(scenario: Scenario) -> dict[str, Any]:
    """Make the four tools for one scenario. They check their inputs like the real tools,
    but return the scenario's data instead of calling the internet.
    """
    trail = make_trail(scenario)
    climb_m = scenario.climb_m
    weather = make_weather(scenario)

    def check_trail_id(trail_id: str) -> None:
        """Fail like the real tools do when the trail id is unknown."""
        if trail is None or trail_id != trail.id:
            raise TrailNotFoundError(f"Unknown trail id {trail_id!r}. Call find_trail first.")

    async def find_trail(arguments: dict[str, Any]) -> dict[str, Any]:
        """The scenario's trail, or 'not found' if it has none."""
        inputs = FindTrailInput.model_validate(arguments)
        if trail is None:
            raise TrailNotFoundError(f"No trail found matching {inputs.name!r}")
        return trail.model_dump(mode="json", exclude={"segments"})

    async def get_weather(arguments: dict[str, Any]) -> dict[str, Any]:
        """The scenario's weather, or 'unavailable' if it has none."""
        GetWeatherInput.model_validate(arguments)
        if weather is None:
            raise DataUnavailableError("Weather unavailable: the weather service did not answer")
        return weather.model_dump(mode="json")

    async def get_elevation_gain(arguments: dict[str, Any]) -> dict[str, Any]:
        """The scenario's climb, or 'unavailable' if it has none."""
        inputs = TrailIdInput.model_validate(arguments)
        check_trail_id(inputs.trail_id)
        if climb_m is None:
            raise DataUnavailableError("Elevation unavailable: the elevation service did not answer")
        return {"gain_m": climb_m, "min_m": 100.0, "max_m": 100.0 + climb_m, "samples": 50}

    async def get_difficulty_score(arguments: dict[str, Any]) -> dict[str, Any]:
        """Our REAL rules engine, run on the scenario's data."""
        inputs = TrailIdInput.model_validate(arguments)
        check_trail_id(inputs.trail_id)
        return score_difficulty(trail, climb_m, weather).model_dump(mode="json")

    return {
        "find_trail": find_trail,
        "get_weather": get_weather,
        "get_elevation_gain": get_elevation_gain,
        "get_difficulty_score": get_difficulty_score,
    }


# ---------- Checking an answer ----------

def numbers_in(text: str) -> list[str]:
    """Every number written in the text, e.g. "8.9 km and 360 m" -> ["8.9", "360"]."""
    return re.findall(r"\d+(?:\.\d+)?", text)


def is_in_data(number: str, data_numbers: list[float]) -> bool:
    """True if the number appears in the tool data, allowing for rounding (8.87 -> "8.9" or "9")."""
    value = float(number)
    decimals = 0
    if "." in number:
        decimals = len(number.split(".")[1])
    for data_number in data_numbers:
        if round(data_number, decimals) == value:
            return True
    return False


def invented_numbers(summary: str, tool_data: str) -> list[str]:
    """Numbers in the summary that don't appear anywhere in the tool results."""
    data_numbers = []
    for number in numbers_in(tool_data):
        data_numbers.append(float(number))
    invented = []
    for number in numbers_in(summary):
        if not is_in_data(number, data_numbers):
            invented.append(number)
    return invented


def missing_words(answer_text: str, must_mention: list[str]) -> list[str]:
    """Required words the answer never mentions. "a|b" is fine if either word is there."""
    text = answer_text.lower()
    missing = []
    for wanted in must_mention:
        found = False
        for option in wanted.split("|"):
            if option.lower() in text:
                found = True
        if not found:
            missing.append(wanted)
    return missing


def tool_data_text(messages: list[Message]) -> str:
    """All the tool results in the conversation, joined into one piece of text."""
    parts = []
    for message in messages:
        if message.role == "tool":
            parts.append(message.content)
    return "\n".join(parts)


def tools_called(messages: list[Message]) -> list[str]:
    """The names of the tools that were asked for, in order."""
    names = []
    for message in messages:
        for call in message.tool_calls:
            names.append(call.name)
    return names


# ---------- Running one scenario ----------

class ScenarioResult(BaseModel):
    """How the AI did on one scenario."""

    id: str
    expected_level: Level
    level: Level | None = None  # the final level (after the engine overrides the AI)
    ai_level: Level | None = None  # what the AI itself said
    valid_json: bool = False
    level_ok: bool = False
    ai_agreed: bool = False  # did the AI's own level match the engine? (not part of "passed")
    invented: list[str] = []
    missing: list[str] = []
    tools: list[str] = []
    summary: str = ""
    seconds: float = 0.0
    passed: bool = False


async def run_scenario(scenario: Scenario, client: LLMClient) -> ScenarioResult:
    """Run one scenario through the real AI with fixed tool data, and check the answer."""
    tools.TOOL_FUNCTIONS = fixed_tools(scenario)  # run_tool now uses this scenario's data
    result = ScenarioResult(id=scenario.id, expected_level=scenario.expected_level)
    started = time.perf_counter()

    # The same steps as explain() in agent.py, kept separate so we can see the AI's own level.
    agent_result = await run_agent(scenario.question, client, TOOL_SPECS)
    messages = agent_result.messages
    await add_weather_if_missing(messages)
    engine = await get_engine_result(messages)
    result.tools = tools_called(messages)
    try:
        answer = await ask_for_answer(client, messages)
    except AnswerError:
        result.seconds = time.perf_counter() - started
        return result  # valid_json stays False, so the scenario fails

    response = build_response(answer, engine)
    result.seconds = time.perf_counter() - started
    result.valid_json = True
    result.level = response.level
    result.ai_level = answer.level
    result.summary = response.summary
    result.level_ok = response.level == scenario.expected_level
    result.ai_agreed = answer.level == response.level

    result.invented = invented_numbers(response.summary, tool_data_text(messages))

    answer_parts = [response.summary] + response.reasons + response.warnings + response.gear_tips
    result.missing = missing_words(" ".join(answer_parts), scenario.must_mention)

    result.passed = result.level_ok and not result.invented and not result.missing
    return result


# ---------- The report ----------

def yes_no(ok: bool) -> str:
    """'yes' or 'NO', so failures stand out in the table."""
    if ok:
        return "yes"
    return "NO"


def print_report(results: list[ScenarioResult], model_name: str) -> None:
    """Print a pass/fail table, the details of each failure, and the overall score."""
    print(f"\nEvals for {model_name}\n")
    header = f"{'Scenario':<26} {'Expected':<10} {'Got':<10} {'AI said':<10} {'JSON':<5} {'Level':<6} {'Numbers':<8} {'Words':<6} {'Time':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.id:<26} {r.expected_level:<10} {str(r.level):<10} {str(r.ai_level):<10} "
            f"{yes_no(r.valid_json):<5} {yes_no(r.level_ok):<6} {yes_no(not r.invented):<8} "
            f"{yes_no(not r.missing):<6} {r.seconds:>5.1f}s"
        )

    failures = []
    for r in results:
        if not r.passed:
            failures.append(r)
    if failures:
        print("\nWhat went wrong:")
        for r in failures:
            print(f"  {r.id}  (tools: {', '.join(r.tools) or 'none'})")
            if not r.valid_json:
                print("    - no valid JSON answer, even after the retry")
            if r.invented:
                print(f"    - numbers not in the tool data: {', '.join(r.invented)}")
            if r.missing:
                print(f"    - never mentioned: {', '.join(r.missing)}")
            if r.valid_json and not r.level_ok:
                print(f"    - level {r.level}, expected {r.expected_level}")
            if r.summary:
                print(f"    summary: {r.summary}")

    total = len(results)
    passed = 0
    agreed = 0
    seconds = 0.0
    for r in results:
        if r.passed:
            passed += 1
        if r.ai_agreed:
            agreed += 1
        seconds += r.seconds
    print(f"\nScore: {passed}/{total} passed ({passed / total:.0%})")
    print(f"AI's own level matched the engine: {agreed}/{total}")
    print(f"Average time per question: {seconds / total:.1f} s")


def save_results(results: list[ScenarioResult], model_name: str) -> Path:
    """Save the results as JSON, so runs (and models) can be compared later."""
    RESULTS_DIR.mkdir(exist_ok=True)
    safe_name = model_name.replace(":", "-").replace("/", "-")
    path = RESULTS_DIR / f"{safe_name}.json"
    passed = 0
    for r in results:
        if r.passed:
            passed += 1
    data = {
        "model": model_name,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "total": len(results),
        "results": [],
    }
    for r in results:
        data["results"].append(r.model_dump())
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


async def main(only_ids: list[str]) -> None:
    """Run the chosen scenarios (or all of them), then print and save the results."""
    settings = get_settings()
    model_name = f"{settings.llm_provider}-{settings.ollama_model}"
    client = get_llm_client()

    scenarios = []
    for scenario in load_scenarios():
        if not only_ids or scenario.id in only_ids:
            scenarios.append(scenario)

    results = []
    for number, scenario in enumerate(scenarios, start=1):
        print(f"[{number}/{len(scenarios)}] {scenario.id}...", flush=True)
        try:
            results.append(await run_scenario(scenario, client))
        except LLMError as error:
            print(f"\nThe AI couldn't be reached: {error}\nIs Ollama running?")
            return

    print_report(results, model_name)
    if not only_ids:  # only save full runs, so the saved file is always complete
        print(f"Saved to {save_results(results, model_name)}")


if __name__ == "__main__":
    # Show warnings (engine overrides, the safety net, JSON retries) but not every tool call.
    logging.basicConfig(level=logging.WARNING, format="  %(levelname)s %(name)s: %(message)s")
    asyncio.run(main(sys.argv[1:]))
