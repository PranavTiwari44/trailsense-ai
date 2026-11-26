"""Tests for the eval runner's checks and scenario file. No model: only our own code is tested here."""

import json

import pytest

from app.engine.difficulty import score_difficulty
from app.llm import tools
from app.llm.base import ToolCall
from evals.run_evals import (
    fixed_tools,
    invented_numbers,
    is_in_data,
    load_scenarios,
    make_trail,
    make_weather,
    missing_words,
    numbers_in,
)

SCENARIOS = load_scenarios()


def scenario_by_id(scenario_id: str):
    """Find one scenario in the file by its id."""
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(scenario_id)


# ---------- The checks ----------

def test_numbers_in_finds_whole_and_decimal_numbers():
    assert numbers_in("8.9 km with 360 m of climb, 35 points") == ["8.9", "360", "35"]


def test_rounded_numbers_count_as_in_the_data():
    assert is_in_data("8.9", [8.87])  # rounded to 1 decimal
    assert is_in_data("9", [8.87])  # rounded to a whole number
    assert is_in_data("360", [360.0])
    assert not is_in_data("8.5", [8.87])


def test_invented_numbers_lists_only_numbers_not_in_the_tool_data():
    tool_data = '{"length_km": 8.87, "gain_m": 360.0, "temp_c": 9.0}'

    assert invented_numbers("8.9 km, 360 m of climb, 9 C.", tool_data) == []
    assert invented_numbers("8.9 km and about 15 km/h of wind.", tool_data) == ["15"]


def test_missing_words_ignores_case_and_accepts_either_option():
    assert missing_words("Light RAIN expected", ["rain"]) == []
    assert missing_words("Watch for icy rocks", ["ice|icy"]) == []
    assert missing_words("Nice and sunny", ["rain", "wind"]) == ["rain", "wind"]


# ---------- The scenario file ----------

def test_there_are_15_to_20_scenarios_with_unique_ids():
    ids = []
    for scenario in SCENARIOS:
        ids.append(scenario.id)

    assert 15 <= len(ids) <= 20
    assert len(set(ids)) == len(ids)  # a set drops duplicates, so the sizes match only if all are unique


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_expected_level_matches_the_rules_engine(scenario):
    # If this fails, either the scenario file or the engine is wrong. Fix it before running evals.
    result = score_difficulty(make_trail(scenario), scenario.climb_m, make_weather(scenario))

    assert result.level == scenario.expected_level


# ---------- The fixed tools ----------

async def test_fixed_tools_return_the_scenario_data(monkeypatch):
    monkeypatch.setattr(tools, "TOOL_FUNCTIONS", fixed_tools(scenario_by_id("wolf-trail-mild")))

    trail = json.loads(await tools.run_tool(ToolCall(name="find_trail", arguments={"name": "Wolf Trail"})))
    score = json.loads(await tools.run_tool(ToolCall(name="get_difficulty_score",
                                                     arguments={"trail_id": "wolf-trail"})))

    assert trail["id"] == "wolf-trail"
    assert score["level"] == "Moderate"


async def test_fixed_tools_report_missing_data_as_errors(monkeypatch):
    monkeypatch.setattr(tools, "TOOL_FUNCTIONS", fixed_tools(scenario_by_id("missing-weather")))

    result = json.loads(await tools.run_tool(ToolCall(name="get_weather", arguments={"lat": 45.5, "lon": -75.9})))

    assert "Weather unavailable" in result["error"]


async def test_fixed_tools_say_not_found_for_a_missing_trail(monkeypatch):
    monkeypatch.setattr(tools, "TOOL_FUNCTIONS", fixed_tools(scenario_by_id("trail-not-found")))

    result = json.loads(await tools.run_tool(ToolCall(name="find_trail", arguments={"name": "Narnia Loop"})))

    assert result == {"error": "No trail found matching 'Narnia Loop'"}
