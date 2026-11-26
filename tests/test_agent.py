"""Tests for the tool-calling loop. No model and no internet:
a FakeClient plays the AI, and run_tool is replaced with a fake.
"""

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from app.llm import agent
from app.llm.agent import (
    AnswerError,
    ask_for_answer,
    build_response,
    explain,
    find_tool_result,
    get_engine_result,
    run_agent,
    start_conversation,
)
from app.llm.base import LLMResponse, Message, ToolCall, ToolSpec
from app.models import SAFETY_NOTE, AIAnswer, DifficultyResult

WEATHER_CALL = ToolCall(name="get_weather", arguments={"lat": 45.5, "lon": -75.9})
SOME_TOOLS = [ToolSpec(name="get_weather", description="Weather at a spot", parameters={"type": "object"})]


class FakeClient:
    """A pretend AI for tests. Gives back replies we wrote in advance, in order,
    and remembers every conversation it was sent."""

    def __init__(self, replies: list[LLMResponse], json_replies: list[str] | None = None):
        """Store the replies to hand out, one per call. json_replies are for chat_json."""
        self.replies = replies
        self.json_replies = json_replies or []
        self.conversations_seen: list[list[Message]] = []
        self.tools_seen: list[ToolSpec] = []

    async def chat(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        """Remember what we were sent, then give back the next scripted reply."""
        # list(...) saves a copy. The agent keeps adding to the same list,
        # so without a copy every saved conversation would end up looking the same.
        self.conversations_seen.append(list(messages))
        self.tools_seen = tools
        return self.replies.pop(0)

    async def chat_json(self, messages: list[Message], schema: dict) -> str:
        """Remember what we were sent, then give back the next scripted JSON text."""
        self.conversations_seen.append(list(messages))
        return self.json_replies.pop(0)


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch):
    """Replace the real run_tool so no test touches the internet. Every tool returns the same small result."""
    async def fake_run_tool(call: ToolCall) -> str:
        return f'{{"tool": "{call.name}", "temp_c": 4}}'

    monkeypatch.setattr(agent, "run_tool", fake_run_tool)


def roles(messages: list[Message]) -> list[str]:
    """Just the roles, so tests can check the order of messages at a glance."""
    result = []
    for message in messages:
        result.append(message.role)
    return result


def test_conversation_starts_with_instructions_then_the_question():
    messages = start_conversation("How hard is Wolf Trail?")

    assert roles(messages) == ["system", "user"]
    assert messages[0].content == agent.SYSTEM_PROMPT
    assert messages[1].content == "How hard is Wolf Trail?"


async def test_answers_straight_away_when_no_tool_is_needed():
    client = FakeClient([LLMResponse(text="Hello, hiker!")])

    result = await run_agent("Hi", client, SOME_TOOLS)

    assert result.answer == "Hello, hiker!"
    assert len(client.conversations_seen) == 1
    assert roles(result.messages) == ["system", "user", "assistant"]


async def test_runs_the_tool_and_sends_the_result_back():
    client = FakeClient([
        LLMResponse(tool_calls=[WEATHER_CALL]),  # step 1: the AI asks for the weather
        LLMResponse(text="It's 4 °C and wet."),  # step 2: the AI answers
    ])

    result = await run_agent("Weather at Wolf Trail?", client, SOME_TOOLS)

    assert result.answer == "It's 4 °C and wet."
    second_call = client.conversations_seen[1]
    assert roles(second_call) == ["system", "user", "assistant", "tool"]
    assert second_call[2].tool_calls == [WEATHER_CALL]  # the AI's own request is kept
    assert second_call[2].content == ""  # it had no text (None), stored as ""
    assert second_call[3].tool_name == "get_weather"
    assert second_call[3].content == '{"tool": "get_weather", "temp_c": 4}'


async def test_runs_every_tool_call_in_one_reply_in_order():
    first = ToolCall(name="get_weather", arguments={"lat": 1, "lon": 2})
    second = ToolCall(name="find_trail", arguments={"name": "Wolf Trail"})
    client = FakeClient([
        LLMResponse(tool_calls=[first, second]),
        LLMResponse(text="Done."),
    ])

    result = await run_agent("Question", client, SOME_TOOLS)

    assert roles(result.messages) == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert result.messages[3].tool_name == "get_weather"
    assert result.messages[4].tool_name == "find_trail"


async def test_gives_up_after_max_steps():
    # This AI asks for the weather forever and never answers.
    client = FakeClient([LLMResponse(tool_calls=[WEATHER_CALL])] * 10)

    result = await run_agent("Question", client, SOME_TOOLS, max_steps=3)

    assert result.answer is None
    assert len(client.conversations_seen) == 3  # called the AI exactly 3 times, then stopped


async def test_sends_the_tools_to_the_ai():
    client = FakeClient([LLMResponse(text="Hi")])

    await run_agent("Question", client, SOME_TOOLS)

    assert client.tools_seen == SOME_TOOLS


async def test_logs_each_tool_it_runs(caplog):
    # caplog collects log messages during the test so we can check them.
    caplog.set_level(logging.INFO)
    client = FakeClient([LLMResponse(tool_calls=[WEATHER_CALL]), LLMResponse(text="Done.")])

    await run_agent("Question", client, SOME_TOOLS)

    assert "get_weather" in caplog.text
    assert "45.5" in caplog.text  # the inputs are logged too


# ---------- build_response: the engine always wins (Milestone 6) ----------

DATA_TIME = datetime(2026, 11, 25, 10, 0, tzinfo=timezone(timedelta(hours=-5)))


def make_ai_answer(level: str = "Moderate") -> AIAnswer:
    """What the AI might write at the end. Only the level changes between tests."""
    return AIAnswer(trail_name="Wolf Trail", level=level, summary="Moderate: 8.9 km with 360 m of climb.",
                    reasons=["8.9 km long"], warnings=["Rocky sections"], gear_tips=["Water"])


def make_engine_result(level: str = "Moderate", score: int | None = 35) -> DifficultyResult:
    """What the rules engine might decide."""
    return DifficultyResult(score=score, level=level, reasons=["8.87 km long (+8)"], data_time=DATA_TIME)


def test_engine_level_wins_when_the_ai_disagrees(caplog):
    caplog.set_level(logging.WARNING)

    response = build_response(make_ai_answer(level="Easy"), make_engine_result(level="Hard", score=55))

    assert response.level == "Hard"
    assert "Easy" in caplog.text and "Hard" in caplog.text  # the mismatch is logged


def test_score_and_data_time_come_from_the_engine():
    response = build_response(make_ai_answer(), make_engine_result(score=35))

    assert response.score == 35
    assert response.data_time == DATA_TIME


def test_the_words_come_from_the_ai():
    response = build_response(make_ai_answer(), make_engine_result())

    assert response.trail_name == "Wolf Trail"
    assert response.summary == "Moderate: 8.9 km with 360 m of climb."
    assert response.reasons == ["8.9 km long"]
    assert response.warnings == ["Rocky sections"]
    assert response.gear_tips == ["Water"]


def test_no_engine_result_means_unknown_even_if_the_ai_says_easy():
    response = build_response(make_ai_answer(level="Easy"), None)

    assert response.level == "Unknown"
    assert response.score is None
    assert response.data_time is None


def test_no_warning_when_the_ai_and_engine_agree(caplog):
    caplog.set_level(logging.WARNING)

    build_response(make_ai_answer(level="Moderate"), make_engine_result(level="Moderate"))

    assert caplog.text == ""


def test_every_response_has_the_safety_note():
    response = build_response(make_ai_answer(), make_engine_result())

    assert response.note == SAFETY_NOTE


# ---------- Milestone 7: retry, safety net, and the whole flow ----------

# Realistic tool results, shaped like what our real tools return.
TRAIL_RESULT = {"id": "wolf-trail", "name": "Wolf Trail", "length_km": 8.87, "sac_scale": "demanding_mountain_hiking",
                "start_lat": 45.54, "start_lon": -75.93, "source": "fallback"}
ENGINE_RESULT = {"score": 35, "level": "Moderate", "reasons": ["8.87 km long (+8)"],
                 "data_time": "2026-11-25T10:00:00-05:00"}

FIND_TRAIL_CALL = ToolCall(name="find_trail", arguments={"name": "Wolf Trail"})
DIFFICULTY_CALL = ToolCall(name="get_difficulty_score", arguments={"trail_id": "wolf-trail"})

GOOD_JSON = make_ai_answer().model_dump_json()  # a valid AIAnswer, as JSON text
CUT_OFF_JSON = '{"trail_name": "Wolf Trail", "summary": "The trail is'  # the model stopped halfway
MISSING_FIELDS_JSON = '{"trail_name": "Wolf Trail"}'  # valid JSON, but not a full AIAnswer


class ToolRecorder:
    """A fake run_tool that returns realistic results and remembers which tools were run."""

    def __init__(self):
        """Start with no tools run."""
        self.names_run: list[str] = []

    async def run_tool(self, call: ToolCall) -> str:
        """Remember the tool's name and return a realistic result for it."""
        self.names_run.append(call.name)
        if call.name == "find_trail":
            return json.dumps(TRAIL_RESULT)
        if call.name == "get_difficulty_score":
            return json.dumps(ENGINE_RESULT)
        return json.dumps({"error": f"no fake result for {call.name}"})


@pytest.fixture
def recorder(monkeypatch):
    """Use ToolRecorder instead of the simple fake tools, for tests that need real-looking results."""
    tool_recorder = ToolRecorder()
    monkeypatch.setattr(agent, "run_tool", tool_recorder.run_tool)
    return tool_recorder


def tool_message(tool_name: str, result: dict) -> Message:
    """A tool result message, as our loop would add it."""
    return Message(role="tool", content=json.dumps(result), tool_name=tool_name)


# --- find_tool_result ---

def test_find_tool_result_gives_the_newest_successful_result():
    messages = [
        tool_message("get_difficulty_score", {"score": 10, "level": "Easy"}),
        tool_message("get_difficulty_score", {"score": 35, "level": "Moderate"}),
    ]

    assert find_tool_result(messages, "get_difficulty_score")["score"] == 35


def test_find_tool_result_skips_errors():
    messages = [
        tool_message("find_trail", TRAIL_RESULT),
        tool_message("find_trail", {"error": "Overpass is down"}),  # newer, but failed
    ]

    assert find_tool_result(messages, "find_trail")["id"] == "wolf-trail"


def test_find_tool_result_is_none_when_the_tool_was_never_called():
    messages = [tool_message("find_trail", TRAIL_RESULT)]

    assert find_tool_result(messages, "get_difficulty_score") is None


# --- get_engine_result (the safety net) ---

async def test_uses_the_engine_result_the_ai_already_asked_for(recorder):
    messages = [tool_message("find_trail", TRAIL_RESULT), tool_message("get_difficulty_score", ENGINE_RESULT)]

    engine = await get_engine_result(messages)

    assert engine.level == "Moderate"
    assert recorder.names_run == []  # no extra tool call needed


async def test_calls_get_difficulty_score_itself_when_the_ai_forgot(recorder, caplog):
    caplog.set_level(logging.WARNING)
    messages = [tool_message("find_trail", TRAIL_RESULT)]  # the AI found the trail, then stopped

    engine = await get_engine_result(messages)

    assert engine.level == "Moderate"
    assert recorder.names_run == ["get_difficulty_score"]
    assert "never called get_difficulty_score" in caplog.text
    # The request is added BEFORE its result, as if the AI had asked.
    assert messages[-2].role == "assistant"
    assert messages[-2].tool_calls == [DIFFICULTY_CALL]
    assert messages[-1].role == "tool"
    assert messages[-1].tool_name == "get_difficulty_score"


async def test_no_trail_found_means_no_engine_result(recorder):
    messages = [tool_message("find_trail", {"error": "No trail found matching 'Narnia Loop'"})]

    engine = await get_engine_result(messages)

    assert engine is None
    assert recorder.names_run == []  # no trail id, so nothing to ask the engine about


# --- ask_for_answer (structured output + retry) ---

async def test_valid_json_on_the_first_try():
    client = FakeClient(replies=[], json_replies=[GOOD_JSON])

    answer = await ask_for_answer(client, start_conversation("Wolf Trail?"))

    assert answer.trail_name == "Wolf Trail"
    assert len(client.conversations_seen) == 1  # no retry needed


@pytest.mark.parametrize("bad_json", [CUT_OFF_JSON, MISSING_FIELDS_JSON])
async def test_invalid_json_then_valid_on_the_retry(bad_json):
    client = FakeClient(replies=[], json_replies=[bad_json, GOOD_JSON])

    answer = await ask_for_answer(client, start_conversation("Wolf Trail?"))

    assert answer.summary == "Moderate: 8.9 km with 360 m of climb."
    # On the retry, the AI sees its own bad answer and what was wrong with it.
    retry = client.conversations_seen[1]
    assert retry[-2].role == "assistant"
    assert retry[-2].content == bad_json
    assert retry[-1].role == "user"
    assert "not valid" in retry[-1].content


async def test_invalid_json_twice_gives_up_with_answer_error():
    client = FakeClient(replies=[], json_replies=[CUT_OFF_JSON, MISSING_FIELDS_JSON])

    with pytest.raises(AnswerError):
        await ask_for_answer(client, start_conversation("Wolf Trail?"))

    assert len(client.conversations_seen) == 2  # one try + one retry, then stop


# --- explain: the whole flow, with no model at all ---

async def test_explain_engine_overrides_the_ai_level(recorder, caplog):
    caplog.set_level(logging.WARNING)
    client = FakeClient(
        replies=[
            LLMResponse(tool_calls=[FIND_TRAIL_CALL]),
            LLMResponse(tool_calls=[DIFFICULTY_CALL]),
            LLMResponse(text="It's easy!"),
        ],
        json_replies=[make_ai_answer(level="Easy").model_dump_json()],  # the AI says Easy...
    )

    response = await explain("How hard is Wolf Trail?", client)

    assert response.level == "Moderate"  # ...but the engine said Moderate, and the engine wins
    assert response.score == 35
    assert "mismatch" in caplog.text


async def test_explain_when_the_ai_never_calls_get_difficulty_score(recorder):
    client = FakeClient(
        replies=[
            LLMResponse(tool_calls=[FIND_TRAIL_CALL]),
            LLMResponse(text="Looks fine to me."),  # skipped the difficulty tool
        ],
        json_replies=[GOOD_JSON],
    )

    response = await explain("How hard is Wolf Trail?", client)

    assert recorder.names_run == ["find_trail", "get_difficulty_score"]  # our code filled the gap
    assert response.level == "Moderate"
    assert response.score == 35
    # The AI's final JSON call could see the engine's result.
    final_call = client.conversations_seen[-1]
    tool_names = []
    for message in final_call:
        tool_names.append(message.tool_name)
    assert "get_difficulty_score" in tool_names


async def test_explain_unknown_trail_is_unknown(monkeypatch):
    async def trail_not_found(call: ToolCall) -> str:
        return json.dumps({"error": "No trail found matching 'Narnia Loop'"})

    monkeypatch.setattr(agent, "run_tool", trail_not_found)
    client = FakeClient(
        replies=[LLMResponse(tool_calls=[ToolCall(name="find_trail", arguments={"name": "Narnia Loop"})]),
                 LLMResponse(text="I couldn't find it.")],
        json_replies=[make_ai_answer(level="Easy").model_dump_json()],  # a confused AI says Easy
    )

    response = await explain("How hard is the Narnia Loop?", client)

    assert response.level == "Unknown"  # never Easy when data is missing
    assert response.score is None
