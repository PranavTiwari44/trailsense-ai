"""Tests for the tool-calling loop. No model and no internet:
a FakeClient plays the AI, and run_tool is replaced with a fake.
"""

import logging

import pytest

from app.llm import agent
from app.llm.agent import run_agent, start_conversation
from app.llm.base import LLMResponse, Message, ToolCall, ToolSpec

WEATHER_CALL = ToolCall(name="get_weather", arguments={"lat": 45.5, "lon": -75.9})
SOME_TOOLS = [ToolSpec(name="get_weather", description="Weather at a spot", parameters={"type": "object"})]


class FakeClient:
    """A pretend AI for tests. Gives back replies we wrote in advance, in order,
    and remembers every conversation it was sent."""

    def __init__(self, replies: list[LLMResponse]):
        """Store the replies to hand out, one per call."""
        self.replies = replies
        self.conversations_seen: list[list[Message]] = []
        self.tools_seen: list[ToolSpec] = []

    async def chat(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        """Remember what we were sent, then give back the next scripted reply."""
        # list(...) saves a copy. The agent keeps adding to the same list,
        # so without a copy every saved conversation would end up looking the same.
        self.conversations_seen.append(list(messages))
        self.tools_seen = tools
        return self.replies.pop(0)


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
