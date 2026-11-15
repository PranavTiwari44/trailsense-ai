"""Tests for the AI interface. No model is used: we only check our format conversions and the factory."""

from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.llm.base import Message, ToolCall, ToolSpec
from app.llm.factory import get_llm_client
from app.llm.ollama_client import OllamaClient, from_ollama_message, to_ollama_messages, to_ollama_tools


def test_messages_become_ollama_dicts():
    messages = [
        Message(role="user", content="How hard is Wolf Trail?"),
        Message(role="assistant", tool_calls=[ToolCall(name="find_trail", arguments={"name": "Wolf Trail"})]),
        Message(role="tool", content='{"id": "wolf-trail"}', tool_name="find_trail"),
    ]

    result = to_ollama_messages(messages)

    assert result[0] == {"role": "user", "content": "How hard is Wolf Trail?"}
    assert result[1]["tool_calls"] == [{"function": {"name": "find_trail", "arguments": {"name": "Wolf Trail"}}}]
    assert result[2]["tool_name"] == "find_trail"


def test_tools_become_ollama_function_format():
    tool = ToolSpec(name="get_weather", description="Weather at a spot",
                    parameters={"type": "object", "properties": {"lat": {"type": "number"}}})

    result = to_ollama_tools([tool])

    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["parameters"]["properties"]["lat"] == {"type": "number"}


def test_ollama_reply_with_text_only():
    # SimpleNamespace makes a quick fake object with the attributes we give it.
    reply = SimpleNamespace(content="Bring a rain jacket.", tool_calls=None)

    response = from_ollama_message(reply)

    assert response.text == "Bring a rain jacket."
    assert response.tool_calls == []


def test_ollama_reply_with_a_tool_call():
    call = SimpleNamespace(function=SimpleNamespace(name="get_weather", arguments={"lat": 45.5, "lon": -75.9}))
    reply = SimpleNamespace(content="", tool_calls=[call])

    response = from_ollama_message(reply)

    assert response.text is None  # an empty reply counts as no text
    assert response.tool_calls == [ToolCall(name="get_weather", arguments={"lat": 45.5, "lon": -75.9})]


def test_factory_gives_ollama_client_by_default():
    assert isinstance(get_llm_client(), OllamaClient)


def test_factory_says_claude_is_not_ready_yet(monkeypatch):
    # monkeypatch temporarily changes something for this test only, then puts it back.
    monkeypatch.setattr(get_settings(), "llm_provider", "claude")

    with pytest.raises(NotImplementedError):
        get_llm_client()
