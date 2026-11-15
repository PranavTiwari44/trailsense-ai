"""The shapes the rest of the app uses to talk to ANY AI (Ollama now, Claude later).

The agent only uses these classes. Each AI client turns them into its own
format, so swapping the AI never changes the agent's code.
"""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """The AI asking us to run one of our tools, with the inputs it chose."""

    id: str | None = None  # Claude gives each call an id; Ollama doesn't
    name: str  # which tool, e.g. "get_weather"
    arguments: dict[str, Any]  # its inputs, e.g. {"lat": 45.5, "lon": -75.9}


class Message(BaseModel):
    """One message in the conversation."""

    # system = instructions for the AI, user = the person, assistant = the AI,
    # tool = the result of a tool we ran for the AI
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # only on assistant messages
    tool_name: str | None = None  # only on tool messages: which tool this result came from
    tool_call_id: str | None = None  # only on tool messages: the id of the call it answers (Claude needs this)


class ToolSpec(BaseModel):
    """Describes one tool to the AI: its name, what it does, and what inputs it takes."""

    name: str
    description: str
    parameters: dict[str, Any]  # the inputs, written as a JSON Schema


class LLMResponse(BaseModel):
    """What the AI sent back: some text, tool calls, or both."""

    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLMError(Exception):
    """The AI couldn't be reached or failed to answer."""


class LLMClient(Protocol):
    """What every AI client must be able to do.

    Any class with a matching `chat` method counts as an LLMClient,
    without having to say so (no "implements" needed).
    """

    async def chat(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        """Send the conversation and the available tools; get the AI's next reply."""
        ...
