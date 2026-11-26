"""Talks to a model running in Ollama on this computer (free, no internet needed)."""

from typing import Any

import httpx
from ollama import AsyncClient, ResponseError

from app.llm.base import LLMError, LLMResponse, Message, ToolCall, ToolSpec


class OllamaClient:
    """Sends conversations to a local Ollama model and reads its replies."""

    def __init__(self, host: str, model: str, think: bool = False, timeout_s: float = 300.0):
        """Remember which model to use and set up the connection to Ollama."""
        self.model = model
        self.think = think
        self.client = AsyncClient(host=host, timeout=timeout_s)

    async def chat(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        """Send the conversation and tools to Ollama and return its reply in our shape."""
        ollama_tools = to_ollama_tools(tools)
        if not ollama_tools:
            ollama_tools = None  # Ollama prefers "no tools" over an empty list
        try:
            response = await self.client.chat(
                model=self.model,
                messages=to_ollama_messages(messages),
                tools=ollama_tools,
                think=self.think,
            )
        except (ResponseError, httpx.HTTPError, ConnectionError) as error:
            raise LLMError(f"Ollama call failed ({self.model}): {error}") from error
        return from_ollama_message(response.message)

    async def chat_json(self, messages: list[Message], schema: dict[str, Any]) -> str:
        """Send the conversation and force Ollama to reply with JSON in the shape of `schema`."""
        try:
            response = await self.client.chat(
                model=self.model,
                messages=to_ollama_messages(messages),
                format=schema,  # Ollama only lets the model write JSON that fits this schema
                think=self.think,
            )
        except (ResponseError, httpx.HTTPError, ConnectionError) as error:
            raise LLMError(f"Ollama call failed ({self.model}): {error}") from error
        return response.message.content or ""


# ---------- Converting between our shapes and Ollama's ----------

def to_ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Turn our Message objects into the dicts Ollama expects."""
    result = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = []
            for call in message.tool_calls:
                item["tool_calls"].append({"function": {"name": call.name, "arguments": call.arguments}})
        if message.tool_name:
            item["tool_name"] = message.tool_name
        result.append(item)
    return result


def to_ollama_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Turn our ToolSpec objects into Ollama's tool format: {"type": "function", "function": {...}}."""
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        })
    return result


def from_ollama_message(message: Any) -> LLMResponse:
    """Turn Ollama's reply message into our LLMResponse."""
    tool_calls = []
    for call in message.tool_calls or []:  # Ollama sends None when there are no tool calls
        tool_calls.append(ToolCall(name=call.function.name, arguments=dict(call.function.arguments)))
    text = message.content or None  # treat an empty reply as "no text"
    return LLMResponse(text=text, tool_calls=tool_calls)
