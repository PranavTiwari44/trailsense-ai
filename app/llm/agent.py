"""The tool-calling loop: let the AI ask for tools until it has enough to answer.

This file only talks to the LLMClient interface, never to Ollama or Claude directly.
"""

import logging
import time

from pydantic import BaseModel

from app.llm.base import LLMClient, Message, ToolCall, ToolSpec
from app.llm.tools import run_tool

logger = logging.getLogger(__name__)

# Safety limit: the most times we call the AI for one question, so a confused model can't loop forever.
MAX_STEPS = 8

SYSTEM_PROMPT = (
    "You are a hiking-conditions assistant. "
    "Always use the tools to get facts. Never make up numbers. "
    "For a trail, call find_trail first: it gives the trail id and start location the other tools need. "
    "The difficulty level must come from get_difficulty_score, never from your own judgement. "
    "If a tool returns an error or data is missing, say so and call the level Unknown. "
    "Keep your answer to one or two short sentences."
)


class AgentResult(BaseModel):
    """What the agent ends with: the AI's final answer and the whole conversation."""

    answer: str | None  # None if the AI never gave a final answer (hit MAX_STEPS)
    messages: list[Message]  # every message, so we can see what the AI did


def start_conversation(question: str) -> list[Message]:
    """The first two messages: our instructions (system) and the person's question (user)."""
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=question),
    ]


async def run_tool_calls(calls: list[ToolCall]) -> list[Message]:
    """Run each tool the AI asked for, in order, and turn each result into a "tool" message.

    Logs each tool's name, inputs and how long it took.
    """
    messages = []
    for call in calls:
        start_time = time.perf_counter()
        content = await run_tool(call)
        seconds = time.perf_counter() - start_time
        logger.info(f"Tool {call.name} with {call.arguments} took {seconds:.2f} s")
        messages.append(Message(role="tool", content=content, tool_name=call.name, tool_call_id=call.id))
    return messages


async def run_agent(question: str, client: LLMClient, tools: list[ToolSpec],
                    max_steps: int = MAX_STEPS) -> AgentResult:
    """Answer a question, letting the AI call tools as many times as it needs (up to max_steps).

    Each step: send the conversation to the AI. If it asks for tools, run them,
    add the results to the conversation and go again. If it answers with text, we're done.
    """
    messages = start_conversation(question)
    for _ in range(max_steps):
        response = await client.chat(messages, tools)

        # Always keep the AI's reply, so next time it can see what it asked for.
        content = response.text or ""  # text is None when the AI only asks for tools
        messages.append(Message(role="assistant", content=content, tool_calls=response.tool_calls))

        # No tools requested means this is the final answer.
        if not response.tool_calls:
            return AgentResult(answer=response.text, messages=messages)

        messages.extend(await run_tool_calls(response.tool_calls))

    logger.warning(f"Gave up after {max_steps} steps without a final answer")
    return AgentResult(answer=None, messages=messages)
