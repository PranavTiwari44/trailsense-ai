"""The tool-calling loop: let the AI ask for tools until it has enough to answer.

This file only talks to the LLMClient interface, never to Ollama or Claude directly.
"""

import json
import logging
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from app.llm.base import LLMClient, Message, ToolCall, ToolSpec
from app.llm.tools import TOOL_SPECS, run_tool
from app.models import AIAnswer, DifficultyResult, ExplainResponse, ScopeCheck

logger = logging.getLogger(__name__)

# Safety limit: the most times we call the AI for one question, so a confused model can't loop forever.
MAX_STEPS = 8

SYSTEM_PROMPT = (
    "You are a hiking-conditions assistant. "
    "Always use the tools to get facts. Never make up numbers. "
    "For a trail, call find_trail first: it gives the trail id and start location the other tools need. "
    "If the question is about something your tools can't check (like parking or opening hours), "
    "still look up the trail and its difficulty, so you can say what you do know. "
    "Call get_weather at the trail's start so you can describe the conditions. "
    "The difficulty level must come from get_difficulty_score, never from your own judgement. "
    "If a tool returns an error or data is missing, say so and call the level Unknown. "
    "Keep your answer to one or two short sentences."
)


# Sent once if the AI tries to answer without using any tools.
NUDGE_TO_USE_TOOLS = (
    "Before answering, use your tools: call find_trail for the trail in the question, then "
    "get_difficulty_score. Then answer what you can, and say clearly what you can't check."
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
    used_a_tool = False
    nudged = False
    for _ in range(max_steps):
        response = await client.chat(messages, tools)

        # Always keep the AI's reply, so next time it can see what it asked for.
        content = response.text or ""  # text is None when the AI only asks for tools
        messages.append(Message(role="assistant", content=content, tool_calls=response.tool_calls))

        # Small models sometimes answer straight away without looking anything up
        # (e.g. "my tools can't check parking"). Nudge them once to use the tools.
        if not response.tool_calls and not used_a_tool and not nudged:
            logger.warning("The AI answered without using any tools; asking it to look the trail up")
            messages.append(Message(role="user", content=NUDGE_TO_USE_TOOLS))
            nudged = True
            continue

        # No tools requested means this is the final answer.
        if not response.tool_calls:
            return AgentResult(answer=response.text, messages=messages)

        used_a_tool = True

        messages.extend(await run_tool_calls(response.tool_calls))

    logger.warning(f"Gave up after {max_steps} steps without a final answer")
    return AgentResult(answer=None, messages=messages)


# ---------- Milestone 6: the final JSON answer, with the engine in charge ----------

# How many times we ask the AI for valid JSON before giving up (the first try + one retry).
MAX_ANSWER_TRIES = 2

FINAL_ANSWER_PROMPT = (
    "Now write your final answer as JSON. Use only facts from the tool results above. "
    "Copy the level from get_difficulty_score (Unknown if it is missing). "
    "In the summary, say the level, then describe the weather right now from get_weather: "
    "the temperature, and any rain, snow or strong wind, even light amounts. "
    "If data is missing or the trail wasn't found, say so. Keep it to one or two sentences. "
    "When you write a number, copy it exactly as it appears in the tool results; never round "
    "or estimate it. Warnings: real dangers for these conditions. Gear tips: what to bring for "
    "this weather and this trail. Never guess anything the tools didn't give you."
)

# The small separate check for questions about things we can't look up (like parking).
# Examples help small models far more than descriptions do.
SCOPE_PROMPT = """You read a hiking question and find any topic this app can't answer.
The app CAN answer: how hard a trail is, its length and climb, the conditions and weather, and what to bring.
Anything else is an other_topic. Reply with JSON only.

Examples:
"How hard is the Wolf Trail right now?" -> {"other_topic": null}
"Is it icy on King Mountain today?" -> {"other_topic": null}
"What should I bring for Pink Lake trail?" -> {"other_topic": null}
"Is there parking at the Beltline Trail?" -> {"other_topic": "parking"}
"Can I bring my dog on Wolf Trail?" -> {"other_topic": "dogs"}
"When does Luskville Falls trail open, and is it hard?" -> {"other_topic": "opening hours"}"""

# The longest topic we accept. Anything longer means the AI misunderstood the job.
MAX_TOPIC_WORDS = 3


class AnswerError(Exception):
    """The AI could not give a valid JSON answer, even after a retry."""


def find_tool_result(messages: list[Message], tool_name: str) -> dict[str, Any] | None:
    """The most recent successful result of a tool in the conversation, as a dict.

    Returns None if the tool was never called, or every call to it returned an error.
    """
    for message in reversed(messages):  # newest first
        if message.role != "tool" or message.tool_name != tool_name:
            continue
        data = json.loads(message.content)
        if "error" in data:
            continue
        return data
    return None


async def run_tool_for_the_ai(messages: list[Message], call: ToolCall) -> None:
    """Run a tool the AI forgot, and add it to the conversation as if the AI had asked.

    The request goes in before the result, so the AI never sees a result it didn't request.
    """
    messages.append(Message(role="assistant", tool_calls=[call]))
    messages.extend(await run_tool_calls([call]))


async def add_weather_if_missing(messages: list[Message]) -> None:
    """If the AI never got the weather, fetch it at the trail's start so the answer can describe it.

    Small models often skip get_weather because get_difficulty_score already uses the weather
    inside our code. But then the AI can't mention light rain, cold or snow. Does nothing if
    the AI already has the weather, or if no trail was found.
    """
    if find_tool_result(messages, "get_weather") is not None:
        return
    trail = find_tool_result(messages, "find_trail")
    if trail is None:
        return
    logger.warning("The AI never got the weather; calling get_weather ourselves")
    call = ToolCall(name="get_weather", arguments={"lat": trail["start_lat"], "lon": trail["start_lon"]})
    await run_tool_for_the_ai(messages, call)


async def get_engine_result(messages: list[Message]) -> DifficultyResult | None:
    """The rules engine's decision for this question.

    Small models sometimes forget to call get_difficulty_score. If so, and we know
    the trail, we call it ourselves and add it to the conversation so the AI sees it.
    Returns None if no trail was found (the answer will then be "Unknown").
    """
    data = find_tool_result(messages, "get_difficulty_score")
    if data is not None:
        return DifficultyResult.model_validate(data)

    trail = find_tool_result(messages, "find_trail")
    if trail is None:
        logger.warning("No trail was found, so there is no difficulty score")
        return None

    logger.warning("The AI never called get_difficulty_score; calling it ourselves")
    call = ToolCall(name="get_difficulty_score", arguments={"trail_id": trail["id"]})
    await run_tool_for_the_ai(messages, call)

    data = find_tool_result(messages, "get_difficulty_score")
    if data is None:
        return None  # the tool itself failed, e.g. the trail id was unknown
    return DifficultyResult.model_validate(data)


async def ask_for_answer(client: LLMClient, messages: list[Message]) -> AIAnswer:
    """Ask the AI for its final answer as JSON, check it, and retry once if it's invalid.

    Raises AnswerError if it is still invalid after the retry.
    """
    schema = AIAnswer.model_json_schema()
    messages.append(Message(role="user", content=FINAL_ANSWER_PROMPT))

    error_text = ""
    for attempt in range(1, MAX_ANSWER_TRIES + 1):
        text = await client.chat_json(messages, schema)
        try:
            return AIAnswer.model_validate_json(text)
        except ValidationError as error:
            error_text = str(error)
            logger.warning(f"Answer try {attempt} was not valid: {error_text}")
            # Show the AI what it sent and what was wrong, so the next try can fix it.
            messages.append(Message(role="assistant", content=text))
            messages.append(Message(role="user", content=f"That JSON was not valid: {error_text}. "
                                                         "Send the corrected JSON only."))

    raise AnswerError(f"The AI gave no valid answer after {MAX_ANSWER_TRIES} tries: {error_text}")


def build_response(answer: AIAnswer, engine: DifficultyResult | None) -> ExplainResponse:
    """Combine the AI's words with the rules engine's numbers. The engine always wins.

    - No engine result (no trail, or data missing): level "Unknown", no score, no data time.
    - Otherwise: level, score and data time come from the engine. If the AI's level
      is different, use the engine's and log a warning about the mismatch.
    - Everything else (trail name, summary, reasons, warnings, gear tips) comes from the AI.
    """
    if engine is None:
        level = "Unknown"
        score = None
        data_time = None
    else:   
        if engine.level != answer.level:
            logger.warning(f"There is level mismatch between AI:{answer.level} and Engine:{engine.level}, used engine's level.")
        level = engine.level
        score = engine.score
        data_time = engine.data_time

    return ExplainResponse(trail_name=answer.trail_name, level=level, score=score, summary=answer.summary, reasons=answer.reasons, warnings=answer.warnings, gear_tips=answer.gear_tips, data_time=data_time)


async def check_question_scope(client: LLMClient, question: str) -> str | None:
    """Ask the AI, in a separate small call, whether the question wants something we can't check.

    Returns a note like "I can't check parking, but here's how hard the trail is right now.",
    or None. The AI only names the topic; our code writes the sentence, so it's always clear.
    A bad reply here just means no note; it never fails the whole answer.
    """
    messages = [
        Message(role="system", content=SCOPE_PROMPT),
        Message(role="user", content=f"Question: {question}"),
    ]
    text = await client.chat_json(messages, ScopeCheck.model_json_schema())
    try:
        topic = ScopeCheck.model_validate_json(text).other_topic
    except ValidationError as error:
        logger.warning(f"The question check gave invalid JSON, so no note: {error}")
        return None

    if not topic:
        return None
    topic = topic.strip().strip(".").lower()
    if len(topic.split()) > MAX_TOPIC_WORDS:
        logger.warning(f"The question check gave a sentence, not a topic, so no note: {topic!r}")
        return None
    return f"I can't check {topic}, but here's how hard the trail is right now."


class ExplainRun(BaseModel):
    """Everything from answering one question: the final response, plus what the AI itself
    said and the whole conversation (the evals need those to score the AI)."""

    response: ExplainResponse
    ai_answer: AIAnswer
    messages: list[Message]


async def explain_with_details(question: str, client: LLMClient) -> ExplainRun:
    """Answer a trail question from start to finish, and keep the details.

    Tools first, then the safety nets, then the JSON answer, then the check for parts of
    the question we can't answer.
    """
    result = await run_agent(question, client, TOOL_SPECS)
    await add_weather_if_missing(result.messages)
    engine = await get_engine_result(result.messages)
    answer = await ask_for_answer(client, result.messages)
    response = build_response(answer, engine)
    response.question_note = await check_question_scope(client, question)
    return ExplainRun(response=response, ai_answer=answer, messages=result.messages)


async def explain(question: str, client: LLMClient) -> ExplainResponse:
    """Answer a trail question from start to finish and return just the final answer."""
    run = await explain_with_details(question, client)
    return run.response
