"""Asks a question through our own AI interface (not the ollama package directly).

Run from the project root:  uv run python -m scripts.try_llm_client "Your question"
"""

import asyncio
import sys
import time

from app.llm.base import Message
from app.llm.factory import get_llm_client


async def main(question: str) -> None:
    """Send one question through get_llm_client() and print the reply."""
    client = get_llm_client()  # this script doesn't know or care which AI it gets
    messages = [
        Message(role="system", content="You are a hiking assistant. Answer in one or two short sentences."),
        Message(role="user", content=question),
    ]
    started = time.perf_counter()
    response = await client.chat(messages, tools=[])
    print(f"[{time.perf_counter() - started:.1f}s] {type(client).__name__}")
    print(response)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "Is it safe to hike in a thunderstorm?"
    asyncio.run(main(question))
