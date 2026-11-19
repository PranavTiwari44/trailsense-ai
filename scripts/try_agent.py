"""Ask the real local AI a question and let it use our tools. Prints every step.

Run from the project root:  uv run python -m scripts.try_agent "Your question"
"""

import asyncio
import logging
import sys
import time

from app.llm.agent import run_agent
from app.llm.factory import get_llm_client
from app.llm.tools import TOOL_SPECS


async def main(question: str) -> None:
    """Run the agent on one question and print the conversation it had."""
    started = time.perf_counter()
    result = await run_agent(question, get_llm_client(), TOOL_SPECS)
    seconds = time.perf_counter() - started

    print("\n--- The conversation ---")
    for message in result.messages:
        if message.role == "system":
            continue  # skip our long instructions
        if message.tool_calls:
            for call in message.tool_calls:
                print(f"[assistant] wants {call.name}({call.arguments})")
        elif message.role == "tool":
            print(f"[tool {message.tool_name}] {message.content[:200]}...")
        else:
            print(f"[{message.role}] {message.content}")

    print(f"\nAnswer: {result.answer}")
    print(f"Took {seconds:.1f} s")


if __name__ == "__main__":
    # Show our INFO logs (the tool calls) in the terminal.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "What's the weather like right now at latitude 45.51, longitude -75.92?"
    asyncio.run(main(question))
