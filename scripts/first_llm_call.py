""""The smallest possible AI call: send one message to the local model, print the reply.

Run from the project root:  uv run python -m scripts.first_llm_call
Add --think to let the model "think" before answering (slower, see the difference).
"""

import sys
import time

from ollama import Client

from app.config import get_settings


def main() -> None:
    """Ask the local model one question and print everything that comes back."""
    settings = get_settings()
    client = Client(host=settings.ollama_host)
    think = "--think" in sys.argv

    # The conversation so far. Each message has a "role" (who is speaking) and "content" (what they said).
    #   system    = instructions for the model (how to behave)
    #   user      = the person asking
    #   assistant = the model's earlier replies (none yet)
    messages = [
        {"role": "system", "content": "You are a hiking assistant. Answer in one short sentence."},
        {"role": "user", "content": "What should I bring for a 9 km hike in light rain?"},
    ]

    print(f"Asking {settings.ollama_model} (think={think})...")
    started = time.perf_counter()
    response = client.chat(model=settings.ollama_model, messages=messages, think=think)
    seconds = time.perf_counter() - started

    print("\n--- The reply ---")
    print("role:    ", response.message.role)
    print("content: ", response.message.content)
    if response.message.thinking:
        print("thinking:", response.message.thinking[:300], "...")

    print("\n--- Extra details Ollama sends back ---")
    print("model:        ", response.model)
    print("done_reason:  ", response.done_reason)  # "stop" = the model finished on its own
    print("prompt tokens:", response.prompt_eval_count)  # how much text we sent, in tokens
    print("reply tokens: ", response.eval_count)  # how much text it wrote, in tokens
    print(f"took:          {seconds:.1f} s")


if __name__ == "__main__":
    main()
