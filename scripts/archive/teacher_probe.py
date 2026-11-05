"""Does the teacher (Aya Expanse 32B via Cohere) do native tool calling? 3 API calls (SEL-37).

    python scripts/archive/teacher_probe.py                            # needs COHERE_API_KEY in .env
    python scripts/archive/teacher_probe.py --model tiny-aya-global    # the student instead

1. Plain chat: is the key and model ID right? What latency?
2. Native: the same question with ``tools=``. Do structured ``tool_calls`` come back?
3. Prompted: the tools in the prompt, in Aya's own Cohere 'Action:' format. Does the reply parse?

Prints a recommendation for TeacherClient's ``mode``.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "src"))
from reflexive.teacher import TeacherClient  # noqa: E402

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Get the status and details of an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."}
                },
                "required": ["order_id"],
            },
        },
    }
]
ASK = [
    {"role": "system", "content": "You are a customer service agent. Use tools to look things up."},
    {"role": "user", "content": "Hi, what's the status of my order #W2378156?"},
]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="c4ai-aya-expanse-32b", help="a Cohere model ID")
    model = ap.parse_args().model
    results = {}
    print(f"model={model}\n1. plain chat")
    t = TeacherClient("native", model=model)
    start = time.monotonic()
    try:
        r = t._call([{"role": "user", "content": "Reply with the single word OK."}], None)
        print(
            f"   ✓ reply={r.choices[0].message.content!r}  {time.monotonic() - start:.1f}s  "
            f"tokens={t.usage.prompt_tokens}/{t.usage.completion_tokens}"
        )
    except Exception as e:  # noqa: BLE001 - report any API error plainly
        print(f"   ✗ {type(e).__name__}: {e}\n   Check COHERE_API_KEY and the model ID.")
        return 1

    print("2. native tool calling (tools=)")
    try:
        turn = t.turn(ASK, TOOLS)
        results["native"] = bool(turn.get("tool_calls"))
        print(
            f"   {'✓ structured tool call' if results['native'] else '✗ no tool call'}: "
            f"{json.dumps(turn)[:300]}"
        )
    except Exception as e:  # noqa: BLE001
        results["native"] = False
        print(f"   ✗ request failed: {type(e).__name__}: {str(e)[:300]}")

    print("3. prompted tool calling (Aya's own 'Action:' format)")
    p = TeacherClient("prompted", model=model)
    try:
        turn = p.turn(ASK, TOOLS)
        results["prompted"] = bool(turn.get("tool_calls"))
        print(
            f"   {'✓ parsed tool call' if results['prompted'] else '✗ did not parse'}: "
            f"{json.dumps(turn)[:300]}"
        )
    except Exception as e:  # noqa: BLE001
        results["prompted"] = False
        print(f"   ✗ request failed: {type(e).__name__}: {str(e)[:300]}")

    if results.get("native"):
        rec = "native works. Prefer prompted anyway if the student trains on prompted transcripts (SEL-35)."
    elif results.get("prompted"):
        rec = "use mode='prompted' (no native tool calling)."
    else:
        rec = "neither worked: paste this output so we can adjust the prompt or route."
    print(f"\nRecommendation: {rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
