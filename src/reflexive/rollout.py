"""Eval-time rollout loop: run a model (base or fine-tuned) on tasks and record trajectories.

Training rollouts are AgentTune's job. This loop exists so the baseline and the tuned
model are evaluated with the *same* harness and prompt, and so trajectories are written
in OpenAI message format with full tool arguments — which AuditKIT needs for its
argument-sensitive scorers (``redundant_tool_calls``, ``tool_call_f1``).

The prompt format mirrors AgentTune's fallback for Cohere models (JSON list of calls,
results returned as "Tool result for <name>: ..." user turns), so it works with models
whose chat template ignores ``tools``.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

from .trajectory import parse_text_tool_calls

GenerateFn = Callable[[list[dict[str, str]]], str]

TOOLS_PROMPT = (
    "You can call these tools:\n{tools}\n\n"
    "To call tools, reply with only a JSON list, one object per call, and no other text: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    'A user turn that starts with "Tool result" is the output of your call. '
    "When you have enough information, answer the original question as plain text."
)
RESULT_FORMAT = "Tool result for {name}: {content}\nUse this result to answer the original question."


def describe_tools(tools: list[Callable]) -> str:
    lines = []
    for fn in tools:
        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn) or ""
        lines.append(f"- {fn.__name__}{sig}: {doc.splitlines()[0] if doc else ''}")
    return "\n".join(lines)


def _call_tool(tools: dict[str, Callable], name: str, args: dict[str, Any]) -> str:
    fn = tools.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool {name!r}"})
    try:
        return json.dumps(fn(**args), default=str)[:4000]
    except Exception as exc:  # the agent should see its own failures
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def run_episode(generate: GenerateFn, tools: list[Callable], prompt: list[dict[str, str]],
                max_steps: int = 6) -> dict[str, Any]:
    """Returns {"messages": OpenAI-format transcript, "final_answer": str, "steps": int}."""
    by_name = {f.__name__: f for f in tools}
    model_view = [{"role": "system", "content": TOOLS_PROMPT.format(tools=describe_tools(tools))},
                  *prompt]
    record: list[dict[str, Any]] = [dict(m) for m in prompt]
    final, call_id = "", 0
    for step in range(max_steps):
        text = generate(model_view)
        model_view.append({"role": "assistant", "content": text})
        calls = parse_text_tool_calls(text)
        if calls is None:
            final = text.strip()
            record.append({"role": "assistant", "content": final})
            return {"messages": record, "final_answer": final, "steps": step + 1}
        tool_calls, results = [], []
        for c in calls:
            call_id += 1
            cid = f"call_{call_id}"
            tool_calls.append({"id": cid, "type": "function",
                               "function": {"name": c.name, "arguments": json.dumps(c.arguments)}})
            out = (_call_tool(by_name, c.name, c.arguments) if c.well_formed
                   else json.dumps({"error": "malformed tool call"}))
            results.append({"role": "tool", "tool_call_id": cid, "name": c.name, "content": out})
            model_view.append({"role": "user", "content": RESULT_FORMAT.format(name=c.name, content=out)})
        record.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        record.extend(results)
    return {"messages": record, "final_answer": final, "steps": max_steps}


def hf_generate_fn(model_id: str, adapter: str | None = None, max_new_tokens: int = 512,
                   temperature: float = 0.0) -> GenerateFn:
    """A transformers-backed generate function. ``adapter`` = path or Hub id of a PEFT adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter or model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    @torch.no_grad()
    def generate(messages: list[dict[str, str]]) -> str:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        kw = {"do_sample": temperature > 0, "max_new_tokens": max_new_tokens}
        if temperature > 0:
            kw["temperature"] = temperature
        out = model.generate(ids, **kw)
        return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)

    return generate
