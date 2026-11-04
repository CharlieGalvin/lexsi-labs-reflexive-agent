"""reflexive.arms + the Arm C prompt variant (SEL-41).

No API calls: the customer is LiteLLM's `mock_response`, and the agent is scripted to play the
task's correct actions. The episode tests need τ²-bench with its data; skipped otherwise.
"""

from __future__ import annotations

import json

import pytest
import yaml

from reflexive.arms import get_arm, load_config, make_student

CUSTOMER = {"user_llm": "gpt-4.1", "user_llm_args": {"mock_response": "Yes, please go ahead."}}


# --- Config (no τ²-bench needed) ------------------------------------------------------------


def test_config_has_arms_a_and_c_sharing_one_student():
    a, c = get_arm("A"), get_arm("C")
    assert (a.prompt_variant, c.prompt_variant) == ("default", "careful")
    assert a.student == c.student and a.customer == c.customer
    assert set(load_config()["arms"]) >= {"A", "C"}


def test_student_uses_the_data_contract_format():
    # SEL-35: Cohere `Action:` format, the same prompt Aya's baseline used
    for key in load_config()["arms"]:
        s = get_arm(key).student
        assert (s["mode"], s["style"]) == ("prompted", "cohere")


def test_unknown_arm_and_unknown_student_setting_are_refused(tmp_path):
    with pytest.raises(KeyError, match="Unknown arm"):
        get_arm("Z")
    cfg = load_config()
    cfg["student"]["temprature"] = 0.7  # typo: must not be silently ignored
    path = tmp_path / "arms.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="temprature"):
        get_arm("A", path)


def test_student_needs_its_api_key(monkeypatch):
    student = get_arm("A").student
    monkeypatch.delenv(student["api_key_env"], raising=False)
    with pytest.raises(RuntimeError, match=student["api_key_env"]):
        make_student(student)
    self_hosted = {**student, "api_key_env": None, "api_base": "http://localhost:8000/v1"}
    agent = make_student(self_hosted)  # a local server needs no key
    assert agent.api_base == "http://localhost:8000/v1" and agent.model == student["model"]


# --- With τ²-bench ----------------------------------------------------------------------------


@pytest.fixture
def tau2():
    pytest.importorskip("tau2")
    from tau2.utils.utils import DATA_DIR

    if not (DATA_DIR / "tau2" / "domains").exists():
        pytest.skip("τ²-bench data folder missing (run `just tau2`)")
    from loguru import logger

    logger.disable("tau2")


class ScriptedAgent:
    """Plays a task's correct tool calls, then ends the conversation. Records each prompt."""

    def __init__(self, domain: str, task_id: str):
        from reflexive.outcome import scorer

        self.calls = [m["tool_calls"] for m in scorer(domain).gold_trajectory(task_id)
                      if m["role"] == "assistant" and m.get("tool_calls")]
        self.prompts: list[list[dict]] = []

    def turn(self, messages, tool_schemas):
        self.prompts.append(messages)
        if self.calls:
            return {"role": "assistant", "content": None, "tool_calls": self.calls.pop(0)}
        return {"role": "assistant", "content": "That's all done for you. ###STOP###"}


def test_careful_prompt_is_the_default_prompt_plus_the_instruction(tau2):
    from reflexive.episode import CAREFUL_INSTRUCTION, Tau2Episode

    default = Tau2Episode("retail", "0", **CUSTOMER)
    careful = Tau2Episode("retail", "0", prompt_variant="careful", **CUSTOMER)
    for ep in (default, careful):
        ep.reset()
    assert CAREFUL_INSTRUCTION not in default.system_prompt
    assert careful.system_prompt.replace(f"\n\n{CAREFUL_INSTRUCTION}", "") == default.system_prompt
    head, policy = careful.system_prompt.split("</instructions>")
    assert CAREFUL_INSTRUCTION in head and CAREFUL_INSTRUCTION not in policy  # inside <instructions>
    for ep in (default, careful):
        ep.finish()


def test_unknown_prompt_variant_is_refused(tau2):
    from reflexive.episode import Tau2Episode

    with pytest.raises(ValueError, match="prompt_variant"):
        Tau2Episode("retail", "0", prompt_variant="extra-careful", **CUSTOMER)


@pytest.mark.parametrize("arm_key", ["A", "C"])
def test_an_arm_rollout_produces_a_scored_record(tau2, arm_key):
    from reflexive.arms import run_episode
    from reflexive.episode import CAREFUL_INSTRUCTION

    arm = get_arm(arm_key)
    agent = ScriptedAgent("retail", "0")
    rec = run_episode(arm, agent, "retail", "0", judges=False, sample=2,
                      user_llm_args=CUSTOMER["user_llm_args"], user_llm=CUSTOMER["user_llm"])
    assert rec["arm"] == arm_key and rec["sample"] == 2 and rec["error"] is None
    assert rec["prompt_variant"] == arm.prompt_variant and rec["hints"] == "none"
    assert rec["termination"] == "agent_stop" and rec["outcome"]["passed"]
    assert rec["diagnostics"]["tools_ran"] > 0
    careful_seen = CAREFUL_INSTRUCTION in agent.prompts[0][0]["content"]
    assert careful_seen == (arm_key == "C")
    json.dumps(rec)  # saveable as a JSON line

    # the agent's view (data contract input): system prompt first, untrimmed
    view = rec["agent_view"]
    assert view[0]["role"] == "system" and view[0]["content"] == agent.prompts[0][0]["content"]
    assert any(m.get("tool_calls") for m in view)


def test_the_agent_view_renders_as_cohere_actions(tau2):
    from reflexive.arms import run_episode
    from reflexive.episode import CAREFUL_INSTRUCTION, Tau2Episode
    from reflexive.teacher import to_prompted

    rec = run_episode(get_arm("C"), ScriptedAgent("retail", "0"), "retail", "0", judges=False,
                      **CUSTOMER)
    ep = Tau2Episode("retail", "0", **CUSTOMER)
    ep.reset()
    text = to_prompted(rec["agent_view"], ep.tool_schemas, style="cohere")
    ep.finish()
    assert CAREFUL_INSTRUCTION in text[0]["content"]
    actions = [m["content"] for m in text if m["role"] == "assistant" and "Action:" in m["content"]]
    assert actions and all(a.startswith("Action: ```json") for a in actions)
    assert any("<results>" in m["content"] for m in text if m["role"] == "user")


def test_a_crashing_agent_is_recorded_not_raised(tau2):
    from reflexive.arms import run_episode

    class Broken:
        def turn(self, messages, tool_schemas):
            raise RuntimeError("server down")

    rec = run_episode(get_arm("A"), Broken(), "retail", "0", judges=False, **CUSTOMER)
    assert "server down" in rec["error"] and not rec["outcome"]["passed"]
