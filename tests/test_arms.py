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


# Each prompt version's text, pinned: editing a variant in configs/prompts.yaml in place would
# make earlier runs incomparable. Add a new variant with a new version (and a line here) instead.
PINNED_PROMPTS = {
    "careful/v1": "10be39575fcd3760225f5cc64043b1049f300a1573be709d8fb54829ff631c86",
}


def test_prompt_versions_are_never_edited_in_place():
    import hashlib

    from reflexive.episode import PROMPT_VARIANTS, PROMPT_VERSIONS

    for variant, text in PROMPT_VARIANTS.items():
        version = PROMPT_VERSIONS[variant]
        if text is None:  # τ²-bench's own prompt, nothing added
            assert version == "tau2/default"
            continue
        assert version in PINNED_PROMPTS, f"{version} is new: pin its hash in PINNED_PROMPTS"
        assert hashlib.sha256(text.encode()).hexdigest() == PINNED_PROMPTS[version], (
            f"{variant} ({version}) was edited in place: add a new variant and version instead")
    assert len(set(PROMPT_VERSIONS.values())) == len(PROMPT_VERSIONS)  # versions are unique


def test_every_arm_uses_a_known_prompt_variant():
    from reflexive.episode import PROMPT_VARIANTS

    for key in load_config()["arms"]:
        assert get_arm(key).prompt_variant in PROMPT_VARIANTS


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


# --- Rate limits and crashed conversations (no τ²-bench needed) -------------------------------


def test_student_rpm_comes_from_the_config_not_teacher_rpm_and_can_be_shared(monkeypatch):
    from reflexive.teacher import RateLimiter

    monkeypatch.setenv("TEACHER_RPM", "5")  # the teacher's cap must not apply to the student
    student = {**get_arm("A").student, "api_key_env": None, "rpm": 20}
    assert make_student(student).limiter.interval == pytest.approx(3.0)
    shared = RateLimiter(20)
    a, b = make_student(student, limiter=shared), make_student(student, limiter=shared)
    assert a.limiter is b.limiter is shared  # one cap for every worker in a run


def _load_rollout_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "rollout.py"
    spec = importlib.util.spec_from_file_location("rollout_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rollout_reruns_crashed_conversations_and_counts_them_apart(tmp_path, monkeypatch, capsys):
    mod = _load_rollout_script()
    out = tmp_path / "A.jsonl"

    def rec(task_id, passed=False, error=None):
        return {"task_id": task_id, "sample": 0, "termination": "agent_stop", "error": error,
                "outcome": {"passed": passed}, "flags": {"context_overflow": False}, "wall_s": 1.0}

    out.write_text("".join(json.dumps(r) + "\n" for r in [
        rec("t1", passed=True), rec("t2", error="RateLimitError: trial key")]))
    calls = []

    def fake_run(arm, agent, domain, task_id, **kw):
        calls.append(task_id)
        return rec(task_id, error="RateLimitError: again" if task_id == "t3" else None)

    monkeypatch.setattr(mod.splits, "task_ids", lambda domain, split: ["t1", "t2", "t3"])
    monkeypatch.setattr(mod, "make_student", lambda student, **kw: object())
    monkeypatch.setattr(mod, "run_episode", fake_run)
    monkeypatch.setattr("sys.argv", ["rollout.py", "A", "telecom", "--out", str(out), "--workers", "1"])
    assert mod.main() == 1  # a conversation crashed: non-zero exit
    assert sorted(calls) == ["t2", "t3"]  # t1 was clean and kept; t2 crashed before, so re-run
    saved = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted(r["task_id"] for r in saved) == ["t1", "t2", "t3"]  # the old crash was replaced
    printed = capsys.readouterr().out
    assert "0/1 passed" in printed and "1 crashed" in printed


def test_rollout_runs_the_frozen_dev_split(tmp_path, monkeypatch, capsys):
    from reflexive import splits

    mod = _load_rollout_script()
    out = tmp_path / "A.jsonl"
    calls = []

    def fake_run(arm, agent, domain, task_id, **kw):
        calls.append(task_id)
        return {"task_id": task_id, "sample": 0, "termination": "agent_stop", "error": None,
                "outcome": {"passed": False}, "flags": {"context_overflow": False}, "wall_s": 1.0}

    monkeypatch.setattr(mod, "make_student", lambda student, **kw: object())
    monkeypatch.setattr(mod, "run_episode", fake_run)
    monkeypatch.setattr("sys.argv", ["rollout.py", "A", "telecom", "--split", "dev", "--out",
                                     str(out), "--workers", "1"])
    assert mod.main() == 0
    dev = splits.task_ids("telecom", "dev")
    assert sorted(calls) == sorted(dev) and not set(dev) & set(splits.task_ids("telecom", "train"))
