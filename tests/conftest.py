"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_intent_judge_calls(monkeypatch):
    """Tests never call the "no yes" rule's LLM judge (SEL-88): word match only, unless a test
    passes its own fake judge."""
    monkeypatch.setenv("REFLEXIVE_INTENT_JUDGE", "off")
