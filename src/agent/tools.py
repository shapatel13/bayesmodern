from __future__ import annotations

from pydantic import BaseModel


class ToolDescriptor(BaseModel):
    name: str
    purpose: str


DEFAULT_TOOLS = [
    ToolDescriptor(name="bayesian_updater", purpose="Deterministic posterior updates."),
    ToolDescriptor(name="next_best_test_engine", purpose="Ranks candidate tests by information gain and stewardship."),
    ToolDescriptor(name="contradiction_checker", purpose="Validates coherence and already-done-test logic."),
]

