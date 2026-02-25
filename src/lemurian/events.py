"""Streaming events emitted during agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamEvent:
    """Base for all streaming events."""


@dataclass
class RawResponseEvent(StreamEvent):
    """Token-level delta from the provider stream."""

    content: str = ""


@dataclass
class RunItemEvent(StreamEvent):
    """A discrete step in the agent loop.

    ``name`` values: ``"tool_call"``, ``"tool_result"``,
    ``"message"``, ``"handoff"``.
    """

    name: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class RunCompleteEvent(StreamEvent):
    """Final event — always the last event yielded."""

    result: Any = None
