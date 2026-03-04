"""Streaming events emitted during agent execution."""

from __future__ import annotations

from dataclasses import dataclass
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
    """Base for discrete execution step events."""


@dataclass
class ToolCallEvent(RunItemEvent):
    """A tool was called and produced a result."""

    tool_name: str = ""
    tool_call_id: str = ""
    arguments: str = ""
    output: str = ""
    is_error: bool = False


@dataclass
class MessageEvent(RunItemEvent):
    """The agent produced a text response."""

    content: str = ""


@dataclass
class HandoffEvent(RunItemEvent):
    """The agent is handing off to another agent."""

    source_agent: str = ""
    target_agent: str = ""
    message: str = ""


@dataclass
class RunCompleteEvent(StreamEvent):
    """Final event — always the last event yielded."""

    result: Any = None
