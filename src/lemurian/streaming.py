"""Streaming primitives for provider responses.

Providers yield :class:`StreamChunk` objects.  The
:class:`ToolCallAccumulator` reassembles tool calls whose arguments
arrive in fragments across multiple chunks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolCallFragment:
    """A fragment of a tool call from a streaming chunk."""

    index: int
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str | None = None


@dataclass
class StreamChunk:
    """Normalised streaming chunk from any provider."""

    content_delta: str | None = None
    tool_call_fragments: list[ToolCallFragment] | None = None
    finish_reason: str | None = None


@dataclass
class ToolCall:
    """A resolved tool call ready for the transcript."""

    id: str = ""
    name: str = ""
    arguments: str = ""


class ToolCallAccumulator:
    """Assembles complete tool calls from streaming fragments."""

    def __init__(self) -> None:
        self._pending: dict[int, ToolCall] = {}

    def feed(self, fragment: ToolCallFragment) -> None:
        if fragment.index not in self._pending:
            self._pending[fragment.index] = ToolCall()
        tc = self._pending[fragment.index]
        if fragment.call_id is not None:
            tc.id = fragment.call_id
        if fragment.name is not None:
            tc.name = fragment.name
        if fragment.arguments_delta is not None:
            tc.arguments += fragment.arguments_delta

    def finalize(self) -> list[ToolCall]:
        """Return completed tool calls in index order."""
        return [self._pending[i] for i in sorted(self._pending)]
