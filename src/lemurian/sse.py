"""Server-Sent Events adapter for streaming events."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict

from lemurian.events import RunCompleteEvent, StreamEvent


async def sse_generator(
    event_stream: AsyncIterator[StreamEvent],
) -> AsyncIterator[str]:
    """Convert a StreamEvent async iterator into SSE-formatted strings."""
    async for event in event_stream:
        event_type = type(event).__name__
        if isinstance(event, RunCompleteEvent):
            data = "{}"
        else:
            data = json.dumps(asdict(event))
        yield f"event: {event_type}\ndata: {data}\n\n"
    yield "event: done\ndata: {}\n\n"
