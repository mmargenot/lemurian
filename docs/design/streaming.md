# Streaming Design

## Goals

- Token-level streaming from providers, enabling real-time UIs and SSE transport.
- Parallel tool-call batch semantics (all requests recorded upfront, execution
  short-circuits on handoff).
- Provider-agnostic `stream_complete()` contract.
- Minimal new abstractions — streaming adds to existing code, not beside it.

## Non-goals

- Structured output streaming (partial JSON for `structured_completion()`).
- Backpressure / flow control.

---

## Design Principles

1. **`run()` stays canonical.** The agent loop lives in `Runner.run()` and
   `Swarm.run()`. Streaming wraps it — it does not replace it. This avoids
   restructuring tool execution to work inside an async generator.

2. **Three event types, not nine.** Following the OpenAI Agents SDK pattern:
   a raw stream event for token deltas, a run-item event for discrete steps
   (tool calls, messages, handoffs), and a completion signal. Consumers
   filter on `event.name` rather than `isinstance` checks against a class
   hierarchy.

3. **No adapter types.** The accumulator produces `ToolCall` objects that the
   transcript consumes directly. No mock/shim conversions.

4. **Tool outcomes are typed.** `_execute_one_tool` returns a dataclass, not
   a dict with magic string keys.

---

## New Abstractions

### Provider layer (streaming.py)

```python
@dataclass
class ToolCallFragment:
    index: int
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str | None = None

@dataclass
class StreamChunk:
    content_delta: str | None = None
    tool_call_fragments: list[ToolCallFragment] | None = None
    finish_reason: str | None = None

@dataclass
class ToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""

class ToolCallAccumulator:
    def feed(self, fragment: ToolCallFragment) -> None: ...
    def finalize(self) -> list[ToolCall]: ...
```

### Provider base class

```python
class ModelProvider:
    async def stream_complete(self, model, messages, tools=None):
        """Yield StreamChunk objects. Subclasses override."""
        raise NotImplementedError
```

All four providers (OpenAI, OpenRouter, vLLM, Modal) share a single
`_openai_stream_complete()` helper since they all use the OpenAI SDK.

### Events (events.py)

```python
@dataclass
class StreamEvent:
    """Base for all streaming events."""

@dataclass
class RawResponseEvent(StreamEvent):
    """Token-level delta from the provider."""
    content: str = ""

@dataclass
class RunItemEvent(StreamEvent):
    """Discrete step in the agent loop."""
    name: str = ""        # "tool_call", "tool_result", "message", "handoff"
    data: dict = field(default_factory=dict)

@dataclass
class RunCompleteEvent(StreamEvent):
    """Final event. Always last."""
    result: Any = None
```

### Runner changes

- `run()` stays as the core loop. Refactored to use `stream_complete()`
  instead of `complete()`, with `ToolCallAccumulator` to reassemble tool
  calls.
- `parallel_tool_calls` parameter controls batch vs interleaved transcript
  format. Both modes execute sequentially and short-circuit on handoff to
  prevent state mutations after a handoff signal.
- `iter()` is a thin wrapper that runs `run()` logic while yielding events
  through a callback or queue.
- `ToolCallRequestMessage.serialize_tool_calls` works with `ToolCall`
  directly — no adapter.

### Swarm changes

- `iter()` wraps `Runner.iter()`, intercepts `RunCompleteEvent` during
  handoffs (does not re-yield it), emits `RunItemEvent(name="handoff")`,
  and only yields `RunCompleteEvent` when the chain is truly done.
- `run()` drains `iter()` with a `None` guard.

### SSE (sse.py)

```python
async def sse_generator(event_stream) -> AsyncIterator[str]:
    async for event in event_stream:
        event_type = type(event).__name__
        data = json.dumps(asdict(event))
        yield f"event: {event_type}\ndata: {data}\n\n"
    yield "event: done\ndata: {}\n\n"
```

---

## Lessons from first implementation

1. Making `iter()` canonical forced the entire tool execution path into
   batch results and outcome collection, adding ~100 lines of scaffolding.
   Keeping `run()` canonical avoids this.

2. `_PendingToolCall` vs OpenAI-shaped objects in `ToolCallRequestMessage`
   created an impedance mismatch requiring `_MockToolCall`/`_MockFunction`.
   A single `ToolCall` dataclass used by both the accumulator and transcript
   eliminates this.

3. Nine event dataclasses were overkill. Most consumers only care about
   text deltas, tool activity, and completion. Three event types with a
   `name` discriminator covers all use cases.

4. `_execute_one_tool` returning untyped dicts led to `"handoff" in outcome`
   checks. A `_ToolOutcome` dataclass catches typos at definition time.

5. `Swarm.iter()` re-yielding `RunComplete` from the Runner during handoffs
   broke the invariant that `RunComplete` = stream is done. The Swarm must
   intercept and suppress intermediate completions.
