"""Tests for streaming (iter), parallel tool execution, and SSE."""

import json

import pytest

from lemurian.agent import Agent
from lemurian.events import (
    RawResponseEvent,
    RunCompleteEvent,
    RunItemEvent,
)
from lemurian.message import MessageRole
from lemurian.runner import Runner
from lemurian.session import Session
from lemurian.state import State
from lemurian.swarm import Swarm
from lemurian.tools import tool

from tests.conftest import (
    MockProvider,
    echo,
    make_multi_tool_call_response,
    make_text_response,
    make_tool_call_response,
)


# ---------------------------------------------------------------------------
# Runner.iter() — event stream
# ---------------------------------------------------------------------------

class TestRunnerIter:
    @pytest.mark.asyncio
    async def test_text_response_events(self):
        provider = MockProvider()
        provider.responses = [make_text_response("Hello!")]
        agent = Agent(name="a", system_prompt="Help.", model="m", provider=provider)
        session = Session(session_id="s1")

        events = [e async for e in Runner().iter(agent, session, State())]
        types = [type(e) for e in events]

        assert types == [
            RawResponseEvent,
            RunItemEvent,
            RunCompleteEvent,
        ]
        assert events[-1].result.last_message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_tool_call_events(self):
        provider = MockProvider()
        provider.responses = [
            make_tool_call_response("echo", {"text": "hi"}),
            make_text_response("Done"),
        ]
        agent = Agent(
            name="a", system_prompt="Help.", model="m",
            provider=provider, tools=[echo],
        )
        session = Session(session_id="s1")

        events = [e async for e in Runner().iter(agent, session, State())]
        types = [type(e) for e in events]

        # Turn 1: tool call (no content delta), Turn 2: text
        assert types == [
            RunItemEvent,       # tool_call
            RawResponseEvent,   # text delta
            RunItemEvent,       # message
            RunCompleteEvent,
        ]
        assert events[0].name == "tool_call"
        assert events[0].data["tool_name"] == "echo"
        assert events[0].data["output"] == "hi"

    @pytest.mark.asyncio
    async def test_run_equals_iter_result(self):
        provider = MockProvider()
        agent = Agent(
            name="a", system_prompt="Help.", model="m",
            provider=provider, tools=[echo],
        )

        provider.responses = [
            make_tool_call_response("echo", {"text": "x"}),
            make_text_response("Done"),
        ]
        run_result = await Runner().run(agent, Session(session_id="s1"), State())

        provider.responses = [
            make_tool_call_response("echo", {"text": "x"}),
            make_text_response("Done"),
        ]
        iter_result = None
        async for event in Runner().iter(agent, Session(session_id="s2"), State()):
            if isinstance(event, RunCompleteEvent):
                iter_result = event.result

        assert run_result.last_message.content == iter_result.last_message.content

    @pytest.mark.asyncio
    async def test_max_turns_events(self):
        """iter() emits RunCompleteEvent when max turns exceeded."""
        provider = MockProvider()
        provider.responses = [
            make_tool_call_response("echo", {"text": "loop"})
            for _ in range(5)
        ]
        agent = Agent(
            name="a", system_prompt="Help.", model="m",
            provider=provider, tools=[echo],
        )
        events = [
            e async for e in Runner(max_turns=2).iter(
                agent, Session(session_id="s1"), State(),
            )
        ]
        types = [type(e) for e in events]

        # 2 turns of tool_call events, then RunCompleteEvent
        assert types[-1] is RunCompleteEvent
        assert "Maximum turns" in events[-1].result.last_message.content

    @pytest.mark.asyncio
    async def test_tool_error_events(self):
        """iter() emits tool_call event with is_error for failures."""

        @tool
        def explode():
            """Always fails."""
            raise RuntimeError("boom")

        provider = MockProvider()
        provider.responses = [
            make_tool_call_response("explode", {}),
            make_text_response("Recovered"),
        ]
        agent = Agent(
            name="a", system_prompt="Help.", model="m",
            provider=provider, tools=[explode],
        )
        events = [
            e async for e in Runner().iter(
                agent, Session(session_id="s1"), State(),
            )
        ]

        tool_events = [
            e for e in events
            if isinstance(e, RunItemEvent) and e.name == "tool_call"
        ]
        assert len(tool_events) == 1
        assert tool_events[0].data["is_error"] is True
        assert "boom" in tool_events[0].data["output"]


# ---------------------------------------------------------------------------
# Parallel tool execution
# ---------------------------------------------------------------------------

class TestParallelTools:
    @pytest.mark.asyncio
    async def test_parallel_all_results_in_transcript(self):
        provider = MockProvider()
        provider.responses = [
            make_multi_tool_call_response([
                ("echo", {"text": "one"}, "c1"),
                ("echo", {"text": "two"}, "c2"),
            ]),
            make_text_response("Done"),
        ]
        agent = Agent(
            name="a", system_prompt="Help.", model="m",
            provider=provider, tools=[echo],
        )
        session = Session(session_id="s1")
        await Runner(parallel_tool_calls=True).run(agent, session, State())

        tool_results = [m for m in session.transcript if m.role == MessageRole.TOOL]
        assert {r.content for r in tool_results} == {"one", "two"}

    @pytest.mark.asyncio
    async def test_parallel_preserves_order(self):
        provider = MockProvider()
        provider.responses = [
            make_multi_tool_call_response([
                ("echo", {"text": "first"}, "c1"),
                ("echo", {"text": "second"}, "c2"),
                ("echo", {"text": "third"}, "c3"),
            ]),
            make_text_response("Done"),
        ]
        agent = Agent(
            name="a", system_prompt="Help.", model="m",
            provider=provider, tools=[echo],
        )
        session = Session(session_id="s1")
        await Runner(parallel_tool_calls=True).run(agent, session, State())

        tool_results = [m for m in session.transcript if m.role == MessageRole.TOOL]
        assert [r.content for r in tool_results] == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_single_call_uses_sequential(self):
        """A single tool call goes through sequential even with parallel=True."""
        call_log = []

        @tool
        def tracked(label: str):
            """Track call."""
            call_log.append(label)
            return label

        provider = MockProvider()
        provider.responses = [
            make_tool_call_response("tracked", {"label": "one"}),
            make_text_response("Done"),
        ]
        agent = Agent(
            name="a", system_prompt="Help.", model="m",
            provider=provider, tools=[tracked],
        )
        session = Session(session_id="s1")
        result = await Runner(parallel_tool_calls=True).run(
            agent, session, State(),
        )

        assert result.last_message.content == "Done"
        assert call_log == ["one"]

    @pytest.mark.asyncio
    async def test_parallel_error_recorded(self):
        """Tool errors in parallel execution are recorded correctly."""

        @tool
        def explode():
            """Always fails."""
            raise RuntimeError("parallel boom")

        provider = MockProvider()
        provider.responses = [
            make_multi_tool_call_response([
                ("echo", {"text": "ok"}, "c1"),
                ("explode", {}, "c2"),
            ]),
            make_text_response("Recovered"),
        ]
        agent = Agent(
            name="a", system_prompt="Help.", model="m",
            provider=provider, tools=[echo, explode],
        )
        session = Session(session_id="s1")
        result = await Runner(parallel_tool_calls=True).run(
            agent, session, State(),
        )

        assert result.last_message.content == "Recovered"
        tool_results = [
            m for m in session.transcript
            if m.role == MessageRole.TOOL
        ]
        outputs = [r.content for r in tool_results]
        assert "ok" in outputs
        assert any("parallel boom" in o for o in outputs)


# ---------------------------------------------------------------------------
# Swarm.iter()
# ---------------------------------------------------------------------------

class TestSwarmIter:
    @pytest.mark.asyncio
    async def test_iter_yields_events(self):
        provider = MockProvider()
        provider.responses = [make_text_response("Hi")]
        agent = Agent(name="a", system_prompt="Help.", model="m", provider=provider)
        swarm = Swarm(agents=[agent])

        events = [e async for e in swarm.iter("hello", agent="a")]
        types = [type(e) for e in events]

        assert types == [
            RawResponseEvent,
            RunItemEvent,
            RunCompleteEvent,
        ]

    @pytest.mark.asyncio
    async def test_iter_emits_handoff_event(self):
        provider = MockProvider()
        a = Agent(name="a", description="A", system_prompt="A.", model="m", provider=provider)
        b = Agent(name="b", description="B", system_prompt="B.", model="m", provider=provider)
        provider.responses = [
            make_tool_call_response("handoff", {"agent_name": "b", "message": "go to b"}),
            make_text_response("Hello from b"),
        ]
        swarm = Swarm(agents=[a, b])

        events = [e async for e in swarm.iter("hello", agent="a")]
        types = [type(e) for e in events]

        # Agent A: tool_call, then Swarm emits handoff
        # Agent B: raw delta, message, complete
        assert types == [
            RunItemEvent,       # tool_call (handoff tool)
            RunItemEvent,       # handoff
            RawResponseEvent,   # text delta from b
            RunItemEvent,       # message from b
            RunCompleteEvent,
        ]
        assert events[1].name == "handoff"
        assert events[1].data["target_agent"] == "b"

    @pytest.mark.asyncio
    async def test_no_intermediate_run_complete(self):
        """RunCompleteEvent is not emitted during handoffs — only at the end."""
        provider = MockProvider()
        a = Agent(name="a", description="A", system_prompt="A.", model="m", provider=provider)
        b = Agent(name="b", description="B", system_prompt="B.", model="m", provider=provider)
        provider.responses = [
            make_tool_call_response("handoff", {"agent_name": "b", "message": "go"}),
            make_text_response("Done"),
        ]
        swarm = Swarm(agents=[a, b])

        completes = [
            e async for e in swarm.iter("hello", agent="a")
            if isinstance(e, RunCompleteEvent)
        ]
        assert len(completes) == 1  # only the final one

    @pytest.mark.asyncio
    async def test_run_and_iter_same_result(self):
        provider = MockProvider()
        agent = Agent(name="a", system_prompt="Help.", model="m", provider=provider)

        provider.responses = [make_text_response("Hi")]
        run_result = await Swarm(agents=[agent]).run("hello", agent="a")

        provider.responses = [make_text_response("Hi")]
        iter_result = None
        async for event in Swarm(agents=[agent]).iter("hello", agent="a"):
            if isinstance(event, RunCompleteEvent):
                iter_result = event.result

        assert run_result.last_message.content == iter_result.last_message.content


# ---------------------------------------------------------------------------
# SSE adapter
# ---------------------------------------------------------------------------

class TestSSE:
    @pytest.mark.asyncio
    async def test_sse_format(self):
        from lemurian.sse import sse_generator

        provider = MockProvider()
        provider.responses = [make_text_response("Hi")]
        agent = Agent(
            name="a", system_prompt="Help.", model="m",
            provider=provider,
        )

        frames = [
            f async for f in sse_generator(
                Runner().iter(
                    agent, Session(session_id="s1"), State(),
                ),
            )
        ]

        # 3 event frames + done sentinel
        assert len(frames) == 4
        assert frames[0].startswith("event: RawResponseEvent\n")
        assert frames[1].startswith("event: RunItemEvent\n")
        assert frames[2].startswith("event: RunCompleteEvent\n")
        assert frames[3] == "event: done\ndata: {}\n\n"
        for frame in frames:
            assert frame.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_sse_raw_response_payload(self):
        from lemurian.sse import sse_generator

        provider = MockProvider()
        provider.responses = [make_text_response("Hello world")]
        agent = Agent(
            name="a", system_prompt="Help.", model="m",
            provider=provider,
        )

        frames = [
            f async for f in sse_generator(
                Runner().iter(
                    agent, Session(session_id="s1"), State(),
                ),
            )
        ]

        data_line = frames[0].split("\n")[1]
        payload = json.loads(data_line.removeprefix("data: "))
        assert payload["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_sse_run_complete_has_empty_data(self):
        """RunCompleteEvent serializes as empty JSON."""
        from lemurian.sse import sse_generator

        provider = MockProvider()
        provider.responses = [make_text_response("Hi")]
        agent = Agent(
            name="a", system_prompt="Help.", model="m",
            provider=provider,
        )

        frames = [
            f async for f in sse_generator(
                Runner().iter(
                    agent, Session(session_id="s1"), State(),
                ),
            )
        ]

        # RunCompleteEvent frame (index 2) has empty data
        assert frames[2] == (
            "event: RunCompleteEvent\ndata: {}\n\n"
        )
