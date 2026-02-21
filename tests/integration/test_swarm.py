import pytest
from pydantic import BaseModel, Field

from lemurian.agent import Agent
from lemurian.context import Context
from lemurian.message import MessageRole
from lemurian.session import Session
from lemurian.state import State
from lemurian.swarm import Swarm
from lemurian.tools import tool

from tests.conftest import (
    MockProvider,
    make_text_response,
    make_tool_call_response,
)


# ---------------------------------------------------------------------------
# Swarm.run — session lifecycle
# ---------------------------------------------------------------------------

class TestSwarmLifecycle:
    @pytest.mark.asyncio
    async def test_first_call_requires_agent(self, make_agent):
        swarm = Swarm(agents=[make_agent(name="a")])
        with pytest.raises(
            ValueError, match="agent must be specified"
        ):
            await swarm.run("hello")

    @pytest.mark.asyncio
    async def test_creates_session_on_first_call(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [make_text_response("Hi")]
        swarm = Swarm(agents=[make_agent(name="a")])

        await swarm.run("hello", agent="a")

        assert swarm.session is not None
        assert isinstance(swarm.session, Session)

    @pytest.mark.asyncio
    async def test_multi_turn_reuses_session(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [
            make_text_response("First"),
            make_text_response("Second"),
        ]
        swarm = Swarm(agents=[make_agent(name="a")])

        await swarm.run("msg1", agent="a")
        session_after_first = swarm.session
        await swarm.run("msg2")

        assert swarm.session is session_after_first
        # transcript: user1, assistant1, user2, assistant2
        assert len(swarm.session.transcript) == 4

    @pytest.mark.asyncio
    async def test_switch_agent_on_subsequent_call(
        self, mock_provider
    ):
        provider = MockProvider()
        provider.responses = [
            make_text_response("From A"),
            make_text_response("From B"),
        ]
        from lemurian.agent import Agent

        a = Agent(
            name="a", system_prompt="A", model="m",
            provider=provider,
        )
        b = Agent(
            name="b", system_prompt="B", model="m",
            provider=provider,
        )
        swarm = Swarm(agents=[a, b])

        await swarm.run("hello", agent="a")
        assert swarm.active_agent_name == "a"

        await swarm.run("switch", agent="b")
        assert swarm.active_agent_name == "b"


# ---------------------------------------------------------------------------
# Single-agent swarm — no handoff tool injected
# ---------------------------------------------------------------------------

class TestSwarmSingleAgent:
    @pytest.mark.asyncio
    async def test_no_handoff_tool(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [make_text_response("Solo")]
        swarm = Swarm(agents=[make_agent(name="solo")])

        await swarm.run("hello", agent="solo")

        sent_tools = mock_provider.call_log[0]["tools"]
        assert sent_tools is None


# ---------------------------------------------------------------------------
# Handoff tool generation
# ---------------------------------------------------------------------------

class TestHandoffTool:
    def test_schema_excludes_current_agent(self, make_agent):
        a = make_agent(name="triage", description="Routes")
        b = make_agent(name="billing", description="Bills")
        c = make_agent(name="support", description="Helps")
        swarm = Swarm(agents=[a, b, c])

        ht = swarm._create_handoff_tool("triage")
        schema = ht.model_dump()
        enum = schema["function"]["parameters"]["properties"][
            "agent_name"
        ]["enum"]

        assert "triage" not in enum
        assert "billing" in enum
        assert "support" in enum

    def test_augment_agent_independent_copy(self, make_agent):
        a = make_agent(name="a")
        b = make_agent(name="b")
        swarm = Swarm(agents=[a, b])

        ht = swarm._create_handoff_tool("a")
        augmented = swarm._augment_agent(a, ht)

        assert len(augmented.tools) == 1  # handoff tool
        assert len(a.tools) == 0  # original unchanged


# ---------------------------------------------------------------------------
# Handoff flow
# ---------------------------------------------------------------------------

class TestSwarmHandoff:
    @pytest.mark.asyncio
    async def test_handoff_switches_agent(self):
        """Triage hands off to billing, billing responds."""
        provider = MockProvider()
        from lemurian.agent import Agent

        triage = Agent(
            name="triage", description="Routes",
            system_prompt="Route.", model="m", provider=provider,
        )
        billing = Agent(
            name="billing", description="Bills",
            system_prompt="Bill.", model="m", provider=provider,
        )
        swarm = Swarm(agents=[triage, billing])

        provider.responses = [
            make_tool_call_response(
                "handoff",
                {
                    "agent_name": "billing",
                    "message": "Customer needs billing help",
                },
            ),
            make_text_response("I can help with your bill."),
        ]

        result = await swarm.run(
            "Help with my bill", agent="triage"
        )

        assert result.active_agent == "billing"
        assert (
            result.last_message.content
            == "I can help with your bill."
        )
        assert swarm.context_start > 0

        handoff_msg = (
            swarm.session.transcript[swarm.context_start]
        )
        assert handoff_msg.role == MessageRole.USER
        assert (
            "Customer needs billing help" in handoff_msg.content
        )

    @pytest.mark.asyncio
    async def test_handoff_to_nonexistent_agent(self):
        """Handoff to an unregistered agent returns an error message."""
        provider = MockProvider()
        from lemurian.agent import Agent

        a = Agent(
            name="a", description="A",
            system_prompt="A.", model="m", provider=provider,
        )
        b = Agent(
            name="b", description="B",
            system_prompt="B.", model="m", provider=provider,
        )
        swarm = Swarm(agents=[a, b])

        # The LLM hallucinates a handoff to an agent that doesn't exist
        provider.responses = [
            make_tool_call_response(
                "handoff",
                {"agent_name": "ghost", "message": "go to ghost"},
            ),
        ]

        result = await swarm.run("hello", agent="a")

        assert "not found" in result.last_message.content
        assert result.active_agent == "a"

    @pytest.mark.asyncio
    async def test_max_handoffs_exceeded(self):
        provider = MockProvider()
        from lemurian.agent import Agent

        a = Agent(
            name="a", description="A", system_prompt="A.",
            model="m", provider=provider,
        )
        b = Agent(
            name="b", description="B", system_prompt="B.",
            model="m", provider=provider,
        )
        swarm = Swarm(agents=[a, b], max_handoffs=2)

        provider.responses = [
            make_tool_call_response(
                "handoff",
                {"agent_name": "b", "message": "go to b"},
            ),
            make_tool_call_response(
                "handoff",
                {"agent_name": "a", "message": "go to a"},
            ),
            make_tool_call_response(
                "handoff",
                {"agent_name": "b", "message": "go to b again"},
            ),
        ]

        result = await swarm.run("start", agent="a")

        assert "Maximum handoffs" in result.last_message.content


# ---------------------------------------------------------------------------
# Custom state through context
# ---------------------------------------------------------------------------

class TestSwarmState:
    @pytest.mark.asyncio
    async def test_custom_state_accessible_via_tools(self):
        class CounterState(State):
            counter: int = 0

        @tool
        def increment(context: Context):
            """Increment the counter."""
            context.state.counter += 1
            return context.state.counter

        provider = MockProvider()
        provider.responses = [
            make_tool_call_response("increment", {}),
            make_text_response("Counter is 1"),
        ]

        agent = Agent(
            name="a", system_prompt="Count.", model="m",
            provider=provider, tools=[increment],
        )
        state = CounterState()
        swarm = Swarm(agents=[agent], state=state)

        await swarm.run("increment", agent="a")

        assert state.counter == 1


# ---------------------------------------------------------------------------
# State persistence across handoffs and turns
# ---------------------------------------------------------------------------

class TestSwarmStatePersistence:
    @pytest.mark.asyncio
    async def test_state_persists_across_handoffs(self):
        """State mutated by agent A is visible to agent B after handoff."""

        class SharedState(State):
            notes: list[str] = Field(default_factory=list)

        @tool
        def add_note(context: Context, note: str):
            """Adds a note to state."""
            context.state.notes.append(note)
            return f"Added: {note}"

        @tool
        def read_notes(context: Context):
            """Read all notes from state."""
            return f"Notes: {context.state.notes}"

        provider = MockProvider()
        writer = Agent(
            name="writer", description="Writes notes",
            system_prompt="Write.", model="m",
            provider=provider, tools=[add_note],
        )
        reader = Agent(
            name="reader", description="Reads notes",
            system_prompt="Read.", model="m",
            provider=provider, tools=[read_notes],
        )
        state = SharedState()
        swarm = Swarm(agents=[writer, reader], state=state)

        provider.responses = [
            # Writer adds a note
            make_tool_call_response(
                "add_note", {"note": "hello"}, "call_1",
            ),
            # Writer hands off to reader
            make_tool_call_response(
                "handoff",
                {
                    "agent_name": "reader",
                    "message": "Read the notes",
                },
            ),
            # Reader reads notes
            make_tool_call_response("read_notes", {}),
            make_text_response("Done"),
        ]

        result = await swarm.run("Write and read", agent="writer")

        assert result.active_agent == "reader"
        assert state.notes == ["hello"]
        # Reader's tool output should contain the note
        tool_results = [
            m for m in swarm.session.transcript
            if m.role == MessageRole.TOOL
        ]
        read_result = [
            r for r in tool_results
            if "Notes:" in (r.content or "")
        ]
        assert len(read_result) == 1
        assert "hello" in read_result[0].content

    @pytest.mark.asyncio
    async def test_nested_state_mutation(self):
        """Nested Pydantic models in state can be mutated via tools."""

        class Preferences(BaseModel):
            theme: str = "light"
            font_size: int = 12

        class AppState(State):
            prefs: Preferences = Field(
                default_factory=Preferences
            )

        @tool
        def set_theme(context: Context, theme: str):
            """Update theme."""
            context.state.prefs.theme = theme
            return f"Theme set to {theme}"

        provider = MockProvider()
        agent = Agent(
            name="a", system_prompt="Help.", model="m",
            provider=provider, tools=[set_theme],
        )
        state = AppState()
        swarm = Swarm(agents=[agent], state=state)

        provider.responses = [
            make_tool_call_response(
                "set_theme", {"theme": "dark"},
            ),
            make_text_response("Done"),
        ]

        await swarm.run("Set dark mode", agent="a")

        assert state.prefs.theme == "dark"
        assert state.prefs.font_size == 12  # unchanged

    @pytest.mark.asyncio
    async def test_multiple_state_mutations_accumulate(self):
        """Multiple tool calls across turns accumulate state changes."""

        class CounterState(State):
            counter: int = 0

        @tool
        def increment(context: Context):
            """Increment counter."""
            context.state.counter += 1
            return str(context.state.counter)

        provider = MockProvider()
        agent = Agent(
            name="a", system_prompt="Count.", model="m",
            provider=provider, tools=[increment],
        )
        state = CounterState()
        swarm = Swarm(agents=[agent], state=state)

        # Turn 1: two increments
        provider.responses = [
            make_tool_call_response("increment", {}),
            make_tool_call_response("increment", {}),
            make_text_response("Counter is 2"),
        ]
        await swarm.run("Increment twice", agent="a")
        assert state.counter == 2

        # Turn 2: one more increment
        provider.responses = [
            make_tool_call_response("increment", {}),
            make_text_response("Counter is 3"),
        ]
        await swarm.run("One more")
        assert state.counter == 3
