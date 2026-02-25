import pytest
from pydantic import BaseModel, Field

from lemurian.agent import Agent
from lemurian.capability import Capability
from lemurian.context import Context
from lemurian.handoff import handoff
from lemurian.message import MessageRole
from lemurian.session import Session
from lemurian.state import State
from lemurian.swarm import Swarm
from lemurian.tools import tool

from tests.conftest import (
    MockCapability,
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
# Single-agent swarm — no handoff tools
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
# Handoff resolution
# ---------------------------------------------------------------------------

class TestHandoffResolution:
    def test_resolve_handoffs_from_agent(self, make_agent):
        billing_ho = handoff("billing", "Bills")
        support_ho = handoff("support", "Helps")
        a = make_agent(
            name="triage",
            handoffs=[billing_ho, support_ho],
        )
        swarm = Swarm(agents=[a])

        resolved = swarm._resolve_handoffs(a)

        assert len(resolved) == 2
        assert resolved[0].tool_name == "transfer_to_billing"
        assert resolved[1].tool_name == "transfer_to_support"

    def test_resolve_agent_independent_copy(self, make_agent):
        a = make_agent(name="a")
        b = make_agent(name="b")
        swarm = Swarm(agents=[a, b])

        resolved = swarm._resolve_agent(a)

        assert len(resolved.tools) == 0
        assert len(a.tools) == 0  # original unchanged


# ---------------------------------------------------------------------------
# Handoff flow
# ---------------------------------------------------------------------------

class TestSwarmHandoff:
    @pytest.mark.asyncio
    async def test_handoff_switches_agent(self):
        """Triage hands off to billing, billing responds."""
        provider = MockProvider()

        billing = Agent(
            name="billing", description="Bills",
            system_prompt="Bill.", model="m", provider=provider,
        )
        triage = Agent(
            name="triage", description="Routes",
            system_prompt="Route.", model="m", provider=provider,
            handoffs=[handoff("billing", "Bills")],
        )
        swarm = Swarm(agents=[triage, billing])

        provider.responses = [
            make_tool_call_response(
                "transfer_to_billing",
                {"message": "Customer needs billing help"},
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
        """Handoff to an unregistered agent returns an error."""
        provider = MockProvider()

        a = Agent(
            name="a", description="A",
            system_prompt="A.", model="m", provider=provider,
            handoffs=[handoff("ghost", "Does not exist")],
        )
        b = Agent(
            name="b", description="B",
            system_prompt="B.", model="m", provider=provider,
        )
        swarm = Swarm(agents=[a, b])

        provider.responses = [
            make_tool_call_response(
                "transfer_to_ghost",
                {"message": "go to ghost"},
            ),
        ]

        result = await swarm.run("hello", agent="a")

        assert "not found" in result.last_message.content
        assert result.active_agent == "a"

    @pytest.mark.asyncio
    async def test_max_handoffs_exceeded(self):
        provider = MockProvider()

        a = Agent(
            name="a", description="A", system_prompt="A.",
            model="m", provider=provider,
            handoffs=[handoff("b", "B")],
        )
        b = Agent(
            name="b", description="B", system_prompt="B.",
            model="m", provider=provider,
            handoffs=[handoff("a", "A")],
        )
        swarm = Swarm(agents=[a, b], max_handoffs=2)

        provider.responses = [
            make_tool_call_response(
                "transfer_to_b",
                {"message": "go to b"},
            ),
            make_tool_call_response(
                "transfer_to_a",
                {"message": "go to a"},
            ),
            make_tool_call_response(
                "transfer_to_b",
                {"message": "go to b again"},
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
            handoffs=[handoff("reader", "Reads notes")],
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
                "transfer_to_reader",
                {"message": "Read the notes"},
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


# ---------------------------------------------------------------------------
# Swarm.add_capability
# ---------------------------------------------------------------------------

class TestSwarmAddCapability:
    def test_add_capability_to_all_agents(
        self, make_agent, cap_tool_raven
    ):
        montresor = make_agent(name="montresor")
        usher = make_agent(name="usher")
        swarm = Swarm(agents=[montresor, usher])
        cap = MockCapability(name="raven", tool_list=[cap_tool_raven])
        swarm.add_capability(cap)
        resolved_montresor = swarm._resolve_agent(montresor)
        resolved_usher = swarm._resolve_agent(usher)
        assert any(t.name == "raven" for t in resolved_montresor.tools)
        assert any(t.name == "raven" for t in resolved_usher.tools)

    def test_add_capability_to_specific_agents(
        self, make_agent, cap_tool_annabel
    ):
        montresor = make_agent(name="montresor")
        usher = make_agent(name="usher")
        swarm = Swarm(agents=[montresor, usher])
        cap = MockCapability(name="annabel_lee", tool_list=[cap_tool_annabel])
        swarm.add_capability(cap, agents=["montresor"])
        resolved_montresor = swarm._resolve_agent(montresor)
        resolved_usher = swarm._resolve_agent(usher)
        montresor_tools = [t.name for t in resolved_montresor.tools]
        usher_tools = [t.name for t in resolved_usher.tools]
        assert "annabel" in montresor_tools
        assert "annabel" not in usher_tools

    def test_on_attach_called_per_agent(self, make_agent):
        montresor = make_agent(name="montresor")
        usher = make_agent(name="usher")
        state = State()
        swarm = Swarm(agents=[montresor, usher], state=state)
        cap = MockCapability(name="raven")
        swarm.add_capability(cap)
        assert len(cap.attach_log) == 2
        assert all(s is state for s in cap.attach_log)

    def test_invalid_agent_raises(self, make_agent):
        swarm = Swarm(agents=[make_agent(name="montresor")])
        cap = MockCapability(name="raven")
        with pytest.raises(ValueError, match="fortunato"):
            swarm.add_capability(cap, agents=["fortunato"])

    def test_add_multiple_capabilities_to_swarm(
        self, make_agent, cap_tool_raven, cap_tool_annabel
    ):
        montresor = make_agent(name="montresor")
        swarm = Swarm(agents=[montresor])
        raven_cap = MockCapability(name="raven", tool_list=[cap_tool_raven])
        annabel_cap = MockCapability(name="annabel_lee", tool_list=[cap_tool_annabel])
        swarm.add_capability(raven_cap)
        swarm.add_capability(annabel_cap)
        resolved = swarm._resolve_agent(montresor)
        tool_names = [t.name for t in resolved.tools]
        assert "raven" in tool_names
        assert "annabel" in tool_names

    def test_add_different_capabilities_to_different_agents(
        self, make_agent, cap_tool_raven, cap_tool_annabel
    ):
        montresor = make_agent(name="montresor")
        usher = make_agent(name="usher")
        swarm = Swarm(agents=[montresor, usher])
        raven_cap = MockCapability(name="raven", tool_list=[cap_tool_raven])
        annabel_cap = MockCapability(name="annabel_lee", tool_list=[cap_tool_annabel])
        swarm.add_capability(raven_cap, agents=["montresor"])
        swarm.add_capability(annabel_cap, agents=["usher"])
        montresor_tools = [t.name for t in swarm._resolve_agent(montresor).tools]
        usher_tools = [t.name for t in swarm._resolve_agent(usher).tools]
        assert "raven" in montresor_tools
        assert "annabel" not in montresor_tools
        assert "annabel" in usher_tools
        assert "raven" not in usher_tools


# ---------------------------------------------------------------------------
# Swarm.remove_capability
# ---------------------------------------------------------------------------

class TestSwarmRemoveCapability:
    def test_on_detach_called_once(self, make_agent):
        montresor = make_agent(name="montresor")
        usher = make_agent(name="usher")
        state = State()
        swarm = Swarm(agents=[montresor, usher], state=state)
        cap = MockCapability(name="raven")
        swarm.add_capability(cap)
        swarm.remove_capability("raven")
        assert len(cap.detach_log) == 1
        assert cap.detach_log[0] is state

    def test_removed_cap_tools_not_in_resolved_agent(
        self, make_agent, cap_tool_raven
    ):
        montresor = make_agent(name="montresor")
        swarm = Swarm(agents=[montresor])
        cap = MockCapability(name="raven", tool_list=[cap_tool_raven])
        swarm.add_capability(cap)
        swarm.remove_capability("raven")
        resolved = swarm._resolve_agent(montresor)
        assert all(t.name != "raven" for t in resolved.tools)

    def test_remove_nonexistent_raises_keyerror(self, make_agent):
        swarm = Swarm(agents=[make_agent(name="montresor")])
        with pytest.raises(KeyError):
            swarm.remove_capability("black_cat")


# ---------------------------------------------------------------------------
# Swarm._resolve_agent — capability placement combinations
# ---------------------------------------------------------------------------

class TestSwarmResolveCapabilities:
    def test_resolve_swarm_caps_only(
        self, make_agent, cap_tool_raven, cap_tool_annabel
    ):
        montresor = make_agent(name="montresor")
        usher = make_agent(name="usher")
        swarm = Swarm(agents=[montresor, usher])
        swarm.add_capability(
            MockCapability(name="raven", tool_list=[cap_tool_raven]),
            agents=["montresor"],
        )
        swarm.add_capability(
            MockCapability(name="annabel_lee", tool_list=[cap_tool_annabel]),
            agents=["montresor"],
        )
        resolved = swarm._resolve_agent(montresor)
        tool_names = [t.name for t in resolved.tools]
        assert "raven" in tool_names
        assert "annabel" in tool_names

    def test_resolve_agent_caps_only(
        self, make_agent, cap_tool_raven, cap_tool_annabel
    ):
        raven_cap = MockCapability(name="raven", tool_list=[cap_tool_raven])
        annabel_cap = MockCapability(name="annabel_lee", tool_list=[cap_tool_annabel])
        montresor = make_agent(name="montresor", capabilities=[raven_cap, annabel_cap])
        swarm = Swarm(agents=[montresor])
        resolved = swarm._resolve_agent(montresor)
        tool_names = [t.name for t in resolved.tools]
        assert "raven" in tool_names
        assert "annabel" in tool_names

    def test_resolve_swarm_cap_with_agent_tools(
        self, make_agent, sample_tool, cap_tool_raven
    ):
        montresor = make_agent(name="montresor", tools=[sample_tool])
        swarm = Swarm(agents=[montresor])
        swarm.add_capability(
            MockCapability(name="raven", tool_list=[cap_tool_raven]),
        )
        resolved = swarm._resolve_agent(montresor)
        tool_names = [t.name for t in resolved.tools]
        assert "greet" in tool_names
        assert "raven" in tool_names

    def test_resolve_agent_cap_with_agent_tools(
        self, make_agent, sample_tool, cap_tool_raven
    ):
        raven_cap = MockCapability(name="raven", tool_list=[cap_tool_raven])
        montresor = make_agent(
            name="montresor", tools=[sample_tool], capabilities=[raven_cap],
        )
        swarm = Swarm(agents=[montresor])
        resolved = swarm._resolve_agent(montresor)
        tool_names = [t.name for t in resolved.tools]
        assert "greet" in tool_names
        assert "raven" in tool_names

    def test_resolve_both_level_caps_with_agent_tools(
        self, make_agent, sample_tool, cap_tool_raven, cap_tool_annabel
    ):
        raven_cap = MockCapability(name="raven", tool_list=[cap_tool_raven])
        annabel_cap = MockCapability(name="annabel_lee", tool_list=[cap_tool_annabel])
        montresor = make_agent(
            name="montresor", tools=[sample_tool], capabilities=[raven_cap],
        )
        usher = make_agent(name="usher")
        swarm = Swarm(agents=[montresor, usher])
        swarm.add_capability(annabel_cap, agents=["montresor"])
        resolved = swarm._resolve_agent(montresor)
        tool_names = [t.name for t in resolved.tools]
        assert "greet" in tool_names
        assert "raven" in tool_names
        assert "annabel" in tool_names

    def test_resolve_multiple_swarm_caps_on_same_agent(
        self, make_agent, cap_tool_raven, cap_tool_lenore
    ):
        montresor = make_agent(name="montresor")
        swarm = Swarm(agents=[montresor])
        swarm.add_capability(
            MockCapability(name="raven", tool_list=[cap_tool_raven]),
        )
        swarm.add_capability(
            MockCapability(name="lenore", tool_list=[cap_tool_lenore]),
        )
        resolved = swarm._resolve_agent(montresor)
        tool_names = [t.name for t in resolved.tools]
        assert "raven" in tool_names
        assert "lenore" in tool_names

    def test_resolve_caps_on_swarm_but_not_agent(
        self, make_agent, cap_tool_raven
    ):
        montresor = make_agent(name="montresor")
        swarm = Swarm(agents=[montresor])
        swarm.add_capability(
            MockCapability(name="raven", tool_list=[cap_tool_raven]),
        )
        resolved = swarm._resolve_agent(montresor)
        tool_names = [t.name for t in resolved.tools]
        assert "raven" in tool_names

    def test_resolve_does_not_mutate_original(
        self, make_agent, cap_tool_raven
    ):
        raven_cap = MockCapability(name="raven", tool_list=[cap_tool_raven])
        montresor = make_agent(name="montresor", capabilities=[raven_cap])
        usher = make_agent(name="usher")
        swarm = Swarm(agents=[montresor, usher])
        swarm._resolve_agent(montresor)
        assert len(montresor.tools) == 0
        assert len(montresor.capabilities) == 1

    def test_resolve_single_agent_no_handoff_with_caps(
        self, make_agent, cap_tool_raven
    ):
        montresor = make_agent(name="montresor")
        swarm = Swarm(agents=[montresor])
        swarm.add_capability(
            MockCapability(name="raven", tool_list=[cap_tool_raven]),
        )
        resolved = swarm._resolve_agent(montresor)
        tool_names = [t.name for t in resolved.tools]
        assert "raven" in tool_names

    def test_resolve_multi_agent_caps_no_auto_handoff(
        self, make_agent, cap_tool_raven
    ):
        """Handoffs are no longer injected as tools by _resolve_agent."""
        montresor = make_agent(name="montresor")
        usher = make_agent(name="usher")
        swarm = Swarm(agents=[montresor, usher])
        swarm.add_capability(
            MockCapability(name="raven", tool_list=[cap_tool_raven]),
            agents=["montresor"],
        )
        resolved = swarm._resolve_agent(montresor)
        tool_names = [t.name for t in resolved.tools]
        assert "raven" in tool_names
        # _resolve_agent no longer injects handoff tools
        assert "handoff" not in tool_names

    def test_resolve_duplicate_cap_on_agent_and_swarm_raises(
        self, make_agent, cap_tool_raven
    ):
        """Same raven tool on both agent.capabilities and swarm level."""
        raven_cap = MockCapability(name="raven_agent", tool_list=[cap_tool_raven])
        raven_swarm = MockCapability(name="raven_swarm", tool_list=[cap_tool_raven])
        montresor = make_agent(name="montresor", capabilities=[raven_cap])
        usher = make_agent(name="usher")
        swarm = Swarm(agents=[montresor, usher])
        swarm.add_capability(raven_swarm, agents=["montresor"])
        with pytest.raises(ValueError, match="Duplicate tool name"):
            swarm._resolve_agent(montresor)


# ---------------------------------------------------------------------------
# Swarm._resolve_agent — duplicate tool detection
# ---------------------------------------------------------------------------

class TestSwarmResolveDuplicateDetection:
    def test_duplicate_agent_tool_vs_swarm_cap(
        self, make_agent, sample_tool
    ):
        """Agent has 'greet' tool; swarm cap also provides 'greet'."""
        @tool
        def greet():
            """Quoth the Raven."""
            return "Quoth the Raven, 'Nevermore.'"
        cap = MockCapability(name="raven", tool_list=[greet])
        montresor = make_agent(name="montresor", tools=[sample_tool])
        usher = make_agent(name="usher")
        swarm = Swarm(agents=[montresor, usher])
        swarm.add_capability(cap, agents=["montresor"])
        with pytest.raises(ValueError, match="Duplicate tool name"):
            swarm._resolve_agent(montresor)

    def test_duplicate_agent_cap_vs_swarm_cap(
        self, make_agent, cap_tool_raven
    ):
        """Agent-level and swarm-level capabilities both provide 'raven'."""
        agent_cap = MockCapability(name="raven_agent", tool_list=[cap_tool_raven])
        swarm_cap = MockCapability(name="tell_tale_heart", tool_list=[cap_tool_raven])
        montresor = make_agent(name="montresor", capabilities=[agent_cap])
        usher = make_agent(name="usher")
        swarm = Swarm(agents=[montresor, usher])
        swarm.add_capability(swarm_cap, agents=["montresor"])
        with pytest.raises(ValueError, match="Duplicate tool name"):
            swarm._resolve_agent(montresor)

    def test_duplicate_between_two_swarm_caps(
        self, make_agent, cap_tool_raven
    ):
        """Two swarm caps both provide the same tool name."""
        cap1 = MockCapability(name="raven", tool_list=[cap_tool_raven])
        cap2 = MockCapability(name="tell_tale_heart", tool_list=[cap_tool_raven])
        montresor = make_agent(name="montresor")
        swarm = Swarm(agents=[montresor])
        swarm.add_capability(cap1)
        swarm.add_capability(cap2)
        with pytest.raises(ValueError, match="Duplicate tool name"):
            swarm._resolve_agent(montresor)

    @pytest.mark.asyncio
    async def test_handoff_tool_name_collides_with_regular_tool(
        self, make_agent, mock_provider
    ):
        """A regular tool named 'transfer_to_billing' collides with
        a handoff to billing."""
        @tool
        def transfer_to_billing():
            """Fake transfer."""
            return "not a real handoff"

        billing_ho = handoff("billing", "Bills")
        montresor = make_agent(
            name="montresor",
            tools=[transfer_to_billing],
            handoffs=[billing_ho],
        )
        mock_provider.responses = [
            make_text_response("never reached"),
        ]
        swarm = Swarm(agents=[montresor])

        with pytest.raises(
            ValueError, match="collides with an existing tool"
        ):
            await swarm.run("hello", agent="montresor")


# ---------------------------------------------------------------------------
# Swarm capability end-to-end
# ---------------------------------------------------------------------------

class TestSwarmCapabilityEndToEnd:
    @pytest.mark.asyncio
    async def test_swarm_cap_tool_dispatched(self):
        @tool
        def raven():
            """Recite the Raven."""
            return "Quoth the Raven, 'Nevermore.'"

        provider = MockProvider()
        provider.responses = [
            make_tool_call_response("raven", {}),
            make_text_response("Darkness there and nothing more"),
        ]
        montresor = Agent(
            name="montresor",
            system_prompt="Once upon a midnight dreary.",
            model="m", provider=provider,
        )
        swarm = Swarm(agents=[montresor])
        cap = MockCapability(name="raven", tool_list=[raven])
        swarm.add_capability(cap)

        result = await swarm.run("Tell me a tale", agent="montresor")

        assert result.last_message.content == "Darkness there and nothing more"
        tool_results = [
            m for m in swarm.session.transcript
            if m.role == MessageRole.TOOL
        ]
        assert any("Nevermore" in r.content for r in tool_results)

    @pytest.mark.asyncio
    async def test_agent_cap_tool_dispatched_through_swarm(self):
        @tool
        def annabel():
            """Recite Annabel Lee."""
            return "It was many and many a year ago, in a kingdom by the sea"

        provider = MockProvider()
        provider.responses = [
            make_tool_call_response("annabel", {}),
            make_text_response("A wind blew out of a cloud, chilling my beautiful Annabel Lee"),
        ]
        cap = MockCapability(name="annabel_lee", tool_list=[annabel])
        prospero = Agent(
            name="prospero",
            description="The Red Death had long devastated the country",
            system_prompt="Deep into that darkness peering.",
            model="m", provider=provider, capabilities=[cap],
        )
        swarm = Swarm(agents=[prospero])

        result = await swarm.run("Tell me a tale", agent="prospero")

        assert "Annabel Lee" in result.last_message.content
        tool_results = [
            m for m in swarm.session.transcript
            if m.role == MessageRole.TOOL
        ]
        assert any("kingdom by the sea" in r.content for r in tool_results)

    @pytest.mark.asyncio
    async def test_both_level_caps_dispatched(self):
        @tool
        def raven():
            """Recite the Raven."""
            return "Quoth the Raven, 'Nevermore.'"

        @tool
        def annabel():
            """Recite Annabel Lee."""
            return "It was many and many a year ago, in a kingdom by the sea"

        provider = MockProvider()
        provider.responses = [
            make_tool_call_response("raven", {}),
            make_tool_call_response("annabel", {}),
            make_text_response("Darkness there and nothing more"),
        ]
        agent_cap = MockCapability(name="raven", tool_list=[raven])
        montresor = Agent(
            name="montresor",
            system_prompt="Once upon a midnight dreary.",
            model="m", provider=provider, capabilities=[agent_cap],
        )
        swarm = Swarm(agents=[montresor])
        swarm_cap = MockCapability(name="annabel_lee", tool_list=[annabel])
        swarm.add_capability(swarm_cap)

        result = await swarm.run("Tell me a tale of terror", agent="montresor")

        assert result.last_message.content == "Darkness there and nothing more"
        tool_results = [
            m for m in swarm.session.transcript
            if m.role == MessageRole.TOOL
        ]
        contents = [r.content for r in tool_results]
        assert any("Nevermore" in c for c in contents)
        assert any("kingdom by the sea" in c for c in contents)

    @pytest.mark.asyncio
    async def test_removed_cap_tools_not_sent(self):
        @tool
        def raven():
            """Recite the Raven."""
            return "Quoth the Raven, 'Nevermore.'"

        provider = MockProvider()
        montresor = Agent(
            name="montresor",
            system_prompt="Once upon a midnight dreary.",
            model="m", provider=provider,
        )
        swarm = Swarm(agents=[montresor])
        cap = MockCapability(name="raven", tool_list=[raven])
        swarm.add_capability(cap)
        swarm.remove_capability("raven")

        provider.responses = [make_text_response("Darkness there and nothing more")]
        await swarm.run("Tell me a tale", agent="montresor")

        sent_tools = provider.call_log[0]["tools"]
        assert sent_tools is None

    @pytest.mark.asyncio
    async def test_capability_tool_receives_context(self):
        @tool
        def who_am_i(context: Context, verse: str):
            """Report the agent's name."""
            return f"agent={context.agent.name}"

        provider = MockProvider()
        provider.responses = [
            make_tool_call_response(
                "who_am_i",
                {"verse": "Is all that we see or seem but a dream within a dream?"},
            ),
            make_text_response("Darkness there and nothing more"),
        ]
        cap = MockCapability(name="tell_tale_heart", tool_list=[who_am_i])
        montresor = Agent(
            name="montresor",
            system_prompt="Once upon a midnight dreary.",
            model="m", provider=provider, capabilities=[cap],
        )
        swarm = Swarm(agents=[montresor])

        await swarm.run("Who are you?", agent="montresor")

        tool_results = [
            m for m in swarm.session.transcript
            if m.role == MessageRole.TOOL
        ]
        assert any("agent=montresor" in r.content for r in tool_results)


# ---------------------------------------------------------------------------
# Capability state integration (bookshop example patterns)
# ---------------------------------------------------------------------------

class TestSwarmCapabilityStateIntegration:
    @pytest.mark.asyncio
    async def test_agent_level_cap_tools_self_contained(self):
        catalog = {"isbn-1": {"title": "The Raven", "author": "Poe"}}

        class SelfContainedCap(Capability):
            def __init__(self):
                super().__init__("library")
                self._catalog = catalog

            def tools(self):
                data = self._catalog

                @tool
                def lookup(isbn: str):
                    """Look up a book."""
                    return data.get(isbn, "Not found")

                return [lookup]

        provider = MockProvider()
        provider.responses = [
            make_tool_call_response("lookup", {"isbn": "isbn-1"}),
            make_text_response("Found it"),
        ]
        agent = Agent(
            name="montresor",
            system_prompt="Deep into that darkness peering.",
            model="m", provider=provider,
            capabilities=[SelfContainedCap()],
        )
        swarm = Swarm(agents=[agent])

        await swarm.run("Find The Raven", agent="montresor")

        tool_results = [
            m for m in swarm.session.transcript
            if m.role == MessageRole.TOOL
        ]
        assert any("The Raven" in r.content for r in tool_results)

    @pytest.mark.asyncio
    async def test_agent_level_cap_on_attach_not_called(self):
        cap = MockCapability(name="raven")
        agent = Agent(
            name="montresor",
            system_prompt="Once upon a midnight dreary.",
            model="m", provider=MockProvider(),
            capabilities=[cap],
        )
        Swarm(agents=[agent])

        assert len(cap.attach_log) == 0

    @pytest.mark.asyncio
    async def test_swarm_level_on_attach_populates_state(self):
        class ShopState(State):
            catalog: dict[str, str] = {}

        class CatalogCap(Capability):
            def __init__(self):
                super().__init__("catalog")

            def tools(self):
                return []

            def on_attach(self, state):
                if isinstance(state, ShopState):
                    state.catalog["isbn-1"] = "The Raven"

        @tool
        def read_catalog(context: Context):
            """Read from shared catalog."""
            return f"catalog={context.state.catalog}"

        provider = MockProvider()
        provider.responses = [
            make_tool_call_response("read_catalog", {}),
            make_text_response("Done"),
        ]
        reader = Agent(
            name="usher",
            system_prompt="The Fall of the House of Usher.",
            model="m", provider=provider, tools=[read_catalog],
        )
        state = ShopState()
        swarm = Swarm(agents=[reader], state=state)
        swarm.add_capability(CatalogCap())

        assert state.catalog == {"isbn-1": "The Raven"}

        await swarm.run("What's in the catalog?", agent="usher")

        tool_results = [
            m for m in swarm.session.transcript
            if m.role == MessageRole.TOOL
        ]
        assert any("The Raven" in r.content for r in tool_results)

    @pytest.mark.asyncio
    async def test_cross_capability_state_sharing_via_handoff(self):
        class SharedState(State):
            catalog: dict[str, str] = {}

        @tool
        def add_book(context: Context, isbn: str, title: str):
            """Add a book to the shared catalog."""
            context.state.catalog[isbn] = title
            return f"Added '{title}'"

        @tool
        def read_catalog(context: Context):
            """Read the shared catalog."""
            return f"catalog={context.state.catalog}"

        provider = MockProvider()
        writer = Agent(
            name="montresor", description="Adds books",
            system_prompt="For the love of God, Montresor!",
            model="m", provider=provider, tools=[add_book],
            handoffs=[handoff("usher", "Reads catalog")],
        )
        reader = Agent(
            name="usher", description="Reads catalog",
            system_prompt="The Fall of the House of Usher.",
            model="m", provider=provider, tools=[read_catalog],
        )
        state = SharedState()
        swarm = Swarm(agents=[writer, reader], state=state)

        provider.responses = [
            make_tool_call_response(
                "add_book",
                {"isbn": "isbn-1", "title": "The Raven"},
            ),
            make_tool_call_response(
                "transfer_to_usher",
                {"message": "Check the catalog"},
            ),
            make_tool_call_response("read_catalog", {}),
            make_text_response("I see The Raven"),
        ]

        result = await swarm.run("Add and check", agent="montresor")

        assert result.active_agent == "usher"
        assert state.catalog == {"isbn-1": "The Raven"}
        tool_results = [
            m for m in swarm.session.transcript
            if m.role == MessageRole.TOOL
        ]
        assert any("The Raven" in r.content for r in tool_results)

    @pytest.mark.asyncio
    async def test_swarm_cap_mutates_state_visible_to_agent_cap(self):
        class InventoryState(State):
            orders: list[str] = []

        @tool
        def place_order(context: Context, title: str):
            """Place an order (swarm-level cap)."""
            context.state.orders.append(title)
            return f"Ordered '{title}'"

        provider = MockProvider()
        order_cap = MockCapability(name="ordering", tool_list=[place_order])

        agent = Agent(
            name="montresor",
            system_prompt="Once upon a midnight dreary.",
            model="m", provider=provider,
        )
        state = InventoryState()
        swarm = Swarm(agents=[agent], state=state)
        swarm.add_capability(order_cap)

        provider.responses = [
            make_tool_call_response("place_order", {"title": "The Raven"}),
            make_text_response("Order placed"),
        ]

        await swarm.run("Order The Raven", agent="montresor")

        assert state.orders == ["The Raven"]
