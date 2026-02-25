"""Full agent lifecycle integration tests.

Each test exercises multiple features working together across the complete
agent lifecycle: creation → initialisation → tool execution → state
mutation → handoff → multi-turn → capability management → cleanup.

The tests are organised by the *combination* of features they exercise
rather than by individual feature, ensuring the framework behaves correctly
when features interact.
"""

import json

import pytest
from pydantic import BaseModel, Field

from lemurian.agent import Agent
from lemurian.capability import Capability
from lemurian.context import Context
from lemurian.message import Message, MessageRole, ToolCallResultMessage
from lemurian.runner import Runner
from lemurian.session import Session
from lemurian.state import State
from lemurian.swarm import Swarm
from lemurian.tools import HandoffResult, LLMRecoverableError, tool

from tests.conftest import (
    MockCapability,
    MockProvider,
    make_multi_tool_call_response,
    make_text_response,
    make_tool_call_response,
)


# ---------------------------------------------------------------------------
# Shared state models
# ---------------------------------------------------------------------------


class TicketState(State):
    """Multi-field state used across lifecycle tests."""

    tickets: list[dict] = Field(default_factory=list)
    assignments: dict[str, str] = Field(default_factory=dict)
    audit_log: list[str] = Field(default_factory=list)


class CounterState(State):
    counter: int = 0


class CatalogState(State):
    catalog: dict[str, str] = Field(default_factory=dict)
    orders: list[str] = Field(default_factory=list)


class NestedPrefsState(State):
    class Prefs(BaseModel):
        theme: str = "light"
        lang: str = "en"
        notifications: bool = True

    prefs: Prefs = Field(default_factory=Prefs)
    history: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared tool definitions
# ---------------------------------------------------------------------------


@tool
def create_ticket(context: Context, title: str, priority: str):
    """Create a support ticket."""
    ticket = {
        "id": len(context.state.tickets) + 1,
        "title": title,
        "priority": priority,
    }
    context.state.tickets.append(ticket)
    context.state.audit_log.append(f"created:{title}")
    return json.dumps(ticket)


@tool
def assign_ticket(context: Context, ticket_id: int, agent_name: str):
    """Assign a ticket to an agent."""
    context.state.assignments[str(ticket_id)] = agent_name
    context.state.audit_log.append(f"assigned:{ticket_id}->{agent_name}")
    return f"Ticket {ticket_id} assigned to {agent_name}"


@tool
def read_tickets(context: Context):
    """Read all tickets."""
    return json.dumps(context.state.tickets)


@tool
def read_audit_log(context: Context):
    """Read the full audit log."""
    return json.dumps(context.state.audit_log)


@tool
def increment(context: Context):
    """Increment the counter."""
    context.state.counter += 1
    return str(context.state.counter)


@tool
async def async_increment(context: Context):
    """Async increment the counter."""
    context.state.counter += 1
    return str(context.state.counter)


@tool
def add_to_catalog(context: Context, isbn: str, title: str):
    """Add a book to the catalog."""
    context.state.catalog[isbn] = title
    return f"Added '{title}'"


@tool
def place_order(context: Context, isbn: str):
    """Place an order for a book."""
    title = context.state.catalog.get(isbn, "Unknown")
    context.state.orders.append(isbn)
    return f"Ordered '{title}'"


@tool
def read_catalog(context: Context):
    """Read the catalog."""
    return json.dumps(context.state.catalog)


@tool
def update_prefs(context: Context, theme: str):
    """Update theme preference."""
    context.state.prefs.theme = theme
    context.state.history.append(f"theme->{theme}")
    return f"Theme set to {theme}"


@tool
def read_prefs(context: Context):
    """Read current preferences."""
    return json.dumps(context.state.prefs.model_dump())


# ===================================================================
# MULTI-TURN + STATE + TOOLS — Runner level
# ===================================================================


class TestRunnerMultiTurnStateMutation:
    """Runner-level lifecycle: tool calls mutating state across multiple
    manual runner invocations (simulates multi-turn without Swarm)."""

    @pytest.mark.asyncio
    async def test_state_accumulates_across_manual_turns(self):
        """State mutations from turn 1 are visible in turn 2."""
        provider = MockProvider()
        agent = Agent(
            name="counter_agent",
            system_prompt="Count things.",
            model="m",
            provider=provider,
            tools=[increment],
        )
        state = CounterState()
        session = Session(session_id="s1")
        runner = Runner()

        # Turn 1
        session.transcript.append(
            Message(role=MessageRole.USER, content="increment")
        )
        provider.responses = [
            make_tool_call_response("increment", {}),
            make_text_response("Counter is 1"),
        ]
        result = await runner.run(agent, session, state)
        assert state.counter == 1
        assert result.last_message.content == "Counter is 1"

        # Turn 2 — state carries over
        session.transcript.append(
            Message(role=MessageRole.USER, content="again")
        )
        provider.responses = [
            make_tool_call_response("increment", {}),
            make_text_response("Counter is 2"),
        ]
        await runner.run(agent, session, state)
        assert state.counter == 2

        # Transcript integrity
        user_msgs = [
            m for m in session.transcript if m.role == MessageRole.USER
        ]
        tool_msgs = [
            m for m in session.transcript if m.role == MessageRole.TOOL
        ]
        assert len(user_msgs) == 2
        assert len(tool_msgs) == 2
        assert tool_msgs[0].content == "1"
        assert tool_msgs[1].content == "2"

    @pytest.mark.asyncio
    async def test_async_tool_state_mutation_with_context(self):
        """Async tools mutate state and receive correct context."""
        provider = MockProvider()
        agent = Agent(
            name="async_agent",
            system_prompt="Async counting.",
            model="m",
            provider=provider,
            tools=[async_increment],
        )
        state = CounterState()
        session = Session(session_id="s1")

        session.transcript.append(Message(role=MessageRole.USER, content="go"))
        provider.responses = [
            make_tool_call_response("async_increment", {}),
            make_tool_call_response("async_increment", {}),
            make_text_response("Done"),
        ]
        await Runner().run(agent, session, state)
        assert state.counter == 2

    @pytest.mark.asyncio
    async def test_capability_tools_dispatched_through_runner(self):
        """Agent-level capability tools are dispatched and mutate state
        when run through Runner directly (no Swarm)."""

        @tool
        def cap_increment(context: Context):
            """Capability-provided increment by 10."""
            context.state.counter += 10
            return str(context.state.counter)

        cap = MockCapability(name="counter_cap", tool_list=[cap_increment])
        provider = MockProvider()
        agent = Agent(
            name="agent",
            system_prompt="Count.",
            model="m",
            provider=provider,
            tools=[increment],
            capabilities=[cap],
        )
        state = CounterState()
        session = Session(session_id="s1")
        session.transcript.append(Message(role=MessageRole.USER, content="go"))

        provider.responses = [
            make_tool_call_response("increment", {}, "c1"),
            make_tool_call_response("cap_increment", {}, "c2"),
            make_text_response("Done"),
        ]
        await Runner().run(agent, session, state)

        assert state.counter == 11  # 1 + 10
        tool_results = [
            m for m in session.transcript if m.role == MessageRole.TOOL
        ]
        assert tool_results[0].content == "1"
        assert tool_results[1].content == "11"

    @pytest.mark.asyncio
    async def test_capability_state_accumulates_across_runner_turns(self):
        """Capability tool mutations from turn 1 are visible in turn 2
        at the Runner level (no Swarm)."""

        @tool
        def cap_append(context: Context, item: str):
            """Capability tool that appends to catalog."""
            context.state.catalog[item] = item.upper()
            return f"added:{item}"

        cap = MockCapability(name="cat_cap", tool_list=[cap_append])
        provider = MockProvider()
        agent = Agent(
            name="agent",
            system_prompt="Catalog.",
            model="m",
            provider=provider,
            capabilities=[cap],
        )
        state = CatalogState()
        session = Session(session_id="s1")
        runner = Runner()

        # Turn 1
        session.transcript.append(
            Message(role=MessageRole.USER, content="add foo")
        )
        provider.responses = [
            make_tool_call_response("cap_append", {"item": "foo"}),
            make_text_response("Added foo"),
        ]
        await runner.run(agent, session, state)
        assert state.catalog == {"foo": "FOO"}

        # Turn 2 — state from turn 1 persists
        session.transcript.append(
            Message(role=MessageRole.USER, content="add bar")
        )
        provider.responses = [
            make_tool_call_response("cap_append", {"item": "bar"}),
            make_text_response("Added bar"),
        ]
        await runner.run(agent, session, state)
        assert state.catalog == {"foo": "FOO", "bar": "BAR"}

    @pytest.mark.asyncio
    async def test_batch_direct_and_capability_tools(self):
        """Direct tool and capability tool in the same batch both
        mutate state through a single Runner call."""

        @tool
        def cap_double(context: Context):
            """Capability tool: double the counter."""
            context.state.counter *= 2
            return str(context.state.counter)

        cap = MockCapability(name="doubler", tool_list=[cap_double])
        provider = MockProvider()
        agent = Agent(
            name="agent",
            system_prompt="Math.",
            model="m",
            provider=provider,
            tools=[increment],
            capabilities=[cap],
        )
        state = CounterState()
        session = Session(session_id="s1")
        session.transcript.append(Message(role=MessageRole.USER, content="go"))

        # increment (0→1), cap_double (1→2), increment (2→3)
        provider.responses = [
            make_multi_tool_call_response(
                [
                    ("increment", {}, "c1"),
                    ("cap_double", {}, "c2"),
                    ("increment", {}, "c3"),
                ]
            ),
            make_text_response("Done"),
        ]
        await Runner().run(agent, session, state)

        assert state.counter == 3
        tool_results = [
            m for m in session.transcript if m.role == MessageRole.TOOL
        ]
        assert [r.content for r in tool_results] == ["1", "2", "3"]


# ===================================================================
# MULTI-AGENT HANDOFF CHAINS — state flows through A→B→C
# ===================================================================


class TestMultiAgentHandoffChain:
    """Three-agent handoff chains and round-trips with shared state."""

    @pytest.mark.asyncio
    async def test_three_agent_chain_state_flows(self):
        provider = MockProvider()

        triage = Agent(
            name="triage",
            description="Routes requests",
            system_prompt="Route.",
            model="m",
            provider=provider,
        )
        writer = Agent(
            name="writer",
            description="Creates tickets",
            system_prompt="Write.",
            model="m",
            provider=provider,
            tools=[create_ticket],
        )
        reader = Agent(
            name="reader",
            description="Reads tickets",
            system_prompt="Read.",
            model="m",
            provider=provider,
            tools=[read_tickets],
        )

        state = TicketState()
        swarm = Swarm(agents=[triage, writer, reader], state=state)

        provider.responses = [
            make_tool_call_response(
                "handoff",
                {
                    "agent_name": "writer",
                    "message": "Create a ticket for billing issue",
                },
            ),
            make_tool_call_response(
                "create_ticket",
                {"title": "Billing issue", "priority": "high"},
            ),
            make_tool_call_response(
                "handoff",
                {
                    "agent_name": "reader",
                    "message": "Read back the tickets",
                },
            ),
            make_tool_call_response("read_tickets", {}),
            make_text_response("Found 1 ticket"),
        ]

        result = await swarm.run("I have a billing issue", agent="triage")

        assert result.active_agent == "reader"
        assert len(state.tickets) == 1
        assert state.tickets[0]["title"] == "Billing issue"
        assert "created:Billing issue" in state.audit_log

        tool_results = [
            m for m in swarm.session.transcript if m.role == MessageRole.TOOL
        ]
        assert any("Billing issue" in (r.content or "") for r in tool_results)

    @pytest.mark.asyncio
    async def test_round_trip_handoff_a_to_b_back_to_a(self):
        """A→B→A round-trip with state shared throughout."""
        provider = MockProvider()

        writer = Agent(
            name="writer",
            description="Writes",
            system_prompt="Write.",
            model="m",
            provider=provider,
            tools=[add_to_catalog],
        )
        orderer = Agent(
            name="orderer",
            description="Orders",
            system_prompt="Order.",
            model="m",
            provider=provider,
            tools=[place_order],
        )

        state = CatalogState()
        swarm = Swarm(agents=[writer, orderer], state=state)

        provider.responses = [
            make_tool_call_response(
                "add_to_catalog",
                {"isbn": "isbn-1", "title": "The Raven"},
            ),
            make_tool_call_response(
                "handoff",
                {"agent_name": "orderer", "message": "Order isbn-1"},
            ),
            make_tool_call_response("place_order", {"isbn": "isbn-1"}),
            make_tool_call_response(
                "handoff",
                {
                    "agent_name": "writer",
                    "message": "Add another book",
                },
            ),
            make_tool_call_response(
                "add_to_catalog",
                {"isbn": "isbn-2", "title": "Annabel Lee"},
            ),
            make_text_response("Both books added, one ordered."),
        ]

        result = await swarm.run("Add books and order", agent="writer")

        assert result.active_agent == "writer"
        assert state.catalog == {
            "isbn-1": "The Raven",
            "isbn-2": "Annabel Lee",
        }
        assert state.orders == ["isbn-1"]


# ===================================================================
# CONTEXT WINDOWING ACROSS MULTI-HOP HANDOFFS
# ===================================================================


class TestContextWindowingMultiHop:
    """Each agent after a handoff sees only messages from context_start
    onward — not the previous agent's full history."""

    @pytest.mark.asyncio
    async def test_each_agent_sees_fresh_context_window(self):
        provider = MockProvider()
        call_log = provider.call_log

        agent_a = Agent(
            name="a",
            description="Agent A",
            system_prompt="I am A.",
            model="m",
            provider=provider,
        )
        agent_b = Agent(
            name="b",
            description="Agent B",
            system_prompt="I am B.",
            model="m",
            provider=provider,
        )

        swarm = Swarm(agents=[agent_a, agent_b])

        provider.responses = [
            make_tool_call_response(
                "handoff",
                {
                    "agent_name": "b",
                    "message": "Handle this for me",
                },
            ),
            make_text_response("B here, handled it."),
        ]

        await swarm.run("Hello from user", agent="a")

        assert len(call_log) == 2

        # A sees the user message
        a_non_sys = [
            m for m in call_log[0]["messages"] if m["role"] != "system"
        ]
        assert any(
            "Hello from user" in (m.get("content") or "") for m in a_non_sys
        )

        # B sees only the handoff message, not the original
        b_non_sys = [
            m for m in call_log[1]["messages"] if m["role"] != "system"
        ]
        assert any(
            "Handle this for me" in (m.get("content") or "") for m in b_non_sys
        )
        assert not any(
            "Hello from user" in (m.get("content") or "") for m in b_non_sys
        )

    @pytest.mark.asyncio
    async def test_three_hop_windowing(self):
        """A→B→C: C sees only its handoff message."""
        provider = MockProvider()
        call_log = provider.call_log

        agents = [
            Agent(
                name=n,
                description=n,
                system_prompt=f"I am {n}.",
                model="m",
                provider=provider,
            )
            for n in ("a", "b", "c")
        ]
        swarm = Swarm(agents=agents)

        provider.responses = [
            make_tool_call_response(
                "handoff",
                {"agent_name": "b", "message": "Msg for B"},
            ),
            make_tool_call_response(
                "handoff",
                {"agent_name": "c", "message": "Msg for C"},
            ),
            make_text_response("C done."),
        ]

        await swarm.run("Original user message", agent="a")

        c_non_sys = [
            m for m in call_log[2]["messages"] if m["role"] != "system"
        ]
        assert any("Msg for C" in (m.get("content") or "") for m in c_non_sys)
        assert not any(
            "Original user message" in (m.get("content") or "")
            for m in c_non_sys
        )
        assert not any(
            "Msg for B" in (m.get("content") or "") for m in c_non_sys
        )


# ===================================================================
# ERROR RECOVERY + HANDOFF
# ===================================================================


class TestErrorRecoveryWithHandoff:
    """Tool errors in one agent followed by handoff to another."""

    @pytest.mark.asyncio
    async def test_tool_exception_then_handoff(self):
        """Tool raises in agent A, A hands off to B."""

        @tool
        def fragile_tool(context: Context, msg: str):
            """Might fail."""
            if msg == "bad":
                raise RuntimeError("tool failure")
            return f"ok: {msg}"

        provider = MockProvider()
        agent_a = Agent(
            name="a",
            description="Agent A",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[fragile_tool],
        )
        agent_b = Agent(
            name="b",
            description="Agent B",
            system_prompt="B.",
            model="m",
            provider=provider,
        )
        swarm = Swarm(agents=[agent_a, agent_b])

        provider.responses = [
            make_tool_call_response("fragile_tool", {"msg": "bad"}),
            make_tool_call_response(
                "handoff",
                {
                    "agent_name": "b",
                    "message": "I had an error, you handle it",
                },
            ),
            make_text_response("B handled it."),
        ]

        result = await swarm.run("Do something", agent="a")

        assert result.active_agent == "b"
        assert result.last_message.content == "B handled it."

        error_msgs = [
            m
            for m in swarm.session.transcript
            if isinstance(m, ToolCallResultMessage)
            and "tool failure" in (m.content or "")
        ]
        assert len(error_msgs) == 1

    @pytest.mark.asyncio
    async def test_recoverable_error_then_handoff(self):
        """LLMRecoverableError guides retry, then handoff succeeds."""

        @tool
        def strict_tool(context: Context, value: int):
            """Only accepts positive values."""
            if value <= 0:
                raise LLMRecoverableError("Value must be positive. Try again.")
            context.state.counter += value
            return str(context.state.counter)

        provider = MockProvider()
        agent_a = Agent(
            name="a",
            description="Agent A",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[strict_tool],
        )
        agent_b = Agent(
            name="b",
            description="Agent B",
            system_prompt="B.",
            model="m",
            provider=provider,
        )
        state = CounterState()
        swarm = Swarm(agents=[agent_a, agent_b], state=state)

        provider.responses = [
            make_tool_call_response("strict_tool", {"value": -1}),
            make_tool_call_response("strict_tool", {"value": 5}),
            make_tool_call_response(
                "handoff",
                {
                    "agent_name": "b",
                    "message": "Counter updated to 5",
                },
            ),
            make_text_response("B confirms counter is 5."),
        ]

        result = await swarm.run("Set counter", agent="a")

        assert result.active_agent == "b"
        assert state.counter == 5

        recovery_msgs = [
            m
            for m in swarm.session.transcript
            if isinstance(m, ToolCallResultMessage)
            and "Value must be positive" in (m.content or "")
        ]
        assert len(recovery_msgs) == 1
        assert not recovery_msgs[0].content.startswith("Error")


# ===================================================================
# BATCH TOOL CALLS + STATE MUTATION
# ===================================================================


class TestBatchToolCallsStateMutation:
    """Multiple tool calls in a single provider response, all
    mutating state."""

    @pytest.mark.asyncio
    async def test_batch_tool_calls_all_mutate_state(self):
        provider = MockProvider()
        agent = Agent(
            name="batch",
            system_prompt="Batch.",
            model="m",
            provider=provider,
            tools=[increment],
        )
        state = CounterState()
        swarm = Swarm(agents=[agent], state=state)

        provider.responses = [
            make_multi_tool_call_response(
                [
                    ("increment", {}, "c1"),
                    ("increment", {}, "c2"),
                    ("increment", {}, "c3"),
                ]
            ),
            make_text_response("Counter is 3"),
        ]

        await swarm.run("Increment thrice", agent="batch")

        assert state.counter == 3
        tool_results = [
            m for m in swarm.session.transcript if m.role == MessageRole.TOOL
        ]
        assert [r.content for r in tool_results] == ["1", "2", "3"]

    @pytest.mark.asyncio
    async def test_batch_with_error_mid_sequence(self):
        """One tool in a batch fails; others still execute."""

        @tool
        def maybe_fail(context: Context, label: str):
            """Fails if label is 'bad'."""
            if label == "bad":
                raise RuntimeError("bad label")
            context.state.counter += 1
            return f"ok:{label}"

        provider = MockProvider()
        agent = Agent(
            name="batch",
            system_prompt="Batch.",
            model="m",
            provider=provider,
            tools=[maybe_fail],
        )
        state = CounterState()
        session = Session(session_id="s1")
        session.transcript.append(Message(role=MessageRole.USER, content="go"))

        provider.responses = [
            make_multi_tool_call_response(
                [
                    ("maybe_fail", {"label": "good1"}, "c1"),
                    ("maybe_fail", {"label": "bad"}, "c2"),
                    ("maybe_fail", {"label": "good2"}, "c3"),
                ]
            ),
            make_text_response("Done"),
        ]

        await Runner().run(agent, session, state)

        assert state.counter == 2
        tool_results = [
            m
            for m in session.transcript
            if isinstance(m, ToolCallResultMessage)
        ]
        assert any("ok:good1" in r.content for r in tool_results)
        assert any("bad label" in r.content for r in tool_results)
        assert any("ok:good2" in r.content for r in tool_results)

    @pytest.mark.asyncio
    async def test_batch_handoff_stops_remaining_state_mutations(
        self,
    ):
        """Handoff mid-batch prevents later tools from executing.

        Runner iterates tool_calls sequentially; when any tool returns a
        HandoffResult the Runner returns immediately, skipping remaining
        calls in the batch.  This is intentional — a handoff transfers
        control, so later tools from the *old* agent must not execute.
        """
        call_log = []

        @tool
        def tracked_increment(context: Context, label: str):
            """Increment and track."""
            call_log.append(label)
            context.state.counter += 1
            return str(context.state.counter)

        @tool
        def trigger_handoff(context: Context, target: str):
            """Trigger handoff."""
            call_log.append("handoff")
            return HandoffResult(target_agent=target, message="Handing off")

        provider = MockProvider()
        agent_a = Agent(
            name="a",
            description="Agent A",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[tracked_increment, trigger_handoff],
        )
        agent_b = Agent(
            name="b",
            description="Agent B",
            system_prompt="B.",
            model="m",
            provider=provider,
        )
        state = CounterState()
        swarm = Swarm(agents=[agent_a, agent_b], state=state)

        provider.responses = [
            make_multi_tool_call_response(
                [
                    ("trigger_handoff", {"target": "b"}, "c1"),
                    (
                        "tracked_increment",
                        {"label": "should_not_run"},
                        "c2",
                    ),
                ]
            ),
            make_text_response("B here."),
        ]

        result = await swarm.run("Go", agent="a")

        assert result.active_agent == "b"
        assert state.counter == 0
        assert "handoff" in call_log
        assert "should_not_run" not in call_log


# ===================================================================
# CAPABILITY LIFECYCLE — add / remove / hooks / all tool sources
# ===================================================================


class TestCapabilityLifecycle:
    """Capability add/remove between turns, lifecycle hooks interacting
    with state, and all three tool-source levels through a handoff."""

    @pytest.mark.asyncio
    async def test_add_capability_after_first_turn(self):
        """Capability added between turns becomes available on next turn."""

        @tool
        def search_catalog(context: Context):
            """Search the catalog."""
            return json.dumps(context.state.catalog)

        provider = MockProvider()
        agent = Agent(
            name="clerk",
            system_prompt="Help.",
            model="m",
            provider=provider,
        )
        state = CatalogState()
        swarm = Swarm(agents=[agent], state=state)

        # Turn 1: no tools
        provider.responses = [make_text_response("No tools available yet.")]
        await swarm.run("Search catalog", agent="clerk")

        assert provider.call_log[0]["tools"] is None

        # Add capability between turns
        cap = MockCapability(name="catalog", tool_list=[search_catalog])
        swarm.add_capability(cap)
        state.catalog["isbn-1"] = "The Raven"

        # Turn 2: capability tool now available
        provider.responses = [
            make_tool_call_response("search_catalog", {}),
            make_text_response("Found The Raven"),
        ]
        await swarm.run("Search again")

        sent_tools_t2 = provider.call_log[1]["tools"]
        tool_names_t2 = [t["function"]["name"] for t in sent_tools_t2]
        assert "search_catalog" in tool_names_t2

        tool_results = [
            m for m in swarm.session.transcript if m.role == MessageRole.TOOL
        ]
        assert any("The Raven" in r.content for r in tool_results)

    @pytest.mark.asyncio
    async def test_remove_capability_between_turns(self):
        """Capability removed between turns is no longer available."""

        @tool
        def secret_tool():
            """Top secret."""
            return "classified"

        provider = MockProvider()
        agent = Agent(
            name="agent",
            system_prompt="Help.",
            model="m",
            provider=provider,
        )
        swarm = Swarm(agents=[agent])
        cap = MockCapability(name="secret", tool_list=[secret_tool])
        swarm.add_capability(cap)

        # Turn 1: tool available
        provider.responses = [
            make_tool_call_response("secret_tool", {}),
            make_text_response("Here's the secret"),
        ]
        await swarm.run("Tell me the secret", agent="agent")

        sent_tools_t1 = provider.call_log[0]["tools"]
        assert any(
            t["function"]["name"] == "secret_tool" for t in sent_tools_t1
        )

        # Remove capability
        swarm.remove_capability("secret")
        assert len(cap.detach_log) == 1

        # Turn 2: tool gone
        provider.responses = [make_text_response("No secrets here")]
        await swarm.run("Tell me again")

        sent_tools_t2 = provider.call_log[2]["tools"]
        assert sent_tools_t2 is None

    @pytest.mark.asyncio
    async def test_add_cap_to_specific_agent_in_multi_agent_swarm(
        self,
    ):
        """Adding a capability to one agent doesn't affect another."""

        @tool
        def special_tool():
            """Only for agent A."""
            return "special"

        provider = MockProvider()
        a = Agent(
            name="a",
            description="A",
            system_prompt="A.",
            model="m",
            provider=provider,
        )
        b = Agent(
            name="b",
            description="B",
            system_prompt="B.",
            model="m",
            provider=provider,
        )
        swarm = Swarm(agents=[a, b])
        cap = MockCapability(name="special", tool_list=[special_tool])
        swarm.add_capability(cap, agents=["a"])

        # Agent A: tool available
        provider.responses = [
            make_tool_call_response("special_tool", {}),
            make_text_response("A used it"),
        ]
        await swarm.run("Use tool", agent="a")

        tool_names_a = [
            t["function"]["name"] for t in provider.call_log[0]["tools"]
        ]
        assert "special_tool" in tool_names_a

        # Agent B: only handoff tool, no special_tool
        provider.responses = [make_text_response("B can't use it")]
        await swarm.run("Use tool", agent="b")

        tool_names_b = [
            t["function"]["name"] for t in provider.call_log[-1]["tools"]
        ]
        assert "special_tool" not in tool_names_b
        assert "handoff" in tool_names_b

    def test_on_detach_cleans_state(self):
        """on_detach modifies state on removal."""

        class CleanupCap(Capability):
            def __init__(self):
                super().__init__("cleaner")

            def tools(self):
                return []

            def on_attach(self, state):
                if isinstance(state, CatalogState):
                    state.catalog["temp"] = "Temporary"

            def on_detach(self, state):
                if isinstance(state, CatalogState):
                    state.catalog.pop("temp", None)

        state = CatalogState()
        provider = MockProvider()
        agent = Agent(
            name="a",
            system_prompt="A.",
            model="m",
            provider=provider,
        )
        swarm = Swarm(agents=[agent], state=state)

        swarm.add_capability(CleanupCap())
        assert "temp" in state.catalog

        swarm.remove_capability("cleaner")
        assert "temp" not in state.catalog

    @pytest.mark.asyncio
    async def test_capability_with_closure_tools_and_state(self):
        """Capability tool closes over internal state and reads app
        state — the real-world database pattern."""
        internal_db = {"isbn-1": "The Raven", "isbn-2": "Lenore"}

        class LibraryCap(Capability):
            def __init__(self):
                super().__init__("library")
                self._db = internal_db

            def tools(self):
                db = self._db

                @tool
                def search_library(context: Context, query: str):
                    """Search internal DB and log to app state."""
                    results = {
                        k: v
                        for k, v in db.items()
                        if query.lower() in v.lower()
                    }
                    context.state.audit_log.append(f"searched:{query}")
                    return json.dumps(results)

                return [search_library]

        provider = MockProvider()
        agent = Agent(
            name="librarian",
            system_prompt="Search books.",
            model="m",
            provider=provider,
        )
        state = TicketState()
        swarm = Swarm(agents=[agent], state=state)
        swarm.add_capability(LibraryCap())

        provider.responses = [
            make_tool_call_response("search_library", {"query": "Raven"}),
            make_text_response("Found The Raven"),
        ]
        await swarm.run("Find Raven", agent="librarian")

        assert "searched:Raven" in state.audit_log
        tool_results = [
            m for m in swarm.session.transcript if m.role == MessageRole.TOOL
        ]
        assert any("The Raven" in r.content for r in tool_results)

    @pytest.mark.asyncio
    async def test_all_tool_source_levels_plus_handoff(self):
        """Agent with direct tools + agent-level cap + swarm-level cap
        hands off to another agent; state accumulated from all three
        tool sources is visible to the second agent."""

        @tool
        def direct_tool(context: Context, msg: str):
            """Direct tool on agent."""
            context.state.audit_log.append(f"direct:{msg}")
            return f"direct:{msg}"

        @tool
        def agent_cap_tool(context: Context, msg: str):
            """Agent-level capability tool."""
            context.state.audit_log.append(f"agent_cap:{msg}")
            return f"agent_cap:{msg}"

        @tool
        def swarm_cap_tool(context: Context, msg: str):
            """Swarm-level capability tool."""
            context.state.audit_log.append(f"swarm_cap:{msg}")
            return f"swarm_cap:{msg}"

        @tool
        def check_log(context: Context):
            """Read audit log."""
            return json.dumps(context.state.audit_log)

        provider = MockProvider()
        a_cap = MockCapability(name="a_cap", tool_list=[agent_cap_tool])
        s_cap = MockCapability(name="s_cap", tool_list=[swarm_cap_tool])

        agent_a = Agent(
            name="a",
            description="All tools",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[direct_tool],
            capabilities=[a_cap],
        )
        agent_b = Agent(
            name="b",
            description="Checker",
            system_prompt="B.",
            model="m",
            provider=provider,
            tools=[check_log],
        )
        state = TicketState()
        swarm = Swarm(agents=[agent_a, agent_b], state=state)
        swarm.add_capability(s_cap, agents=["a"])

        provider.responses = [
            make_tool_call_response("direct_tool", {"msg": "hello"}, "c1"),
            make_tool_call_response("agent_cap_tool", {"msg": "world"}, "c2"),
            make_tool_call_response("swarm_cap_tool", {"msg": "foo"}, "c3"),
            make_tool_call_response(
                "handoff",
                {"agent_name": "b", "message": "Check the log"},
            ),
            make_tool_call_response("check_log", {}, "check_log_call"),
            make_text_response("Log verified."),
        ]

        result = await swarm.run("Do everything", agent="a")

        assert result.active_agent == "b"
        assert state.audit_log == [
            "direct:hello",
            "agent_cap:world",
            "swarm_cap:foo",
        ]

        # Find check_log result by its tool_call_id.
        log_result = [
            m
            for m in swarm.session.transcript
            if isinstance(m, ToolCallResultMessage)
            and m.tool_call_id == "check_log_call"
        ]
        assert len(log_result) == 1
        parsed = json.loads(log_result[0].content)
        assert parsed == [
            "direct:hello",
            "agent_cap:world",
            "swarm_cap:foo",
        ]


# ===================================================================
# MULTI-TURN SWARM — SESSION + STATE PERSISTENCE
# ===================================================================


class TestSwarmMultiTurnPersistence:
    """Multiple swarm.run() calls with state and session persisting."""

    @pytest.mark.asyncio
    async def test_multi_turn_with_agent_switch(self):
        """Switch agents between run() calls; state persists."""
        provider = MockProvider()
        a = Agent(
            name="writer",
            description="Writes",
            system_prompt="W.",
            model="m",
            provider=provider,
            tools=[add_to_catalog],
        )
        b = Agent(
            name="reader",
            description="Reads",
            system_prompt="R.",
            model="m",
            provider=provider,
            tools=[read_catalog],
        )
        state = CatalogState()
        swarm = Swarm(agents=[a, b], state=state)

        # Turn 1: writer adds a book
        provider.responses = [
            make_tool_call_response(
                "add_to_catalog",
                {"isbn": "1", "title": "Book One"},
            ),
            make_text_response("Added"),
        ]
        await swarm.run("Add a book", agent="writer")
        assert state.catalog == {"1": "Book One"}

        # Turn 2: switch to reader
        provider.responses = [
            make_tool_call_response("read_catalog", {}),
            make_text_response("Found Book One"),
        ]
        await swarm.run("What do we have?", agent="reader")

        tool_results = [
            m for m in swarm.session.transcript if m.role == MessageRole.TOOL
        ]
        assert any("Book One" in (r.content or "") for r in tool_results)

    @pytest.mark.asyncio
    async def test_handoff_persists_active_agent_for_next_turn(self):
        """Turn 1: A→B handoff. Turn 2: B continues implicitly."""
        provider = MockProvider()
        a = Agent(
            name="a",
            description="A",
            system_prompt="A.",
            model="m",
            provider=provider,
        )
        b = Agent(
            name="b",
            description="B",
            system_prompt="B.",
            model="m",
            provider=provider,
            tools=[increment],
        )
        state = CounterState()
        swarm = Swarm(agents=[a, b], state=state)

        # Turn 1: A hands off to B
        provider.responses = [
            make_tool_call_response(
                "handoff",
                {"agent_name": "b", "message": "Your turn"},
            ),
            make_text_response("B here in turn 1"),
        ]
        r1 = await swarm.run("Start", agent="a")
        assert r1.active_agent == "b"
        assert swarm.active_agent_name == "b"

        # Turn 2: B continues (no explicit agent)
        provider.responses = [
            make_tool_call_response("increment", {}),
            make_text_response("Incremented"),
        ]
        r2 = await swarm.run("Do something")
        assert r2.active_agent == "b"
        assert state.counter == 1


# ===================================================================
# NESTED STATE MUTATIONS ACROSS HANDOFFS
# ===================================================================


class TestNestedStateMutationsAcrossHandoffs:
    """Complex nested Pydantic state modified by different agents."""

    @pytest.mark.asyncio
    async def test_nested_pydantic_state_across_agents(self):
        """Agent A sets theme, hands off to B which reads full prefs."""
        provider = MockProvider()
        a = Agent(
            name="settings",
            description="Settings manager",
            system_prompt="Manage settings.",
            model="m",
            provider=provider,
            tools=[update_prefs],
        )
        b = Agent(
            name="display",
            description="Display manager",
            system_prompt="Show settings.",
            model="m",
            provider=provider,
            tools=[read_prefs],
        )
        state = NestedPrefsState()
        swarm = Swarm(agents=[a, b], state=state)

        provider.responses = [
            make_tool_call_response("update_prefs", {"theme": "dark"}),
            make_tool_call_response(
                "handoff",
                {
                    "agent_name": "display",
                    "message": "Show current prefs",
                },
            ),
            make_tool_call_response("read_prefs", {}),
            make_text_response("Dark theme active"),
        ]

        result = await swarm.run("Set dark mode", agent="settings")

        assert result.active_agent == "display"
        assert state.prefs.theme == "dark"
        assert state.prefs.lang == "en"  # unchanged
        assert state.prefs.notifications is True  # unchanged
        assert state.history == ["theme->dark"]

        tool_results = [
            m for m in swarm.session.transcript if m.role == MessageRole.TOOL
        ]
        prefs_result = [
            r
            for r in tool_results
            if "dark" in (r.content or "") and "en" in (r.content or "")
        ]
        assert len(prefs_result) >= 1


# ===================================================================
# MAX TURNS + MAX HANDOFFS + CUSTOM RUNNER
# ===================================================================


class TestMaxTurnsHandoffInteraction:
    """Interactions between max_turns (Runner) and max_handoffs (Swarm),
    including custom Runner configuration."""

    @pytest.mark.asyncio
    async def test_max_turns_within_agent_before_handoff(self):
        """Runner hits max_turns; Swarm returns without handoff."""
        provider = MockProvider()
        a = Agent(
            name="a",
            description="A",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[increment],
        )
        b = Agent(
            name="b",
            description="B",
            system_prompt="B.",
            model="m",
            provider=provider,
        )
        state = CounterState()
        runner = Runner(max_turns=2)
        swarm = Swarm(agents=[a, b], state=state, runner=runner)

        provider.responses = [
            make_tool_call_response("increment", {}),
            make_tool_call_response("increment", {}),
            make_tool_call_response("increment", {}),
        ]

        result = await swarm.run("Count forever", agent="a")

        assert "Maximum turns" in result.last_message.content
        assert result.active_agent == "a"
        assert state.counter == 2

    @pytest.mark.asyncio
    async def test_max_handoffs_with_productive_work_between(self):
        """Agents do work between handoffs; max_handoffs kicks in."""
        provider = MockProvider()
        a = Agent(
            name="a",
            description="A",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[increment],
        )
        b = Agent(
            name="b",
            description="B",
            system_prompt="B.",
            model="m",
            provider=provider,
            tools=[increment],
        )
        state = CounterState()
        swarm = Swarm(agents=[a, b], state=state, max_handoffs=2)

        provider.responses = [
            make_tool_call_response("increment", {}),
            make_tool_call_response(
                "handoff",
                {"agent_name": "b", "message": "your turn"},
            ),
            make_tool_call_response("increment", {}),
            make_tool_call_response(
                "handoff",
                {"agent_name": "a", "message": "back to you"},
            ),
            make_tool_call_response("increment", {}),
            make_tool_call_response(
                "handoff",
                {"agent_name": "b", "message": "once more"},
            ),
        ]

        result = await swarm.run("Ping pong", agent="a")

        assert "Maximum handoffs" in result.last_message.content
        assert state.counter == 3


# ===================================================================
# TOOL SCHEMA VERIFICATION
# ===================================================================


class TestToolSchemaIntegration:
    """Correct tool schemas sent to the provider at each lifecycle
    stage."""

    @pytest.mark.asyncio
    async def test_resolved_tools_sent_to_provider(self):
        """All resolved tools (direct + cap + handoff) appear in the
        provider call."""

        @tool
        def direct_tool(msg: str):
            """Direct tool."""
            return msg

        @tool
        def cap_tool():
            """Capability tool."""
            return "cap"

        provider = MockProvider()
        cap = MockCapability(name="test_cap", tool_list=[cap_tool])
        a = Agent(
            name="a",
            description="A",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[direct_tool],
        )
        b = Agent(
            name="b",
            description="B",
            system_prompt="B.",
            model="m",
            provider=provider,
        )
        swarm = Swarm(agents=[a, b])
        swarm.add_capability(cap, agents=["a"])

        provider.responses = [make_text_response("Done")]
        await swarm.run("Hello", agent="a")

        sent_tools = provider.call_log[0]["tools"]
        tool_names = {t["function"]["name"] for t in sent_tools}
        assert tool_names == {
            "direct_tool",
            "cap_tool",
            "handoff",
        }


# ===================================================================
# TRANSCRIPT INTEGRITY
# ===================================================================


class TestTranscriptIntegrity:
    """End-to-end transcript verification: message ordering, roles,
    and content correctness across complex scenarios."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_transcript_order(self):
        """Exact message ordering through: user → tool_call →
        tool_result → handoff → user(handoff_msg) → tool_call →
        tool_result → assistant."""
        provider = MockProvider()

        @tool
        def ping(context: Context):
            """Ping."""
            return "pong"

        a = Agent(
            name="a",
            description="Pinger",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[ping],
        )
        b = Agent(
            name="b",
            description="Ponger",
            system_prompt="B.",
            model="m",
            provider=provider,
            tools=[ping],
        )
        swarm = Swarm(agents=[a, b])

        provider.responses = [
            make_tool_call_response("ping", {}, "c1"),
            make_tool_call_response(
                "handoff",
                {"agent_name": "b", "message": "Your turn"},
            ),
            make_tool_call_response("ping", {}, "c3"),
            make_text_response("All done"),
        ]

        await swarm.run("Start", agent="a")

        transcript = swarm.session.transcript
        roles = [m.role for m in transcript]

        expected_roles = [
            MessageRole.USER,  # "Start"
            MessageRole.ASSISTANT,  # tool call: ping
            MessageRole.TOOL,  # result: pong
            MessageRole.ASSISTANT,  # tool call: handoff
            MessageRole.TOOL,  # result: "Transferring to b"
            MessageRole.USER,  # handoff msg: "Your turn"
            MessageRole.ASSISTANT,  # tool call: ping
            MessageRole.TOOL,  # result: pong
            MessageRole.ASSISTANT,  # final: "All done"
        ]
        assert roles == expected_roles

        assert transcript[0].content == "Start"
        assert transcript[2].content == "pong"
        assert "Transferring to b" in transcript[4].content
        assert transcript[5].content == "Your turn"
        assert transcript[7].content == "pong"
        assert transcript[8].content == "All done"

    @pytest.mark.asyncio
    async def test_tool_call_ids_are_preserved(self):
        """Tool call IDs match between request and result messages."""
        provider = MockProvider()

        @tool
        def my_tool(msg: str):
            """Tool."""
            return f"echo:{msg}"

        agent = Agent(
            name="a",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[my_tool],
        )
        session = Session(session_id="s1")
        session.transcript.append(Message(role=MessageRole.USER, content="go"))

        provider.responses = [
            make_multi_tool_call_response(
                [
                    ("my_tool", {"msg": "alpha"}, "call_abc"),
                    ("my_tool", {"msg": "beta"}, "call_def"),
                ]
            ),
            make_text_response("Done"),
        ]

        await Runner().run(agent, session, State())

        tool_results = [
            m
            for m in session.transcript
            if isinstance(m, ToolCallResultMessage)
        ]
        assert tool_results[0].tool_call_id == "call_abc"
        assert tool_results[0].content == "echo:alpha"
        assert tool_results[1].tool_call_id == "call_def"
        assert tool_results[1].content == "echo:beta"


# ===================================================================
# SYSTEM PROMPT HANDLING
# ===================================================================


class TestSystemPromptLifecycle:
    """System prompts are agent-specific and never stored in the
    transcript."""

    @pytest.mark.asyncio
    async def test_each_agent_gets_its_system_prompt_after_handoff(
        self,
    ):
        """After handoff, the new agent's system prompt is used."""
        provider = MockProvider()
        call_log = provider.call_log

        a = Agent(
            name="a",
            description="A",
            system_prompt="You are Agent A.",
            model="m",
            provider=provider,
        )
        b = Agent(
            name="b",
            description="B",
            system_prompt="You are Agent B.",
            model="m",
            provider=provider,
        )
        swarm = Swarm(agents=[a, b])

        provider.responses = [
            make_tool_call_response(
                "handoff",
                {"agent_name": "b", "message": "Go to B"},
            ),
            make_text_response("B here."),
        ]

        await swarm.run("Hello", agent="a")

        a_sys = call_log[0]["messages"][0]
        assert a_sys["role"] == "system"
        assert a_sys["content"] == "You are Agent A."

        b_sys = call_log[1]["messages"][0]
        assert b_sys["role"] == "system"
        assert b_sys["content"] == "You are Agent B."

    @pytest.mark.asyncio
    async def test_system_prompt_never_in_transcript(self):
        """After multiple turns and handoffs, no system messages
        in transcript."""
        provider = MockProvider()
        a = Agent(
            name="a",
            description="A",
            system_prompt="Sys A.",
            model="m",
            provider=provider,
        )
        b = Agent(
            name="b",
            description="B",
            system_prompt="Sys B.",
            model="m",
            provider=provider,
        )
        swarm = Swarm(agents=[a, b])

        provider.responses = [
            make_tool_call_response(
                "handoff",
                {"agent_name": "b", "message": "Go B"},
            ),
            make_text_response("B done"),
        ]
        await swarm.run("Turn 1", agent="a")

        provider.responses = [make_text_response("B again")]
        await swarm.run("Turn 2")

        for msg in swarm.session.transcript:
            assert msg.role != MessageRole.SYSTEM


# ===================================================================
# CONTEXT INJECTION — AGENT IDENTITY ACROSS HANDOFFS
# ===================================================================


class TestContextAgentIdentityHandoff:
    """Tools receiving context see the correct agent identity, even
    after handoffs."""

    @pytest.mark.asyncio
    async def test_context_agent_changes_after_handoff(self):
        """context.agent reflects the new agent after handoff."""

        @tool
        def who_am_i(context: Context):
            """Report agent name."""
            return f"I am {context.agent.name}"

        provider = MockProvider()
        a = Agent(
            name="alpha",
            description="Alpha",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[who_am_i],
        )
        b = Agent(
            name="beta",
            description="Beta",
            system_prompt="B.",
            model="m",
            provider=provider,
            tools=[who_am_i],
        )
        swarm = Swarm(agents=[a, b])

        provider.responses = [
            make_tool_call_response("who_am_i", {}, "c1"),
            make_tool_call_response(
                "handoff",
                {
                    "agent_name": "beta",
                    "message": "Who are you?",
                },
            ),
            make_tool_call_response("who_am_i", {}, "c3"),
            make_text_response("Done"),
        ]

        await swarm.run("Who am I?", agent="alpha")

        tool_results = [
            m
            for m in swarm.session.transcript
            if isinstance(m, ToolCallResultMessage)
        ]
        identity_results = [
            r
            for r in tool_results
            if r.content and r.content.startswith("I am")
        ]
        assert identity_results[0].content == "I am alpha"
        assert identity_results[1].content == "I am beta"


# ===================================================================
# ASYNC CAPABILITY TOOLS THROUGH FULL LIFECYCLE
# ===================================================================


class TestAsyncCapabilityTools:
    """Async tools provided by capabilities work end-to-end."""

    @pytest.mark.asyncio
    async def test_async_cap_tool_through_swarm(self):
        @tool
        async def async_lookup(context: Context, key: str):
            """Async lookup."""
            return f"found:{context.state.catalog.get(key, 'nothing')}"

        provider = MockProvider()
        cap = MockCapability(name="async_cap", tool_list=[async_lookup])
        agent = Agent(
            name="agent",
            system_prompt="Help.",
            model="m",
            provider=provider,
        )
        state = CatalogState()
        state.catalog["k1"] = "Value1"
        swarm = Swarm(agents=[agent], state=state)
        swarm.add_capability(cap)

        provider.responses = [
            make_tool_call_response("async_lookup", {"key": "k1"}),
            make_text_response("Found it"),
        ]
        await swarm.run("Look up k1", agent="agent")

        tool_results = [
            m for m in swarm.session.transcript if m.role == MessageRole.TOOL
        ]
        assert any("found:Value1" in r.content for r in tool_results)

    @pytest.mark.asyncio
    async def test_mixed_sync_async_capability_tools(self):
        """Capability with both sync and async tools."""

        @tool
        def sync_tool():
            """Sync."""
            return "sync_result"

        @tool
        async def async_tool():
            """Async."""
            return "async_result"

        provider = MockProvider()
        cap = MockCapability(
            name="mixed_cap", tool_list=[sync_tool, async_tool]
        )
        agent = Agent(
            name="agent",
            system_prompt="Help.",
            model="m",
            provider=provider,
        )
        swarm = Swarm(agents=[agent])
        swarm.add_capability(cap)

        provider.responses = [
            make_tool_call_response("sync_tool", {}, "c1"),
            make_tool_call_response("async_tool", {}, "c2"),
            make_text_response("Both worked"),
        ]
        await swarm.run("Use both", agent="agent")

        tool_results = [
            m for m in swarm.session.transcript if m.role == MessageRole.TOOL
        ]
        contents = [r.content for r in tool_results]
        assert "sync_result" in contents
        assert "async_result" in contents


# ===================================================================
# EDGE CASES
# ===================================================================


class TestEdgeCases:
    """Edge cases at feature boundaries."""

    @pytest.mark.asyncio
    async def test_empty_state_survives_full_lifecycle(self):
        """Default State() with no custom fields works through the
        entire lifecycle including handoffs."""
        provider = MockProvider()

        @tool
        def noop_tool(context: Context):
            """Accesses state but doesn't write."""
            return f"state_type={type(context.state).__name__}"

        a = Agent(
            name="a",
            description="A",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[noop_tool],
        )
        b = Agent(
            name="b",
            description="B",
            system_prompt="B.",
            model="m",
            provider=provider,
            tools=[noop_tool],
        )
        swarm = Swarm(agents=[a, b])  # default State()

        provider.responses = [
            make_tool_call_response("noop_tool", {}),
            make_tool_call_response(
                "handoff",
                {"agent_name": "b", "message": "Go B"},
            ),
            make_tool_call_response("noop_tool", {}),
            make_text_response("Done"),
        ]

        result = await swarm.run("Go", agent="a")
        assert result.active_agent == "b"

        tool_results = [
            m
            for m in swarm.session.transcript
            if isinstance(m, ToolCallResultMessage)
            and "state_type" in (m.content or "")
        ]
        assert all("State" in r.content for r in tool_results)

    @pytest.mark.asyncio
    async def test_invalid_json_mid_lifecycle_recovers(self):
        """Invalid JSON in tool args doesn't crash; error is recorded
        and conversation continues."""
        from tests.conftest import (
            MockFunction,
            MockResponse,
            MockToolCall,
        )

        provider = MockProvider()

        @tool
        def my_tool(msg: str):
            """Tool."""
            return f"ok:{msg}"

        agent = Agent(
            name="a",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[my_tool],
        )
        swarm = Swarm(agents=[agent])

        bad_response = MockResponse(
            tool_calls=[
                MockToolCall(
                    id="call_bad",
                    type="function",
                    function=MockFunction(
                        name="my_tool", arguments="{{not json"
                    ),
                )
            ]
        )
        provider.responses = [
            bad_response,
            make_tool_call_response("my_tool", {"msg": "retry"}),
            make_text_response("Recovered"),
        ]

        result = await swarm.run("Try", agent="a")

        assert result.last_message.content == "Recovered"
        error_msgs = [
            m
            for m in swarm.session.transcript
            if isinstance(m, ToolCallResultMessage)
            and "invalid arguments" in (m.content or "")
        ]
        assert len(error_msgs) == 1

    @pytest.mark.asyncio
    async def test_tool_not_found_mid_lifecycle_recovers(self):
        """LLM hallucinates a tool name; error recorded, conversation
        continues."""
        provider = MockProvider()

        @tool
        def real_tool():
            """Real tool."""
            return "real"

        agent = Agent(
            name="a",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[real_tool],
        )
        swarm = Swarm(agents=[agent])

        provider.responses = [
            make_tool_call_response("ghost_tool", {}),
            make_tool_call_response("real_tool", {}),
            make_text_response("Found the real one"),
        ]

        result = await swarm.run("Use tools", agent="a")

        assert result.last_message.content == "Found the real one"
        error_msgs = [
            m
            for m in swarm.session.transcript
            if isinstance(m, ToolCallResultMessage)
            and "not found" in (m.content or "")
        ]
        assert len(error_msgs) == 1

    @pytest.mark.asyncio
    async def test_tool_reads_session_transcript(self):
        """A tool can read prior messages via context.session."""

        @tool
        def read_history(context: Context):
            """Count user messages in transcript."""
            user_msgs = [
                m
                for m in context.session.transcript
                if m.role == MessageRole.USER
            ]
            return f"user_messages={len(user_msgs)}"

        provider = MockProvider()
        agent = Agent(
            name="a",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[read_history],
        )
        swarm = Swarm(agents=[agent])

        # Turn 1
        provider.responses = [
            make_tool_call_response("read_history", {}),
            make_text_response("1 user message"),
        ]
        await swarm.run("Hello", agent="a")

        # Turn 2 — should see 2 user messages now
        provider.responses = [
            make_tool_call_response("read_history", {}),
            make_text_response("2 user messages"),
        ]
        await swarm.run("Hello again")

        tool_results = [
            m
            for m in swarm.session.transcript
            if isinstance(m, ToolCallResultMessage)
            and "user_messages=" in (m.content or "")
        ]
        assert tool_results[0].content == "user_messages=1"
        assert tool_results[1].content == "user_messages=2"

    @pytest.mark.asyncio
    async def test_on_attach_seeds_state_then_tool_reads_it(self):
        """on_attach writes to state; a capability tool in the same
        capability reads that seeded data on first execution."""

        class SeedingCap(Capability):
            def __init__(self):
                super().__init__("seeder")

            def on_attach(self, state):
                if isinstance(state, CatalogState):
                    state.catalog["seed"] = "Seeded Book"

            def tools(self):
                @tool
                def read_seed(context: Context):
                    """Read the seeded catalog entry."""
                    return context.state.catalog.get("seed", "MISSING")

                return [read_seed]

        provider = MockProvider()
        agent = Agent(
            name="a",
            system_prompt="A.",
            model="m",
            provider=provider,
        )
        state = CatalogState()
        swarm = Swarm(agents=[agent], state=state)
        swarm.add_capability(SeedingCap())

        assert state.catalog["seed"] == "Seeded Book"

        provider.responses = [
            make_tool_call_response("read_seed", {}),
            make_text_response("Got it"),
        ]
        await swarm.run("Read seed", agent="a")

        tool_results = [
            m for m in swarm.session.transcript if m.role == MessageRole.TOOL
        ]
        assert tool_results[0].content == "Seeded Book"

    @pytest.mark.asyncio
    async def test_tool_returning_non_string_output(self):
        """Runner JSON-serialises non-string tool outputs (dicts, lists)."""

        @tool
        def dict_tool():
            """Returns a dict."""
            return {"key": "value", "n": 42}

        @tool
        def list_tool():
            """Returns a list."""
            return [1, "two", 3]

        provider = MockProvider()
        agent = Agent(
            name="a",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[dict_tool, list_tool],
        )
        session = Session(session_id="s1")
        session.transcript.append(Message(role=MessageRole.USER, content="go"))

        provider.responses = [
            make_multi_tool_call_response(
                [
                    ("dict_tool", {}, "c1"),
                    ("list_tool", {}, "c2"),
                ]
            ),
            make_text_response("Done"),
        ]
        await Runner().run(agent, session, State())

        tool_results = [
            m
            for m in session.transcript
            if isinstance(m, ToolCallResultMessage)
        ]
        assert json.loads(tool_results[0].content) == {
            "key": "value",
            "n": 42,
        }
        assert json.loads(tool_results[1].content) == [1, "two", 3]

    @pytest.mark.asyncio
    async def test_handoff_to_invalid_agent_returns_error(self):
        """Handoff to a non-existent agent returns an error message
        without crashing."""

        @tool
        def force_handoff(context: Context):
            """Force a handoff to a bogus target."""
            return HandoffResult(
                target_agent="does_not_exist",
                message="Go to ghost",
            )

        provider = MockProvider()
        agent = Agent(
            name="a",
            description="A",
            system_prompt="A.",
            model="m",
            provider=provider,
            tools=[force_handoff],
        )
        b = Agent(
            name="b",
            description="B",
            system_prompt="B.",
            model="m",
            provider=provider,
        )
        swarm = Swarm(agents=[agent, b])

        provider.responses = [
            make_tool_call_response("force_handoff", {}),
        ]
        result = await swarm.run("Go", agent="a")

        assert "not found" in result.last_message.content
        assert result.active_agent == "a"

    @pytest.mark.asyncio
    async def test_capability_tools_refreshed_each_resolution(self):
        """Capability.tools() is called on every agent resolution,
        so dynamic tool sets are supported."""
        call_count = 0

        class DynamicCap(Capability):
            def __init__(self):
                super().__init__("dynamic")

            def tools(self):
                nonlocal call_count
                call_count += 1

                @tool
                def dynamic_tool():
                    """Dynamic tool."""
                    return f"call_{call_count}"

                return [dynamic_tool]

        provider = MockProvider()
        agent = Agent(
            name="a",
            system_prompt="A.",
            model="m",
            provider=provider,
        )
        swarm = Swarm(agents=[agent])
        swarm.add_capability(DynamicCap())

        provider.responses = [
            make_tool_call_response("dynamic_tool", {}),
            make_text_response("Turn 1 done"),
        ]
        await swarm.run("Turn 1", agent="a")

        provider.responses = [
            make_tool_call_response("dynamic_tool", {}),
            make_text_response("Turn 2 done"),
        ]
        await swarm.run("Turn 2")

        # tools() was called at least twice (once per resolution)
        assert call_count >= 2


# ===================================================================
# FULL END-TO-END LIFECYCLE — THE "KITCHEN SINK"
# ===================================================================


class TestFullEndToEndLifecycle:
    """Comprehensive lifecycle test: agents with all feature
    combinations, multiple turns with handoffs, state mutations,
    capability management, and transcript verification."""

    @pytest.mark.asyncio
    async def test_kitchen_sink(self):
        """
        Lifecycle:
        1. Create swarm with 3 agents, custom state
        2. Turn 1: intake creates ticket, hands off to dispatch,
           dispatch assigns ticket, hands off to auditor
        3. Between turns: add audit_log reader capability to auditor
        4. Turn 2: auditor uses the new capability tool

        Verified: state persists, capabilities resolve, system
        prompts are agent-specific, no system messages in transcript.
        """
        provider = MockProvider()
        call_log = provider.call_log

        agent_a = Agent(
            name="intake",
            description="Creates tickets",
            system_prompt="You create tickets.",
            model="m",
            provider=provider,
            tools=[create_ticket],
        )
        agent_b = Agent(
            name="dispatch",
            description="Assigns tickets",
            system_prompt="You assign tickets.",
            model="m",
            provider=provider,
            tools=[assign_ticket],
        )
        agent_c = Agent(
            name="auditor",
            description="Audits logs",
            system_prompt="You audit logs.",
            model="m",
            provider=provider,
        )

        state = TicketState()
        swarm = Swarm(agents=[agent_a, agent_b, agent_c], state=state)

        # --- Turn 1 ---
        provider.responses = [
            make_tool_call_response(
                "create_ticket",
                {"title": "Server down", "priority": "critical"},
            ),
            make_tool_call_response(
                "handoff",
                {
                    "agent_name": "dispatch",
                    "message": "Assign ticket #1 to ops team",
                },
            ),
            make_tool_call_response(
                "assign_ticket",
                {"ticket_id": 1, "agent_name": "ops_team"},
            ),
            make_tool_call_response(
                "handoff",
                {
                    "agent_name": "auditor",
                    "message": "Audit the actions taken",
                },
            ),
            make_text_response("Audit complete."),
        ]

        r1 = await swarm.run("Server is down!", agent="intake")

        assert r1.active_agent == "auditor"
        assert len(state.tickets) == 1
        assert state.tickets[0]["title"] == "Server down"
        assert state.assignments == {"1": "ops_team"}
        assert "created:Server down" in state.audit_log
        assert "assigned:1->ops_team" in state.audit_log

        # Each agent's system prompt appeared in at least one call
        sys_prompts_seen = {
            entry["messages"][0]["content"]
            for entry in call_log
            if entry["messages"] and entry["messages"][0]["role"] == "system"
        }
        assert "You create tickets." in sys_prompts_seen
        assert "You assign tickets." in sys_prompts_seen
        assert "You audit logs." in sys_prompts_seen

        # --- Between turns: add capability to auditor ---
        audit_cap = MockCapability(name="audit", tool_list=[read_audit_log])
        swarm.add_capability(audit_cap, agents=["auditor"])

        # --- Turn 2 ---
        provider.responses = [
            make_tool_call_response("read_audit_log", {}),
            make_text_response("Full audit: 2 actions recorded."),
        ]

        r2 = await swarm.run("Show me the audit log")

        assert r2.active_agent == "auditor"
        assert r2.last_message.content == "Full audit: 2 actions recorded."

        tool_results = [
            m for m in swarm.session.transcript if m.role == MessageRole.TOOL
        ]
        audit_result = [
            r
            for r in tool_results
            if "created:Server down" in (r.content or "")
        ]
        assert len(audit_result) >= 1
        parsed_log = json.loads(audit_result[-1].content)
        assert parsed_log == [
            "created:Server down",
            "assigned:1->ops_team",
        ]

        # No system prompt ever stored in transcript
        for msg in swarm.session.transcript:
            assert msg.role != MessageRole.SYSTEM
