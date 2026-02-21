import pytest

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

        from lemurian.agent import Agent

        agent = Agent(
            name="a", system_prompt="Count.", model="m",
            provider=provider, tools=[increment],
        )
        state = CounterState()
        swarm = Swarm(agents=[agent], state=state)

        await swarm.run("increment", agent="a")

        assert state.counter == 1
