import json
from dataclasses import dataclass

import pytest

from lemurian.capability import Capability
from lemurian.context import Context
from lemurian.provider import ModelProvider
from lemurian.agent import Agent
from lemurian.state import State
from lemurian.tools import Tool, tool


# ---------------------------------------------------------------------------
# Mock response dataclasses (mirrors OpenAI response shape)
# ---------------------------------------------------------------------------

@dataclass
class MockFunction:
    name: str
    arguments: str


@dataclass
class MockToolCall:
    id: str
    type: str
    function: MockFunction


@dataclass
class MockResponse:
    content: str | None = None
    tool_calls: list[MockToolCall] | None = None


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

class MockProvider(ModelProvider):
    """Provider that returns pre-queued responses. No network calls."""

    def __init__(self):
        self.responses: list[MockResponse] = []
        self.call_log: list[dict] = []

    async def complete(self, model, messages, tools=None):
        self.call_log.append({"messages": messages, "tools": tools})
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Response builder helpers
# ---------------------------------------------------------------------------

def make_text_response(content: str) -> MockResponse:
    """Fake provider response with text only (no tool calls)."""
    return MockResponse(content=content)


def make_tool_call_response(
    name: str,
    args: dict,
    call_id: str = "call_1",
    content: str | None = None,
) -> MockResponse:
    """Fake provider response containing a single tool call."""
    tc = MockToolCall(
        id=call_id,
        type="function",
        function=MockFunction(
            name=name,
            arguments=json.dumps(args),
        ),
    )
    return MockResponse(content=content, tool_calls=[tc])


def make_multi_tool_call_response(
    calls: list[tuple[str, dict, str]],
    content: str | None = None,
) -> MockResponse:
    """Fake provider response containing multiple tool calls.

    Each item in *calls* is ``(func_name, args_dict, call_id)``.
    """
    tool_calls = [
        MockToolCall(
            id=call_id,
            type="function",
            function=MockFunction(
                name=name,
                arguments=json.dumps(args),
            ),
        )
        for name, args, call_id in calls
    ]
    return MockResponse(content=content, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Mock capability
# ---------------------------------------------------------------------------

class MockCapability(Capability):
    """Capability test double that records lifecycle calls and returns
    configurable tools."""

    def __init__(
        self,
        name: str = "mock_cap",
        tool_list: list[Tool] | None = None,
    ):
        super().__init__(name)
        self._tool_list = tool_list or []
        self.attach_log: list[State] = []
        self.detach_log: list[State] = []
        self.tools_call_count: int = 0

    def tools(self) -> list[Tool]:
        self.tools_call_count += 1
        return list(self._tool_list)

    def on_attach(self, state: State) -> None:
        self.attach_log.append(state)

    def on_detach(self, state: State) -> None:
        self.detach_log.append(state)


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def sample_tool():
    @tool
    def greet(name: str):
        """Say hello."""
        return f"Hello {name}"
    return greet


@pytest.fixture
def sample_async_tool():
    @tool
    async def async_greet(name: str):
        """Async greeting."""
        return f"Hello async {name}"
    return async_greet


@pytest.fixture
def sample_context_tool():
    @tool
    def stateful(context: Context, query: str):
        """Tool that uses context."""
        return f"state={context.state}, query={query}"
    return stateful


@pytest.fixture
def cap_tool_raven():
    @tool
    def raven():
        """Recite the Raven."""
        return "Quoth the Raven, 'Nevermore.'"
    return raven


@pytest.fixture
def cap_tool_annabel():
    @tool
    def annabel():
        """Recite Annabel Lee."""
        return "It was many and many a year ago, in a kingdom by the sea"
    return annabel


@pytest.fixture
def cap_tool_lenore():
    @tool
    def lenore():
        """Recite Lenore."""
        return "Nameless here for evermore"
    return lenore


@pytest.fixture
def mock_capability(cap_tool_raven):
    """Pre-configured MockCapability with a single tool."""
    return MockCapability(name="raven", tool_list=[cap_tool_raven])


@pytest.fixture
def make_agent(mock_provider):
    """Factory fixture to build agents with the mock provider.

    Pass ``provider=`` to override the default mock_provider (e.g. when
    multiple agents need to share a single provider in swarm tests).
    """
    def _make(
        name="test_agent",
        tools=None,
        handoffs=None,
        capabilities=None,
        system_prompt="You are helpful.",
        description="",
        provider=None,
    ):
        return Agent(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools or [],
            handoffs=handoffs or [],
            capabilities=capabilities or [],
            model="mock-model",
            provider=provider or mock_provider,
        )
    return _make
