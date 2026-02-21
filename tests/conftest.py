import json
from dataclasses import dataclass

import pytest

from lemurian.context import Context
from lemurian.provider import ModelProvider
from lemurian.agent import Agent
from lemurian.tools import tool


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
def make_agent(mock_provider):
    """Factory fixture to build agents with the mock provider.

    Pass ``provider=`` to override the default mock_provider (e.g. when
    multiple agents need to share a single provider in swarm tests).
    """
    def _make(
        name="test_agent",
        tools=None,
        system_prompt="You are helpful.",
        description="",
        provider=None,
    ):
        return Agent(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools or [],
            model="mock-model",
            provider=provider or mock_provider,
        )
    return _make
