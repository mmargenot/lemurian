import json

import pytest

from lemurian.context import Context
from lemurian.message import Message, MessageRole, ToolCallResultMessage
from lemurian.runner import Runner
from lemurian.session import Session
from lemurian.state import State
from lemurian.tools import HandoffResult, tool

from tests.conftest import make_text_response, make_tool_call_response


# ---------------------------------------------------------------------------
# Tool fixtures (module-level, reused across test classes)
# ---------------------------------------------------------------------------

@tool
def echo(text: str):
    """Echo text back."""
    return text


@tool
def get_data():
    """Return structured data."""
    return {"items": [1, 2, 3]}


@tool
def context_reader(context: Context, query: str):
    """Reads agent name from context."""
    return f"agent={context.agent.name}, query={query}"


@tool
async def async_echo(text: str):
    """Async echo."""
    return f"async: {text}"


@tool
def do_handoff(context: Context, target: str):
    """Triggers a handoff."""
    return HandoffResult(target_agent=target, message="handing off")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunnerSimpleResponse:
    @pytest.mark.asyncio
    async def test_appends_assistant_message(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [make_text_response("Hello!")]
        agent = make_agent()
        session = Session(session_id="s1")

        result = await Runner().run(agent, session, State())

        assert result.last_message.content == "Hello!"
        assert result.last_message.role == MessageRole.ASSISTANT
        assert result.hand_off is None
        assert len(session.transcript) == 1

    @pytest.mark.asyncio
    async def test_system_prompt_not_in_transcript(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [make_text_response("Hi")]
        agent = make_agent(system_prompt="Be helpful.")
        session = Session(session_id="s1")

        await Runner().run(agent, session, State())

        for msg in session.transcript:
            assert msg.role != MessageRole.SYSTEM

        sent_messages = mock_provider.call_log[0]["messages"]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[0]["content"] == "Be helpful."


class TestRunnerToolCalls:
    @pytest.mark.asyncio
    async def test_tool_call_then_response(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [
            make_tool_call_response("echo", {"text": "hi"}),
            make_text_response("Done"),
        ]
        agent = make_agent(tools=[echo])
        session = Session(session_id="s1")

        result = await Runner().run(agent, session, State())

        assert result.last_message.content == "Done"
        # transcript: tool_call_request, tool_result, assistant
        assert len(session.transcript) == 3
        assert session.transcript[1].role == MessageRole.TOOL
        assert session.transcript[1].content == "hi"

    @pytest.mark.asyncio
    async def test_context_injection(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [
            make_tool_call_response(
                "context_reader", {"query": "test"}
            ),
            make_text_response("Done"),
        ]
        agent = make_agent(name="myagent", tools=[context_reader])
        session = Session(session_id="s1")

        await Runner().run(agent, session, State())

        tool_output = session.transcript[1].content
        assert "agent=myagent" in tool_output
        assert "query=test" in tool_output

    @pytest.mark.asyncio
    async def test_tool_not_found(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [
            make_tool_call_response("nonexistent", {}),
            make_text_response("Recovered"),
        ]
        agent = make_agent(tools=[echo])
        session = Session(session_id="s1")

        result = await Runner().run(agent, session, State())

        assert result.last_message.content == "Recovered"
        error_msg = session.transcript[1]
        assert isinstance(error_msg, ToolCallResultMessage)
        assert "not found" in error_msg.content

    @pytest.mark.asyncio
    async def test_non_string_output_serialized(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [
            make_tool_call_response("get_data", {}),
            make_text_response("Done"),
        ]
        agent = make_agent(tools=[get_data])
        session = Session(session_id="s1")

        await Runner().run(agent, session, State())

        tool_result = session.transcript[1]
        assert json.loads(tool_result.content) == {
            "items": [1, 2, 3]
        }

    @pytest.mark.asyncio
    async def test_async_tool(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [
            make_tool_call_response(
                "async_echo", {"text": "hello"}
            ),
            make_text_response("Done"),
        ]
        agent = make_agent(tools=[async_echo])
        session = Session(session_id="s1")

        await Runner().run(agent, session, State())

        assert session.transcript[1].content == "async: hello"


class TestRunnerHandoff:
    @pytest.mark.asyncio
    async def test_handoff_detection(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [
            make_tool_call_response(
                "do_handoff", {"target": "billing"}
            ),
        ]
        agent = make_agent(tools=[do_handoff])
        session = Session(session_id="s1")

        result = await Runner().run(agent, session, State())

        assert result.hand_off is not None
        assert result.hand_off.target_agent == "billing"
        assert result.hand_off.message == "handing off"
        assert (
            "Transferring to billing"
            in session.transcript[-1].content
        )


class TestRunnerWindowing:
    @pytest.mark.asyncio
    async def test_context_start_windows_transcript(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [make_text_response("Reply")]
        agent = make_agent()
        session = Session(session_id="s1")
        session.transcript = [
            Message(role=MessageRole.USER, content="old msg 0"),
            Message(
                role=MessageRole.ASSISTANT, content="old msg 1"
            ),
            Message(role=MessageRole.USER, content="new msg"),
        ]

        await Runner().run(agent, session, State(), context_start=2)

        sent_messages = mock_provider.call_log[0]["messages"]
        # system prompt + transcript[2:]
        assert len(sent_messages) == 2
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1]["content"] == "new msg"


class TestRunnerMaxTurns:
    @pytest.mark.asyncio
    async def test_max_turns_exceeded(
        self, make_agent, mock_provider
    ):
        mock_provider.responses = [
            make_tool_call_response("echo", {"text": "loop"})
            for _ in range(5)
        ]
        agent = make_agent(tools=[echo])
        session = Session(session_id="s1")

        result = await Runner(max_turns=3).run(
            agent, session, State()
        )

        assert "Maximum turns" in result.last_message.content
        assert result.hand_off is None
