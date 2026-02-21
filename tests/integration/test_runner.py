import json

import pytest

from lemurian.context import Context
from lemurian.message import Message, MessageRole, ToolCallResultMessage
from lemurian.runner import Runner
from lemurian.session import Session
from lemurian.state import State
from lemurian.tools import HandoffResult, tool

from tests.conftest import (
    MockFunction,
    MockResponse,
    MockToolCall,
    make_multi_tool_call_response,
    make_text_response,
    make_tool_call_response,
)


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


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestRunnerErrorPaths:
    @pytest.mark.asyncio
    async def test_tool_exception_recorded_in_transcript(
        self, make_agent, mock_provider
    ):
        """When a tool raises, the error is recorded and the loop continues."""

        @tool
        def explode(msg: str):
            """Always fails."""
            raise RuntimeError(f"boom: {msg}")

        mock_provider.responses = [
            make_tool_call_response("explode", {"msg": "test"}),
            make_text_response("Recovered"),
        ]
        agent = make_agent(tools=[explode])
        session = Session(session_id="s1")

        result = await Runner().run(agent, session, State())

        assert result.last_message.content == "Recovered"
        error_msg = session.transcript[1]
        assert isinstance(error_msg, ToolCallResultMessage)
        assert "boom: test" in error_msg.content

    @pytest.mark.asyncio
    async def test_invalid_json_arguments_recorded_in_transcript(
        self, make_agent, mock_provider
    ):
        """Malformed JSON in tool_call arguments is recorded as an error."""
        bad_response = MockResponse(
            tool_calls=[
                MockToolCall(
                    id="call_1",
                    type="function",
                    function=MockFunction(
                        name="echo",
                        arguments="not valid json{{{",
                    ),
                )
            ]
        )
        mock_provider.responses = [bad_response, make_text_response("OK")]
        agent = make_agent(tools=[echo])
        session = Session(session_id="s1")

        result = await Runner().run(agent, session, State())

        assert result.last_message.content == "OK"
        error_msg = session.transcript[1]
        assert isinstance(error_msg, ToolCallResultMessage)
        assert "invalid arguments" in error_msg.content

    @pytest.mark.asyncio
    async def test_tool_returns_none_serializes_as_null(
        self, make_agent, mock_provider
    ):
        """A tool returning None serializes as 'null' in the transcript."""

        @tool
        def return_nothing():
            """Returns None."""
            return None

        mock_provider.responses = [
            make_tool_call_response("return_nothing", {}),
            make_text_response("Got null"),
        ]
        agent = make_agent(tools=[return_nothing])
        session = Session(session_id="s1")

        result = await Runner().run(agent, session, State())

        assert result.last_message.content == "Got null"
        tool_result_msg = session.transcript[1]
        assert tool_result_msg.content == "null"


# ---------------------------------------------------------------------------
# Multiple tool calls in a single response
# ---------------------------------------------------------------------------

class TestRunnerMultiToolCall:
    @pytest.mark.asyncio
    async def test_all_tool_calls_dispatched(
        self, make_agent, mock_provider
    ):
        """Multiple tool calls in one response all get executed."""
        mock_provider.responses = [
            make_multi_tool_call_response([
                ("echo", {"text": "first"}, "call_1"),
                ("echo", {"text": "second"}, "call_2"),
            ]),
            make_text_response("Done"),
        ]
        agent = make_agent(tools=[echo])
        session = Session(session_id="s1")

        result = await Runner().run(agent, session, State())

        assert result.last_message.content == "Done"
        tool_results = [
            m for m in session.transcript
            if m.role == MessageRole.TOOL
        ]
        assert len(tool_results) == 2
        assert tool_results[0].content == "first"
        assert tool_results[1].content == "second"

    @pytest.mark.asyncio
    async def test_tool_call_ids_match_results(
        self, make_agent, mock_provider
    ):
        """Each tool result references the correct call_id."""
        mock_provider.responses = [
            make_multi_tool_call_response([
                ("echo", {"text": "a"}, "call_aaa"),
                ("echo", {"text": "b"}, "call_bbb"),
            ]),
            make_text_response("Done"),
        ]
        agent = make_agent(tools=[echo])
        session = Session(session_id="s1")

        await Runner().run(agent, session, State())

        tool_results = [
            m for m in session.transcript
            if isinstance(m, ToolCallResultMessage)
        ]
        assert tool_results[0].tool_call_id == "call_aaa"
        assert tool_results[1].tool_call_id == "call_bbb"

    @pytest.mark.asyncio
    async def test_handoff_in_batch_stops_remaining_calls(
        self, make_agent, mock_provider
    ):
        """A handoff mid-batch prevents later tool calls from running."""
        call_log = []

        @tool
        def track_call(label: str):
            """Tracks that it was called."""
            call_log.append(label)
            return f"tracked: {label}"

        @tool
        def trigger_handoff(context: Context, target: str):
            """Triggers a handoff."""
            call_log.append("handoff")
            return HandoffResult(
                target_agent=target, message="handing off"
            )

        mock_provider.responses = [
            make_multi_tool_call_response([
                (
                    "trigger_handoff",
                    {"target": "billing"},
                    "call_1",
                ),
                (
                    "track_call",
                    {"label": "should_not_run"},
                    "call_2",
                ),
            ]),
        ]
        agent = make_agent(tools=[track_call, trigger_handoff])
        session = Session(session_id="s1")

        result = await Runner().run(agent, session, State())

        assert result.hand_off is not None
        assert result.hand_off.target_agent == "billing"
        assert "handoff" in call_log
        assert "should_not_run" not in call_log
