from lemurian.message import (
    Message,
    MessageRole,
    ToolCallRequestMessage,
    ToolCallResultMessage,
)
from lemurian.session import Session
from tests.conftest import MockFunction, MockToolCall


def test_plain_messages_round_trip():
    """A transcript of plain messages survives dump/validate."""
    session = Session(session_id="s1")
    session.transcript = [
        Message(role=MessageRole.USER, content="hello"),
        Message(role=MessageRole.ASSISTANT, content="hi there"),
        Message(role=MessageRole.USER, content="bye"),
    ]

    dumped = session.model_dump()
    restored = Session.model_validate(dumped)

    assert restored.session_id == "s1"
    assert len(restored.transcript) == 3
    assert restored.transcript[0].content == "hello"
    assert restored.transcript[1].content == "hi there"
    assert restored.transcript[2].content == "bye"
    for orig, rest in zip(
        session.transcript, restored.transcript
    ):
        assert orig.role == rest.role
        assert orig.content == rest.content


def test_tool_result_fields_lost_on_dump():
    """ToolCallResultMessage.tool_call_id is dropped by model_dump()
    because transcript is typed as list[Message].

    Pydantic v2 serializes using the declared type (Message), not the
    runtime type (ToolCallResultMessage), so subclass fields are lost.
    This documents a current limitation — Session cannot faithfully
    round-trip transcripts that contain tool-call subclass messages.
    """
    session = Session(session_id="s1")
    session.transcript.append(
        Message(role=MessageRole.USER, content="hi")
    )
    session.transcript.append(
        ToolCallResultMessage(
            role=MessageRole.TOOL,
            content="result data",
            tool_call_id="call_42",
        )
    )

    dumped = session.model_dump()

    # The tool_call_id is already lost at dump time
    assert "tool_call_id" not in dumped["transcript"][1]
    # Content (on the base Message) survives
    assert dumped["transcript"][1]["content"] == "result data"
    assert dumped["transcript"][1]["role"] == "tool"
