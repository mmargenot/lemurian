from lemurian.message import MessageRole, ToolCallRequestMessage
from tests.conftest import MockFunction, MockToolCall


def test_tool_call_request_serializes_raw_objects():
    """The custom serializer transforms OpenAI-style objects into dicts."""
    tc = MockToolCall(
        id="call_abc",
        type="function",
        function=MockFunction(
            name="greet", arguments='{"name": "world"}'
        ),
    )
    msg = ToolCallRequestMessage(
        role=MessageRole.ASSISTANT,
        tool_calls=[tc],
    )
    dumped = msg.model_dump()
    assert dumped["tool_calls"] == [
        {
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "greet",
                "arguments": '{"name": "world"}',
            },
        }
    ]
