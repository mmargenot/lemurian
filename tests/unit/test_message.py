from lemurian.message import MessageRole, ToolCallRequestMessage
from lemurian.streaming import ToolCall


def test_tool_call_request_serializes_raw_objects():
    """The custom serializer transforms ToolCall objects into dicts."""
    tc = ToolCall(
        id="call_abc",
        name="greet",
        arguments='{"name": "world"}',
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
