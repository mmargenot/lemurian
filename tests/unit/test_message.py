from lemurian.message import MessageRole, ToolCallRequestMessage
from lemurian.streaming import ToolCall


def test_tool_call_request_serializes_tool_calls():
    """The serializer transforms ToolCall objects into OpenAI-compatible dicts."""
    msg = ToolCallRequestMessage(
        role=MessageRole.ASSISTANT,
        tool_calls=[ToolCall(id="call_abc", name="greet", arguments='{"name": "world"}')],
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
