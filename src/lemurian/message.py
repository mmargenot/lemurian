from enum import Enum
from pydantic import BaseModel, field_serializer


class MessageRole(str, Enum):
    """Roles that a message in the transcript can have."""

    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in the conversation transcript.

    Args:
        role: The role of the message sender.
        content: The text content of the message, or None for
            tool-call-only assistant messages.
    """

    role: MessageRole
    content: str | None = None


class ToolCallRequestMessage(Message):
    """An assistant message containing one or more tool call requests.

    Args:
        tool_calls: Raw tool call objects from the provider response.
    """

    tool_calls: list

    @field_serializer("tool_calls")
    def serialize_tool_calls(self, tool_calls: list) -> list[dict]:
        """Serialize ToolCall objects into OpenAI-compatible dicts."""
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in tool_calls
        ]


class ToolCallResultMessage(Message):
    """The result of a tool call, sent back to the provider.

    Args:
        tool_call_id: The ID of the tool call this result corresponds to.
    """

    tool_call_id: str

