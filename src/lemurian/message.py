from enum import Enum
from pydantic import BaseModel, field_serializer


class MessageRole(Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"


class Message(BaseModel):
    role: MessageRole
    content: str

    @field_serializer('role')
    def serialize_role(self, role: MessageRole, _info) -> str:
        return role.value


class ToolCallRequestMessage(Message):
    tool_calls: list

    @field_serializer("tool_calls")
    def serialize_tool_calls(self, tool_calls: list) -> list[dict]:
        return [
            {
                "id": t.id,
                "type": "function",
                "function": {
                    "arguments": t.function.arguments,
                    "name": t.function.name
                }
            }
            for t in tool_calls
        ]

class ToolCallResultMessage(Message):
    tool_call_id: str

