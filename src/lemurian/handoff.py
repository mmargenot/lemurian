import re
from dataclasses import dataclass, field


def _normalize_tool_name(name: str) -> str:
    """Normalize an agent name into a valid tool-name string.

    Lowercases, replaces whitespace/hyphens with underscores, strips
    non-alphanumeric characters.  E.g. ``"Billing Support"`` becomes
    ``"billing_support"``.
    """
    name = name.lower().strip()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


@dataclass
class Handoff:
    """A declared transfer from one agent to another.

    Converted to a tool schema for the LLM but never executed through
    the tool pipeline.  The Runner classifies handoff tool calls by
    name and returns immediately without executing any function.

    Args:
        tool_name: Tool name sent to the LLM (e.g. ``"transfer_to_billing"``).
        tool_description: Description shown to the LLM.
        target_agent: Name of the agent to hand off to.
        input_json_schema: JSON schema for the tool parameters.
    """

    tool_name: str
    tool_description: str
    target_agent: str
    input_json_schema: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": (
                    "Context and instructions for the next agent"
                ),
            },
        },
        "required": ["message"],
    })

    def tool_schema(self) -> dict:
        """Return an OpenAI-compatible function tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description,
                "parameters": self.input_json_schema,
            },
        }


@dataclass
class HandoffResult:
    """Constructed by the Runner when it classifies a handoff tool call.

    Not returned by a tool function — the Runner builds this from the
    :class:`Handoff` metadata and the parsed tool-call arguments.

    Args:
        target_agent: Name of the agent to hand off to.
        message: Context and instructions for the next agent.
    """

    target_agent: str
    message: str


def handoff(agent_name: str, description: str = "") -> Handoff:
    """Create a :class:`Handoff` from an agent name and description.

    Normalises the agent name to produce a valid tool name
    (e.g. ``"Billing Support"`` → ``"transfer_to_billing_support"``).
    """
    normalized = _normalize_tool_name(agent_name)
    return Handoff(
        tool_name=f"transfer_to_{normalized}",
        tool_description=(
            f"Hand off to {agent_name}. {description}"
            if description
            else f"Hand off to {agent_name}."
        ),
        target_agent=agent_name,
    )
