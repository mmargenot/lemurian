from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, Field


class ToolCallResult(BaseModel):
    """The result of executing a tool.

    Args:
        tool_name: Name of the tool that produced this result.
        output: The return value of the tool function. Can be any type.
    """

    tool_name: str
    output: Any


@dataclass
class HandoffResult:
    """Returned by a tool to signal an agent handoff.

    When the Runner detects a HandoffResult as a tool's output, it
    stops execution and returns control to the Swarm for agent switching.

    Args:
        target_agent: Name of the agent to hand off to.
        message: Context and instructions for the next agent.
    """

    target_agent: str
    message: str


# ---------------------------------------------------------------------------
# Type mapping helpers
# ---------------------------------------------------------------------------

_PYTHON_TO_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "NoneType": "null",
    "dict": "object",
    "list": "array",
    "tuple": "array",
    "set": "array",
}


def _json_type(python_type_name: str) -> str:
    """Map a Python type name to its JSON schema type string.

    Args:
        python_type_name: The ``__name__`` of a Python type (e.g. ``"str"``).

    Returns:
        The corresponding JSON schema type, or ``"string"`` if unknown.
    """
    return _PYTHON_TO_JSON_TYPE.get(python_type_name, "string")


# ---------------------------------------------------------------------------
# Tool model
# ---------------------------------------------------------------------------

class Tool(BaseModel):
    """A callable tool with an OpenAI-compatible function schema.

    Created by the ``@tool`` decorator. Wraps a sync or async function
    and generates the JSON schema from its signature.

    Args:
        func: The underlying Python function.
        name: Tool name exposed to the LLM.
        description: Tool description exposed to the LLM.
        parameters_schema: JSON schema for the tool's parameters.
    """

    model_config = {"arbitrary_types_allowed": True}

    func: Callable = Field(exclude=True)
    name: str
    description: str
    parameters_schema: dict

    def model_dump(self, **kwargs) -> dict:
        """Return OpenAI-compatible function tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def model_dump_json(self, **kwargs) -> str:
        """Return the OpenAI function schema as a JSON string."""
        return json.dumps(self.model_dump())

    async def __call__(self, **kwargs) -> ToolCallResult:
        """Execute the wrapped function and return a ToolCallResult.

        Handles both sync and async functions transparently.

        Args:
            **kwargs: Arguments to pass to the underlying function.

        Returns:
            A ToolCallResult containing the tool name and output.
        """
        if asyncio.iscoroutinefunction(self.func):
            result = await self.func(**kwargs)
        else:
            result = self.func(**kwargs)
        return ToolCallResult(tool_name=self.name, output=result)


# ---------------------------------------------------------------------------
# Schema extraction from a function signature
# ---------------------------------------------------------------------------

def _build_parameters_schema(func: Callable) -> tuple[dict, list[str]]:
    """Extract JSON-schema properties and required list from *func*.

    Parameters named ``context`` are treated as magic (injected at call time)
    and excluded from the schema.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "context":
            continue

        annotation = hints.get(name)
        type_name = getattr(annotation, "__name__", "str") if annotation else "str"

        properties[name] = {
            "type": _json_type(type_name),
            "description": "",
        }

        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    return schema, required


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

def tool(func: Callable | None = None, *, name: str | None = None, description: str | None = None) -> Tool | Callable:
    """Decorator that turns a function into a :class:`Tool`.

    Can be used bare (``@tool``) or with keyword arguments
    (``@tool(name="...", description="...")``).
    """
    def _wrap(fn: Callable) -> Tool:
        tool_name = name or fn.__name__
        tool_description = description or fn.__doc__ or ""
        params_schema, _ = _build_parameters_schema(fn)

        return Tool(
            func=fn,
            name=tool_name,
            description=tool_description,
            parameters_schema=params_schema,
        )

    if func is not None:
        # @tool  (no parentheses)
        return _wrap(func)
    # @tool(...)  (with parentheses)
    return _wrap
