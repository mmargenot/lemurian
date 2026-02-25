import asyncio
import inspect
import json
from typing import Any, Callable, get_type_hints

from docstring_parser import parse as parse_docstring
from pydantic import BaseModel, Field


class ToolCallResult(BaseModel):
    """The result of executing a tool.

    Args:
        tool_name: Name of the tool that produced this result.
        output: The return value of the tool function. Can be any type.
    """

    tool_name: str
    output: Any


class LLMRecoverableError(Exception):
    """Raised by a tool to send recovery guidance back to the LLM.

    Unlike unexpected exceptions (which produce a generic error message),
    this carries an intentional message crafted by the tool author to
    help the LLM retry with correct arguments or approach.

    Example::

        @tool
        def lookup_user(user_id: int):
            if user_id <= 0:
                raise LLMRecoverableError(
                    "user_id must be a positive integer. "
                    "Use the get_user_id tool first if you only have a name."
                )
            return db.get(user_id)
    """


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
# Docstring parameter description extraction
# ---------------------------------------------------------------------------


def _parse_param_descriptions(func: Callable) -> dict[str, str]:
    """Extract ``{param_name: description}`` from *func*'s docstring.

    Uses ``docstring_parser`` with auto-detection so Google, NumPy,
    Sphinx/reST, and Epydoc styles are all supported. Returns an
    empty dict when there is no docstring or no documented params.
    """
    doc = func.__doc__
    if not doc:
        return {}
    parsed = parse_docstring(doc)
    return {p.arg_name: (p.description or "") for p in parsed.params}


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
    param_descs = _parse_param_descriptions(func)

    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "context":
            continue

        annotation = hints.get(name)
        type_name = (
            getattr(annotation, "__name__", "str") if annotation else "str"
        )

        properties[name] = {
            "type": _json_type(type_name),
            "description": param_descs.get(name, ""),
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


def tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable:
    """Decorator that turns a function into a :class:`Tool`.

    Can be used bare (``@tool``) or with keyword arguments
    (``@tool(name="...", description="...")``).
    """

    def _wrap(fn: Callable) -> Tool:
        tool_name = name or fn.__name__  # ty: ignore[unresolved-attribute]
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
