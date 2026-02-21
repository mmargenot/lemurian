import pytest

from lemurian.tools import (
    Tool,
    ToolCallResult,
    _build_parameters_schema,
    tool,
)


# ---------------------------------------------------------------------------
# Schema generation (_build_parameters_schema)
# ---------------------------------------------------------------------------

class TestBuildParametersSchema:
    def test_python_types_map_to_json_schema_types(self):
        def func(
            a: str, b: int, c: float, d: bool, e: list, f: dict
        ):
            pass

        schema, _ = _build_parameters_schema(func)
        assert schema["properties"]["a"]["type"] == "string"
        assert schema["properties"]["b"]["type"] == "integer"
        assert schema["properties"]["c"]["type"] == "number"
        assert schema["properties"]["d"]["type"] == "boolean"
        assert schema["properties"]["e"]["type"] == "array"
        assert schema["properties"]["f"]["type"] == "object"

    def test_context_param_excluded(self):
        def func(context, query: str):
            pass

        schema, _ = _build_parameters_schema(func)
        assert "context" not in schema["properties"]
        assert "query" in schema["properties"]

    def test_optional_params_not_required(self):
        def func(name: str, greeting: str = "hi"):
            pass

        _, required = _build_parameters_schema(func)
        assert required == ["name"]

    def test_unannotated_param_defaults_to_string(self):
        def func(x):
            pass

        schema, _ = _build_parameters_schema(func)
        assert schema["properties"]["x"]["type"] == "string"


# ---------------------------------------------------------------------------
# @tool decorator — two code paths
# ---------------------------------------------------------------------------

class TestToolDecorator:
    def test_bare_decorator(self):
        @tool
        def greet(name: str):
            """Say hello."""
            return f"Hello {name}"

        assert isinstance(greet, Tool)
        assert greet.name == "greet"
        assert greet.description == "Say hello."

    def test_decorator_with_args(self):
        @tool(name="custom_name", description="Custom desc")
        def greet(name: str):
            """Original docstring."""
            return f"Hello {name}"

        assert greet.name == "custom_name"
        assert greet.description == "Custom desc"


# ---------------------------------------------------------------------------
# Tool.model_dump — OpenAI-compatible schema
# ---------------------------------------------------------------------------

def test_tool_model_dump_openai_format():
    @tool
    def greet(name: str):
        """Say hello."""
        pass

    schema = greet.model_dump()
    assert schema == {
        "type": "function",
        "function": {
            "name": "greet",
            "description": "Say hello.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": ""},
                },
                "required": ["name"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Tool.__call__ — sync/async dispatch and output type
# ---------------------------------------------------------------------------

class TestToolCall:
    @pytest.mark.asyncio
    async def test_sync_function(self):
        @tool
        def add(a: int, b: int):
            """Add numbers."""
            return a + b

        result = await add(a=2, b=3)
        assert isinstance(result, ToolCallResult)
        assert result.tool_name == "add"
        assert result.output == 5

    @pytest.mark.asyncio
    async def test_async_function(self):
        @tool
        async def fetch(url: str):
            """Fake fetch."""
            return {"status": 200, "url": url}

        result = await fetch(url="http://example.com")
        assert result.tool_name == "fetch"
        assert result.output == {
            "status": 200,
            "url": "http://example.com",
        }

    @pytest.mark.asyncio
    async def test_output_preserves_type(self):
        """Output is Any — not stringified like v1."""

        @tool
        def get_data():
            """Return structured data."""
            return {"key": [1, 2, 3]}

        result = await get_data()
        assert isinstance(result.output, dict)
        assert result.output["key"] == [1, 2, 3]
