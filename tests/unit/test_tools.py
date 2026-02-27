import inspect

import pytest

from lemurian.tools import (
    Tool,
    ToolCallResult,
    _build_parameters_schema,
    _parse_param_descriptions,
    tool,
)


# ---------------------------------------------------------------------------
# Schema generation (_build_parameters_schema)
# ---------------------------------------------------------------------------


class TestBuildParametersSchema:
    def test_python_types_map_to_json_schema_types(self):
        def func(a: str, b: int, c: float, d: bool, e: list, f: dict):
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
# Docstring param description parsing (_parse_param_descriptions)
# ---------------------------------------------------------------------------


class TestParseParamDescriptions:
    def test_google_style(self):
        def func(name: str, age: int):
            """Do something.

            Args:
                name: The user's name.
                age: The user's age.
            """

        descs = _parse_param_descriptions(func)
        assert descs == {
            "name": "The user's name.",
            "age": "The user's age.",
        }

    def test_google_style_with_type_in_docstring(self):
        def func(name, age):
            """Do something.

            Args:
                name (str): The user's name.
                age (int): The user's age.
            """

        descs = _parse_param_descriptions(func)
        assert descs == {
            "name": "The user's name.",
            "age": "The user's age.",
        }

    def test_sphinx_rest_style(self):
        def func(name: str, age: int):
            """Do something.

            :param name: The user's name.
            :param age: The user's age.
            """

        descs = _parse_param_descriptions(func)
        assert descs == {
            "name": "The user's name.",
            "age": "The user's age.",
        }

    def test_numpy_style(self):
        def func(name: str, age: int):
            """Do something.

            Parameters
            ----------
            name : str
                The user's name.
            age : int
                The user's age.
            """

        descs = _parse_param_descriptions(func)
        assert descs == {
            "name": "The user's name.",
            "age": "The user's age.",
        }

    def test_no_docstring(self):
        def func(x: str):
            pass

        assert _parse_param_descriptions(func) == {}

    def test_docstring_without_params_section(self):
        def func(x: str):
            """Just a summary."""

        assert _parse_param_descriptions(func) == {}

    def test_multiline_description(self):
        def func(query: str):
            """Search.

            Args:
                query: The search query string.
                    Supports boolean operators
                    and wildcards.
            """

        descs = _parse_param_descriptions(func)
        assert descs == {
            "query": (
                "The search query string.\n"
                "Supports boolean operators\n"
                "and wildcards."
            ),
        }

    def test_context_param_in_docstring_ignored_by_schema(self):
        """Documented context param doesn't leak into the schema."""

        def func(context, query: str):
            """Search.

            Args:
                context: The runtime context.
                query: The search query.
            """

        schema, _ = _build_parameters_schema(func)
        assert "context" not in schema["properties"]
        assert schema["properties"]["query"]["description"] == (
            "The search query."
        )

    def test_undocumented_param_gets_empty_description(self):
        def func(a: str, b: int):
            """Do something.

            Args:
                a: Documented param.
            """

        schema, _ = _build_parameters_schema(func)
        assert schema["properties"]["a"]["description"] == (
            "Documented param."
        )
        assert schema["properties"]["b"]["description"] == ""


# ---------------------------------------------------------------------------
# Param descriptions flow through to @tool schema
# ---------------------------------------------------------------------------


class TestToolParamDescriptions:
    def test_google_descriptions_in_tool_schema(self):
        @tool
        def search(query: str, max_results: int = 10):
            """Search the knowledge base.

            Args:
                query: The search query string.
                max_results: Maximum results to return.
            """

        schema = search.model_dump()
        params = schema["function"]["parameters"]["properties"]
        assert params["query"]["description"] == ("The search query string.")
        assert params["max_results"]["description"] == (
            "Maximum results to return."
        )

    def test_rest_descriptions_in_tool_schema(self):
        @tool
        def lookup(user_id: int):
            """Look up a user.

            :param user_id: The ID of the user to look up.
            """

        schema = lookup.model_dump()
        params = schema["function"]["parameters"]["properties"]
        assert params["user_id"]["description"] == (
            "The ID of the user to look up."
        )

    def test_no_params_section_still_works(self):
        @tool
        def greet(name: str):
            """Say hello."""

        schema = greet.model_dump()
        params = schema["function"]["parameters"]["properties"]
        assert params["name"]["description"] == ""


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


# ---------------------------------------------------------------------------
# Tool.bind — partial application with schema pruning
# ---------------------------------------------------------------------------


class TestToolBind:
    def test_bind_removes_param_from_schema(self):
        @tool
        def search(db: str, query: str):
            """Search."""
            return f"{db}:{query}"

        bound = search.bind(db="my_db")
        schema = bound.model_dump()
        props = schema["function"]["parameters"]["properties"]
        assert "db" not in props
        assert "query" in props

    def test_bind_removes_param_from_required(self):
        @tool
        def search(db: str, query: str):
            """Search."""
            return f"{db}:{query}"

        bound = search.bind(db="my_db")
        required = bound.parameters_schema["required"]
        assert "db" not in required
        assert "query" in required

    def test_bind_preserves_name_and_description(self):
        @tool(name="custom_search", description="Custom desc")
        def search(db: str, query: str):
            """Search."""
            return f"{db}:{query}"

        bound = search.bind(db="my_db")
        assert bound.name == "custom_search"
        assert bound.description == "Custom desc"

    @pytest.mark.asyncio
    async def test_bound_tool_is_callable(self):
        @tool
        def search(db: str, query: str):
            """Search."""
            return f"{db}:{query}"

        bound = search.bind(db="my_db")
        result = await bound(query="hello")
        assert result.tool_name == "custom_search" or result.tool_name == "search"
        assert result.output == "my_db:hello"

    @pytest.mark.asyncio
    async def test_bound_tool_with_async_func(self):
        @tool
        async def fetch(base_url: str, path: str):
            """Fetch a URL."""
            return f"{base_url}/{path}"

        bound = fetch.bind(base_url="https://example.com")
        result = await bound(path="api/data")
        assert result.output == "https://example.com/api/data"

    def test_bind_does_not_mutate_original(self):
        @tool
        def search(db: str, query: str):
            """Search."""
            return f"{db}:{query}"

        original_props = set(search.parameters_schema["properties"].keys())
        _ = search.bind(db="my_db")
        assert set(search.parameters_schema["properties"].keys()) == original_props

    def test_chained_bind(self):
        @tool
        def query(db: str, table: str, column: str):
            """Query a column."""
            return f"{db}.{table}.{column}"

        bound = query.bind(db="my_db").bind(table="users")
        props = bound.parameters_schema["properties"]
        assert "db" not in props
        assert "table" not in props
        assert "column" in props
        assert bound.parameters_schema["required"] == ["column"]

    @pytest.mark.asyncio
    async def test_chained_bind_callable(self):
        @tool
        def query(db: str, table: str, column: str):
            """Query a column."""
            return f"{db}.{table}.{column}"

        bound = query.bind(db="my_db").bind(table="users")
        result = await bound(column="email")
        assert result.output == "my_db.users.email"

    def test_bind_with_context_param_preserved(self):
        """Context parameter stays in the signature for runner injection."""

        @tool
        def stateful(context, db: str, query: str):
            """Stateful search."""
            return f"{db}:{query}"

        bound = stateful.bind(db="my_db")
        # context should still be visible in the func signature
        sig = inspect.signature(bound.func)
        assert "context" in sig.parameters
        # but not in the schema (excluded by @tool)
        assert "context" not in bound.parameters_schema["properties"]
        # db should be gone from the schema (hidden from LLM)
        assert "db" not in bound.parameters_schema["properties"]

    def test_bind_optional_param(self):
        @tool
        def search(db: str, query: str, limit: int = 10):
            """Search."""
            return f"{db}:{query}:{limit}"

        bound = search.bind(db="my_db")
        required = bound.parameters_schema["required"]
        assert "db" not in required
        assert "query" in required
        assert "limit" not in required
        assert "limit" in bound.parameters_schema["properties"]
