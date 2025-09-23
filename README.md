# `lemurian`

`lemurian` is a framework for building artificial intelligence (AI) agents in Python.

Create an Agent by subclassing `lemurian.agents.Agent` and implementing the `get_all_tools` function. This function could return a collection of tools that broadly have the shape:

```
from lemurian.tools import Tool
from lemurian.session import Session


@Tool
def my_first_tool(session: Session, param_1: int) -> int:
    """
    Detailed documentation around `my_first_tool` using Google-style
    docstrings. Make sure to _exclude_ `session` here, as it is a "magic"
    variable that is ignored by the `Tool` decorator.

    Args:
        param_1: Description of what `param_1` is and what it's for.

    Returns:
        return_1: Description of object / information that is returned by the
            function.
    """
    pass
```

It is important to have a detailed docstring, as this description is sent to the tool-calling provider to give context to the large language model (LLM) that will call your tool.

## Defining Tools

Within the the `lemurian` agent framework, there are two key classes:
- `lemurian.tools.Tool`
- `lemurian.session.Session`

The `Tool` class is used as a Python decorator that does a lot of heavy lifting for us. Adding `@Tool` to a function turns that function into a `pydantic.BaseModel` class whose schema is automatically generated from the type hints and docstrings that you provide. The end result is that you don't need to define that schema yourself.

The `Session` class is a way to preserve state within your agent system. The default `respond` implementation of an `Agent` looks at the `Message` transcript that is stored in the state and uses that to generate the tool calls and response from your model provider.

You can extend this state object to add other information that you may need within tools, such as metadata about a user for whom an agent is running a completion or an object that different tools act upon. The `session` parameter in a tool is automatically ignored by the `Tool` decorator when constructing the schema.

## Model Providers
You can run `lemurian` agents using vLLM. Make sure that you are serving the model that you want via vLLM on the machine, like so:
```
uv run --extra local vllm serve "Qwen/Qwen3-8B" --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
```

Then you can define `VLLMProvider` that points to the address of the served model and define an agent with that provider.

Make sure to refer to the [vLLM docs](https://qwen.readthedocs.io/en/latest/deployment/vllm.html#parsing-tool-calls) to pair the appropriate tool call parser with the model that you want to serve. Additionally, if the model performs reasoning (or "thinking"), make sure to include a **reasoning parser** to remove the thinking sections if you don't expressly want them in your transcripts.

# Development

`lemurian` is work in progress for agent-based experimentation. Feel free to suggest issues or modifications.
