# `lemurian`

`lemurian` is a framework for building AI agents in Python.

## Architecture

| Layer | Components | Role |
|---|---|---|
| **Orchestration** | Swarm | Registers agents, creates handoff tool, manages handoff loop, tracks active agent |
| **Execution** | Runner | Mediates all transcript access: builds messages for provider, injects system prompt, appends responses and tool results, detects handoffs |
| **Interface** | Context, Tools, Provider | Tools use Context to access state/session/agent. Provider receives pre-built messages and schemas. |
| **Data** | Session, State | Session holds the transcript (ground truth). State holds typed application data. |

**Tool** — A `@tool`-decorated Python function exposed to an LLM. Schema is generated from the function signature. Receives a `Context` for accessing state, session, and agent.

**Agent** — Declarative Pydantic model bundling a system prompt, tools, model name, and provider. Agents don't run themselves — the Runner executes them.

**Runner** — The agent loop. Builds messages for the provider, dispatches tool calls, serializes results, detects handoffs. The only component that reads or writes the transcript.

**Session** — A single conversation transcript shared across all agents in a Swarm.

**State** — Typed Pydantic model for application data. Subclass it to add fields. Mutated in-place through `context.state`.

**Context** — Passed automatically to any tool declaring a `context` parameter. Holds references to session, state, and agent.

**Swarm** — Multi-agent orchestrator. Creates a dynamic `handoff` tool with an enum of available agents. Fresh-context handoffs: each agent sees only the handoff message onward.

**Provider** — Abstraction over LLM APIs. Implementations for OpenAI, OpenRouter, local vLLM, Modal vLLM, any OpenAI-compatible endpoint, and 100+ providers via LiteLLM.

## Defining Tools

Use the `@tool` decorator on any function. The schema is generated from the type hints and docstring:

```python
from lemurian.tools import tool

@tool
def lookup_customer(email: str):
    """Find a customer by their email address."""
    return {"customer_id": "CUST-123", "name": "Jane Doe"}
```

You can override the name and description:

```python
@tool(name="search", description="Search the knowledge base")
def kb_search(query: str, limit: int = 10):
    ...
```

Async functions work the same way:

```python
@tool
async def fetch_data(url: str):
    """Fetch data from a URL."""
    ...
```

To access state, session, or the current agent, add a `context` parameter. It is automatically excluded from the schema and injected at call time:

```python
from lemurian.context import Context

@tool
def update_counter(context: Context, amount: int):
    """Increment the counter in state."""
    context.state.counter += amount
    return context.state.counter
```

## Single Agent

```python
import asyncio
from lemurian.tools import tool
from lemurian.agent import Agent
from lemurian.runner import Runner
from lemurian.session import Session
from lemurian.state import State
from lemurian.message import Message, MessageRole
from lemurian.provider import OpenAIProvider

@tool
def greet(name: str):
    """Greet someone by name."""
    return f"Hello, {name}!"

agent = Agent(
    name="greeter",
    system_prompt="You are a friendly greeter. Use the greet tool when asked.",
    tools=[greet],
    model="gpt-4o-mini",
    provider=OpenAIProvider(),
)

async def main():
    session = Session(session_id="demo")
    session.transcript.append(
        Message(role=MessageRole.USER, content="Say hi to Alice")
    )
    result = await Runner().run(agent, session, State())
    print(result.last_message.content)

asyncio.run(main())
```

## Multi-Agent Swarm

```python
import asyncio
from lemurian.tools import tool
from lemurian.agent import Agent
from lemurian.swarm import Swarm
from lemurian.state import State
from lemurian.provider import OpenAIProvider

@tool
def check_invoice(invoice_id: str):
    """Check the status of an invoice."""
    return {"amount": 49.99, "status": "paid"}

provider = OpenAIProvider()

triage = Agent(
    name="triage",
    description="Routes customer requests to the right agent",
    system_prompt="Route customer requests to the appropriate agent.",
    model="gpt-4o-mini",
    provider=provider,
)

billing = Agent(
    name="billing",
    description="Handles billing and invoice questions",
    system_prompt="Help customers with billing inquiries.",
    tools=[check_invoice],
    model="gpt-4o-mini",
    provider=provider,
)

async def main():
    swarm = Swarm(agents=[triage, billing])
    result = await swarm.run("I have a question about my invoice", agent="triage")
    print(f"[{result.active_agent}] {result.last_message.content}")

asyncio.run(main())
```

The Swarm automatically creates a `handoff` tool for each agent with an enum of available targets. When triage calls `handoff(agent_name="billing", message="...")`, the Swarm switches context to the billing agent.

## Model Providers

### OpenAI / OpenRouter

```python
from lemurian.provider import OpenAIProvider, OpenRouter

provider = OpenAIProvider()    # uses OPENAI_API_KEY env var
provider = OpenRouter()        # uses OPENROUTER_API_KEY env var
```

### Local vLLM

Serve a model with vLLM:
```bash
uv run --extra local vllm serve "Qwen/Qwen3-8B" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3
```

```python
from lemurian.provider import VLLMProvider

provider = VLLMProvider(url="localhost", port=8000)
```

Refer to the [vLLM docs](https://qwen.readthedocs.io/en/latest/deployment/vllm.html#parsing-tool-calls) to pair the appropriate tool call parser with your model.

### Modal vLLM (Remote/Serverless)

Deploy vLLM to [Modal](https://modal.com) for serverless GPU inference:

```bash
uv sync --group modal
modal setup
modal deploy examples/modal_vllm.py
```

```python
from lemurian.provider import ModalVLLMProvider

provider = ModalVLLMProvider(
    endpoint_url="https://your-workspace--lemurian-vllm-vllmserver-serve.modal.run"
)
```

Customize deployments with environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-8B` | HuggingFace model ID |
| `GPU_TYPE` | `A100` | Modal GPU type (A10G, A100, H100) |
| `GPU_COUNT` | `1` | Number of GPUs for tensor parallelism |
| `MAX_MODEL_LEN` | `8192` | Maximum sequence length |

Pre-download model weights for faster cold starts:
```bash
modal run examples/modal_vllm.py --download
```

### Any OpenAI-Compatible Server

Works with Ollama, Together, Groq, Fireworks, LM Studio, or any server with an OpenAI-compatible `/chat/completions` endpoint:

```python
from lemurian.provider import OpenAICompatibleProvider

# Ollama
provider = OpenAICompatibleProvider(
    base_url="http://localhost:11434/v1"
)

# Together AI
provider = OpenAICompatibleProvider(
    base_url="https://api.together.xyz/v1",
    api_key="your-together-key",
)

# Groq
provider = OpenAICompatibleProvider(
    base_url="https://api.groq.com/openai/v1",
    api_key="your-groq-key",
)
```

### LiteLLM (100+ Providers)

Access Anthropic, Gemini, Cohere, Bedrock, Mistral, Azure, and [100+ other providers](https://docs.litellm.ai/docs/providers) through LiteLLM. Install it separately:

```bash
pip install litellm
# or: uv sync --group litellm
```

```python
from lemurian.provider import LiteLLMProvider

# Anthropic
provider = LiteLLMProvider()  # uses ANTHROPIC_API_KEY env var
agent = Agent(..., model="anthropic/claude-sonnet-4-20250514", provider=provider)

# Google Gemini
provider = LiteLLMProvider()  # uses GEMINI_API_KEY env var
agent = Agent(..., model="gemini/gemini-2.0-flash", provider=provider)

# Explicit API key
provider = LiteLLMProvider(api_key="sk-...")
```

## Testing

```bash
uv run pytest
uv run pytest --cov=lemurian --cov-report=term-missing
```

## Development

`lemurian` is work in progress for agent-based experimentation. Feel free to suggest issues or modifications.
