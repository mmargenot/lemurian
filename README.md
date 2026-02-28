# `lemurian`

`lemurian` is a framework for building AI agents in Python.

## Architecture

| Layer | Components | Role |
|---|---|---|
| **Orchestration** | Swarm | Registers agents, creates handoff tool, manages handoff loop, tracks active agent |
| **Execution** | Runner | Mediates all transcript access: builds messages for provider, injects system prompt, appends responses and tool results, detects handoffs |
| **Interface** | Context, Tools, Capabilities, Provider | Tools use Context to access state/session/agent. Capabilities define collections of tools with their own state. Provider receives pre-built messages and schemas. |
| **Data** | Session, State | Session holds the transcript (ground truth). State holds typed application data. |

**Tool** — A `@tool`-decorated Python function exposed to an LLM. Schema is generated from the function signature. Receives a `Context` for accessing state, session, and agent.

**Capability** - A `Capability` groups tools into a coherent subcomponent of an `Agent` that can be applied to a single agent or across all Agents in a swarm. Tools within a Capability share access to state that they can manipulate, independently of the `Swarm` state.

**Agent** — Declarative Pydantic model bundling a system prompt, tools, model name, and provider. Agents don't run themselves — the Runner executes them.

**Runner** — The agent loop. Builds messages for the provider, dispatches tool calls, serializes results, detects handoffs. The only component that reads or writes the transcript.

**Session** — A single conversation transcript shared across all agents in a Swarm.

**State** — Typed Pydantic model for application data. Subclass it to add fields. Mutated in-place through `context.state`.

**Context** — Passed automatically to any tool declaring a `context` parameter. Holds references to session, state, and agent.

**Swarm** — Multi-agent orchestrator. Creates a dynamic `handoff` tool with an enum of available agents. Fresh-context handoffs: each agent sees only the handoff message onward.

**Provider** — Abstraction over LLM APIs. Implementations for OpenAI, OpenRouter, local vLLM, and Modal vLLM.

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

## Capabilities

A `Capability` groups related tools with their own encapsulated resources:

```python
import sqlite3
from lemurian.capability import Capability
from lemurian.tools import Tool, tool

class KnowledgeBaseCapability(Capability):
    def __init__(self, db_path: str = ":memory:"):
        super().__init__("knowledge_base")
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS notes (topic TEXT, content TEXT)"
        )

    def tools(self) -> list[Tool]:
        conn = self._conn

        @tool
        def store_note(topic: str, content: str):
            """Store a note under a topic."""
            conn.execute(
                "INSERT INTO notes VALUES (?, ?)", (topic, content)
            )
            conn.commit()
            return f"Stored note under '{topic}'."

        @tool
        def search_notes(query: str):
            """Search notes by keyword."""
            rows = conn.execute(
                "SELECT topic, content FROM notes WHERE content LIKE ?",
                (f"%{query}%",),
            ).fetchall()
            return [{"topic": r[0], "content": r[1]} for r in rows]

        return [store_note, search_notes]

    def on_attach(self, state):
        print("[KnowledgeBase] Connected.")

    def on_detach(self, state):
        self._conn.close()
```

Attach capabilities to individual agents or across an entire swarm:

```python
agent = Agent(
    name="researcher",
    system_prompt="You are a research assistant.",
    capabilities=[KnowledgeBaseCapability("notes.db")],
    model="gpt-4o-mini",
    provider=OpenAIProvider(),
)
```

## Model Providers

### OpenAI / OpenRouter

```python
from lemurian.provider import OpenAIProvider, OpenRouter

provider = OpenAIProvider()    # uses OPENAI_API_KEY env var
provider = OpenRouter()        # uses OPENROUTER_API_KEY env var
```

### vLLM

`VLLMProvider` connects to any vLLM endpoint — local or remote:

```python
from lemurian.provider import VLLMProvider

provider = VLLMProvider("localhost:8000")
provider = VLLMProvider("https://your-workspace-vllm-server.modal.run")
```

Serve a model locally:
```bash
uv run --group local vllm serve "Qwen/Qwen3-8B" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3
```

Refer to the [vLLM docs](https://qwen.readthedocs.io/en/latest/deployment/vllm.html#parsing-tool-calls) to pair the appropriate tool call parser with your model.

For serverless GPU inference on [Modal](https://modal.com), see [`scripts/modal_deploy.py`](scripts/modal_deploy.py).

## Testing

```bash
uv run pytest
uv run pytest --cov=lemurian --cov-report=term-missing
```

## Development

`lemurian` is work in progress for agent-based experimentation. Feel free to suggest issues or modifications.
